"""The ocean under the air (stage V2-3, kernels/ocean.cu; decisions #64–#74).

Three Boussinesq layers on the air's grid — the mixed layer (k = 0), the thermocline's water below it and the abyss
(decision #71) — over a flat bottom. Each interface has a fixed density step, g' (decision #64): the interfaces move
freely (wind-driven Ekman pumping, upwelling, Kelvin and Rossby waves), and each layer's temperature T (K) is heat only,
carried with the water. (Density following T within a layer, tried first, grew a spurious instability in thick, weakly
stratified convective columns: the inhomogeneous-layer model's known weakness.) The mixed layer's temperature is the
surface the air sees: the air's column physics heats it (sunlight, longwave, sensible heat, evaporation) and freezes or
melts sea ice over it with its heat capacity ρ_w c_w h₀. The ocean steps on its own interval (1 h; the air's fluxes are
summed over it, nothing is lost): its layers move (the same time-split RK3 core as the air), T is carried by FCT
transport, and the column then mixes heat wherever a layer is colder than the one below (convection; the interfaces stay),
and the mixed layer entrains thermocline water by wind stirring and surface cooling, or shoals in calm, warm weather
(Kraus–Turner, on the temperature jump, with the sunlight counted where it is absorbed, decision #74). Sunlight passing
the mixed layer's base heats the layers below (decision #70).
The wind stress joins the air's bottom layer to the mixed layer every air step, exactly (`couple`). Salinity is not
modelled yet; the water the air evaporates and rains is a budget only (V2-4).
"""
import math

import cupy as cp
import numpy as np

from .constants import GRAV
from .gpu import grid_dims
from .layers import Layers
from .transport import Transport

OCEAN = dict(depth=4000.0, h_mixed=100.0, h_thermo=600.0, T_thermo=285.15, T_deep=277.15, alpha=2.0e-4, rho=1025.0, cw=3990.0, dt=3600.0, mix_dt=86400.0,
             gprime=(0.016, 0.016),
             hmin=10.0, hmax=1000.0, dTmin=0.1, m_kt=0.5, n_kt=0.2, visc4_U=0.5, bottom_stress=2.5e-4,
             interface_stress=5.0e-4, thin=2.0, sw_R=0.58, sw_z1=0.35, sw_z2=23.0)


class Ocean:
    def __init__(self, dg, cfg=None, Tsurf=None, state=None):
        """cfg overrides OCEAN; Tsurf (N, K) starts the mixed layer's temperature (from the air case's surface);
        state = (h, u, T) restores one."""
        self.dg = dg
        self.cfg = c = dict(OCEAN, **(cfg or {}))
        N, E = dg.N, dg.E
        g = dg.host
        from .world import visc4                                 # (world imports this module lazily: no cycle)
        nu4 = visc4(g, c['visc4_U'])
        self.b = cp.full(N, -c['depth'])
        # the thermocline's turbulent stress carries the wind's momentum down (decision #69): c ~ C_d |du| ~ 1e-3 x 0.5
        # m/s; without it a mixed layer thinned to its floor by equatorial upwelling ran away toward the wind's speed
        # g' per interface (decision #64): g α ΔT for 8 K steps (≈ 20 → 12 → 4 °C), baroclinic gravity waves 3.1 and 1.2 m/s
        L = self.L = 3
        assert len(c['gprime']) == L - 1
        self.core = Layers(dg, L, b=self.b, gprime=list(c['gprime']), stress=[0.0] * (L - 1) + [c['bottom_stress']], nu4=nu4,
                           floor=c['hmin'], vmix=c['interface_stress'], thin=c['thin'])
        if nu4:
            lam = self.core.lap_max()
            assert nu4 * lam * lam * c['dt'] <= 1.0, 'ocean nu4 unstable at its step'
        self.tr = Transport(dg, layers=L)
        if state is None:
            # the thermocline water at T_thermo, no warmer than the surface above it nor colder than the abyss
            Ts = np.asarray(Tsurf, dtype=float)
            h = np.stack([np.full(N, c['h_mixed']), np.full(N, c['h_thermo']), np.full(N, c['depth'] - c['h_mixed'] - c['h_thermo'])])
            u = np.zeros((L, E))
            T = np.stack([Ts, np.clip(Ts, c['T_deep'], c['T_thermo']), np.full(N, c['T_deep'])])
        else:
            h, u, T = state
        self.h, self.u, self.T = cp.asarray(h, dtype=float), cp.asarray(u, dtype=float), cp.asarray(T, dtype=float)
        self.M = self.h * cp.asarray(g.areaCell)[None, :]
        self.nfast = int(math.ceil(1.2 * self.core.substeps(self.h, c['dt'])))
        z = lambda *s: cp.zeros(s)
        self._Fm, self._M1 = z(L, E), z(L, N)
        self.cap = z(N)                                         # ρ_w c_w h₀, for the air's surface physics
        self.qocean, self.qsw = z(N), z(N)                      # W/m² into the mixed layer, and its sunlight
        self.q_acc, self.sun_acc = z(N), z(N)                   # their non-solar part and the sunlight, summed for the mixing (J/m²)
        self.tau_acc, self.taum, self.diss = z(E), z(N), z(E)   # Σ τ² dt on edges; the step's rms stress at cells
        self.wx = z(L - 1, N)
        self.galpha = GRAV * c['alpha']
        # The mixed layer is mixed on daily-mean forcing (decision #67): mixed hourly, the day's sunshine shoaled it at
        # once to the Monin-Obukhov depth and folded its warm water into the deep layer, and the night's slow
        # entrainment never won it back (the bulk two-layer model's known diurnal pathology).
        ratio = c['mix_dt'] / c['dt']
        assert abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1, 'mix_dt must be a whole number of ocean steps'
        self.mix_every = int(round(ratio))
        k = lambda n: dg.kernel(n, 'ocean.cu')
        self.k = {n: k(n) for n in ('couple_stress', 'ocn_column', 'ocn_cap', 'ocn_overturn', 'ocn_pen')}
        self.k['cell_speed'] = dg.kernel('cell_speed', 'air.cu')
        self.update_cap()

    def update_cap(self):
        g = self.dg
        self.k['ocn_cap'](*grid_dims(g.N), (np.int32(g.N), self.h, g.dtype.type(self.cfg['rho'] * self.cfg['cw']), self.cap))

    def couple(self, air_h, air_u, c, dt):
        """The wind stress over one air step (exactly, momentum conserved); the kinetic energy it removed is left in
        self.diss (J/m² per edge) for the air to take as heat."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        self.k['couple_stress'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(air_h.shape[0]), air_h, air_u, self.h, self.u,
                                                  g.eI, g.eJ, R(c), R(self.cfg['rho']), R(dt), self.diss, self.tau_acc))

    def accumulate(self, dt):
        """Pass the air step's sunlight below the mixed layer's base to the deep layer, and add the flux the mixed layer
        keeps to the mixing's sum."""
        g, c, R = self.dg, self.cfg, self.dg.dtype.type
        self.k['ocn_pen'](*grid_dims(g.N), (np.int32(g.N), np.int32(self.L), self.h, self.T, self.qocean, self.qsw, R(c['rho'] * c['cw']),
                                           R(c['sw_R']), R(c['sw_z1']), R(c['sw_z2']), R(dt), self.q_acc, self.sun_acc))

    def step(self, mix=True):
        """One ocean step: the layers move and T is carried; with mix, the mixed layer then overturns / entrains /
        shoals on the forcing summed since the last mixing (mix_dt)."""
        g, c, i32, R = self.dg, self.cfg, np.int32, self.dg.dtype.type
        self.k['ocn_overturn'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.T))
        self.core.step(self.h, self.u, c['dt'], self.nfast, self._Fm)
        self.tr.mass_after(self.M, self._Fm, self._M1)
        self.tr.step(self.M, self._M1, self._Fm, self.T, order=3)
        cp.copyto(self.M, self._M1)
        self.update_cap()
        if not mix:
            return
        dt = c['mix_dt']
        # the mixing interval's rms stress at cells and mean surface flux
        self.tau_acc *= 1.0 / dt                               # in place: the tick is captured as a CUDA graph
        cp.sqrt(self.tau_acc, out=self.tau_acc)
        self.k['cell_speed'](*grid_dims(g.N), (i32(g.N), self.tau_acc, g.cellEdge, g.le, g.de, g.areaCell, self.taum))
        self.q_acc /= dt
        self.sun_acc /= dt
        self.k['ocn_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.T, self.taum, self.q_acc, self.sun_acc, R(self.galpha),
                                               R(c['rho']), R(c['cw']), R(c['m_kt']), R(c['n_kt']), R(c['hmin']), R(c['hmax']), R(c['dTmin']),
                                               R(c['sw_R']), R(c['sw_z1']), R(c['sw_z2']), R(dt), self.wx))
        self.core.exchange(self.wx, self.h, self.u, self.M, self.T)
        self.tau_acc.fill(0)
        self.q_acc.fill(0)
        self.sun_acc.fill(0)
        self.update_cap()

    def heat(self):
        """The ocean's heat content (J, relative to 0 K): ρ_w c_w Σ A h T."""
        A = cp.asarray(self.dg.host.areaCell)
        return float(cp.sum(A * cp.sum(self.h * self.T, axis=0))) * self.cfg['rho'] * self.cfg['cw']

    def momentum(self):
        """ρ_w Σ_e l_e d_e Σ_k h_e,k u_k (the column momentum measure of World.totals, in mass units)."""
        g = self.dg.host
        he = 0.5 * (self.h[:, g.eI] + self.h[:, g.eJ])
        return float(cp.sum(cp.sum(he * self.u, axis=0) * cp.asarray(g.le * g.de))) * self.cfg['rho']
