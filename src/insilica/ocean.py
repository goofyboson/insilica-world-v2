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
The wind stress joins the air's bottom layer to the mixed layer every air step, exactly (`couple`). Each layer carries
a salinity S too (decision #76): rain, evaporation and the sea ice's growth and melt freshen or salt the mixed layer (a
virtual salt flux at S_ref), and the column's convection and mixing use the buoyancy of T and S, with an expansion
coefficient α + α2·(T − 0 °C) that falls toward freezing (0.52e-4/K at 0 °C, 3.3e-4/K at 28 °C) as in seawater.
"""
import math

import cupy as cp
import numpy as np

from .constants import GRAV
from .gpu import NULL, grid_dims
from .layers import Layers
from .transport import Transport

ALPHA_REF = 2.0e-4                                              # 1/K: dTmin's buoyancy (decision #68)
OCEAN = dict(depth=4000.0, h_mixed=100.0, h_thermo=600.0, T_thermo=293.15, T_deep=273.15, alpha=0.52e-4, alpha2=1.0e-5, beta=7.6e-4, S_ref=35.0, rho=1025.0, cw=3990.0, dt=3600.0, mix_dt=86400.0,
             gprime=(0.016, 0.016),
             hmin=10.0, hmax=1000.0, dTmin=0.1, m_kt=0.5, n_kt=0.2, visc4_U=0.5, bottom_stress=2.5e-4,
             interface_stress=5.0e-4, thin=2.0, walls=(), sw_R=0.58, sw_z1=0.35, sw_z2=23.0)


def wall_mask(g, walls):
    """(E,) mask, 0 on the edges whose two cells lie on opposite sides of a wall at one of the longitudes (degrees)."""
    m = np.ones(g.E)
    for lon in walls:
        d = np.angle(np.exp(1j * (g.lon - math.radians(lon))))        # longitude from the wall, in (−π, π]
        a, b = d[g.eI], d[g.eJ]
        m[(np.abs(a) < math.pi / 2) & (np.abs(b) < math.pi / 2) & ((a >= 0) != (b >= 0))] = 0.0   # a cell on the wall is east of it
    return m


class Ocean:
    def __init__(self, dg, cfg=None, Tsurf=None, state=None, land=None, c_land=2.0e6):
        """cfg overrides OCEAN; Tsurf (N, K) starts the mixed layer's temperature (from the air case's surface);
        state = (h, u, T) restores one; land (N, 0/1): cells with no water (#84), walled off, their T[0] the ground's
        temperature with heat capacity c_land."""
        self.dg = dg
        self.cfg = c = dict(OCEAN, **(cfg or {}))
        N, E = dg.N, dg.E
        g = dg.host
        from .world import visc4                                 # (world imports this module lazily: no cycle)
        nu4 = visc4(g, c['visc4_U'])
        self.b = cp.full(N, -c['depth'])
        # the thermocline's turbulent stress carries the wind's momentum down (decision #69): c ~ C_d |du| ~ 1e-3 x 0.5
        # m/s; without it a mixed layer thinned to its floor by equatorial upwelling ran away toward the wind's speed
        # g' per interface (decision #64): g · 2e-4/K · ΔT for 8 K steps (≈ 20 → 12 → 4 °C), baroclinic gravity waves 3.1 and 1.2 m/s
        L = self.L = 3
        assert len(c['gprime']) == L - 1
        # thin walls (decision #79): at each longitude in walls (degrees), pole to pole, the edges between the cells on its two
        # sides carry no water — a basin's coasts, with no land cells (the air still sees open water there)
        self.wall = wall_mask(g, c['walls']) if len(c['walls']) else None
        self.c_land = c_land
        self.land = None
        if land is not None:
            m = self.wall if self.wall is not None else np.ones(g.E)
            m[(land[g.eI] > 0.5) | (land[g.eJ] > 0.5)] = 0.0
            self.wall = m
            self.land = cp.asarray(np.asarray(land, dtype=float))
        self.core = Layers(dg, L, b=self.b, gprime=list(c['gprime']), stress=[0.0] * (L - 1) + [c['bottom_stress']], nu4=nu4,
                           floor=c['hmin'], vmix=c['interface_stress'], thin=c['thin'], wall=self.wall)
        if nu4:
            lam = self.core.lap_max()
            assert nu4 * lam * lam * c['dt'] <= 1.0, 'ocean nu4 unstable at its step'
        self.tr = Transport(dg, layers=L)
        if state is None:
            # the thermocline water at T_thermo, no warmer than the surface above it nor colder than the abyss: started warm,
            # near where the subtropics' mixing pumps it (decision #75), its drift stays slow (~0.05 K/yr); the abyss at the
            # freezing point, where polar winters would take it over centuries, so polar ice can form (decision #77)
            Ts = np.asarray(Tsurf, dtype=float)
            h = np.stack([np.full(N, c['h_mixed']), np.full(N, c['h_thermo']), np.full(N, c['depth'] - c['h_mixed'] - c['h_thermo'])])
            u = np.zeros((L, E))
            T = np.stack([Ts, np.clip(Ts, c['T_deep'], c['T_thermo']), np.full(N, c['T_deep'])])
            S = np.full((L, N), c['S_ref'])
        else:
            h, u, T = state[:3]
            S = state[3] if len(state) > 3 else np.full((L, N), c['S_ref'])
        self.h, self.u, self.T = cp.asarray(h, dtype=float), cp.asarray(u, dtype=float), cp.asarray(T, dtype=float)
        self.core.mask(self.u, L)
        self.S = cp.asarray(S, dtype=float)                     # salinity (psu, decision #76)
        self.M = self.h * cp.asarray(g.areaCell)[None, :]
        self.nfast = int(math.ceil(1.2 * self.core.substeps(self.h, c['dt'])))
        z = lambda *s: cp.zeros(s)
        self._Fm, self._M1 = z(L, E), z(L, N)
        self.cap = z(N)                                         # ρ_w c_w h₀, for the air's surface physics
        self.qocean, self.qsw = z(N), z(N)                      # W/m² into the mixed layer, and its sunlight
        self.q_acc, self.sun_acc = z(N), z(N)                   # their non-solar part and the sunlight, summed for the mixing (J/m²)
        self.fw_acc, self.ice_prev = z(N), z(N)                 # fresh water in (kg/m²), summed; the sea ice at the last step
        self.tau_acc, self.taum, self.diss = z(E), z(N), z(E)   # Σ τ² dt on edges; the step's rms stress at cells
        self.wx = z(L - 1, N)
        # The mixed layer is mixed on daily-mean forcing (decision #67): mixed hourly, the day's sunshine shoaled it at
        # once to the Monin-Obukhov depth and folded its warm water into the deep layer, and the night's slow
        # entrainment never won it back (the bulk two-layer model's known diurnal pathology).
        ratio = c['mix_dt'] / c['dt']
        assert abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1, 'mix_dt must be a whole number of ocean steps'
        self.mix_every = int(round(ratio))
        k = lambda n: dg.kernel(n, 'ocean.cu')
        self.k = {n: k(n) for n in ('couple_stress', 'ocn_column', 'ocn_cap', 'ocn_overturn', 'ocn_pen', 'ocn_fresh', 'ocn_xch_salt',
                                    'ocn_heat')}
        self.conv = None                                        # the currents' heat convergence, when tracked
        self.k['cell_speed'] = dg.kernel('cell_speed', 'air.cu')
        self.update_cap()

    def track_heat_transport(self):
        """From now on, sum the currents' heat convergence per cell in self.conv (J/m², relative to 0 °C; decision #80):
        the ocean's heat transport is its integral over latitude. In a world, use World.track_ocean_heat_transport, which
        also recaptures the step's graph."""
        self.conv = cp.zeros(self.dg.N)

    def _heat(self, sign):
        g = self.dg
        self.k['ocn_heat'](*grid_dims(g.N), (np.int32(g.N), np.int32(self.L), self.h, self.T, g.dtype.type(self.cfg['rho'] * self.cfg['cw']),
                                            g.dtype.type(sign), self.conv))

    def update_cap(self):
        g = self.dg
        self.k['ocn_cap'](*grid_dims(g.N), (np.int32(g.N), self.h, g.dtype.type(self.cfg['rho'] * self.cfg['cw']), self.cap, self._land(),
                                            g.dtype.type(self.c_land)))

    def couple(self, air_h, air_u, c, dt):
        """The wind stress over one air step (exactly, momentum conserved); the kinetic energy it removed is left in
        self.diss (J/m² per edge) for the air to take as heat."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        self.k['couple_stress'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(air_h.shape[0]), air_h, air_u, self.h, self.u,
                                                  g.eI, g.eJ, R(c), R(self.cfg['rho']), R(dt), self.diss, self.tau_acc,
                                                  self.core.wall if self.core.wall is not None else NULL))

    def _land(self):
        return self.land if self.land is not None else NULL

    def eos(self):
        """The column physics' equation of state as kernel arguments: g, α, α2, β (decision #76)."""
        c, R = self.cfg, self.dg.dtype.type
        return R(GRAV), R(c['alpha']), R(c['alpha2']), R(c['beta'])

    def accumulate(self, dt, evap=None, rain=None, ice=None, mouth=None):
        """Pass the air step's sunlight below the mixed layer's base to the layers below, add the fluxes the mixed layer
        keeps to the mixing's sums, and freshen or salt it with rain, evaporation and the sea ice's change (kg/m²/s, kg/m²)."""
        g, c, R = self.dg, self.cfg, self.dg.dtype.type
        if evap is not None:
            self.k['ocn_fresh'](*grid_dims(g.N), (np.int32(g.N), self.h, self.S, evap, rain, ice, self.ice_prev, R(c['S_ref']),
                                                 R(c['rho']), R(dt), self.fw_acc, self._land(), mouth if mouth is not None else NULL))
        self.k['ocn_pen'](*grid_dims(g.N), (np.int32(g.N), np.int32(self.L), self.h, self.T, self.qocean, self.qsw, R(c['rho'] * c['cw']),
                                           R(c['sw_R']), R(c['sw_z1']), R(c['sw_z2']), R(dt), self.q_acc, self.sun_acc, self._land()))

    def step(self, mix=True):
        """One ocean step: the layers move and T is carried; with mix, the mixed layer then overturns / entrains /
        shoals on the forcing summed since the last mixing (mix_dt)."""
        g, c, i32, R = self.dg, self.cfg, np.int32, self.dg.dtype.type
        self.k['ocn_overturn'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.T, self.S) + self.eos() + (self._land(),))
        if self.conv is not None:
            self._heat(-1.0)
        self.core.step(self.h, self.u, c['dt'], self.nfast, self._Fm)
        self.tr.mass_after(self.M, self._Fm, self._M1)
        self.tr.step(self.M, self._M1, self._Fm, self.T, order=3)
        if self.conv is not None:
            self._heat(1.0)
        self.tr.step(self.M, self._M1, self._Fm, self.S, order=3)
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
        self.fw_acc /= dt
        self.k['ocn_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.T, self.S, self.taum, self.q_acc, self.fw_acc,
                                               self.sun_acc) + self.eos() + (R(c['S_ref']), R(ALPHA_REF), R(c['rho']), R(c['cw']),
                                               R(c['m_kt']), R(c['n_kt']), R(c['hmin']), R(c['hmax']), R(c['dTmin']), R(c['sw_R']),
                                               R(c['sw_z1']), R(c['sw_z2']), R(dt), self.wx, self._land()))
        self.k['ocn_xch_salt'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.wx, self.h, self.S))
        self.core.exchange(self.wx, self.h, self.u, self.M, self.T)
        self.tau_acc.fill(0)
        self.q_acc.fill(0)
        self.sun_acc.fill(0)
        self.fw_acc.fill(0)
        self.update_cap()

    def heat(self):
        """The ocean's heat content (J, relative to 0 K): ρ_w c_w Σ A h T."""
        A = cp.asarray(self.dg.host.areaCell)
        rc = self.cfg['rho'] * self.cfg['cw']
        if self.land is None:
            return float(cp.sum(A * cp.sum(self.h * self.T, axis=0))) * rc
        return float(cp.sum(A * ((1 - self.land) * rc * cp.sum(self.h * self.T, axis=0) + self.land * self.c_land * self.T[0])))

    def salt(self):
        """The ocean's salt (kg): ρ_w Σ A h S / 1000."""
        A = cp.asarray(self.dg.host.areaCell)
        if self.land is not None:
            A = A * (1 - self.land)
        return float(cp.sum(A * cp.sum(self.h * self.S, axis=0))) * self.cfg['rho'] / 1000.0

    def momentum(self):
        """ρ_w Σ_e l_e d_e Σ_k h_e,k u_k (the column momentum measure of World.totals, in mass units)."""
        g = self.dg.host
        he = 0.5 * (self.h[:, g.eI] + self.h[:, g.eJ])
        return float(cp.sum(cp.sum(he * self.u, axis=0) * cp.asarray(g.le * g.de))) * self.cfg['rho']
