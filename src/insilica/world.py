"""A world: a stack of layers on the layered core (layers.py) carrying one tracer per layer, on the GPU.

State (all 64-bit, layer-major): h (L, N) thickness, u (L, E) normal wind, M (L, N) layer volume per cell
(h·A, moved by the same fluxes as h so the tracer budget closes exactly), psi (L, N) tracer per volume.
A tick is one time-split RK3 step of the layers followed by one FCT transport step with that step's fluxes.
Cases and their settings are in cases.py; `params` overrides a case's settings:
  dt, gprime (list, L − 1), floor (m, default 1), drag (per layer, 1/s), stress (per layer, m/s: linear stress
  coefficient, deceleration c/h), nu4 (m⁴/s) or visc4_U (m/s: ν4 = U·d³/π⁴
  with d the mean cell spacing, the grid-scale wave then damps at U/d), rotation (multiplies f), beta (transport),
  div_damp (the fast mode's divergence damping, default 0.1), nfast (fast sub-steps per step; default from √(gH)).
"""
import hashlib
import math

import cupy as cp
import numpy as np

from . import cases
from . import grid as gridmod
from .cases import PRESETS, cosine_bell, default_dt, edge_east_north, initial_state  # noqa: F401  (re-exported)
from .gpu import DeviceGrid, dsum, grid_dims, source_hash
from .io import save as savefile
from .atmosphere import AirPhysics
from .constants import CP_DRY, GRAV, KAPPA, P0
from .layers import DIV_DAMP, Layers
from .transport import BETA, Transport

FIELDS = ('h', 'z', 'eta', 'psi', 'speed', 'vort', 'theta', 'T', 'Ts', 'Tsurf', 'ps', 'q', 'rh', 'rain', 'evap', 'ice',
          'sst', 'sss', 'mld', 'tthermo', 'tdeep', 'current')
# z: height of the layer's top; eta: the free surface; theta, T, Ts, ps: air (potential temperature, temperature,
# surface temperature, surface pressure in hPa); q, rh, rain, evap, ice: moist air (specific humidity in g/kg, relative
# humidity, rain and evaporation in mm/day, sea ice in m, Tsurf the radiating surface: the ice skin over ice);
# sst, sss, mld, tthermo, tdeep, current: the ocean (V2-3; mixed-layer temperature in °C and salinity in psu, its depth in m, the thermocline
# layer's and the abyss's temperatures in °C, the mixed layer's current speed in m/s)


def visc4(g, U):
    """ν4 = U·d³/π⁴ (d the mean cell spacing): a wave of length 2d damps at rate U/d, the grid's shortest modes
    (the Laplacian's largest eigenvalue, about 27/d²) at about 7.5·U/d (SPEC §9 #25)."""
    d = float(np.mean(g.de))
    return U * d ** 3 / math.pi ** 4


class World:
    def __init__(self, n=4, preset='w5', params=None, layers=None, state=None):
        """A world on level n from case `preset`. layers=K splits a one-layer case into K equal layers with
        g' = 0 (they then move as one: test REPRO). state=(h, b, u) starts from given arrays instead."""
        self.n, self.preset = n, preset
        self.grid = gridmod.get(n)
        self.dg = DeviceGrid(self.grid)
        spec = PRESETS[preset]
        self.params = dict(params or {})
        P = self.params
        built = state if state is not None else spec['build'](self.grid)
        h, b, u = built[:3]
        extra = built[3] if len(built) > 3 else {}
        h, u = np.asarray(h, dtype=float), np.asarray(u, dtype=float)
        self.air = spec.get('air')                       # air layers (V2-2a): h is mass per area, θ per layer
        if layers and layers != h.shape[0]:
            assert h.shape[0] == 1
            h, u = np.repeat(h / layers, layers, axis=0), np.repeat(u, layers, axis=0)
            P.setdefault('gprime', [0.0] * (layers - 1))
        self.L = h.shape[0]
        gp = P.get('gprime', spec.get('gprime', [0.0] * (self.L - 1)))
        gp = gp(self.grid) if callable(gp) else gp
        self.gprime = [float(x) for x in np.atleast_1d(gp)][:self.L - 1]
        dt = P.get('dt', spec['dt'](n))
        self.dt = float(dt)
        nu4 = P.get('nu4', visc4(self.grid, P['visc4_U']) if 'visc4_U' in P else visc4(self.grid, spec.get('visc4_U', 0.0)))
        self.b = cp.asarray(b, dtype=float)
        floor = P.get('floor', 10.0 if self.air else 1.0)
        self.core = Layers(self.dg, self.L, b=self.b, gprime=self.gprime, drag=P.get('drag', spec.get('drag', 0.0)),
                           stress=P.get('stress', spec.get('stress', 0.0)),
                           nu4=nu4, floor=floor, rotation=P.get('rotation', 1.0),
                           div_damp=P.get('div_damp', DIV_DAMP),
                           air=dict(t_ref=self.air.get('t_ref', 288.0)) if self.air else None,
                           vmix=P.get('vmix', spec.get('vmix', 0.0)))
        self.dynamics = P.get('dynamics', spec.get('dynamics', True))
        self.tracer = P.get('tracer', spec.get('tracer', True))   # transport the demo tracer psi (off for air: #42)
        if nu4:
            lam = self.core.lap_max()
            if nu4 * lam * lam * self.dt > 1.0:
                raise ValueError(f'nu4 {nu4:.3g} m^4/s is unstable at dt {self.dt:g} s on n = {n} (nu4*lambda^2*dt = '
                                 f'{nu4 * lam * lam * self.dt:.2f}; keep it <= 1: nu4 <= {1.0 / (lam * lam * self.dt):.3g})')
        self.tr = Transport(self.dg, layers=self.L, beta=P.get('beta', BETA))
        self.h, self.u = cp.asarray(h), cp.asarray(u)
        self.M = self.h * cp.asarray(self.grid.areaCell)[None, :]
        self.psi = cp.asarray(np.tile(cosine_bell(self.grid), (self.L, 1)))
        self.STATE = World.STATE
        if self.air:
            phys = dict(self.air.get('physics', {}), floor_=floor, **P.get('physics', {}))
            phys = {k: (v(self.grid) if callable(v) else v) for k, v in phys.items()}     # e.g. θ* from the case
            self.physics = AirPhysics(self.dg, self.L, phys, mean_sun=P.get('mean_sun', self.air.get('mean_sun', False)),
                                      forcing=self.air.get('forcing', 'radiation'))
            self.theta = cp.asarray(np.asarray(extra['theta'], dtype=float))
            self.Ts = cp.asarray(np.asarray(extra['Ts'], dtype=float))
            self.time = cp.zeros(1)
            self.STATE = World.STATE + ('theta', 'Ts', 'time')
            self.moist = self.physics.moist
            if self.moist:                                # vapor per layer (kg/kg) and sea ice (kg/m²): V2-2b
                self.q = cp.asarray(np.asarray(extra['q'], dtype=float))
                self.ice = cp.asarray(np.asarray(extra.get('ice', np.zeros(self.grid.N)), dtype=float))
                self.STATE = self.STATE + ('q', 'ice')
        else:
            self.moist = False
        # The ocean under the air (V2-3): its mixed layer's temperature is the air's surface temperature
        self.sea = None
        if spec.get('ocean') is not None:
            from .ocean import Ocean
            ocfg = dict(spec['ocean'], **P.get('ocean', {}))
            self.surface_drag = ocfg.pop('surface_drag')
            self.sea = Ocean(self.dg, ocfg, Tsurf=extra['Ts'], state=extra.get('ocean'))
            ratio = self.sea.cfg['dt'] / self.dt
            if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
                raise ValueError(f'the ocean step {self.sea.cfg["dt"]} s must be a whole number of air steps ({self.dt} s)')
            self.ocean_every = int(round(ratio))
            self.Ts = self.sea.T[0]                          # a view: the air's surface physics heats the mixed layer
            self.oh, self.ou, self.oT, self.oM, self.oS = self.sea.h, self.sea.u, self.sea.T, self.sea.M, self.sea.S
            self.STATE = tuple(k for k in self.STATE if k != 'Ts') + ('oh', 'ou', 'oT', 'oM', 'oS')
            self._sync_sea()
        if self.air:
            self.core.update_gfast(self.h, self.theta)
        # fast sub-steps per step: from the fastest column, with 20% room
        self.nfast = int(P.get('nfast', math.ceil(1.2 * self.core.substeps(self.h, self.dt))))
        P['nfast'] = self.nfast                         # saved with the world, so a reload continues identically
        self.tick = self._tick_start = 0
        self._Fm, self._M1 = cp.zeros_like(self.u), cp.zeros_like(self.M)
        # A blocking stream: work on it orders with the default stream, so reads after step() see the results.
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graphs, self._tick0 = {}, 0                 # one captured graph per kind of tick (air; + ocean; + mixing)

    STATE = ('h', 'u', 'M', 'psi')

    def _sync_sea(self):
        """The ocean's quantities derived from the state (after creation or loading): the mixed layer's heat capacity, and
        the sea ice the fresh-water flux counts its change from."""
        self.sea.update_cap()
        if self.moist:
            cp.copyto(self.sea.ice_prev, self.ice)

    def _kind(self, tick):
        """0: air only; 1: with an ocean step; 2: with an ocean step and the mixed layer's mixing."""
        if self.sea is None or (tick + 1) % self.ocean_every:
            return 0
        return 2 if (tick + 1) % (self.ocean_every * self.sea.mix_every) == 0 else 1

    def _tick(self, kind=0):
        """One tick's kernels, on fixed buffers (no allocation), so the sequence can be captured as a graph:
        the layers' dynamics, transport of the tracer (and θ), then for air the column physics and the θ* crossings."""
        if self.dynamics:
            self.core.step(self.h, self.u, self.dt, self.nfast, self._Fm, theta=self.theta if self.air else None)
            self.tr.mass_after(self.M, self._Fm, self._M1)
            if self.tracer:
                self.tr.step(self.M, self._M1, self._Fm, self.psi, order=3)
            if self.air:
                self.tr.step(self.M, self._M1, self._Fm, self.theta, order=3)
            if self.moist:
                self.tr.step(self.M, self._M1, self._Fm, self.q, order=3)
            cp.copyto(self.M, self._M1)
        if self.sea is not None:
            # the wind stress, exactly, with its kinetic energy back as the air's heat
            g, i32, R = self.dg, np.int32, self.dg.dtype.type
            self.sea.couple(self.h, self.u, self.surface_drag, self.dt)
            self.core.k['vmix_heat'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.sea.diss, g.cellEdge, g.le, g.de,
                                                       g.areaCell, R(GRAV), R(CP_DRY), R(KAPPA), R(P0), self.theta))
        if self.air:
            self.physics.column(self.h, self.theta, self.Ts, self.u, self.time, self.dt,
                                q=self.q if self.moist else None, ice=self.ice if self.moist else None,
                                cap=self.sea.cap if self.sea is not None else None,
                                qocean=self.sea.qocean if self.sea is not None else None,
                                qsw=self.sea.qsw if self.sea is not None else None)
            if self.sea is not None:
                self.sea.accumulate(self.dt, self.physics.evap, self.physics.rain, self.ice)
            if self.L > 1:
                self.core.exchange(self.physics.wx, self.h, self.u, self.M, self.psi, theta=self.theta)
        if kind:
            self.sea.step(mix=kind == 2)

    def step(self, k=1, graph=True):
        """k ticks. The first tick runs directly (and compiles the kernels); after that the tick is captured once
        as a CUDA graph and replayed with a single launch per tick (SPEC §9 #18). graph=False runs every tick
        directly; the results are bit-identical (test REPRO)."""
        for _ in range(k):
            kind = self._kind(self.tick)
            gr = self._graphs.get(kind)
            if graph and gr is None and self.tick > self._tick0:
                with self._stream:
                    self._stream.begin_capture()
                    self._tick(kind)
                    gr = self._graphs[kind] = self._stream.end_capture()
            if graph and gr is not None:
                gr.launch(self._stream)
            else:
                with self._stream:
                    self._tick(kind)
            self.tick += 1
        self._stream.synchronize()
        return self

    def track_ocean_heat_transport(self):
        """Sum the ocean currents' heat convergence per cell from now on (Ocean.track_heat_transport); the captured step
        graphs are dropped, so the next steps record the sums."""
        self.sea.track_heat_transport()
        self._graphs.clear()
        self._tick0 = self.tick

    def exchange(self, wx):
        """Move thickness wx (L − 1, N; m, positive up) between neighbouring layers with tracer and momentum."""
        with self._stream:
            self.core.exchange(cp.asarray(wx, dtype=float), self.h, self.u, self.M, self.psi)
        self._stream.synchronize()

    # ---- diagnostics ----
    def totals(self):
        """Deterministic totals per layer and summed: volume Σh·A, prognostic volume ΣM, tracer ΣM·psi; and the
        column's volume-weighted momentum Σ_e l_e d_e Σ_k h_e,k u_k."""
        A = cp.asarray(self.grid.areaCell)
        out = {}
        for k in range(self.L):
            out[f'volume{k}'] = dsum(self.h[k] * A)
            out[f'M{k}'] = dsum(self.M[k])
            out[f'tracer{k}'] = dsum(self.M[k], self.psi[k])
        g = self.grid
        he = 0.5 * (self.h[:, g.eI] + self.h[:, g.eJ])
        out['momentum'] = dsum(cp.sum(he * self.u, axis=0) * cp.asarray(g.le * g.de))
        return out

    def water(self):
        """The air's vapor (kg): Σ A Σ_k μ_k q_k."""
        return dsum(cp.sum(self.h * self.q, axis=0), cp.asarray(self.grid.areaCell))

    def energy(self):
        """Total energy (J): kinetic + potential (+ for air the enthalpy and the slab's heat C·Ts; for moist air the
        vapor's latent energy L_v·μq and the ice's −L_f·m)."""
        if self.air:
            A = cp.asarray(self.grid.areaCell)
            e = self.core.energy(self.h, self.u, self.theta)
            e += self.sea.heat() if self.sea is not None else float(cp.sum(self.Ts * A)) * self.physics.p['c_slab']
            if self.moist:
                e += (self.physics.p['Lv'] * float(cp.sum(A * cp.sum(self.h * self.q, axis=0)))
                      - self.physics.p['Lf'] * float(cp.sum(A * self.ice)))
            return e
        return self.core.energy(self.h, self.u)

    def fields(self):
        """The fields this world can show: air worlds add θ, T, the surface temperature and the surface pressure."""
        base = ['h', 'z', 'eta', 'psi', 'speed', 'vort']
        if not self.air:
            return base
        return (base + ['theta', 'T', 'Ts', 'ps'] + (['Tsurf', 'q', 'rh', 'rain', 'evap', 'ice'] if self.moist else [])
                + (['sst', 'sss', 'mld', 'tthermo', 'tdeep', 'current'] if self.sea is not None else []))

    def field(self, name, layer=0):
        """A cell field (CuPy, float64) for the viewer."""
        g = self.grid
        if name == 'h':
            return self.h[layer]
        if name in ('z', 'eta'):
            top = 0 if name == 'eta' else layer
            return self.b + cp.sum(self.h[top:], axis=0)
        if name == 'psi':
            return self.psi[layer]
        if name == 'theta':
            return self.theta[layer]
        if name == 'Ts':
            return self.Ts
        if name == 'q':
            return self.q[layer] * 1000.0                          # g/kg
        if name == 'sst':
            return self.sea.T[0] - 273.15
        if name == 'sss':
            return self.sea.S[0]
        if name == 'tthermo':
            return self.sea.T[1] - 273.15
        if name == 'tdeep':
            return self.sea.T[-1] - 273.15
        if name == 'mld':
            return self.sea.h[0]
        if name == 'current':
            d = self.dg
            out = cp.zeros(d.N)
            self.sea.k['cell_speed'](*grid_dims(d.N), (np.int32(d.N), self.sea.u[0], d.cellEdge, d.le, d.de, d.areaCell, out))
            return out
        if name == 'Tsurf':                                        # before the first step (of a new or loaded world): the slab
            return self.physics.tsurf if self.tick > self._tick_start else self.Ts
        if name == 'rain':
            return self.physics.rain * 86400.0                      # mm/day
        if name == 'evap':
            return self.physics.evap * 86400.0
        if name == 'ice':
            return self.ice / self.physics.p['rho_ice']             # m
        if name == 'rh':
            T = self.field('T', layer)
            p = GRAV * cp.cumsum(self.h, axis=0)
            pt = p - GRAV * self.h
            pk = CP_DRY * (p ** (KAPPA + 1) - pt ** (KAPPA + 1)) / ((KAPPA + 1) * (p - pt) * P0 ** KAPPA)
            pm = P0 * (pk[layer] / CP_DRY) ** (1 / KAPPA)           # the layer's pressure, as the column physics uses it
            qs = 0.622 * 611.0 * cp.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm
            return self.q[layer] / qs
        if name == 'ps':
            return GRAV * cp.sum(self.h, axis=0) / 100.0          # hPa
        if name == 'T':
            p = GRAV * cp.cumsum(self.h, axis=0)
            pt = p - GRAV * self.h
            pib = CP_DRY * (p ** (KAPPA + 1) - pt ** (KAPPA + 1)) / ((KAPPA + 1) * (p - pt) * P0 ** KAPPA)
            return (pib * self.theta / CP_DRY)[layer]
        if name in ('speed', 'vort'):
            uh = self.u[layer].get()
            if name == 'speed':
                ce = np.where(g.cellEdge >= 0, g.cellEdge, 0)
                ke = np.sum(np.where(g.cellEdge >= 0, g.le[ce] * g.de[ce] * 0.25 * uh[ce] ** 2, 0.0), axis=1)
                return cp.asarray(np.sqrt(2 * ke / g.areaCell))
            z = np.zeros(g.NV)
            np.add.at(z, g.ev1, g.circ1 * g.de * uh)
            np.add.at(z, g.ev2, g.circ2 * g.de * uh)
            z /= g.areaV
            cv = np.where(g.cornerV >= 0, g.cornerV, 0)
            cell = np.sum(np.where(g.cornerV >= 0, z[cv] * g.kite, 0.0), axis=1)
            return cp.asarray(cell / g.areaCell)
        raise KeyError(name)

    def state_hash(self):
        h = hashlib.sha1()
        for k in self.STATE:
            h.update(getattr(self, k).get().tobytes())
        h.update(str(self.tick).encode())
        return h.hexdigest()

    # ---- save / load ----
    def save(self, path):
        meta = dict(kind='insilica-world-v2', n=self.n, preset=self.preset, layers=self.L, dt=self.dt, tick=self.tick,
                    params=self.params, order=self.grid.order, gridHash=gridmod.generator_hash(), sourceHash=source_hash())
        savefile.write(path, meta, {**{k: getattr(self, k).get() for k in self.STATE}, 'b': self.b.get()})

    @classmethod
    def load(cls, path):
        meta, arrays = savefile.read(path)
        if meta.get('gridHash') != gridmod.generator_hash() or meta.get('order') != gridmod.DEFAULT_ORDER:
            raise ValueError('save file was made with a different grid generator or cell order')
        extra = {k: arrays[k] for k in ('theta', 'Ts', 'q', 'ice') if k in arrays}
        if 'oT' in arrays:                                     # coupled: the mixed layer is the surface
            extra['Ts'] = arrays['oT'][0]
        w = cls(n=meta['n'], preset=meta['preset'], params=meta['params'],
                state=(arrays['h'], arrays['b'], arrays['u'], extra))
        for k in w.STATE:
            a = arrays[k]
            if a.shape != getattr(w, k).shape or a.dtype != np.float64:
                raise ValueError(f'save file array {k}: shape {a.shape} {a.dtype}, expected {getattr(w, k).shape} float64')
            getattr(w, k)[...] = cp.asarray(a)          # into the existing buffers a captured graph uses
        if w.sea is not None:
            w._sync_sea()
        w.tick = w._tick0 = w._tick_start = meta['tick']
        return w
