"""A world: a stack of layers on the layered core (layers.py) carrying one tracer per layer, on the GPU.

State (all 64-bit, layer-major): h (L, N) thickness, u (L, E) normal wind, M (L, N) layer volume per cell
(h·A, moved by the same fluxes as h so the tracer budget closes exactly), psi (L, N) tracer per volume.
A tick is one time-split RK3 step of the layers followed by one FCT transport step with that step's fluxes.
Cases and their settings are in cases.py; `params` overrides a case's settings:
  dt, gprime (list, L − 1), floor (m, default 1), drag (per layer, 1/s), stress (per layer, m/s: linear stress
  coefficient, deceleration c/h), nu4 (m⁴/s) or visc4_U (m/s: ν4 = U·d³/π⁴
  with d the mean cell spacing, the grid-scale wave then damps at U/d), rotation (multiplies f), beta (transport),
  div_damp (the fast mode's divergence damping, default 0.1), nfast (fast sub-steps per step; default from √(gH)),
  thin_wind (air: the layer mass below which a layer takes its neighbours' wind, default 100 kg/m², #83; 0 = off).
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
from .constants import C_SOIL, GRAV, K_SOIL, P0, PLANET, air_props
from .land import LandWater
from .layers import DIV_DAMP, Layers
from .transport import BETA, Transport

THIN_WIND = 100.0     # kg/m² (~1 hPa): below it an air layer takes its neighbours' wind, fully at the floor (#83)
# the gases carried as tracers (V2-6e, #217), in the k-tables' order, and their starting mole fractions where the planet gives none
GAS_TRACERS = (('co2', 400e-6), ('n2o', 0.32e-6), ('ch4', 1.8e-6))

FIELDS = ('h', 'z', 'eta', 'psi', 'speed', 'vort', 'theta', 'T', 'Ts', 'Tsurf', 'ps', 'q', 'rh', 'rain', 'evap', 'lightning', 'ice',
          'sst', 'sss', 'mld', 'tthermo', 'tdeep', 'current', 'land', 'elev', 'soil', 'pond', 'snow',
          'sla', 'tide', 'co2', 'ch4', 'n2o', 'tsoil', 'g_rain', 'g_rx1', 'g_river', 'g_ftc', 'g_weather')
# (V2-6g) sla: the sea level's anomaly from its mean over the sea (m); tide: the equilibrium tide, the tidal potential over g (m);
# co2, ch4, n2o: the carried gases (ppm, by layer); tsoil: the land column's soil temperature (°C, by soil layer from the top);
# g_*: geology's accumulators as means (rain mm/day, the largest one-day rain mm, river flow m³/s, freeze-thaw cycles a year,
# weathering mm/day)
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
        spec = PRESETS[preset]
        self.params = dict(params or {})
        P = self.params
        # the planet (V2-6e, #186): radius, rotation and gravity (Earth's unless the preset or params set them); the rest of its
        # settings (day, year, sun, orbit, gases) are the air physics' own
        pl = dict(spec.get('planet', {}), **P.get('planet', {}))
        self.planet = dict(PLANET, **{k: v for k, v in pl.items() if k in PLANET})
        self.grav = self.planet['grav']
        self.grid = gridmod.for_planet(gridmod.get(n), **self.planet)
        self.dg = DeviceGrid(self.grid)
        built = state if state is not None else spec['build'](self.grid)
        h, b, u = built[:3]
        extra = built[3] if len(built) > 3 else {}
        h, u = np.asarray(h, dtype=float), np.asarray(u, dtype=float)
        self.air = spec.get('air')                       # air layers (V2-2a): h is mass per area, θ per layer
        # the dry air's gases (V2-6e, #216): by mole fraction, from the params' physics, the preset's or the planet; none, the old constants
        self.gas = air_props((P.get('physics', {}) or {}).get('gases') or ((self.air or {}).get('physics', {}) or {}).get('gases') or pl.get('gases'),
                             t_cp=pl.get('t_cp'), o3=pl.get('o3'))   # (V2-7) CO2's heat capacity at the air's temperature; ozone
        if self.air and state is None and self.grav != GRAV:
            h = h * (GRAV / self.grav)                   # (#186) the builders' pressures kept: their masses are Earth's
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
        floor = P.get('floor', spec.get('floor', 10.0 if self.air else 1.0))   # (V2-7) a preset's own: a thinner air's (Mars)
        self.core = Layers(self.dg, self.L, b=self.b, grav=self.grav, gprime=self.gprime, drag=P.get('drag', spec.get('drag', 0.0)),
                           stress=P.get('stress', spec.get('stress', 0.0)),
                           nu4=nu4, floor=floor, rotation=P.get('rotation', 1.0),
                           div_damp=P.get('div_damp', DIV_DAMP),
                           air=dict(t_ref=self.air.get('t_ref', 288.0), cp=self.gas['cp'], kappa=self.gas['kappa'], rd=self.gas['rd']) if self.air else None,
                           vmix=P.get('vmix', spec.get('vmix', 0.0)),
                           thin_wind=P.get('thin_wind', spec.get('thin_wind', THIN_WIND if spec.get('air') else 0.0)),
                           shelf=P.get('shelf', spec.get('shelf', 0.0)))   # (air, V2-6b: a layer squeezed below it over terrain loses its pressure force, as the sea's over its floor, #97)
        self.dynamics = P.get('dynamics', spec.get('dynamics', True))
        self.tracer = P.get('tracer', spec.get('tracer', True))   # transport the demo tracer psi (off for air: #42)
        if nu4:
            lam = self.core.lap_max()
            if nu4 * lam * lam * self.dt > 1.0:
                raise ValueError(f'nu4 {nu4:.3g} m^4/s is unstable at dt {self.dt:g} s on n = {n} (nu4*lambda^2*dt = '
                                 f'{nu4 * lam * lam * self.dt:.2f}; keep it <= 1: nu4 <= {1.0 / (lam * lam * self.dt):.3g})')
        # gas tracers (V2-6e, #217): CO2, N2O and CH4 carried by the air, per layer, with exact budgets; the k-tables read them
        self.gas_on = bool(self.air) and P.get('gas_tracers', spec.get('gas_tracers', 0)) > 0
        self.tr = Transport(self.dg, layers=self.L, beta=P.get('beta', BETA), tracers=4 + (len(GAS_TRACERS) if self.gas_on else 0))
        self.h, self.u = cp.asarray(h), cp.asarray(u)
        self.M = self.h * cp.asarray(self.grid.areaCell)[None, :]
        self.psi = cp.asarray(np.tile(cosine_bell(self.grid), (self.L, 1)))
        self.STATE = World.STATE
        if self.air:
            phys = dict(self.air.get('physics', {}), floor_=floor, **P.get('physics', {}))
            phys.update({k: v for k, v in pl.items() if k not in PLANET}, grav=self.grav)      # the planet's day, year, sun, orbit, gases (#186)
            if self.grav != GRAV:
                phys.setdefault('mu_ref', P0 / self.grav)
                phys.setdefault('mu_bl', 1.0e4 / self.grav)
            phys = {k: (v(self.grid) if callable(v) else v) for k, v in phys.items()}     # e.g. θ* from the case
            if self.gas['defs']:                                                          # (#216) the air's own constants
                phys.update(cp=self.gas['cp'], kappa=self.gas['kappa'], air_rd=self.gas['rd_col'], air_defs=list(self.gas['defs']),
                            air_cia=list(self.gas['rad']['cia']))                   # (#226) its radiation
            if self.gas_on:                                                               # (#217) the column records its moves
                phys['air_defs'] = list(phys.get('air_defs') or []) + ['-DGASX']
            self.physics = AirPhysics(self.dg, self.L, phys, mean_sun=P.get('mean_sun', self.air.get('mean_sun', False)),
                                      forcing=self.air.get('forcing', 'radiation'),
                                      longwave=P.get('longwave', self.air.get('longwave', 'gray')))
            # radiation steps (V2-6a): the k-tables' longwave every rad_dt s (an hour), and on the first tick
            self.rad_every = max(1, int(round(P.get('rad_dt', self.air.get('rad_dt', 3600.0)) / self.dt)))
            self.theta = cp.asarray(np.asarray(extra['theta'], dtype=float))
            self.Ts = cp.asarray(np.asarray(extra['Ts'], dtype=float))
            self.time = cp.zeros(1)
            self.STATE = World.STATE + ('theta', 'Ts', 'time')
            if self.gas_on:                               # (#217) each gas's mole fraction per layer, starting uniform
                self.gx = cp.stack([cp.full((self.L, self.grid.N), float(self.physics.p.get(k, v))) for k, v in GAS_TRACERS])
                self.gas_src = cp.zeros((len(GAS_TRACERS), self.grid.N))   # surface sources, mole fraction x kg/m^2/s of air
                self.k_gas = {n: self.dg.kernel(n, 'air.cu', self.physics.defs) for n in ('gas_src',)}
                self.STATE = self.STATE + ('gx',)
            self.moist = self.physics.moist
            if self.moist:                                # vapor per layer (kg/kg) and sea ice (kg/m²): V2-2b
                self.q = cp.asarray(np.asarray(extra['q'], dtype=float))
                self.ice = cp.asarray(np.asarray(extra.get('ice', np.zeros(self.grid.N)), dtype=float))
                self.STATE = self.STATE + ('q', 'ice')
                self.cw = None
                if self.physics.p['cloud'] > 0:              # cloud water per layer (kg/kg), #95
                    self.cw = cp.zeros_like(self.q)
                    self.STATE = self.STATE + ('cw',)
        else:
            self.moist = False
        # Land (V2-4a, #84): a mask from the preset, with a soil bucket whose capacity grows with height (v1's rule)
        land = spec['land'](self.grid) if spec.get('land') is not None else None
        self.land = self.soil = self.soilcap = self.cap = self.pond = self.snow = self.routing = self.mouth = None
        if land is not None:
            p = self.physics.p
            self.land = cp.asarray(np.asarray(land, dtype=float))
            self.soilcap = cp.asarray(p['soil_base'] + p['soil_per_m'] * np.maximum(cp.asarray(b).get(), 0.0))
            self.soil = 0.5 * self.soilcap
            self.pond, self.snow = cp.zeros(self.grid.N), cp.zeros(self.grid.N)
            if p.get('snow_start', 0) > 0 and state is None and self.moist:  # (#222) an ice sheet starts under its firn, as snow
                self.snow[...] = cp.minimum(self.ice * self.land, p['rho_firn'] * p['firn'] if p['firn'] > 0 else p['snow_cover'])
            self.routing = LandWater(self.dg, self.b, self.land, self.soilcap, self.pond, self.soil, self.ice, rho_ice=p['rho_ice'],
                                     **({} if self.grav == GRAV else dict(grav=self.grav)))   # (Earth: the ice flow's own 9.81)
            self.mouth = self.routing.mouth
            self.STATE = self.STATE + ('soil', 'pond', 'snow', 'mouth')
            self.firn = None
            if p['firn'] > 0:                                    # the firn's temperature (V2-6d, #160)
                self.firn = cp.full(self.grid.N, p['Tf'])
                self.STATE = self.STATE + ('firn',)
            self.cap = self.land * p['c_land'] + (1 - self.land) * p['c_slab']   # (the ocean's own when coupled)
            if p['cover'] > 0:                                   # land cover (V2-6b, #132)
                if spec.get('cover') is None:
                    raise ValueError('land cover (cover) needs the preset cover map (cover)')
                self.physics.cover = cp.asarray(np.asarray(spec['cover'](self.grid), dtype=float))
        # The ocean under the air (V2-3): its mixed layer's temperature is the air's surface temperature
        self.sea = None
        if spec.get('ocean') is not None:
            from .ocean import Ocean
            ocfg = dict(spec['ocean'], **P.get('ocean', {}))
            self.surface_drag = ocfg.pop('surface_drag')
            self.sea = Ocean(self.dg, ocfg, Tsurf=extra['Ts'], state=extra.get('ocean'), land=land,
                             c_land=self.physics.p['c_land'], c_deep=self.physics.p['c_deep'], grav=self.grav,
                             day_s=self.physics.p['day_s'] if self.air else 86400.0,
                             soil=dict(dz=self.physics.p.get('soil_dz') or (), c=self.physics.p.get('c_soil', C_SOIL),
                                       k=self.physics.p.get('k_soil', K_SOIL)) if self.air else None)
            ratio = self.sea.cfg['dt'] / self.dt
            if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
                raise ValueError(f'the ocean step {self.sea.cfg["dt"]} s must be a whole number of air steps ({self.dt} s)')
            self.ocean_every = int(round(ratio))
            if self.routing is not None:
                self.routing.ice_prev = self.sea.ice_prev
            self.Ts = self.sea.T[0]                          # a view: the air's surface physics heats the mixed layer
            self.oh, self.ou, self.oT, self.oM, self.oS = self.sea.h, self.sea.u, self.sea.T, self.sea.M, self.sea.S
            self.STATE = tuple(k for k in self.STATE if k != 'Ts') + ('oh', 'ou', 'oT', 'oM', 'oS')
            if self.sea.soil is not None:                    # (#218) the land column's temperatures
                self.oTg = self.sea.Tg
                self.STATE = self.STATE + ('oTg',)
            self._sync_sea()
        # Moon and tides (V2-6f, #228): the tide-generating force on the ocean and the mixing its dissipation drives
        self.tides = None
        tides = P.get('tides', spec.get('tides'))
        if tides is not None and tides is not False:
            if self.sea is None or not self.air:
                raise ValueError('tides need the coupled ocean under air')
            from .tides import Tides
            self.tides = Tides(self.dg, self.sea, self.physics.p, self.planet['radius'], tides,
                               rough=spec['rough'](self.grid) if spec.get('rough') is not None else None,
                               nbot=spec['nbot'](self.grid) if spec.get('nbot') is not None else None)
        # the mountains' drag (V2-6b, #127): Lott & Miller (1997) from the sub-grid orography
        self.oro = None
        if self.air and self.physics.p['oro'] > 0:
            from .orodrag import OroDrag
            if self.sea is None or spec.get('sso') is None:
                raise ValueError('the mountain drag (oro) needs the coupled ocean and the preset sub-grid orography (sso)')
            self.oro = OroDrag(self.dg, self.L, spec['sso'](self.grid), self.physics.p, self.dt)
        # the gravity waves not tied to the ground (#128): Alexander & Dunkerton (1999)
        self.gw = None
        if self.air and self.physics.p['gw'] > 0:
            from .orodrag import WaveDrag
            if self.sea is None:
                raise ValueError('the wave drag (gw) books its heat with the wind stress: it needs the coupled ocean')
            self.gw = WaveDrag(self.dg, self.L, self.physics.p, self.dt)
            if self.physics.p.get('gw_conv', 0.0) > 0 or self.physics.p.get('gw_kel', 0.0) > 0:   # (#188, #233) the column's convective rain
                self.gw.conv = self.physics.rainp[2]
        if self.air:
            self.core.update_gfast(self.h, self.theta)
        # fast sub-steps per step: from the fastest column, with 20% room
        self.nfast = int(P.get('nfast', math.ceil(1.2 * self.core.substeps(self.h, self.dt))))
        P['nfast'] = self.nfast                         # saved with the world, so a reload continues identically
        self.tick = self._tick_start = 0
        self._Fm, self._M1 = cp.zeros_like(self.u), cp.zeros_like(self.M)
        # A blocking stream: work on it orders with the default stream, so reads after step() see the results.
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graphs, self._tick0 = {}, 0                 # one captured graph per kind of tick (air; + ocean; + mixing; + radiation)
        self._rad0 = 0                                    # the tick radiation steps count from (the first after a build or load)
        self.recorder = None
        self.geology = None                               # geology's accumulators (V2-6e, #220), off unless asked for
        if P.get('geology'):
            self._geology_on()

    STATE = ('h', 'u', 'M', 'psi')

    def _geology_on(self):
        from .geology import Geology
        if not self.moist:
            raise ValueError("geology's accumulators (geology) need moist air")
        p = self.physics.p
        self.geology = Geology(self.dg, self.dt, p['day_s'], self.Ts, p['Tf'], land=self.land)
        self.geo, self.geo_ring, self.geo_clock = self.geology.acc, self.geology.ring, self.geology.clock
        self.STATE = self.STATE + ('geo', 'geo_ring', 'geo_clock')
        if self.routing is not None:
            self.routing.geo = self.geology

    def enable_geology(self):
        """Geology's accumulators (V2-6e, #220) on in a built world, from zero; saved with it from now on, and the captured
        step graphs dropped so the sums are in them."""
        if self.geology is None:
            self.params['geology'] = 1
            self._geology_on()
            self._graphs.clear()
            self._tick0 = self.tick
        return self.geology

    def _sync_sea(self):
        """The ocean's quantities derived from the state (after creation or loading): the mixed layer's heat capacity, and
        the sea ice the fresh-water flux counts its change from."""
        self.sea.update_cap()
        if self.moist:
            cp.copyto(self.sea.ice_prev, self.ice)

    def enable_ice_drift(self, wind=0.02):
        """Sea-ice drift (V2-6e, #185) on in a built world; its captured step graphs are dropped so the drift is in them."""
        self.sea.enable_ice_drift(wind)
        self._graphs.clear()
        self._tick0 = self.tick

    def _kind(self, tick):
        """0: air only; 1: with an ocean step; 2: with an ocean step and the mixed layer's mixing."""
        if self.sea is None or (tick + 1) % self.ocean_every:
            return 0
        return 2 if (tick + 1) % (self.ocean_every * self.sea.mix_every) == 0 else 1

    def _radiates(self, tick):
        return self.air and self.physics.lw is not None and (tick - self._rad0) % self.rad_every == 0

    def _tick(self, kind=0, rad=False):
        """One tick's kernels, on fixed buffers (no allocation), so the sequence can be captured as a graph:
        the layers' dynamics, transport of the tracer (and θ), then for air the column physics and the θ* crossings."""
        rec = self.recorder                                   # the ocean's forcing being recorded (spinup.Recorder)
        if self.dynamics:
            self.core.step(self.h, self.u, self.dt, self.nfast, self._Fm, theta=self.theta if self.air else None)
            self.tr.mass_after(self.M, self._Fm, self._M1)
            # every tracer through the same launches (V2-5)
            moved = ([self.psi] if self.tracer else []) + ([self.theta] if self.air else [])
            if self.moist:
                moved += [self.q] + ([self.cw] if self.cw is not None else [])
            if self.gas_on:
                moved += [self.gx[j] for j in range(len(GAS_TRACERS))]
            if moved:
                self.tr.step_many(self.M, self._M1, self._Fm, moved, order=3)
            cp.copyto(self.M, self._M1)
        if self.sea is not None:
            # the wind stress, exactly, with its kinetic energy back as the air's heat
            g, i32, R = self.dg, np.int32, self.dg.dtype.type
            if rec:
                rec.pre_couple()
            if self.air and self.physics.sfc:
                self.physics.surface(self.h, self.theta, self.Ts, self.u, q=self.q if self.moist else None,
                                     ice=self.ice if self.moist else None, land=self.land)
            self.sea.couple(self.h, self.u, self.surface_drag, self.dt, cd=self.physics.cd if self.air else None)
            if self.oro is not None:
                self.oro.apply(self.h, self.theta, self.u, self.dt, self.sea.diss)
            if self.gw is not None:
                self.gw.apply(self.h, self.theta, self.u, self.dt, self.sea.diss)
            if rec:
                rec.post_couple()
            self.core.k['vmix_heat'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.h, self.sea.diss, g.cellEdge, g.le, g.de,
                                                       g.areaCell, R(self.grav), R(self.core.cpa), R(self.core.kap), R(P0), self.theta))
        if self.air:
            if rad:
                self.physics.radiate(self.h, self.theta, self.Ts, self.time, self.rad_every * self.dt,
                                     q=self.q if self.moist else None, cw=self.cw if self.moist else None,
                                     gx=self.gx if self.gas_on else None)
            if rec:
                rec.pre_column()
            self.physics.column(self.h, self.theta, self.Ts, self.u, self.time, self.dt,
                                q=self.q if self.moist else None, ice=self.ice if self.moist else None,
                                cap=self.sea.cap if self.sea is not None else self.cap,
                                qocean=self.sea.qocean if self.sea is not None else None,
                                qsw=self.sea.qsw if self.sea is not None else None,
                                land=self.land, soilcap=self.soilcap, soil=self.soil, pond=self.pond, snow=self.snow,
                                cw=self.cw if self.moist else None, elev=self.b, firn=getattr(self, 'firn', None),
                                gx=self.gx if self.gas_on else None)
            if rec:
                rec.post_column()
            if self.geology is not None:
                self.geology.column(self.physics.rain, self.physics.evap, ice=self.ice if self.land is not None else None)
            if self.sea is not None:
                self.sea.ground(self.dt)
                self.sea.accumulate(self.dt, self.physics.evap, self.physics.rain, self.ice, self.mouth)
            if self.L > 1:
                self.core.exchange(self.physics.wx, self.h, self.u, self.M, self.psi, theta=self.theta)
                if self.gas_on:                            # (#217) the exchange's own moves already carried them in the column; their sources
                    gN, i32, R = grid_dims(self.grid.N), np.int32, self.dg.dtype.type
                    G = i32(len(GAS_TRACERS))
                    self.k_gas['gas_src'](*gN, (i32(self.grid.N), i32(self.L), G, self.gas_src, self.h, R(self.dt), self.gx))
        if kind:
            if rec and kind == 2:
                rec.pre_mix()
            if self.tides is not None:                                     # (#228) the force at the ocean step's middle
                self.tides.force(self.time, 0.5 * self.sea.cfg['dt'])
            self.sea.step(mix=kind == 2)
            if self.tides is not None:
                self.tides.mix(self.sea.cfg['dt'])
            if self.moist and self.sea.cfg['ice_drift'] > 0:              # sea-ice drift (V2-6e, #185)
                self.sea.drift_ice(self.ice, self.u, self.physics.p['rho_ice'])
        if self.routing is not None and (kind or self.sea is None):     # on the ocean's step (1 h), or every step
            self.routing.step(self.sea.cfg['dt'] if self.sea is not None else self.dt)

    def step(self, k=1, graph=True):
        """k ticks. The first tick runs directly (and compiles the kernels); after that the tick is captured once
        as a CUDA graph and replayed with a single launch per tick (SPEC §9 #18). graph=False runs every tick
        directly; the results are bit-identical (test REPRO)."""
        for _ in range(k):
            self.launch_tick(graph)
        self._stream.synchronize()
        return self

    def launch_tick(self, graph=True):
        """Queue one tick on this world's stream without waiting for it (step() and Ensemble.step() call this)."""
        kind = (self._kind(self.tick), self._radiates(self.tick))
        gr = self._graphs.get(kind)
        if graph and gr is None and self.tick > self._tick0:
            with self._stream:
                self._stream.begin_capture()
                self._tick(*kind)
                gr = self._graphs[kind] = self._stream.end_capture()
        if graph and gr is not None:
            gr.launch(self._stream)
        else:
            with self._stream:
                self._tick(*kind)
        self.tick += 1

    def set_recorder(self, rec):
        """Record the ocean's forcing with rec (spinup.Recorder) from now on, or stop (None); the captured graphs are
        dropped, so the next ticks include (or leave out) the recording."""
        self.recorder = rec
        self._graphs.clear()
        self._tick0 = self.tick

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
        """The air's vapor and cloud water (kg): Σ A Σ_k μ_k (q_k + cw_k)."""
        return dsum(cp.sum(self.h * (self.q if self.cw is None else self.q + self.cw), axis=0), cp.asarray(self.grid.areaCell))

    def energy(self):
        """Total energy (J): kinetic + potential (+ for air the enthalpy and the slab's heat C·Ts; for moist air the
        vapor's latent energy L_v·μq and the ice's −L_f·m)."""
        if self.air:
            A = cp.asarray(self.grid.areaCell)
            e = self.core.energy(self.h, self.u, self.theta)
            if self.sea is not None:
                e += self.sea.heat()
            elif self.cap is not None:
                e += float(cp.sum(self.Ts * A * self.cap))
            else:
                e += float(cp.sum(self.Ts * A)) * self.physics.p['c_slab']
            if self.moist:
                e += (self.physics.p['Lv'] * float(cp.sum(A * cp.sum(self.h * self.q, axis=0)))
                      - self.physics.p['Lf'] * float(cp.sum(A * self.ice)))
            if getattr(self, 'firn', None) is not None:              # (#160)
                p = self.physics.p
                e += p['c_ice'] * p['rho_firn'] * p['firn'] * float(cp.sum(A * (self.firn - p['Tf'])))
            return e
        return self.core.energy(self.h, self.u)

    def fields(self):
        """The fields this world can show: air worlds add θ, T, the surface temperature and the surface pressure."""
        base = ['h', 'z', 'eta', 'psi', 'speed', 'vort']
        if not self.air:
            return base
        return (base + ['theta', 'T', 'Ts', 'ps'] + (['Tsurf', 'q', 'rh', 'rain', 'evap', 'ice', 'lightning'] if self.moist else [])
                + (['sst', 'sss', 'mld', 'tthermo', 'tdeep', 'current', 'sla'] if self.sea is not None else [])
                + (['tide'] if getattr(self, 'tides', None) is not None else [])
                + (['land', 'elev', 'soil', 'pond', 'snow'] if self.land is not None else [])
                + (['tsoil'] if self.sea is not None and self.sea.soil is not None else [])
                + (['co2', 'ch4', 'n2o'] if getattr(self, 'gas_on', False) else [])
                + (['g_rain', 'g_rx1', 'g_river', 'g_ftc', 'g_weather'] if getattr(self, 'geology', None) is not None else []))

    def sky(self):
        """(V2-6g, the viewer) The Sun's and the Moon's directions in the planet's frame at the model's time, as the tide kernel
        places them (kernels/tides.cu bodies): dict(t, sun, moon, ra_sun, rot, sun_r, moon_r, day_s, year_s, tilt); None without air."""
        if not self.air:
            return None
        from .tides import sky_par
        if not hasattr(self, '_sky'):
            tid = getattr(self, 'tides', None)
            self._sky = (tid.par if tid is not None else sky_par(self.physics.p, self.planet['radius']), cp.zeros(10),
                         self.dg.kernel('sky_dirs', 'tides.cu'))
        par, out, k = self._sky
        k((1,), (32,), (par, self.time, out))
        o, p = out.get(), self.physics.p
        return dict(t=float(self.time.get()[0]), sun=[float(x) for x in o[:3]], moon=[float(x) for x in o[3:6]], ra_sun=float(o[6]),
                    rot=float(o[7]), sun_r=float(o[8]), moon_r=float(o[9]), day_s=float(p['day_s']), year_s=float(p['year_s']),
                    tilt=float(p['tilt']))

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
        if name == 'land':
            return self.land
        if name == 'elev':
            return self.b
        if name == 'soil':
            return self.soil
        if name == 'pond':
            return self.pond
        if name == 'snow':
            return self.snow
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
        if name == 'sla':                                          # (V2-6g) the sea level's anomaly from its mean over the sea
            s = self.sea.h.sum(0) + self.sea.b
            wet = (self.land < 0.5) if self.land is not None else cp.ones(self.grid.N, dtype=bool)
            A = cp.asarray(g.areaCell)
            return cp.where(wet, s - cp.sum(s * A * wet) / cp.sum(A * wet), 0.0)
        if name == 'tide':
            return self.tides.V / self.grav
        if name in ('co2', 'n2o', 'ch4'):
            return self.gx[GAS_TRACERS.index(next(t for t in GAS_TRACERS if t[0] == name)), layer] * 1e6
        if name == 'tsoil':
            return self.sea.Tg[min(layer, self.sea.Tg.shape[0] - 1)] - 273.15
        if name.startswith('g_'):
            s = self.geology.summary()
            key = {'g_rain': 'rain', 'g_rx1': 'rx1day', 'g_river': 'river', 'g_ftc': 'ftc', 'g_weather': 'weathering'}[name]
            return cp.asarray(s[key])
        if name == 'Tsurf':                                        # before the first step (of a new or loaded world): the slab
            return self.physics.tsurf if self.tick > self._tick_start else self.Ts
        if name == 'rain':
            return self.physics.rain * 86400.0                      # mm/day
        if name == 'evap':
            return self.physics.evap * 86400.0
        if name == 'lightning':
            return self.physics.flash * 3.15576e13                  # flashes/km²/year
        if name == 'ice':
            return self.ice / self.physics.p['rho_ice']             # m
        if name == 'rh':
            T = self.field('T', layer)
            p = self.grav * cp.cumsum(self.h, axis=0)
            pt = p - self.grav * self.h
            c_, k_ = self.core.cpa, self.core.kap
            pk = c_ * (p ** (k_ + 1) - pt ** (k_ + 1)) / ((k_ + 1) * (p - pt) * P0 ** k_)
            pm = P0 * (pk[layer] / c_) ** (1 / k_)                   # the layer's pressure, as the column physics uses it
            qs = self.gas['eps'] * 611.0 * cp.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm
            return self.q[layer] / qs
        if name == 'ps':
            return self.grav * cp.sum(self.h, axis=0) / 100.0          # hPa
        if name == 'T':
            p = self.grav * cp.cumsum(self.h, axis=0)
            pt = p - self.grav * self.h
            c_, k_ = self.core.cpa, self.core.kap
            pib = c_ * (p ** (k_ + 1) - pt ** (k_ + 1)) / ((k_ + 1) * (p - pt) * P0 ** k_)
            return (pib * self.theta / c_)[layer]
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
            if k in ('firn', 'gx') and k not in arrays:          # (#217: a save without gas tracers starts them uniform)
                continue
            if k == 'oTg' and k not in arrays:                   # (#218: a save without the land column starts it at its deep ground's)
                w.oTg[...] = cp.asarray(arrays['oT'][1] if w.sea.land is not None else arrays['oT'][0])[None, :]
                continue
            if k == 'firn' and k not in arrays:                  # (a save from before the firn: it starts at freezing)
                continue
            a = arrays[k]
            if a.shape != getattr(w, k).shape or a.dtype != np.float64:
                raise ValueError(f'save file array {k}: shape {a.shape} {a.dtype}, expected {getattr(w, k).shape} float64')
            getattr(w, k)[...] = cp.asarray(a)          # into the existing buffers a captured graph uses
        if w.sea is not None:
            w._sync_sea()
        w.tick = w._tick0 = w._tick_start = w._rad0 = meta['tick']
        return w
