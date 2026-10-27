"""Column physics of the air layers (stage V2-2a, kernels/air.cu): v1's radiation and surface exchange over a slab
ocean, heating as potential temperature, and mass crossing the isentropic interface θ* only when heated or cooled
across it (decisions #33–#35).

Defaults are v1's (SPEC §16 of v1, as calibrated through its Stage 3c): S0 1361 W/m², 20% of the sunlight absorbed in
the air at a full column (ASW0), gray infrared optical depth TAU0 = 1.4 split 0.7 / 0.3 between the lower and upper
layer at an even split, surface emissivity 0.95, ocean albedo 0.06, sensible heat 10·(1 + |u|/10) W/m²K, a 100 m
mixed-layer slab. Sensible heat goes into the bottom 100 hPa of the column (mu_bl).

Moisture (stage V2-2b, decisions #45–#48; moist=1): vapor q in each layer, evaporation from open water
K_EVAP·(1 + |u|/10)·(q_sat(T_s) − q) kg/m²/s into the bottom 100 hPa (v1's form and K_EVAP 2e-3), saturation adjustment
in each layer (the water rains out at once), moist convection across θ* relaxing the lower layer's θ_e excess over the
upper layer's saturated θ_e in τ_conv = 2 h, and thermodynamic sea ice on the slab (v1's rules: freezing below 273.15 K,
albedo 0.6 at full cover 0.05 m, a conducting skin over thicker ice).
"""
import math

import cupy as cp
import numpy as np

from .constants import CP_DRY, GRAV, KAPPA, P0
from .gpu import grid_dims

SIGMA = 5.670e-8
DAY_S, YEAR_S = 86400.0, 365 * 86400.0
TILT = math.radians(23.44)

DEFAULTS = dict(S0=1361.0, asw=0.2, tau0=1.4, mu_ref=P0 / GRAV, eps_s=0.95, h_sens=10.0, u0=10.0, alb=0.06,
                c_slab=1000.0 * 4186.0 * 100.0, mu_bl=1.0e4 / GRAV, theta_star=300.0, floor_=10.0, sigma=SIGMA,
                grav=GRAV, cp=CP_DRY, kappa=KAPPA, p0=P0, tilt=TILT, year_s=YEAR_S, day_s=DAY_S, tw_top=0.6, tw_bottom=1.4,
                moist=0.0, Lv=2.5e6, Lf=3.34e5, Tf=273.15, k_evap=2.0e-3, tau_conv=7200.0, alb_ice=0.6, rho_ice=917.0,
                ice_cover=0.05, k_ice=2.2, Rv=461.5)
PAR_ORDER = ('S0', 'asw', 'tau0', 'mu_ref', 'eps_s', 'h_sens', 'u0', 'alb', 'c_slab', 'mu_bl', 'theta_star', 'floor_', 'sigma',
             'grav', 'cp', 'kappa', 'p0', 'tilt', 'year_s', 'day_s', 'tw_top', 'tw_bottom',
             'moist', 'Lv', 'Lf', 'Tf', 'k_evap', 'tau_conv', 'alb_ice', 'rho_ice', 'ice_cover', 'k_ice', 'Rv')


def annual_mean_insolation(lat, S0=1361.0, tilt=TILT, days=365):
    """Annual- and daily-mean insolation (W/m²) at latitudes lat (radians), averaged over `days` days."""
    lat = np.asarray(lat, dtype=float)
    out = np.zeros_like(lat)
    for d in range(days):
        dec = math.asin(math.sin(tilt) * math.sin(2 * math.pi * (d + 0.5) / days))
        x = np.clip(-np.tan(lat) * math.tan(dec), -1, 1)
        h0 = np.arccos(x)
        out += S0 / math.pi * (h0 * np.sin(lat) * math.sin(dec) + np.cos(lat) * math.cos(dec) * np.sin(h0))
    return out / days


class AirPhysics:
    def __init__(self, dg, layers, params=None, mean_sun=False, forcing='radiation'):
        """forcing: 'radiation' (v1's sun, gray infrared and a slab surface) or 'held_suarez' (Newtonian relaxation of
        θ toward Held & Suarez's equilibrium; no surface)."""
        self.dg, self.L, self.forcing = dg, layers, forcing
        self.p = dict(DEFAULTS, **(params or {}))
        self.par = cp.asarray([float(self.p[k]) for k in PAR_ORDER])
        self.mode = 1 if mean_sun else 0
        N = dg.N
        self.insol = cp.asarray(annual_mean_insolation(np.arcsin(np.clip(dg.host.pos[:, 2], -1, 1)), self.p['S0'], self.p['tilt'])) \
            if mean_sun else cp.zeros(1)
        self.pos = cp.asarray(dg.host.pos)
        z = lambda *s: cp.zeros(s)
        self.uabs, self.wx, self.olr, self.sw_abs, self.stored = z(N), z(max(layers - 1, 1), N), z(N), z(N), z(N)
        self.k = {n: dg.kernel(n, 'air.cu') for n in ('air_column', 'advance_time', 'cell_speed', 'hs_column')}
        self.hs = cp.asarray([1.0 / (40 * DAY_S), 1.0 / (4 * DAY_S)])
        self.lat = cp.asarray(np.arcsin(np.clip(dg.host.pos[:, 2], -1, 1)))
        self.qnet = z(N)
        self.moist = bool(self.p['moist'])
        self.rain, self.evap, self.conv = z(N), z(N), z(N)
        self._none = z(1)                                     # q and ice for a dry column (never read)

    def column(self, mu, theta, Ts, u, t, dt, q=None, ice=None):
        """Radiation, surface exchange, heating, (moist: evaporation, condensation, sea ice, convection) and the θ*
        crossing for one step; the masses moved up go to self.wx (moved with momentum and tracers by Layers.exchange),
        rain and evaporation (kg/m²/s) to self.rain, self.evap."""
        g, k, i32, R = self.dg, self.k, np.int32, self.dg.dtype.type
        if self.forcing == 'held_suarez':
            k['hs_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, self.hs, self.lat, mu, theta, R(dt), self.wx, self.qnet,
                                             self.stored))
            k['advance_time']((1,), (1,), (t, R(dt)))
            return
        k['cell_speed'](*grid_dims(g.N), (i32(g.N), u[self.L - 1], g.cellEdge, g.le, g.de, g.areaCell, self.uabs))
        k['air_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, i32(self.mode), t, self.insol, self.pos, self.uabs,
                                          mu, theta, q if q is not None else self._none, Ts, ice if ice is not None else self._none, R(dt),
                                          self.wx, self.olr, self.sw_abs, self.stored, self.rain, self.evap, self.conv))
        k['advance_time']((1,), (1,), (t, R(dt)))


def held_suarez_theta(p, lat):
    """Held & Suarez's equilibrium potential temperature at pressure p (Pa) and latitude lat (radians)."""
    s2 = np.sin(lat) ** 2
    T = np.maximum(200.0, (315.0 - 60.0 * s2 - 10.0 * np.log(p / P0) * (1 - s2)) * (p / P0) ** KAPPA)
    return T * (P0 / p) ** KAPPA
