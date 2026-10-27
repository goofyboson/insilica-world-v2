"""Column physics of the air layers (stage V2-2a, kernels/air.cu): v1's radiation and surface exchange over a slab
ocean, heating as potential temperature, and mass crossing the isentropic interface θ* only when heated or cooled
across it (decisions #33–#35).

Defaults are v1's (SPEC §16 of v1, as calibrated through its Stage 3c): S0 1361 W/m², 20% of the sunlight absorbed in
the air at a full column (ASW0), gray infrared optical depth TAU0 = 1.4 split 0.7 / 0.3 between the lower and upper
layer at an even split, surface emissivity 0.95, ocean albedo 0.06, sensible heat 10·(1 + |u|/10) W/m²K, a 100 m
mixed-layer slab. Sensible heat goes into the bottom 100 hPa of the column (mu_bl).
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
                grav=GRAV, cp=CP_DRY, kappa=KAPPA, p0=P0, tilt=TILT, year_s=YEAR_S, day_s=DAY_S, tw_top=0.6, tw_bottom=1.4)
PAR_ORDER = ('S0', 'asw', 'tau0', 'mu_ref', 'eps_s', 'h_sens', 'u0', 'alb', 'c_slab', 'mu_bl', 'theta_star', 'floor_', 'sigma',
             'grav', 'cp', 'kappa', 'p0', 'tilt', 'year_s', 'day_s', 'tw_top', 'tw_bottom')


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

    def column(self, mu, theta, Ts, u, t, dt):
        """Radiation, surface exchange, heating and the θ* crossing for one step; the crossing masses go to self.wx
        (moved with momentum and tracers by Layers.exchange)."""
        g, k, i32, R = self.dg, self.k, np.int32, self.dg.dtype.type
        if self.forcing == 'held_suarez':
            k['hs_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, self.hs, self.lat, mu, theta, R(dt), self.wx, self.qnet,
                                             self.stored))
            k['advance_time']((1,), (1,), (t, R(dt)))
            return
        k['cell_speed'](*grid_dims(g.N), (i32(g.N), u[self.L - 1], g.cellEdge, g.le, g.de, g.areaCell, self.uabs))
        k['air_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, i32(self.mode), t, self.insol, self.pos, self.uabs,
                                          mu, theta, Ts, R(dt), self.wx, self.olr, self.sw_abs, self.stored))
        k['advance_time']((1,), (1,), (t, R(dt)))


def held_suarez_theta(p, lat):
    """Held & Suarez's equilibrium potential temperature at pressure p (Pa) and latitude lat (radians)."""
    s2 = np.sin(lat) ** 2
    T = np.maximum(200.0, (315.0 - 60.0 * s2 - 10.0 * np.log(p / P0) * (1 - s2)) * (p / P0) ** KAPPA)
    return T * (P0 / p) ** KAPPA
