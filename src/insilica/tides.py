"""Moon and tides (V2-6f, decision #228): the Moon's and the Sun's tide-generating force on the ocean, and the mixing the tide's
dissipation drives. Off unless a world has `params['tides']` (a dict of TIDES' settings to change, or {} / True for Earth's); its
`moon` None takes the Moon away (the Sun's tide stays; sun_gm 0 takes that away too). Kernels in kernels/tides.cu.

The force is the degree-2 potential's gradient, times 1 + k2 - h2 = 0.69 (the solid Earth's own tide lowers it), on every
ocean layer (through the layers' Bernoulli function), so the tide is the ocean's own response: waves, amphidromes, the
tidal currents; the ocean's self-attraction and loading are left out. The Moon: Earth's (mass, mean distance, eccentricity,
inclination to the ecliptic, sidereal month, the node's 18.61-year regression and the perigee's 8.85-year advance), its
longitude, node and perigee at the March equinox the model starts from (moon_L0, node0, aps0, degrees) free settings, 0 by
default. The Sun's tide uses the air's own orbit (#184). Tidal mixing: the bottom stress's dissipation, a share gamma
(Osborn 1980's mixing efficiency, 0.2) raising the column's potential energy by mixing heat and salt across its interfaces."""
import math

import cupy as cp
import numpy as np

from .gpu import NULL, grid_dims

DAY = 86400.0
TIDES = dict(moon=dict(gm=4.9028e12, a=3.84748e8, e=0.0549, i=5.145, T=27.321662 * DAY, node_T=-6798.38 * DAY, aps_T=3232.6 * DAY,
                       L0=0.0, node0=0.0, aps0=0.0),
             sun_gm=1.32712440018e20, sun_a=1.495978707e11, love=0.69, gamma=0.2,
             cd=2.5e-3, kappa_it=2 * math.pi / 10.0e3)
# cd: the bottom's quadratic drag coefficient (global tide models' 2.5e-3; Taylor 1919, Egbert & Ray 2001); kappa_it the internal
# tide drag's wavenumber, 2 pi / 10 km (Jayne & St Laurent 2001)


def settings(tides):
    """TIDES with a world's changes (tides: dict, or True / {} for Earth's)."""
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TIDES.items()}
    if isinstance(tides, dict):
        for k, v in tides.items():
            if k == 'moon' and isinstance(v, dict):
                c['moon'].update(v)
            else:
                c[k] = v
    return c


class Tides:
    def __init__(self, dg, sea, p, radius, tides=True, rough=None, nbot=None):
        """sea: the world's Ocean; p: the air physics' settings (tilt, year_s, day_s, ecc, lam_peri); radius: the planet's (m);
        rough: the sea floor's roughness h^2 per cell (m^2; the case's), for the internal tide's drag, or None; nbot: the buoyancy
        frequency above the floor (1/s; the case's), or None (from the two lowest layers)."""
        self.dg, self.sea, self.cfg = dg, sea, settings(tides)
        c, m = self.cfg, self.cfg['moon']
        d = math.radians
        mp = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] if m is None else \
             [m['gm'], m['a'], m['e'], d(m['i']), m['T'], m['node_T'], m['aps_T'], d(m['L0']), d(m['node0']), d(m['aps0'])]
        self.par = cp.asarray([float(p['tilt']), float(p['year_s']), float(p['day_s']), float(p.get('ecc', 0.0)), float(p.get('lam_peri', 0.0)),
                               c['sun_gm'], c['sun_a']] + mp + [c['love'], float(radius)], dtype=dg.dtype)
        N = dg.N
        self.V = cp.zeros(N)                                       # the potential (m²/s²), read by the layers' Bernoulli function
        self.dsum = cp.zeros(N)                                    # the bottom stress's dissipation, summed (J/m²)
        self.pos = cp.asarray(dg.host.pos)
        sea.core.tide = self.V
        sea.core.cd_bot = float(c['cd'])                          # the bottom's quadratic drag
        self.h2 = None if rough is None else cp.asarray(np.asarray(rough, dtype=float))
        self.nbot = None if nbot is None else cp.asarray(np.asarray(nbot, dtype=float))
        if self.h2 is not None:
            sea.core.cbot = cp.zeros(dg.E)                        # the internal tide's drag, per edge, set each ocean step
        self.k = {n: dg.kernel(n, 'tides.cu') for n in ('tide_pot', 'tide_mix', 'tide_itd')}

    def force(self, time, back):
        """The potential at time[0] - back (s; the ocean step's middle), and the internal tide's drag from the stratification."""
        g, R, s = self.dg, self.dg.dtype.type, self.sea
        self.k['tide_pot'](*grid_dims(g.N), (np.int32(g.N), self.par, time, R(back), self.pos, self.V))
        if self.h2 is not None:
            self.k['tide_itd'](*grid_dims(g.E), (np.int32(g.N), np.int32(g.E), np.int32(s.L), s.h, s.T, s.S, self.h2, g.eI, g.eJ,
                                                 R(self.cfg['kappa_it'])) + s.eos() + (s.core.cbot, self.nbot if self.nbot is not None else NULL))

    def mix(self, dt):
        """The tidal mixing after an ocean step of dt seconds (the dissipation of the bottom's stress at its total coefficient)."""
        g, s, R, i32 = self.dg, self.sea, self.dg.dtype.type, np.int32
        c = s.cfg
        self.k['tide_mix'](*grid_dims(g.N), (i32(g.N), i32(g.E), i32(s.L), s.h, s.u, s.T, s.S, g.cellEdge, g.le, g.de, g.areaCell,
                                             s._land() if s._land() is not None else NULL, R(c['bottom_stress']), R(c['rho']),
                                             R(self.cfg['gamma']), R(dt)) + s.eos() + (self.dsum, R(self.cfg['cd']),
                                             s.core.cbot if s.core.cbot is not None else NULL))
