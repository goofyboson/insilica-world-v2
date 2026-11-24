"""Longwave and shortwave radiation by k-tables (V2-6a, SPEC #109, #110; kernels/air.cu lw_column): 136 g-points in 10 bands from
HITRAN lines and the MT_CKD continuum, fitted to line-by-line (tools/make_ktables.py; data/lw_ktables.npz). Run every
radiation step; the column physics uses its saved fluxes at the layers' edges until the next one, the surface's emission
spread by the saved shares, so energy is conserved exactly between radiation steps."""
from pathlib import Path

import cupy as cp
import numpy as np

DATA = Path(__file__).resolve().parent / 'data' / 'lw_ktables.npz'
# (#226) another air's: the tables keeping the weak lines of large CO2 and CH4 columns, and collision-induced absorption per g-point
DATA_WIDE = Path(__file__).resolve().parent / 'data' / 'lw_ktables_wide.npz'
CIA_DATA = Path(__file__).resolve().parent / 'data' / 'cia_ktables.npz'
# (V2-7) the other-air table sets: 'wide' (#226) and Mars's own ('mars': grouped and fitted on Mars's columns, 100-340 K), each its
# longwave, shortwave and collision-induced absorption
TABLES = {name: tuple(Path(__file__).resolve().parent / 'data' / f for f in files) for name, files in
          dict(wide=('lw_ktables_wide.npz', 'sw_ktables_wide.npz', 'cia_ktables.npz'),
               mars=('lw_ktables_mars.npz', 'sw_ktables_mars.npz', 'cia_ktables_mars.npz')).items()}


def cia_table(weights, which, tables='wide'):
    """The CIA pairs' tables (T x g, x 1e40) summed with weights x_a x_b (constants.CIA_PAIRS' order), float32 on the GPU."""
    from .constants import CIA_PAIRS
    t = np.load(TABLES[tables][2])
    assert tuple(str(x) for x in t['pairs']) == CIA_PAIRS
    a = np.tensordot(np.asarray(weights, float), t[which].astype(float), axes=1)
    return cp.asarray(a.astype(np.float32))
HAZE_SCALE = 1500.0                         # m, the haze's scale height (tools/swlbl.HAZE)


class Dust:
    """(V2-7, #239) A planet's prescribed dust: maps of its column (9.3 um absorption at 610 Pa, 5-sol means by solar longitude;
    tools/make_mars_dust.py) read onto the cells, and its optics per g-point of the longwave and shortwave tables. Each radiation step
    dust_update (in the captured tick) gives each cell its column at the season, times its surface pressure over 610 Pa (dtau), and
    the profile's top (dzmax, km); lw_setup spreads it by the profile, lw_column absorbs and sw_column scatters and absorbs."""

    def __init__(self, dg, layers, par, maps, optics, lw_ng, sw_ng, defs=()):
        d = np.load(Path(__file__).resolve().parent / 'data' / maps)
        o = np.load(Path(__file__).resolve().parent / 'data' / optics)
        h = dg.host
        lat = np.arcsin(np.clip(h.pos[:, 2], -1, 1))
        lon = np.arctan2(h.pos[:, 1], h.pos[:, 0])
        la, lo = d['lat'], d['lon']                               # (rising; the map's cell centres)
        c = d['cdod610'].astype(float)                            # NB x lat x lon
        # bilinear in latitude and (periodic) longitude
        y = np.clip(np.interp(np.degrees(lat), la, np.arange(len(la))), 0, len(la) - 1 - 1e-9)
        dl = lo[1] - lo[0]
        x = ((np.degrees(lon) - lo[0]) / dl) % len(lo)
        y0, x0 = y.astype(int), x.astype(int)
        fy, fx = y - y0, x - x0
        x1 = (x0 + 1) % len(lo)
        cells = ((1 - fy) * ((1 - fx) * c[:, y0, x0] + fx * c[:, y0, x1]) + fy * ((1 - fx) * c[:, y0 + 1, x0] + fx * c[:, y0 + 1, x1]))
        self.dg, self.L, self.par = dg, layers, par
        self.tab = cp.asarray(np.ascontiguousarray(cells, dtype=np.float32))   # (row-major: fancy indexing gave column-major)
        self.ls = cp.asarray(np.asarray(d['ls'], float))
        self.lat = cp.asarray(lat)
        self.dtau, self.dzmax = cp.zeros(dg.N, dtype=dg.dtype), cp.full(dg.N, 60.0, dtype=dg.dtype)
        f32 = lambda a: cp.asarray(np.asarray(a, np.float32))
        self.lw_abs = f32(np.repeat(o['lw_ext'] * (1 - o['lw_ssa']), lw_ng))
        self.sw_ext, self.sw_ssa, self.sw_g = (f32(np.repeat(o[k], sw_ng)) for k in ('sw_ext', 'sw_ssa', 'sw_g'))
        self.k = dg.kernel('dust_update', 'air.cu', (f'-DNL={layers}',) + tuple(defs))

    def update(self, t, h):
        from .gpu import BLOCK
        g, i32 = self.dg, np.int32
        self.k(((g.N + BLOCK - 1) // BLOCK,), (BLOCK,), (i32(g.N), i32(self.L), self.par, t, i32(len(self.ls)), self.tab, self.ls, self.lat,
                                                       h, self.dtau, self.dzmax))


class Longwave:
    def __init__(self, dg, layers, par, co2=400e-6, n2o=0.32e-6, ch4=1.8e-6, tmode=2, defs=(), cia=None, tables='wide'):
        """cia (#226): another air's CIA pair weights (constants.air_radiation): the wide tables and the pairs' absorption;
        tables (V2-7): which other-air set (TABLES)."""
        t = np.load(DATA if cia is None else TABLES[tables][0])
        self.cia = None if cia is None else cia_table(cia, 'lw', tables)
        self.dg, self.L, self.par, self.tmode = dg, layers, par, int(tmode)
        self.logk = cp.asarray(np.ascontiguousarray(t['logk']))
        self.planck = cp.asarray(np.ascontiguousarray(t['planck']))
        self.o3 = cp.asarray(t['o3'])
        P, T, Tp = t['P'], t['T'], t['T_pl']
        self.NC, self.NP, self.NT, self.NG = self.logk.shape
        self.lp0, self.dlp = float(np.log(P[0])), float((np.log(P[0]) - np.log(P[-1])) / (len(P) - 1))
        self.t0, self.dtt, self.tp0 = float(T[0]), float(T[1] - T[0]), float(Tp[0])
        self.gases = (co2, n2o, ch4)
        assert self.NG <= 144
        self.block = (self.NG + 31) // 32 * 32
        self.radf = cp.zeros((3 * (layers + 1) + 3 + 6 * layers, dg.N), dtype=dg.dtype)   # edges; clear sky; the Planck sum; each layer's edge sub-levels
        self.sub = cp.zeros((4 * layers * 15, dg.N), dtype=cp.float32)        # the sub-levels' properties (lw_setup)
        self.ks = dg.kernel('lw_setup', 'air.cu', (f'-DNL={layers}',) + tuple(defs))
        self.k = dg.kernel('lw_column', 'air.cu', (f'-DNL={layers}',) + tuple(defs))
        # (#250) its reduction rows in dynamic shared memory, 144 floats each; past 48 KB the kernel opts in
        self.shm = (3 * (layers + 1) + 3 + 6 * layers) * 144 * 4
        if self.shm > 48 * 1024:
            stat = self.k.shared_size_bytes
            self.k.max_dynamic_shared_size_bytes = self.shm
            assert stat + self.shm <= 99 * 1024, 'lw_column: shared memory past the 99 KB a block can hold'
        self.ng = np.asarray(t['ng'])
        self.dust = None                                           # (V2-7, #239) a Dust, set by AirPhysics
        self.seps = None                                           # (#240) the surface's emissivity per cell, set by AirPhysics

    def __call__(self, mu, theta, q, cloudo, Ts, gx=None):
        """cloudo: AirPhysics.cloudo (the clouds' cover and longwave optical depth at the column's last step), or None. gx: the
        carried CO2, N2O and CH4 (3, L, N; #217), or None for the planet's own."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        from .gpu import NULL, BLOCK
        self.ks(((g.N + BLOCK - 1) // BLOCK, self.L), (BLOCK,),
                (i32(g.N), i32(self.L), self.par, i32(self.NP), i32(self.NT), R(self.lp0), R(self.dlp), R(self.t0),
                 R(self.dtt), self.o3, i32(len(self.o3)), R(self.lp0), R(self.dlp), *(R(x) for x in self.gases), gx if gx is not None else NULL, mu, theta,
                 q if q is not None else NULL, cloudo if cloudo is not None else NULL, R(HAZE_SCALE), i32(self.tmode), self.sub,
                 self.dust.dzmax if self.dust is not None else NULL))
        self.k((g.N,), (self.block,), (i32(g.N), i32(self.L), self.par, i32(self.NG), self.logk, self.planck,
                                        i32(self.NP), i32(self.NT), i32(self.planck.shape[0]), R(self.tp0), self.sub, Ts,
                                        self.radf, self.cia if self.cia is not None else NULL,
                                        self.dust.dtau if self.dust is not None else NULL, self.dust.lw_abs if self.dust is not None else NULL,
                                        self.seps if self.seps is not None else NULL), shared_mem=self.shm)


SW_DATA = Path(__file__).resolve().parent / 'data' / 'sw_ktables.npz'
SW_DATA_WIDE = Path(__file__).resolve().parent / 'data' / 'sw_ktables_wide.npz'
# the clouds' single-scattering albedo per shortwave band (cm-1 ranges of tools/make_swtables.BANDS; estimates for drops of
# about 10 um: water absorbs strongly from 2.5 um, weakly at 1.2-2.5 um, hardly below; after Slingo 1989's bands)
CLOUD_SSA = (0.60, 0.75, 0.98, 0.994, 0.9995, 0.9998, 0.99999, 0.99999, 0.99999, 0.99999)
# clouds are uneven within a cell: a plane-parallel cloud of the mean water path reflects more than the real one; the
# standard correction scales its optical depth (Cahalan et al. 1994; #111)
CLOUD_INHOM = 0.7
# ice_sw (V2-6c, #154): ice clouds' own single-scattering albedo and asymmetry, Ebert & Curry (1992) as RRTMG's (iceflag 1):
# 1 - ssa = c + d r_e, g = e + f r_e (r_e in um), in five intervals chosen by each band's upper edge as RRTMG does
# (> 14300, > 7700, > 5300, > 4000, >= 2500 cm-1; below 2500 the last); liquid keeps CLOUD_SSA and cloud_g
EC92_C = (1.000e-05, 1.100e-04, 1.240e-02, 3.779e-02, 4.666e-01)
EC92_D = (0.0, 1.405e-05, 6.867e-04, 1.284e-03, 2.050e-05)
EC92_E = (7.661e-01, 7.730e-01, 7.865e-01, 8.172e-01, 9.595e-01)
EC92_F = (5.851e-04, 5.665e-04, 7.204e-04, 7.463e-04, 1.076e-04)
BAND_TOP = (2600, 4000, 5250, 7700, 9350, 12850, 16000, 29000, 38000, 50000)     # cm-1, tools/make_swtables.BANDS


def ice_optics(re_um):
    """Per band (ssa, g) of ice clouds of effective radius re_um (Ebert & Curry 1992)."""
    out = []
    for top in BAND_TOP:
        j = 0 if top > 14300 else 1 if top > 7700 else 2 if top > 5300 else 3 if top > 4000 else 4
        out.append((1 - EC92_C[j] - EC92_D[j] * re_um, min(EC92_E[j] + EC92_F[j] * re_um, 1 - 1e-6)))
    return out


class Shortwave:
    def __init__(self, dg, layers, par, lw, inhom=CLOUD_INHOM, ice_re=None, defs=(), cia=None, tables='wide'):
        t = np.load(SW_DATA if cia is None else TABLES[tables][1])      # (V2-7) tables: the other-air set
        self.cia = None if cia is None else cia_table(cia, 'sw', tables)
        self.dg, self.L, self.par, self.lwm, self.inhom = dg, layers, par, lw, inhom
        self.logk = cp.asarray(np.ascontiguousarray(t['logk']))
        self.NC, self.NP, self.NT, self.NG = self.logk.shape
        f32 = lambda a: cp.asarray(np.asarray(a, np.float32))
        self.sun, self.ray, self.hzf = f32(t['sun'] / t['sun'].sum()), f32(t['ray']), f32(t['haze'])
        self.cssa = f32(np.repeat(CLOUD_SSA, t['ng']))
        self.cssa_i = self.g_i = None                              # (ice_sw, #154: the ice clouds' own optics; None: the drops')
        if ice_re is not None:
            io = ice_optics(ice_re * 1e6)
            self.cssa_i, self.g_i = f32(np.repeat([x[0] for x in io], t['ng'])), f32(np.repeat([x[1] for x in io], t['ng']))
        self.haze = t['haze_par']                                  # tau550, ssa, g, scale height
        self.yscale = float((10.0) / (t['T'][1] - t['T'][0]))      # lw_setup's temperature positions are in 10 K steps
        self.block = (self.NG + 31) // 32 * 32
        self.swf = cp.zeros((3 * layers + 2, dg.N), dtype=dg.dtype)   # layers, surface, clear sky's reflection, each layer's top and bottom sub-level
        self.k = dg.kernel('sw_column', 'air.cu', (f'-DNL={layers}',) + tuple(defs))
        self.ng = np.asarray(t['ng'])

    def __call__(self, mode, t, interval, pos, alb, cloudo):
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        from .gpu import NULL
        self.k((g.N,), (self.block,), (i32(g.N), i32(self.L), self.par, i32(self.NG), self.logk, i32(self.NP), i32(self.NT),
                                        R(self.yscale), self.sun, self.ray, self.hzf, self.cssa, R(self.haze[0]),
                                        R(self.haze[1]), R(self.haze[2]), R(self.inhom), self.lwm.sub, i32(mode), t, R(interval), pos, alb,
                                        self.cssa_i if self.cssa_i is not None else NULL, self.g_i if self.g_i is not None else NULL,
                                        cloudo if cloudo is not None else NULL, self.swf, self.cia if self.cia is not None else NULL,
                                        *((du.dtau, du.sw_ext, du.sw_ssa, du.sw_g) if (du := self.lwm.dust) is not None else (NULL,) * 4)))
