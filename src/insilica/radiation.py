"""Longwave and shortwave radiation by k-tables (V2-6a, SPEC #109, #110; kernels/air.cu lw_column): 136 g-points in 10 bands from
HITRAN lines and the MT_CKD continuum, fitted to line-by-line (tools/make_ktables.py; data/lw_ktables.npz). Run every
radiation step; the column physics uses its saved fluxes at the layers' edges until the next one, the surface's emission
spread by the saved shares, so energy is conserved exactly between radiation steps."""
from pathlib import Path

import cupy as cp
import numpy as np

DATA = Path(__file__).resolve().parent / 'data' / 'lw_ktables.npz'
HAZE_SCALE = 1500.0                         # m, the haze's scale height (tools/swlbl.HAZE)


class Longwave:
    def __init__(self, dg, layers, par, co2=400e-6, n2o=0.32e-6, ch4=1.8e-6):
        t = np.load(DATA)
        self.dg, self.L, self.par = dg, layers, par
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
        self.sub = cp.zeros((4 * layers * 14, dg.N), dtype=cp.float32)        # the sub-levels' properties (lw_setup)
        self.ks = dg.kernel('lw_setup', 'air.cu', (f'-DNL={layers}',))
        self.k = dg.kernel('lw_column', 'air.cu', (f'-DNL={layers}',))

    def __call__(self, mu, theta, q, cloudo, Ts):
        """cloudo: AirPhysics.cloudo (the clouds' cover and longwave optical depth at the column's last step), or None."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        from .gpu import NULL, BLOCK
        self.ks(((g.N + BLOCK - 1) // BLOCK, self.L), (BLOCK,),
                (i32(g.N), i32(self.L), self.par, i32(self.NP), i32(self.NT), R(self.lp0), R(self.dlp), R(self.t0),
                 R(self.dtt), self.o3, i32(len(self.o3)), R(self.lp0), R(self.dlp), *(R(x) for x in self.gases), mu, theta,
                 q if q is not None else NULL, cloudo if cloudo is not None else NULL, R(HAZE_SCALE), self.sub))
        self.k((g.N,), (self.block,), (i32(g.N), i32(self.L), self.par, i32(self.NG), self.logk, self.planck,
                                        i32(self.NP), i32(self.NT), i32(self.planck.shape[0]), R(self.tp0), self.sub, Ts,
                                        self.radf))


SW_DATA = Path(__file__).resolve().parent / 'data' / 'sw_ktables.npz'
# the clouds' single-scattering albedo per shortwave band (cm-1 ranges of tools/make_swtables.BANDS; estimates for drops of
# about 10 um: water absorbs strongly from 2.5 um, weakly at 1.2-2.5 um, hardly below; after Slingo 1989's bands)
CLOUD_SSA = (0.60, 0.75, 0.98, 0.994, 0.9995, 0.9998, 0.99999, 0.99999, 0.99999, 0.99999)


class Shortwave:
    def __init__(self, dg, layers, par, lw):
        t = np.load(SW_DATA)
        self.dg, self.L, self.par, self.lwm = dg, layers, par, lw
        self.logk = cp.asarray(np.ascontiguousarray(t['logk']))
        self.NC, self.NP, self.NT, self.NG = self.logk.shape
        f32 = lambda a: cp.asarray(np.asarray(a, np.float32))
        self.sun, self.ray, self.hzf = f32(t['sun'] / t['sun'].sum()), f32(t['ray']), f32(t['haze'])
        self.cssa = f32(np.repeat(CLOUD_SSA, t['ng']))
        self.haze = t['haze_par']                                  # tau550, ssa, g, scale height
        self.yscale = float((10.0) / (t['T'][1] - t['T'][0]))      # lw_setup's temperature positions are in 10 K steps
        self.block = (self.NG + 31) // 32 * 32
        self.swf = cp.zeros((3 * layers + 2, dg.N), dtype=dg.dtype)   # layers, surface, clear sky's reflection, each layer's top and bottom sub-level
        self.k = dg.kernel('sw_column', 'air.cu', (f'-DNL={layers}',))

    def __call__(self, mode, t, interval, pos, alb, cloudo):
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        from .gpu import NULL
        self.k((g.N,), (self.block,), (i32(g.N), i32(self.L), self.par, i32(self.NG), self.logk, i32(self.NP), i32(self.NT),
                                        R(self.yscale), self.sun, self.ray, self.hzf, self.cssa, R(self.haze[0]),
                                        R(self.haze[1]), R(self.haze[2]), self.lwm.sub, i32(mode), t, R(interval), pos, alb,
                                        cloudo if cloudo is not None else NULL, self.swf))
