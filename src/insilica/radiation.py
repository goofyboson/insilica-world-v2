"""Longwave radiation by k-tables (V2-6a, SPEC #109, #110; kernels/air.cu lw_column): 136 g-points in 10 bands from
HITRAN lines and the MT_CKD continuum, fitted to line-by-line (tools/make_ktables.py; data/lw_ktables.npz). Run every
radiation step; the column physics uses its saved fluxes at the layers' edges until the next one, the surface's emission
spread by the saved shares, so energy is conserved exactly between radiation steps."""
from pathlib import Path

import cupy as cp
import numpy as np

DATA = Path(__file__).resolve().parent / 'data' / 'lw_ktables.npz'


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
        self.block = (self.NG + 31) // 32 * 32
        self.radf = cp.zeros((3 * (layers + 1) + 3, dg.N), dtype=dg.dtype)     # (the last row: the surface's Planck sum)
        self.sub = cp.zeros((4 * layers * 12, dg.N), dtype=cp.float32)        # the sub-levels' properties (lw_setup)
        self.ks = dg.kernel('lw_setup', 'air.cu', (f'-DNL={layers}',))
        self.k = dg.kernel('lw_column', 'air.cu', (f'-DNL={layers}',))

    def __call__(self, mu, theta, q, cloudo, Ts):
        """cloudo: AirPhysics.cloudo (the clouds' cover and longwave optical depth at the column's last step), or None."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        from .gpu import NULL, BLOCK
        self.ks(((g.N + BLOCK - 1) // BLOCK, self.L), (BLOCK,),
                (i32(g.N), i32(self.L), self.par, i32(self.NP), i32(self.NT), R(self.lp0), R(self.dlp), R(self.t0),
                 R(self.dtt), self.o3, i32(len(self.o3)), R(self.lp0), R(self.dlp), *(R(x) for x in self.gases), mu, theta,
                 q if q is not None else NULL, cloudo if cloudo is not None else NULL, self.sub))
        self.k((g.N,), (self.block,), (i32(g.N), i32(self.L), self.par, i32(self.NG), self.logk, self.planck,
                                        i32(self.NP), i32(self.NT), i32(self.planck.shape[0]), R(self.tp0), self.sub, Ts,
                                        self.radf))
