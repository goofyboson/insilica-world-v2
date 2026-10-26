"""Tracer transport on the GPU: a port of v1's `transportStep` (`src/sim/56-swe.js`, v1 SPEC §18 #45).

Works on the mass-weighted tracer M·psi given each edge's mass flux Fm (per step, eI → eJ positive).
order 1: first-order upwind. order 3: 3rd-order upwind-biased fluxes in RK3 stages with Zalesak
flux-corrected transport against the upwind solution, so tracers stay within local bounds (no negatives).
Arrays are layered: psi, M0, M1 (L, N); Fm (L, E).
"""
import cupy as cp
import numpy as np

from .gpu import NULL, grid_dims


class Transport:
    def __init__(self, dg, layers=1):
        self.dg, self.L = dg, layers
        N, E, L, dt = dg.N, dg.E, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        self.FL, self.H, self.A, self.CA = z(L, E), z(L, E), z(L, E), z(L, E)
        self.td, self.p1, self.p2, self.Rp, self.Rm = z(L, N), z(L, N), z(L, N), z(L, N), z(L, N)
        self.k = {n: dg.kernel(n) for n in ('tr_upwind_flux', 'tr_cell_update', 'tr_ho_flux', 'tr_antidiff', 'tr_fct_ratios',
                                            'tr_fct_edge', 'tr_fct_apply', 'mass_update')}

    def mass_after(self, M0, Fm, M1):
        """M1 = M0 moved by Fm (what the cells hold after the step)."""
        g, i32 = self.dg, np.int32
        self.k['mass_update'](*grid_dims(g.N, self.L), (i32(g.N), i32(g.E), M0, Fm, g.cellEdgeAsc, g.eI, M1))

    def step(self, M0, M1, Fm, psi, order=3):
        g, k, L, i32, R = self.dg, self.k, self.L, np.int32, g_real(self.dg)
        cg, eg = grid_dims(g.N, L), grid_dims(g.E, L)
        k['tr_upwind_flux'](*eg, (i32(g.N), i32(g.E), Fm, psi, g.eI, g.eJ, self.FL))
        k['tr_cell_update'](*cg, (i32(g.N), i32(g.E), M0, psi, self.FL, R(1), NULL, M1, g.cellEdgeAsc, g.eI, self.td))
        if order == 1:
            psi[...] = self.td
            return psi
        for src, frac, out in ((psi, 1 / 3, self.p1), (self.p1, 1 / 2, self.p2)):
            k['tr_ho_flux'](*eg, (i32(g.N), i32(g.E), Fm, src, g.d2idx, g.d2wt, g.eI, g.eJ, self.H))
            k['tr_cell_update'](*cg, (i32(g.N), i32(g.E), M0, psi, self.H, R(frac), Fm, M1, g.cellEdgeAsc, g.eI, out))
        k['tr_ho_flux'](*eg, (i32(g.N), i32(g.E), Fm, self.p2, g.d2idx, g.d2wt, g.eI, g.eJ, self.H))
        k['tr_antidiff'](*eg, (i32(g.E), self.H, self.FL, self.A))
        k['tr_fct_ratios'](*cg, (i32(g.N), i32(g.E), psi, self.td, M1, self.A, g.nbr, g.cellEdgeAsc, g.eI, self.Rp, self.Rm))
        k['tr_fct_edge'](*eg, (i32(g.N), i32(g.E), self.A, self.Rp, self.Rm, g.eI, g.eJ, self.CA))
        k['tr_fct_apply'](*cg, (i32(g.N), i32(g.E), self.td, M1, self.CA, g.cellEdgeAsc, g.eI, psi))
        return psi


def g_real(dg):
    return dg.dtype.type
