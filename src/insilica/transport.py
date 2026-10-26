"""Tracer transport on the GPU: v1's `transportStep` (`src/sim/56-swe.js`, v1 SPEC §18 #45).

Works on the mass-weighted tracer M·psi given each edge's mass flux Fm (per step, eI → eJ positive).
order 1: first-order upwind. order 3: 3rd-order upwind-biased fluxes in RK3 stages with Zalesak
flux-corrected transport against the upwind solution, so tracers stay within local bounds (no negatives).
Arrays are layered: psi, M0, M1 (L, N); Fm (L, E). Fused kernels (kernels/fused.cu): five cell passes, the edge
fluxes recomputed where they are summed; bit-identical to reference.py (test E2). Allocates nothing.
"""
import cupy as cp
import numpy as np

from .gpu import grid_dims

BETA = 0.25         # upwind weight of the 3rd-order flux: MPAS's default (SPEC §9 #21)


class Transport:
    def __init__(self, dg, layers=1, beta=BETA):
        """beta: weight of the 3rd-order flux's upwind term, f (mean − a − b) + |f| (b − a) β (SPEC §9 #20, #21):
        1 is 3rd-order upwind, 0 is 4th-order centered; the default 0.25 is MPAS's."""
        self.dg, self.L, self.beta = dg, layers, beta
        N, L, dt = dg.N, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        self.td, self.p1, self.p2, self.Rp, self.Rm, self.out = z(L, N), z(L, N), z(L, N), z(L, N), z(L, N), z(L, N)
        self.k = {n: dg.kernel(n, 'fused.cu') for n in ('tr_td', 'tr_stage', 'tr_ratios', 'tr_apply')}
        self.k['mass_update'] = dg.kernel('mass_update')

    def mass_after(self, M0, Fm, M1):
        """M1 = M0 moved by Fm (what the cells hold after the step)."""
        g, i32 = self.dg, np.int32
        self.k['mass_update'](*grid_dims(g.N, self.L), (i32(g.N), i32(g.E), M0, Fm, g.cellEdgeAsc, g.eI, M1))

    def step(self, M0, M1, Fm, psi, order=3):
        g, k, i32, R = self.dg, self.k, np.int32, self.dg.dtype.type
        cg = grid_dims(g.N, self.L)
        k['tr_td'](*cg, (i32(g.N), i32(g.E), M0, psi, Fm, M1, g.cellEdgeAsc, g.eI, g.eJ, self.td))
        if order == 1:
            cp.copyto(psi, self.td)
            return psi
        k['tr_stage'](*cg, (i32(g.N), i32(g.E), M0, psi, psi, R(1 / 3), Fm, g.d2idx, g.d2wt, g.cellEdgeAsc, g.eI, g.eJ, R(self.beta), self.p1))
        k['tr_stage'](*cg, (i32(g.N), i32(g.E), M0, psi, self.p1, R(1 / 2), Fm, g.d2idx, g.d2wt, g.cellEdgeAsc, g.eI, g.eJ, R(self.beta), self.p2))
        k['tr_ratios'](*cg, (i32(g.N), i32(g.E), psi, self.td, M1, self.p2, Fm, g.d2idx, g.d2wt, g.nbr, g.cellEdgeAsc, g.eI, g.eJ,
                             R(self.beta), self.Rp, self.Rm))
        k['tr_apply'](*cg, (i32(g.N), i32(g.E), psi, self.td, M1, self.p2, Fm, self.Rp, self.Rm, g.d2idx, g.d2wt, g.cellEdgeAsc,
                            g.eI, g.eJ, R(self.beta), self.out))
        cp.copyto(psi, self.out)
        return psi
