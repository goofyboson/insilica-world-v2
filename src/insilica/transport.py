"""Tracer transport on the GPU: v1's `transportStep` (`src/sim/56-swe.js`, v1 SPEC §18 #45).

Works on the mass-weighted tracer M·psi given each edge's mass flux Fm (per step, eI → eJ positive).
order 1: first-order upwind. order 3: 3rd-order upwind-biased fluxes in RK3 stages with Zalesak
flux-corrected transport against the upwind solution, so tracers stay within local bounds (no negatives).
Arrays are layered: psi, M0, M1 (L, N); Fm (L, E). Fused kernels (kernels/fused.cu): every tracer a world carries goes
through the same launches, stacked as rows tracer × layer, and each 3rd-order edge flux is computed once per edge and
summed by the cell passes (V2-5). Allocates nothing per step.
"""
import cupy as cp
import numpy as np

from .gpu import grid_dims

BETA = 0.25         # upwind weight of the 3rd-order flux: MPAS's default (SPEC §9 #21)


class Transport:
    def __init__(self, dg, layers=1, beta=BETA, tracers=1):
        """beta: weight of the 3rd-order flux's upwind term, f (mean − a − b) + |f| (b − a) β (SPEC §9 #20, #21):
        1 is 3rd-order upwind, 0 is 4th-order centered; the default 0.25 is MPAS's. tracers: the most moved together."""
        self.dg, self.L, self.beta, self.K = dg, layers, beta, tracers
        N, E, L, dt = dg.N, dg.E, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        R = tracers * L
        self.psi, self.td, self.p1, self.p2, self.Rp, self.Rm, self.out = (z(R, N) for _ in range(7))
        self.Hf, self.A = z(R, E), z(R, E)
        self.k = {n: dg.kernel(n, 'fused.cu') for n in ('tr_td', 'tr_hflux', 'tr_stage', 'tr_aflux', 'tr_ratios', 'tr_apply')}
        self.k['mass_update'] = dg.kernel('mass_update')

    def mass_after(self, M0, Fm, M1):
        """M1 = M0 moved by Fm (what the cells hold after the step)."""
        g, i32 = self.dg, np.int32
        self.k['mass_update'](*grid_dims(g.N, self.L), (i32(g.N), i32(g.E), M0, Fm, g.cellEdgeAsc, g.eI, M1))

    def step(self, M0, M1, Fm, psi, order=3):
        return self.step_many(M0, M1, Fm, [psi], order)[0]

    def step_many(self, M0, M1, Fm, tracers, order=3):
        """Move each (L, N) tracer in `tracers` by the same mass fluxes, in place, in one set of launches."""
        g, k, i32, R = self.dg, self.k, np.int32, self.dg.dtype.type
        n, L, N, E = len(tracers), self.L, g.N, g.E
        assert n <= self.K
        rows = slice(0, n * L)
        psi, td, p1, p2, Rp, Rm, out, Hf, A = (a[rows] for a in (self.psi, self.td, self.p1, self.p2, self.Rp, self.Rm,
                                                                  self.out, self.Hf, self.A))
        for j, t in enumerate(tracers):
            cp.copyto(psi[j * L:(j + 1) * L], t)
        cg, eg, Lm, b = grid_dims(N, n * L), grid_dims(E, n * L), i32(L), R(self.beta)
        k['tr_td'](*cg, (i32(N), i32(E), Lm, M0, psi, Fm, M1, g.cellEdgeAsc, g.eI, g.eJ, td))
        if order == 1:
            res = td
        else:
            for src, frac, dst in ((psi, 1 / 3, p1), (p1, 1 / 2, p2)):
                k['tr_hflux'](*eg, (i32(E), Lm, src, R(frac), Fm, g.d2idx, g.d2wt_tr, g.eI, g.eJ, i32(N), b, Hf))
                k['tr_stage'](*cg, (i32(N), i32(E), Lm, M0, psi, Hf, R(frac), Fm, g.cellEdgeAsc, g.eI, dst))
            k['tr_aflux'](*eg, (i32(E), Lm, psi, p2, Fm, g.d2idx, g.d2wt_tr, g.eI, g.eJ, i32(N), b, A))
            k['tr_ratios'](*cg, (i32(N), i32(E), Lm, psi, td, M1, A, g.nbr, g.cellEdgeAsc, g.eI, Rp, Rm))
            k['tr_apply'](*cg, (i32(N), i32(E), Lm, td, M1, A, Rp, Rm, g.cellEdgeAsc, g.eI, g.eJ, out))
            res = out
        for j, t in enumerate(tracers):
            cp.copyto(t, res[j * L:(j + 1) * L])
        return tracers
