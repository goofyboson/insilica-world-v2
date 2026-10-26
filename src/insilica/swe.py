"""Shallow water on the TRiSK C-grid, on the GPU: v1's `makeSolver` with one layer (`src/sim/56-swe.js`),
generalized to L independent layers stacked in one array (shape (L, N) and (L, E)).

    dh/dt = −div(h_e u)
    du/dt = Σ w_ee' F_e' (q_e + q_e')/2 − ∂(g(h + b) + K)/∂n − r u − ν4 ∇⁴u

Wicker–Skamarock RK3 with v1's positivity limiter in every stage; the ∇⁴ term is evaluated once per step from the
step's starting wind and held through the stages (as MPAS and v1). Fused kernels (kernels/fused.cu): five
launches per stage, fifteen per step (21 with ∇⁴), doing v1's arithmetic in v1's order (checked against v1 by
V1-ENGINE). The step allocates nothing, so it can be captured in a CUDA graph (world.py).
"""
import cupy as cp
import numpy as np

from .constants import GRAV
from .gpu import NULL, grid_dims


class SWE:
    def __init__(self, dg, layers=1, grav=GRAV, b=None, drag=0.0, nu4=0.0):
        self.dg, self.L = dg, layers
        self.real = dg.dtype.type
        self.grav, self.nu4 = grav, nu4
        self.b = cp.zeros(dg.N, dtype=dg.dtype) if b is None else cp.asarray(b, dtype=dg.dtype)
        self.drag = cp.asarray(np.broadcast_to(np.asarray(drag, dtype=float), (layers,)), dtype=dg.dtype)
        N, E, NV, L, dt = dg.N, dg.E, dg.NV, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        self.F, self.B, self.q, self.scale = z(L, E), z(L, N), z(L, NV), z(L, N)
        self.h1, self.h2, self.u1, self.u2 = z(L, N), z(L, N), z(L, E), z(L, E)
        self.D1, self.Z1, self.L1, self.L2 = z(L, N), z(L, NV), z(L, E), z(L, E)
        self.k = {n: dg.kernel(n, 'fused.cu') for n in ('swe_scale', 'swe_flux', 'swe_cell', 'swe_vertex', 'swe_edge')}
        self.k.update({n: dg.kernel(n) for n in ('cell_div', 'vertex_curl', 'edge_lap')})

    def _lap(self, u, out):
        g, k, L, i32 = self.dg, self.k, self.L, np.int32
        k['cell_div'](*grid_dims(g.N, L), (i32(g.N), i32(g.E), u, g.cellEdge, g.cellSign, g.le, g.areaCell, self.D1))
        k['vertex_curl'](*grid_dims(g.NV, L), (i32(g.E), i32(g.NV), u, g.vEdge, g.vEdgeSign, g.de, g.areaV, self.Z1))
        k['edge_lap'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), i32(g.NV), self.D1, self.Z1, g.eI, g.eJ, g.ev1, g.ev2, g.de, g.le, out))

    def _stage(self, hs, us, h0, u0, c, hn, un, Fm=None, dt=0.0):
        """One RK stage: tendencies of (hs, us), written as hn = h0 + c·dh, un = u0 + c·du (limiter for a stage of c)."""
        g, k, L, i32, R = self.dg, self.k, self.L, np.int32, self.real
        cg, eg, vg = grid_dims(g.N, L), grid_dims(g.E, L), grid_dims(g.NV, L)
        k['swe_scale'](*cg, (i32(g.N), i32(g.E), hs, us, g.cellEdgeAsc, g.eI, g.eJ, g.le, g.areaCell, R(c), self.scale))
        k['swe_flux'](*eg, (i32(g.N), i32(g.E), hs, us, g.eI, g.eJ, self.scale, g.le, R(dt), self.F, Fm if Fm is not None else NULL))
        k['swe_cell'](*cg, (i32(g.N), i32(g.E), hs, self.b, self.F, us, g.cellEdge, g.cellSign, g.le, g.de, g.areaCell, R(self.grav),
                            h0, R(c), hn, self.B))
        k['swe_vertex'](*vg, (i32(g.N), i32(g.E), i32(g.NV), hs, us, g.vCell, g.vSlot, g.kite, g.vEdge, g.vEdgeSign, g.de,
                              g.areaV, g.fV, self.q))
        k['swe_edge'](*eg, (i32(g.N), i32(g.E), i32(g.NV), self.F, self.q, self.B, us, g.eoeStart, g.eoe, g.w, g.eI, g.eJ,
                            g.ev1, g.ev2, g.de, self.drag, R(self.nu4), self.L2 if self.nu4 else NULL, u0, R(c), un))

    def step(self, h, u, dt, Fm=None):
        """One RK3 step in place on h (L, N) and u (L, E). Fm, if given, receives the final stage's thickness
        flux times l·dt (m³), consistent with the new h."""
        if self.nu4:
            self._lap(u, self.L1)
            self._lap(self.L1, self.L2)
        self._stage(h, u, h, u, dt / 3, self.h1, self.u1)
        self._stage(self.h1, self.u1, h, u, dt / 2, self.h2, self.u2)
        self._stage(self.h2, self.u2, h, u, dt, h, u, Fm, dt)
