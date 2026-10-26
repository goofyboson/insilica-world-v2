"""Reference (unfused) kernels, kept for test E2: the fused, graph-captured engine in swe.py and transport.py must
match these bit for bit. They are V2-0's direct ports, one kernel per v1 loop.

Shallow water on the TRiSK C-grid, on the GPU: a port of v1's `makeSolver` with one layer
(`src/sim/56-swe.js`), generalized to L independent layers stacked in one array (shape (L, N) and (L, E)).

    dh/dt = −div(h_e u)
    du/dt = Σ w_ee' F_e' (q_e + q_e')/2 − ∂(g(h + b) + K)/∂n − r u − ν4 ∇⁴u

Time step: Wicker–Skamarock RK3 with v1's positivity limiter in every stage; the ∇⁴ term is evaluated once
per step from the step's starting wind and held through the stages (as MPAS and v1).
"""
import cupy as cp
import numpy as np

from .constants import GRAV
from .gpu import NULL, grid_dims


class SWEReference:
    def __init__(self, dg, layers=1, grav=GRAV, b=None, drag=0.0, nu4=0.0):
        self.dg, self.L = dg, layers
        self.dt_ = dg.dtype.type
        self.grav, self.nu4 = grav, nu4
        self.b = cp.zeros(dg.N, dtype=dg.dtype) if b is None else cp.asarray(b, dtype=dg.dtype)
        self.drag = cp.asarray(np.broadcast_to(np.asarray(drag, dtype=float), (layers,)), dtype=dg.dtype)
        N, E, NV, L, dt = dg.N, dg.E, dg.NV, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        self.F, self.dh, self.B, self.q, self.qe, self.du = z(L, E), z(L, N), z(L, N), z(L, NV), z(L, E), z(L, E)
        self.h1, self.u1, self.scale = z(L, N), z(L, E), z(L, N)
        self.D1, self.Z1, self.L1, self.L2 = z(L, N), z(L, NV), z(L, E), z(L, E)
        k = dg.kernel
        self.k = {name: k(name) for name in ('edge_flux', 'cell_outflow_scale', 'edge_limit', 'cell_tend', 'vertex_pv', 'edge_qe',
                                             'edge_tend', 'cell_div', 'vertex_curl', 'edge_lap', 'axpy', 'edge_volume')}

    def _lap(self, u, out):
        g, k, L, i32 = self.dg, self.k, self.L, np.int32
        k['cell_div'](*grid_dims(g.N, L), (i32(g.N), i32(g.E), u, g.cellEdge, g.cellSign, g.le, g.areaCell, self.D1))
        k['vertex_curl'](*grid_dims(g.NV, L), (i32(g.E), i32(g.NV), u, g.vEdge, g.vEdgeSign, g.de, g.areaV, self.Z1))
        k['edge_lap'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), i32(g.NV), self.D1, self.Z1, g.eI, g.eJ, g.ev1, g.ev2, g.de, g.le, out))

    def tend(self, h, u, dts):
        """Tendencies (self.dh, self.du) of state (h, u), with the positivity limiter for a stage of dts."""
        g, k, L, i32, R = self.dg, self.k, self.L, np.int32, self.dt_
        k['edge_flux'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), h, u, g.eI, g.eJ, self.F))
        if dts:
            k['cell_outflow_scale'](*grid_dims(g.N, L), (i32(g.N), i32(g.E), h, self.F, g.cellEdgeAsc, g.eI, g.le, g.areaCell, R(dts), self.scale))
            k['edge_limit'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), self.F, g.eI, g.eJ, self.scale))
        k['cell_tend'](*grid_dims(g.N, L), (i32(g.N), i32(g.E), h, self.b, self.F, u, g.cellEdge, g.cellSign, g.le, g.de, g.areaCell,
                                            R(self.grav), self.dh, self.B))
        k['vertex_pv'](*grid_dims(g.NV, L), (i32(g.N), i32(g.E), i32(g.NV), h, u, g.vCell, g.vSlot, g.kite, g.vEdge, g.vEdgeSign, g.de,
                                              g.areaV, g.fV, self.q))
        k['edge_qe'](*grid_dims(g.E, L), (i32(g.NV), i32(g.E), self.q, g.ev1, g.ev2, self.qe))
        k['edge_tend'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), self.F, self.qe, self.B, u, g.eoeStart, g.eoe, g.w, g.eI, g.eJ, g.de,
                                             self.drag, R(self.nu4), self.L2 if self.nu4 else NULL, self.du))

    def _axpy(self, x, c, d, out):
        n = x.size
        self.k['axpy'](((n + 255) // 256,), (256,), (np.int64(n), x, self.dt_(c), d, out))

    def step(self, h, u, dt, Fm=None):
        """One RK3 step in place on h (L, N) and u (L, E). Fm, if given, receives the final stage's thickness
        flux times l·dt (m³), consistent with the new h."""
        if self.nu4:
            self._lap(u, self.L1)
            self._lap(self.L1, self.L2)
        self.tend(h, u, dt / 3)
        self._axpy(h, dt / 3, self.dh, self.h1)
        self._axpy(u, dt / 3, self.du, self.u1)
        self.tend(self.h1, self.u1, dt / 2)
        self._axpy(h, dt / 2, self.dh, self.h1)
        self._axpy(u, dt / 2, self.du, self.u1)
        self.tend(self.h1, self.u1, dt)
        if Fm is not None:
            g = self.dg
            self.k['edge_volume'](*grid_dims(g.E, self.L), (np.int32(g.E), self.F, g.le, self.dt_(dt), Fm))
        self._axpy(h, dt, self.dh, h)
        self._axpy(u, dt, self.du, u)


class TransportReference:
    def __init__(self, dg, layers=1, v1_formula=False, beta=1.0):
        """v1_formula: v1's 3rd-order flux, which uses the eI cell's curvature for both flux signs (SPEC §9 #20).
        beta: weight of the upwind term (1: 3rd-order upwind; 0: 4th-order centered)."""
        self.dg, self.L, self.v1bias, self.beta = dg, layers, np.int32(1 if v1_formula else 0), beta
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
            k['tr_ho_flux'](*eg, (i32(g.N), i32(g.E), Fm, src, g.d2idx, g.d2wt, g.eI, g.eJ, self.v1bias, R(self.beta), self.H))
            k['tr_cell_update'](*cg, (i32(g.N), i32(g.E), M0, psi, self.H, R(frac), Fm, M1, g.cellEdgeAsc, g.eI, out))
        k['tr_ho_flux'](*eg, (i32(g.N), i32(g.E), Fm, self.p2, g.d2idx, g.d2wt, g.eI, g.eJ, self.v1bias, R(self.beta), self.H))
        k['tr_antidiff'](*eg, (i32(g.E), self.H, self.FL, self.A))
        k['tr_fct_ratios'](*cg, (i32(g.N), i32(g.E), psi, self.td, M1, self.A, g.nbr, g.cellEdgeAsc, g.eI, self.Rp, self.Rm))
        k['tr_fct_edge'](*eg, (i32(g.N), i32(g.E), self.A, self.Rp, self.Rm, g.eI, g.eJ, self.CA))
        k['tr_fct_apply'](*cg, (i32(g.N), i32(g.E), self.td, M1, self.CA, g.cellEdgeAsc, g.eI, psi))
        return psi


def g_real(dg):
    return dg.dtype.type
