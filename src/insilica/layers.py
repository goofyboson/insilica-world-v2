"""Stacked-layer shallow water with a split-explicit fast mode: V2's fluid core (SPEC §3, decisions #24–#27).

L Boussinesq layers (k = 0 on top), thickness h (L, N) and normal wind u (L, E), over topography b. Layer k feels
the Montgomery potential M_k = g·η + M'_k (η = b + Σh the free surface; M'_k = Σ_{m≤k} g'_{m−1}·z_m, z_m the top
of layer m), so each layer's dynamics are v1's TRiSK shallow water with that potential:

    dh_k/dt = −div(h_e,k u_k)
    du_k/dt = Σ w_ee' F_k,e' (q_k,e + q_k,e')/2 − ∂(M_k + K_k)/∂n − (r_k + c_k/h_k) u_k − ν4 ∇⁴u_k

(r_k a drag rate; c_k a linear stress coefficient, m/s: a stress decelerates a layer in proportion to 1/h.)

Time stepping: time-split RK3 (Wicker & Skamarock 2002; MPAS-Atmosphere). In each RK3 stage the slow tendencies
(everything but g·∇η) are taken from the stage's input state and held fixed while the free surface and the
column's barotropic wind are sub-stepped from tⁿ with forward-backward steps at the true external wave speed
√(gH), with divergence damping (the pressure gradient takes the forward-weighted surface η + γ(η − η_prev),
γ = 0.1; Skamarock & Klemp 1992, as MPAS-Atmosphere and WRF; decision #28). Layers then move with their own baroclinic flux plus their share of the stage's mean barotropic transport,
so Σ_k h_k equals the sub-stepped free surface; η is never a separate prognostic variable. A thickness floor
limits each layer's outflow (no mass is added). With one layer this is plain shallow water.

Exchange between layers (`exchange`) is the conservative hook vertical processes will use (#27).
"""
import math

import cupy as cp
import numpy as np

from .constants import CP_DRY, GRAV, KAPPA, P0, R_DRY
from .gpu import NULL, grid_dims

LMAX = 8
FAST_COURANT = 0.5          # fast sub-step: tau·√(gH)/d_min ≤ this
DIV_DAMP = 0.1              # divergence damping of the fast mode (decision #28)


class Layers:
    def __init__(self, dg, layers=1, b=None, grav=GRAV, gprime=(), drag=0.0, stress=0.0, nu4=0.0, floor=1.0, rotation=1.0,
                 div_damp=DIV_DAMP, air=None, vmix=0.0, thin=2.0, wall=None):
        """air: None for Boussinesq (ocean-like) layers, h in m; or a dict for air layers, h the layer's mass per area
        (kg/m²) with a potential temperature θ per layer passed to step() (decision #32). The fast mode then runs on
        the column mass with each cell's own coefficient Gc = R·T_s/μ (decision #37), at √(R·T_s). thin: the PV flux's cap on a
        neighbour's thickness, in multiples of the edge's own (decision #38; the ocean's, #72). wall: None, or an (E,) mask,
        0 on the edges no water crosses (thin walls between cells, Boussinesq layers only; decision #79)."""
        assert 1 <= layers <= LMAX and len(gprime) == layers - 1
        self.dg, self.L = dg, layers
        self.real = dg.dtype.type
        self.grav, self.nu4, self.floor, self.div_damp, self.thin = grav, nu4, floor, div_damp, thin
        assert wall is None or not air, 'walls are for Boussinesq layers'
        self.wall = None if wall is None else cp.asarray(wall, dtype=dg.dtype)
        N, E, NV, L, dt = dg.N, dg.E, dg.NV, layers, dg.dtype
        z = lambda *s: cp.zeros(s, dtype=dt)
        self.b = z(N) if b is None else cp.asarray(b, dtype=dt)
        self.gp = cp.asarray(np.asarray(gprime, dtype=float).reshape(-1) if layers > 1 else np.zeros(1), dtype=dt)
        self.drag = cp.asarray(np.broadcast_to(np.asarray(drag, dtype=float), (layers,)), dtype=dt)
        self.stress = cp.asarray(np.broadcast_to(np.asarray(stress, dtype=float), (layers,)), dtype=dt)
        self.fV = dg.fV * rotation
        self.air = air
        self.vmix = vmix                    # interfacial stress coefficient c (decision #39), h-units/s
        self.zero = z(N)
        self.Gc = cp.full(N, float(grav))   # the fast mode's coefficient per cell: g, or for air R·T_s/μ
        if air:
            self.pibar = z(L, N)
            # every gravity wave in the fast substeps (decision #55): the column's stiffness, each layer's mass and wind
            self.S, self.muf, self.mup, self.phi = z(L, N), z(L, N), z(L, N), z(L, N)
            self.uf, self.accL = z(L, E), z(L, E)
        self.B, self.q, self.Tp, self.F, self.scale = z(L, N), z(L, NV), z(L, E), z(L, E), z(L, N)
        self.G, self.He, self.ubs, self.ub0, self.ub, self.acc = z(E), z(E), z(E), z(E), z(E), z(E)
        self.eta0, self.eta, self.eta_prev = z(N), z(N), z(N)
        self.h1, self.h2, self.u1, self.u2 = z(L, N), z(L, N), z(L, E), z(L, E)
        self.D1, self.Z1, self.L1, self.L2 = z(L, N), z(L, NV), z(L, E), z(L, E)
        self.hold = z(L, N)
        self.hedge, self.qedge = z(L, E), z(L, E)
        self.diss = z(E)
        k = lambda n, f='layers.cu': dg.kernel(n, f)
        self.k = {n: k(n) for n in ('lay_eta', 'lay_ubar', 'lay_bern', 'air_bern', 'air_gfast', 'lay_slow', 'bt_eta', 'bt_u', 'lay_flux', 'lay_scale',
                                    'lay_limit', 'lay_h', 'lay_u', 'xch_cell', 'xch_edge', 'lay_vmix', 'vmix_heat', 'lay_edge_hq',
                                    'air_stiff', 'af_mu', 'af_phi', 'af_u', 'af_flux', 'lay_mask')}
        self.k['swe_vertex'] = k('swe_vertex', 'fused.cu')
        self.k.update({n: k(n, 'trisk.cu') for n in ('cell_div', 'vertex_curl', 'edge_lap')})
        self.dmin = float(np.min(dg.host.de))

    def mask(self, a, rows):
        """Zero the wall edges of `rows` rows of the edge field a (no-op without walls)."""
        if self.wall is not None:
            self.k['lay_mask'](*grid_dims(self.dg.E), (np.int32(self.dg.E), np.int32(rows), self.wall, a))

    def substeps(self, h, dt):
        """Fast sub-steps per full step for the current deepest column."""
        c2 = float(cp.max(cp.sum(h, axis=0) * self.Gc))           # (external speed)², the fastest column
        return max(1, math.ceil(dt * math.sqrt(c2) / (FAST_COURANT * self.dmin)))

    def update_gfast(self, h, theta):
        """For air: each cell's fast-mode coefficient R·T_s/μ from the current state."""
        if self.air:
            g, i32, R = self.dg, np.int32, self.real
            self.k['air_gfast'](*grid_dims(g.N), (i32(g.N), i32(self.L), h, theta, R(self.grav), R(KAPPA), R(P0), R(R_DRY),
                                                 R(self.floor), self.Gc))

    def lap_max(self, iters=30):
        """Largest |eigenvalue| of the edge vector Laplacian (power iteration from a fixed seed), 1/m². The ∇⁴ term,
        held through the RK3 stages, is stable while ν4·λ²·dt stays well below 2."""
        L = self.L
        v = cp.asarray(np.random.default_rng(0).standard_normal((L, self.dg.E)))
        lam = 0.0
        for _ in range(iters):
            self._lap(v, self.L1)
            lam = float(cp.linalg.norm(self.L1[0]) / cp.linalg.norm(v[0]))
            v = self.L1 / cp.linalg.norm(self.L1[0])
        return lam

    def _lap(self, u, out):
        g, k, L, i32 = self.dg, self.k, self.L, np.int32
        k['cell_div'](*grid_dims(g.N, L), (i32(g.N), i32(g.E), u, g.cellEdge, g.cellSign, g.le, g.areaCell, self.D1))
        k['vertex_curl'](*grid_dims(g.NV, L), (i32(g.E), i32(g.NV), u, g.vEdge, g.vEdgeSign, g.de, g.areaV, self.Z1))
        k['edge_lap'](*grid_dims(g.E, L), (i32(g.N), i32(g.E), i32(g.NV), self.D1, self.Z1, g.eI, g.eJ, g.ev1, g.ev2, g.de, g.le, out))

    def _stage(self, hs, us, h0, u0, c, dt, nfast, hn, un, Fm=None, theta=None):
        g, k, L, i32, R = self.dg, self.k, self.L, np.int32, self.real
        N, E, NV = i32(g.N), i32(g.E), i32(g.NV)
        c1, e1 = grid_dims(g.N), grid_dims(g.E)
        cL, eL = grid_dims(g.N, L), grid_dims(g.E, L)
        if self.air:
            k['air_bern'](*c1, (N, E, i32(L), hs, us, theta, self.b, R(self.grav), R(CP_DRY), R(KAPPA), R(P0), self.S,
                                g.cellEdge, g.le, g.de, g.areaCell, self.B, self.pibar))
        else:
            k['lay_bern'](*c1, (N, E, i32(L), hs, us, self.b, self.gp, g.cellEdge, g.le, g.de, g.areaCell, self.B))
        k['swe_vertex'](*grid_dims(g.NV, L), (N, E, NV, hs, us, g.vCell, g.vSlot, g.kite, g.vEdge, g.vEdgeSign, g.de, g.areaV,
                                              self.fV, self.q))
        k['lay_edge_hq'](*grid_dims(g.E, L), (N, E, NV, hs, self.q, g.eI, g.eJ, g.ev1, g.ev2, self.hedge, self.qedge))
        k['lay_slow'](*e1, (N, E, NV, i32(L), self.hedge, us, self.qedge, self.B, g.eoeStart, g.eoe, g.w, g.eI, g.eJ, g.ev1, g.ev2, g.de,
                            self.drag, self.stress, R(self.floor), R(self.nu4), self.L2 if self.nu4 else NULL,
                            theta if self.air else NULL,
                            self.pibar if self.air else NULL, R(c * dt), R(self.thin), self.Tp, self.G, self.He,
                            self.ubs))
        n = max(1, math.ceil(c * nfast))
        tau = R(c * dt / n)
        if self.air:
            # every layer's mass and wind sub-stepped from t^n over c·dt (decision #55)
            cp.copyto(self.muf, h0)
            cp.copyto(self.uf, u0)
            self.accL.fill(0)
            for _ in range(n):
                k['af_mu'](*cL, (N, E, self.hedge, self.uf, g.cellEdge, g.cellSign, g.le, g.areaCell, tau, self.muf, self.mup))
                k['af_phi'](*c1, (N, i32(L), self.muf, self.mup, R(self.div_damp), self.S, self.phi))
                k['af_u'](*eL, (N, E, self.phi, self.Tp, self.G, g.eI, g.eJ, g.de, tau, self.uf, self.accL))
            k['af_flux'](*eL, (E, self.hedge, self.accL, R(1.0 / n), self.F))
            k['lay_scale'](*cL, (N, E, h0, self.F, g.cellEdgeAsc, g.eI, g.le, g.areaCell, R(c * dt), R(self.floor), self.scale))
            k['lay_limit'](*eL, (N, E, g.eI, g.eJ, self.scale, g.le, R(dt), self.F, Fm if Fm is not None else NULL))
            k['lay_h'](*cL, (N, E, h0, self.F, g.cellEdge, g.cellSign, g.le, g.areaCell, R(c * dt), hn))
            cp.copyto(un, self.uf)
            return
        # fast mode from t^n over c·dt
        cp.copyto(self.eta, self.eta0)
        cp.copyto(self.ub, self.ub0)
        self.acc.fill(0)
        for _ in range(n):
            k['bt_eta'](*c1, (N, self.He, self.ub, g.cellEdge, g.cellSign, g.le, g.areaCell, tau, self.eta, self.eta_prev))
            k['bt_u'](*e1, (E, self.eta, self.eta_prev, R(self.div_damp), self.G, g.eI, g.eJ, g.de, self.Gc, tau, self.ub, self.acc))
            self.mask(self.ub, 1)
            self.mask(self.acc, 1)
        # layers
        k['lay_flux'](*eL, (N, E, hs, us, self.ubs, self.acc, R(1.0 / n), g.eI, g.eJ, self.F))
        self.mask(self.F, L)
        k['lay_scale'](*cL, (N, E, h0, self.F, g.cellEdgeAsc, g.eI, g.le, g.areaCell, R(c * dt), R(self.floor), self.scale))
        k['lay_limit'](*eL, (N, E, g.eI, g.eJ, self.scale, g.le, R(dt), self.F, Fm if Fm is not None else NULL))
        k['lay_h'](*cL, (N, E, h0, self.F, g.cellEdge, g.cellSign, g.le, g.areaCell, R(c * dt), hn))
        k['lay_u'](*eL, (E, u0, self.Tp, self.ub, self.ub0, R(c * dt), un))
        self.mask(un, L)

    def step(self, h, u, dt, nfast, Fm=None, theta=None):
        """One time-split RK3 step in place on h (L, N), u (L, E), with nfast fast sub-steps per full step (stage s
        takes ⌈c_s·nfast⌉). Fm, if given, receives each layer's final-stage volume flux l·dt·F (m³)."""
        g, k, i32, L = self.dg, self.k, np.int32, self.L
        if self.air:
            R = self.real
            k['air_stiff'](*grid_dims(g.N), (i32(g.N), i32(L), h, theta, R(self.grav), R(CP_DRY), R(KAPPA), R(P0), self.S))
        else:
            k['lay_eta'](*grid_dims(g.N), (i32(g.N), i32(L), h, self.b, self.eta0))
            k['lay_ubar'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(L), h, u, g.eI, g.eJ, self.ub0))
            self.mask(self.ub0, 1)
        if self.nu4:
            self._lap(u, self.L1)
            self._lap(self.L1, self.L2)
            self.nu4dt = self.nu4 * dt                     # the ∇⁴ loss is booked as heat with the crossings' (#81)
        self._stage(h, u, h, u, 1 / 3, dt, nfast, self.h1, self.u1, theta=theta)
        self._stage(self.h1, self.u1, h, u, 1 / 2, dt, nfast, self.h2, self.u2, theta=theta)
        self._stage(self.h2, self.u2, h, u, 1.0, dt, nfast, h, u, Fm, theta=theta)
        if self.vmix and L > 1:
            k['lay_vmix'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(L), h, g.eI, g.eJ, self.real(self.vmix), self.real(dt), u, self.diss))
            self.mask(u, L)
            if self.air:
                R = self.real
                k['vmix_heat'](*grid_dims(g.N), (i32(g.N), i32(L), h, self.diss, g.cellEdge, g.le, g.de, g.areaCell, R(self.grav),
                                                 R(CP_DRY), R(KAPPA), R(P0), theta))

    def exchange(self, wx, h, u, M, psi, theta=None):
        """Move thickness wx (L−1, N; m, positive upward across the interface below layer k) between layers with
        its tracer and momentum, conservatively (upwind). The caller keeps donors above the floor. For air (theta
        given) the kinetic energy the momentum mixing removed warms the column (decision #73), with the last step's ∇⁴
        viscosity loss (#81)."""
        g, k, i32 = self.dg, self.k, np.int32
        hold = self.hold
        cp.copyto(hold, h)
        heat = self.air and theta is not None
        k['xch_cell'](*grid_dims(g.N), (i32(g.N), i32(self.L), wx, g.areaCell, h, M, psi))
        v4 = heat and self.nu4 != 0
        k['xch_edge'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(self.L), wx, hold, g.eI, g.eJ, u, self.diss if heat else NULL,
                                         self.L2 if v4 else NULL, self.real(self.nu4dt if v4 else 0.0)))
        self.mask(u, self.L)
        if heat:
            R = self.real
            k['vmix_heat'](*grid_dims(g.N), (i32(g.N), i32(self.L), h, self.diss, g.cellEdge, g.le, g.de, g.areaCell, R(self.grav),
                                             R(CP_DRY), R(KAPPA), R(P0), theta))

    def energy(self, h, u, theta=None):
        """Total energy per unit density (J/ρ0): Σ_k Σ_i A h_k K_k + Σ_i A [g(η² − b²)/2 + Σ_k≥1 g'_{k−1}(z_k² − b²)/2]."""
        gh = self.dg.host
        hh, uu, b = h.get(), u.get(), self.b.get()
        ce = np.where(gh.cellEdge >= 0, gh.cellEdge, 0)
        ok = gh.cellEdge >= 0
        kin = sum(np.sum(hh[k] * np.sum(np.where(ok, gh.le[ce] * gh.de[ce] * 0.25 * uu[k][ce] ** 2, 0.0), axis=1)) for k in range(self.L))
        if self.air:
            # Σ μ K + Σ μ π̄ θ (enthalpy c_p T) + Φ_s·μ (the column's potential energy with the enthalpy)
            th = theta.get()
            p = np.cumsum(self.grav * hh, axis=0)
            pt = np.vstack([np.zeros((1, hh.shape[1])), p[:-1]])
            pib = CP_DRY * (p ** (KAPPA + 1) - pt ** (KAPPA + 1)) / ((KAPPA + 1) * (p - pt) * P0 ** KAPPA)
            return float(kin + np.sum(gh.areaCell * (np.sum(hh * pib * th, axis=0) + self.grav * b * hh.sum(0))))
        z = b + np.cumsum(hh[::-1], axis=0)[::-1]             # z[k] = top of layer k; z[0] = η
        gp = self.gp.get()
        pot = np.sum(gh.areaCell * self.grav * (z[0] ** 2 - b ** 2) / 2)
        for k in range(1, self.L):
            pot += np.sum(gh.areaCell * gp[k - 1] * (z[k] ** 2 - b ** 2) / 2)
        return float(kin + pot)
