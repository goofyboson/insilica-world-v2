"""The mountains' drag on the air (V2-6b, SPEC #127): Lott & Miller (1997) on the isentropic layers (kernels/drag.cu)."""
import cupy as cp
import numpy as np

from .constants import GRAV, KAPPA, P0, R_PLANET
from .gpu import grid_dims

RD = 287.0


def east_north(pos):
    """Unit east and north vectors at points on the unit sphere (N, 3) each."""
    lon, lat = np.arctan2(pos[:, 1], pos[:, 0]), np.arcsin(np.clip(pos[:, 2], -1, 1))
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], 1)
    return east, north


def operators(g):
    """Perot's reconstruction of a cell's wind (east, north) from its edges' normal winds, as grid.trisk.cell_winds:
    (N, 6, 2) coefficients per edge slot; and each edge's normal (the direction of positive u) as (east, north) at its two cells,
    (E, 4)."""
    east, north = east_north(g.pos)
    ce = np.where(g.cellEdge >= 0, g.cellEdge, 0)
    c = np.where(g.cellEdge >= 0, g.cellSign * g.le[ce] * R_PLANET, 0.0) / g.areaCell[:, None]
    d = g.xm[ce] - g.pos[:, None, :]
    op = np.stack([c * np.sum(d * east[:, None, :], 2), c * np.sum(d * north[:, None, :], 2)], 2)
    # the normal from the reconstruction itself: a uniform wind W at a cell gives u_e = W . n_e
    t = g.pos[g.eJ] - g.pos[g.eI]
    nrm = np.zeros((g.E, 4))
    for s, cell in ((0, g.eI), (2, g.eJ)):
        n = t - np.sum(t * g.pos[cell], 1)[:, None] * g.pos[cell]
        n /= np.linalg.norm(n, axis=1)[:, None]
        nrm[:, s], nrm[:, s + 1] = np.sum(n * east[cell], 1), np.sum(n * north[cell], 1)
    return op, nrm


class OroDrag:
    """sso: (mu, sigma, gamma, theta) per cell (cases.earth_sso); p: the air physics' settings (oro_g, oro_cd, oro_hnc, oro_ric)."""

    def __init__(self, dg, L, sso, p, dt):
        g, R = dg.host, dg.dtype
        op, nrm = operators(g)
        self.dg, self.L = dg, L
        self.op, self.nrm = cp.asarray(op.reshape(-1), dtype=R), cp.asarray(nrm.reshape(-1), dtype=R)
        self.sso = cp.asarray(np.stack([np.asarray(x, dtype=float) for x in sso]).reshape(-1), dtype=R)
        self.p = [GRAV, P0, KAPPA, RD, p['oro_g'], p['oro_cd'], p['oro_hnc'], p['oro_ric'], dt, 1.0]
        self.par = cp.asarray(self.p, dtype=R)
        self.beta, self.F = cp.zeros((L, g.N), dtype=R), cp.zeros((2 * L, g.N), dtype=R)
        self.k = {n: dg.kernel(n, 'drag.cu') for n in ('oro_cell', 'oro_edge')}

    def apply(self, h, theta, u, dt, diss):
        """One step's drag on the layers' winds u (in place); the kinetic energy removed is added to diss (J/m² per edge)."""
        g, i32, R = self.dg, np.int32, self.dg.dtype.type
        assert dt == self.p[8]
        self.k['oro_cell'](*grid_dims(g.N), (i32(g.N), i32(g.E), i32(self.L), h, theta, u, g.cellEdge, self.op, self.sso,
                                             self.par, self.beta, self.F))
        self.k['oro_edge'](*grid_dims(g.E), (i32(g.N), i32(g.E), i32(self.L), h, g.eI, g.eJ, self.nrm, self.beta, self.F,
                                             R(dt), u, diss))
