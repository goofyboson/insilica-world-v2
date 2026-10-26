"""TRiSK C-grid geometry (Thuburn et al. 2009, Ringler et al. 2010): a port of v1's `getTrisk`
(`src/sim/12-trisk.js`), plus the gather tables the GPU kernels use instead of v1's scatter loops.

Cells hold mass and tracers, edges the normal wind u_e (positive from eI to eJ; n_e points eI → eJ and
t_e = k × n_e runs from vertex v1 to v2), vertices (the triangles' circumcenters) vorticity and potential
vorticity. Array names follow v1.
"""
import numpy as np

from ..constants import OMEGA, R_PLANET
from .icosa import _cross, _dot, _norm, sph_tri_area


@np.errstate(invalid='ignore', divide='ignore')      # pentagon pad slots are degenerate and masked
def build_trisk(g):
    N, E = g['N'], g['E']
    pos, nbr, deg, corners = g['pos'], g['nbr'].astype(np.int64), g['deg'].astype(np.int64), g['corners']
    eI, eJ = g['eI'].astype(np.int64), g['eJ'].astype(np.int64)
    eKi, eKj = g['eKi'].astype(np.int64), g['eKj'].astype(np.int64)
    k6 = np.arange(6)[None, :]
    valid = k6 < deg[:, None]
    nxt = np.where(valid, (k6 + 1) % deg[:, None], k6)

    # Vertices: one per triangle (unique sorted cell triple), numbered in order of first appearance.
    nb = np.where(valid, nbr, 0)
    tri = np.stack([np.broadcast_to(np.arange(N)[:, None], (N, 6)), nb, np.take_along_axis(nb, nxt, axis=1)], axis=-1)
    tri = np.sort(tri, axis=-1)
    key = (tri[..., 0] * N + tri[..., 1]) * N + tri[..., 2]
    flat = np.where(valid, key, -1).ravel()
    vflat = np.flatnonzero(flat >= 0)
    uniq, first, inv = np.unique(flat[vflat], return_index=True, return_inverse=True)
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(len(uniq))
    corner_v = np.full(N * 6, -1, dtype=np.int64)
    corner_v[vflat] = rank[inv]
    corner_v = corner_v.reshape(N, 6)
    NV = len(uniq)
    v_pos = np.empty((NV, 3))
    v_pos[rank] = corners.reshape(N * 6, 3)[vflat[first]]

    cell_edge = np.full((N, 6), -1, dtype=np.int64)
    cell_sign = np.zeros((N, 6), dtype=np.int8)
    ar = np.arange(E)
    cell_edge[eI, eKi] = ar
    cell_sign[eI, eKi] = 1
    cell_edge[eJ, eKj] = ar
    cell_sign[eJ, eKj] = -1
    m_i = deg[eI]
    ev1 = corner_v[eI, (eKi + m_i - 1) % m_i]
    ev2 = corner_v[eI, eKi]
    de = g['dist'][eI, eKi]
    le = g['edgeLen'][eI, eKi]
    pi, pj = pos[eI], pos[eJ]
    s = pi + pj
    xe = s / _norm(s)[:, None]
    d = pj - pi
    d = d - _dot(d, xe)[:, None] * xe
    ne = d / _norm(d)[:, None]

    # Kites: cell i ∩ triangle at corner k = (x_i, edge-k midpoint, vertex, edge-(k+1) midpoint).
    ce = np.where(valid, cell_edge, 0)
    e0, e1 = ce, np.take_along_axis(ce, nxt, axis=1)
    vv = np.where(valid, corner_v, 0)
    P = np.broadcast_to(pos[:, None, :], (N, 6, 3))
    kite = (sph_tri_area(P, xe[e0], v_pos[vv]) + sph_tri_area(P, v_pos[vv], xe[e1])) * R_PLANET * R_PLANET
    kite = np.where(valid, kite, 0.0)
    area_cell = np.zeros(N)
    for k in range(6):
        area_cell = area_cell + kite[:, k]
    area_v = np.zeros(NV)
    np.add.at(area_v, corner_v[valid], kite[valid])        # in (cell, slot) order, as v1

    # Circulation sign of each edge around its vertices (+1 if n_e runs counterclockwise about it).
    def circ(vi):
        dd = xe - v_pos[vi]
        return np.sign(_dot(ne, _cross(xe, dd))).astype(np.int8)
    circ1, circ2 = circ(ev1), circ(ev2)

    # Tangential weights: for each of the edge's two cells, walk its other edges counterclockwise, adding
    # each passed vertex's kite fraction before using it. Stored CSR (eoeStart, eoe, w) in v1's order:
    # cell eI's edges first, then cell eJ's.
    cols_e, cols_w, cols_ok = [], [], []
    for c, kc, se in ((eI, eKi, 1.0), (eJ, eKj, -1.0)):
        m = deg[c]
        rsum = np.zeros(E)
        for step in range(1, 6):
            ok = step < m
            k = (kc + step) % m
            kv = (k + m - 1) % m
            rsum = rsum + np.where(ok, kite[c, kv] / area_cell[c], 0.0)
            e2 = cell_edge[c, k]
            wt = se * (0.5 - rsum) * cell_sign[c, k] * le[e2] / de
            cols_e.append(e2)
            cols_w.append(wt)
            cols_ok.append(ok)
    ok = np.stack(cols_ok, axis=1)
    eoe = np.stack(cols_e, axis=1)[ok]
    w = np.stack(cols_w, axis=1)[ok]
    cnt = ok.sum(axis=1)
    eoe_start = np.concatenate([[0], np.cumsum(cnt)])
    fV = 2 * OMEGA * v_pos[:, 2]
    # The Voronoi edge's own midpoint (SPEC §9 #17). xe, where the line between the two cell centers crosses
    # the edge, is the true corner of the kites and stays the TRiSK edge point; xm is the centre of the edge,
    # used where a formula wants it: the Perot wind rebuild and sampling analytic winds onto edges. The edge
    # normal ne is the same at both points (p_j − p_i is perpendicular to the edge's whole great circle).
    xm = v_pos[ev1] + v_pos[ev2]
    xm = xm / _norm(xm)[:, None]

    t = dict(NV=NV, vPos=v_pos, xm=xm, cornerV=corner_v.astype(np.int32), cellEdge=cell_edge.astype(np.int32), cellSign=cell_sign,
             ev1=ev1.astype(np.int32), ev2=ev2.astype(np.int32), de=de, le=le, xe=xe, ne=ne, kite=kite, areaCell=area_cell,
             areaV=area_v, circ1=circ1, circ2=circ2, fV=fV, eoeStart=eoe_start.astype(np.int32), eoe=eoe.astype(np.int32), w=w)
    t.update(gather_tables(g, t))
    return t


def gather_tables(g, t):
    """Tables that let a GPU thread gather what v1's loops scatter, in the order v1 adds them:
    - cellEdgeAsc (N,6): each cell's edge slots sorted by edge index (v1's edge loops visit a cell's edges
      in ascending edge order); pads −1.
    - vEdge, vEdgeSign (NV,3): each vertex's three edges, ascending, with their circulation sign.
    - vCell, vSlot (NV,3): the three (cell, slot) kites of each vertex in ascending cell order."""
    N, NV = g['N'], t['NV']
    ce = t['cellEdge'].astype(np.int64)
    ce_key = np.where(ce < 0, np.iinfo(np.int64).max, ce)
    order = np.argsort(ce_key, axis=1, kind='stable')
    cell_edge_asc = np.take_along_axis(ce, order, axis=1)
    E = g['E']
    ev = np.concatenate([t['ev1'], t['ev2']]).astype(np.int64)
    ee = np.concatenate([np.arange(E), np.arange(E)])
    sg = np.concatenate([t['circ1'], t['circ2']])
    o = np.lexsort((ee, ev))
    v_edge = ee[o].reshape(NV, 3)
    v_edge_sign = sg[o].reshape(NV, 3)
    deg = g['deg'].astype(np.int64)
    valid = np.arange(6)[None, :] < deg[:, None]
    cv = t['cornerV'].astype(np.int64)
    cells = np.broadcast_to(np.arange(N)[:, None], (N, 6))[valid]
    slots = np.broadcast_to(np.arange(6)[None, :], (N, 6))[valid]
    vs = cv[valid]
    o = np.lexsort((cells, vs))
    return dict(cellEdgeAsc=cell_edge_asc.astype(np.int32), vEdge=v_edge.astype(np.int32), vEdgeSign=v_edge_sign.astype(np.int8),
                vCell=cells[o].reshape(NV, 3).astype(np.int32), vSlot=slots[o].reshape(NV, 3).astype(np.int8))


def cell_winds(g, u):
    """Wind vectors at cells from edge normal winds (Perot): U_i = (1/A_i) Σ_e l_e (x_e − x_i) u_e^out, with x_e
    the Voronoi edge midpoint xm, projected onto the tangent plane. u (E,) → (N, 3) in m/s.
    (v1's cellWinds uses the kite corner xe instead; its error does not fall with resolution, SPEC §9 #17.)"""
    ce = np.where(g.cellEdge >= 0, g.cellEdge, 0)
    c = np.where(g.cellEdge >= 0, g.cellSign * u[ce] * g.le[ce] * R_PLANET, 0.0)
    v = np.sum(c[..., None] * (g.xm[ce] - g.pos[:, None, :]), axis=1) / g.areaCell[:, None]
    return v - np.sum(v * g.pos, axis=1)[:, None] * g.pos
