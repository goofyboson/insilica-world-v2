"""Icosahedral Voronoi grid with Lloyd/SCVT smoothing: a port of v1's `buildGrid` and `lloydSCVT`
(`src/sim/10-grid.js`, v1 SPEC §5 and §18 #45–#46).

The steps follow v1's order exactly (same icosahedron, the same midpoint numbering, the same Jacobi Lloyd
update with the poles pinned, the same counterclockwise neighbor sort), so cell, edge and vertex numbers match
v1 one to one and the geometry agrees to rounding (test G4). The spherical-triangle formulas are V2's own,
precise for small triangles (SPEC §9 #15). Everything is vectorized with NumPy; the Lloyd
iteration may run on the GPU (`xp=cupy`) for large n, which changes results only by rounding.
"""
import numpy as np

from ..constants import R_PLANET

LLOYD_MAX_ITERS = 400
LLOYD_TOL = 1e-11


def _am(a):
    """The array module (NumPy or CuPy) that array a belongs to."""
    if type(a).__module__.startswith('cupy'):
        import cupy
        return cupy
    return np


def _norm(v):
    return _am(v).sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def _cross(a, b):
    return _am(a).stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


# Small-triangle precision (SPEC §9 #15). v1's formulas take cross products of whole unit vectors, whose
# components are O(1): the small quantities that fix a mesh triangle (its area ~h², its circumcenter's offset
# ~h) then carry absolute rounding of ~1e-16, a relative error of eps/h² in areas and eps/h in corner
# positions, growing 4× and 2× per grid level. The forms below work with chords u = b − a, v = c − a (O(h),
# and differences of nearby floats are nearly exact) in a tangent frame at a, and take the radial parts from
# the exact identity a·b − 1 = −|b − a|²/2 for unit vectors, so every result keeps relative precision ~eps.

def _tangent_basis(a):
    """Orthonormal tangent vectors (e1, e2) at unit vectors a, e2 = a × e1."""
    xp = _am(a)
    k = xp.argmin(xp.abs(a), axis=-1)
    axis = (xp.arange(3) == k[..., None]).astype(a.dtype)          # the coordinate axis least aligned with a
    e1 = _cross(a, axis)
    e1 = e1 / _norm(e1)[..., None]
    return e1, _cross(a, e1)


def _local(u, a, e1, e2):
    """Chord u from a in the frame (e1, e2, a); the radial part from −|u|²/2."""
    return _dot(u, e1), _dot(u, e2), -0.5 * _dot(u, u)


def angle_between(a, b):
    """Great-circle angle between unit vectors, from the chord: 2 asin(|b − a|/2)."""
    xp = _am(a)
    return 2 * xp.arcsin(xp.minimum(1.0, 0.5 * _norm(b - a)))


def sph_tri_area(a, b, c):
    """Area of the spherical triangle (a, b, c) of unit vectors (Van Oosterom & Strackee):
    tan(E/2) = |a·(b × c)| / (1 + a·b + b·c + c·a), with a·(b × c) = a·(u × v) taken in a's tangent frame and
    the denominator as 4 − (|b − a|² + |c − b|² + |a − c|²)/2."""
    xp = _am(a)
    u, v = b - a, c - a
    e1, e2 = _tangent_basis(a)
    u1, u2 = _dot(u, e1), _dot(u, e2)
    v1, v2 = _dot(v, e1), _dot(v, e2)
    triple = u1 * v2 - u2 * v1
    w = c - b
    den = 4 - 0.5 * (_dot(u, u) + _dot(w, w) + _dot(v, v))
    return 2 * xp.arctan2(xp.abs(triple), den)


def circumcenter(pa, pb, pc):
    """Circumcenter of spherical triangles (a, b, c): the unit normal of the plane through them on the
    triangle's side (as v1's `circumcenter`). The normal w = u × v is formed in a's frame (e1, e2, a), where
    its radial part w3 is ~h² and its tangential parts w1, w2 ~h³, each to relative precision eps; the corner
    is a + (w1 e1 + w2 e2)/w3, normalized. Degenerate triangles (the pad slots of pentagons) give NaN; callers
    mask them."""
    xp = _am(pa)
    e1, e2 = _tangent_basis(pa)
    u1, u2, u3 = _local(pb - pa, pa, e1, e2)
    v1, v2, v3 = _local(pc - pa, pa, e1, e2)
    w1 = u2 * v3 - u3 * v2
    w2 = u3 * v1 - u1 * v3
    w3 = u1 * v2 - u2 * v1
    t1, t2 = w1 / w3, w2 / w3
    p = pa + t1[..., None] * e1 + t2[..., None] * e2
    return p / _norm(p)[..., None]


def icosphere(n):
    """Unit icosahedron with vertices at both poles, subdivided n times. Returns (V (N,3), faces (F,3))."""
    V = []

    def add(x, y, z):
        r = (x * x + y * y + z * z) ** 0.5
        V.append((x / r, y / r, z / r))
        return len(V) - 1

    add(0.0, 0.0, 1.0)
    ring_lat = np.arctan(0.5)
    for k in range(5):
        lo = k * 2 * np.pi / 5
        add(np.cos(ring_lat) * np.cos(lo), np.cos(ring_lat) * np.sin(lo), np.sin(ring_lat))
    for k in range(5):
        lo = (k + 0.5) * 2 * np.pi / 5
        add(np.cos(ring_lat) * np.cos(lo), np.cos(ring_lat) * np.sin(lo), -np.sin(ring_lat))
    add(0.0, 0.0, -1.0)
    faces = []
    for k in range(5):
        u0, u1, l0, l1 = 1 + k, 1 + (k + 1) % 5, 6 + k, 6 + (k + 1) % 5
        faces += [(0, u0, u1), (u0, l0, u1), (u1, l0, l1), (11, l1, l0)]
    for _ in range(n):
        mid = {}

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            m = mid.get(key)
            if m is None:
                va, vb = V[a], V[b]
                m = add(va[0] + vb[0], va[1] + vb[1], va[2] + vb[2])
                mid[key] = m
            return m

        nf = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return np.array(V, dtype=np.float64), np.array(faces, dtype=np.int64)


def _tangent_frame(P):
    """v1's local east/north basis; at the exact poles east = (0, 1, 0)."""
    ex, ey = -P[:, 1], P[:, 0].copy()
    er = np.sqrt(ex * ex + ey * ey)
    pole = er < 1e-12
    ex = np.where(pole, 0.0, ex)
    ey = np.where(pole, 1.0, ey)
    er = np.where(pole, 1.0, er)
    east = np.stack([ex / er, ey / er, np.zeros_like(ex)], axis=1)
    north = _cross(P, east)
    return east, north


def _sorted_ring(P, adj, deg):
    """Neighbors of each cell sorted counterclockwise from east (v1's angle sort); pads repeat slot 0."""
    east, north = _tangent_frame(P)
    Q = P[np.maximum(adj, 0)]
    ang = np.arctan2(_dot(Q, north[:, None, :]), _dot(Q, east[:, None, :]))
    ang = np.where(adj < 0, np.inf, ang)
    order = np.argsort(ang, axis=1, kind='stable')
    ring = np.take_along_axis(adj, order, axis=1)
    return ring


def _adjacency(faces, N):
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    src = np.concatenate([a, a, b, b, c, c])
    dst = np.concatenate([b, c, a, c, a, b])
    key = np.unique(src * N + dst)
    src, dst = key // N, key % N
    deg = np.bincount(src, minlength=N)
    start = np.concatenate([[0], np.cumsum(deg)[:-1]])
    slot = np.arange(len(src)) - start[src]
    adj = np.full((N, 6), -1, dtype=np.int64)
    adj[src, slot] = dst
    return adj, deg


def _next_slot(deg):
    """For each (cell, slot k) the slot k+1 mod deg; pad slots point at themselves."""
    k = np.arange(6)[None, :]
    m = deg[:, None]
    return np.where(k < m, (k + 1) % m, k)


def _prev_slot(deg):
    k = np.arange(6)[None, :]
    m = deg[:, None]
    return np.where(k < m, (k + m - 1) % m, k)


def lloyd_scvt(V, ring, deg, xp=np, max_iters=LLOYD_MAX_ITERS, tol=LLOYD_TOL):
    """Jacobi Lloyd iteration toward a spherical centroidal Voronoi tessellation (v1 `lloydSCVT`).
    Each cell's generator moves to the centroid of its Voronoi polygon (the area-weighted mean of its fan
    triangles' centroids). Vertices 0 and 11 (the poles) stay pinned. Returns (V, iterations, last move)."""
    N = V.shape[0]
    ring_d = xp.asarray(np.where(ring < 0, ring[:, :1], ring))
    nxt = xp.asarray(_next_slot(deg))
    valid = xp.asarray(np.arange(6)[None, :] < deg[:, None])
    nxt_idx = xp.take_along_axis(ring_d, nxt, axis=1)
    free = xp.ones(N, dtype=bool)
    free[0] = False
    free[11] = False
    Vd = xp.asarray(V)
    it, moved = 0, np.inf
    for it in range(1, max_iters + 1):
        pa = Vd[:, None, :]
        C = circumcenter(xp.broadcast_to(pa, (N, 6, 3)), Vd[ring_d], Vd[nxt_idx])
        C1 = xp.take_along_axis(C, xp.broadcast_to(nxt[:, :, None], (N, 6, 3)), axis=1)
        s = xp.zeros((N, 3))
        for k in range(6):
            a = sph_tri_area(Vd, C[:, k], C1[:, k])
            s = s + xp.where(valid[:, k, None], a[:, None] * (Vd + C[:, k] + C1[:, k]), 0.0)
        nxt_pos = s / _norm(s)[:, None]
        d = nxt_pos - Vd
        moved = float(xp.max(xp.where(free, _norm(d), 0.0)))
        Vd = xp.where(free[:, None], nxt_pos, Vd)
        if moved < tol:
            break
    V_out = Vd.get() if hasattr(Vd, 'get') else Vd
    return np.ascontiguousarray(V_out), it, moved


@np.errstate(invalid='ignore', divide='ignore')      # pentagon pad slots are degenerate and masked
def build_grid(n, xp=np):
    """The v1 grid at level n as a dict of NumPy arrays (names as in v1's `buildGrid`)."""
    V, faces = icosphere(n)
    N = V.shape[0]
    adj, deg = _adjacency(faces, N)
    ring = _sorted_ring(V, adj, deg)
    V, iters, moved = lloyd_scvt(V, ring, deg, xp=xp)

    pos = V
    lat = np.arcsin(np.clip(pos[:, 2], -1, 1))
    lon = np.arctan2(pos[:, 1], pos[:, 0])
    east, north = _tangent_frame(pos)
    nbr = _sorted_ring(pos, adj, deg)
    valid = np.arange(6)[None, :] < deg[:, None]
    nbr = np.where(valid, nbr, -1)
    nb = np.where(valid, nbr, nbr[:, :1])
    nxt, prv = _next_slot(deg), _prev_slot(deg)
    P = pos[:, None, :]
    corners = circumcenter(np.broadcast_to(P, (N, 6, 3)), pos[nb], pos[np.take_along_axis(nb, nxt, axis=1)])
    corners = np.where(valid[..., None], corners, 0.0)
    ca = np.take_along_axis(corners, prv[..., None], axis=1)            # corner k−1
    cb = corners                                                        # corner k
    dist = np.where(valid, R_PLANET * angle_between(np.broadcast_to(P, (N, 6, 3)), pos[nb]), 0.0)
    edge_len = np.where(valid, R_PLANET * angle_between(ca, cb), 0.0)
    e = cb - ca
    t = _cross(e, np.broadcast_to(P, (N, 6, 3)))                        # edge × up
    t = t - _dot(t, P)[..., None] * P
    t = np.where((_dot(t, pos[nb] - P) < 0)[..., None], -t, t)
    edge_dir = np.where(valid[..., None], t / np.where(valid, _norm(t), 1.0)[..., None], 0.0)
    c1 = np.take_along_axis(corners, nxt[..., None], axis=1)
    area = np.zeros(N)
    for k in range(6):
        a = sph_tri_area(pos, corners[:, k], c1[:, k])
        area = area + np.where(valid[:, k], a, 0.0)
    area = area * R_PLANET * R_PLANET

    g = dict(n=n, N=N, pos=pos, lat=lat, lon=lon, area=area, east=east, north=north,
             nbr=nbr.astype(np.int32), deg=deg.astype(np.uint8), dist=dist, edgeLen=edge_len, edgeDir=edge_dir,
             corners=corners, totalArea=float(np.sum(area)), lloydIters=iters, lloydMoved=moved)
    g.update(edge_list(g['nbr'], g['deg']))
    return g


def edge_list(nbr, deg):
    """Unordered edge list (i < j) in v1's order: by i, then slot k (v1 SPEC §5.3), with each side's slot."""
    N = nbr.shape[0]
    valid = np.arange(6)[None, :] < deg.astype(np.int64)[:, None]
    ii, kk = np.nonzero(valid & (nbr > np.arange(N)[:, None]))
    jj = nbr[ii, kk]
    kj = np.argmax(nbr[jj] == ii[:, None], axis=1)
    return dict(E=len(ii), eI=ii.astype(np.int32), eJ=jj.astype(np.int32), eKi=kk.astype(np.uint8), eKj=kj.astype(np.uint8))


def morton_order(pos):
    """Cell permutation along a 3-D Morton (Z-order) curve of the unit positions, for memory locality."""
    q = np.clip(((pos + 1) * 0.5 * (2 ** 21 - 1)).astype(np.uint64), 0, 2 ** 21 - 1)

    def spread(v):
        v = v & np.uint64(0x1FFFFF)
        for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                            (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):
            v = (v | (v << np.uint64(shift))) & np.uint64(mask)
        return v

    key = spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)) | (spread(q[:, 2]) << np.uint64(2))
    return np.argsort(key, kind='stable')


def reorder(g, perm):
    """The same mesh with cells renumbered: new cell k is old cell perm[k]. Edges and vertices are renumbered
    by the usual rules (edges by their lower cell, vertices by first appearance)."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = dict(g)
    for k in ('pos', 'lat', 'lon', 'area', 'east', 'north', 'deg', 'dist', 'edgeLen', 'edgeDir', 'corners'):
        out[k] = g[k][perm]
    nb = g['nbr'][perm]
    out['nbr'] = np.where(nb >= 0, inv[np.maximum(nb, 0)], -1).astype(np.int32)
    out.update(edge_list(out['nbr'], out['deg']))
    return out
