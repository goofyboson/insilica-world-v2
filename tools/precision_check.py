"""Precision of the mesh formulas against a 40-digit reference (mpmath; `pip install mpmath`), SPEC §9 #15.

1. Random icosphere triangles at n = 5 and 8: corner (circumcenter) position, area and arc length from v1's
   formulas and from V2's, against the reference with every point projected exactly onto the unit sphere.
2. Cell areas at n = 5 computed by v1 and by V2 from v1's own positions, against the same reference
   (needs the v1 dump, built by tests/v1ref.py from the v1 checkout next to this repository).

  python tools/precision_check.py
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from insilica.constants import R_PLANET  # noqa: E402
from insilica.grid import icosa  # noqa: E402
from insilica.grid.icosa import _cross, _dot, _norm  # noqa: E402

mp.mp.dps = 40


def v1_circumcenter(a, b, c):
    w = _cross(b - a, c - a)
    w = np.where((_dot(w, a + b + c) < 0)[..., None], -w, w)
    return w / _norm(w)[..., None]


def v1_area(a, b, c):
    return 2 * np.arctan2(np.abs(_dot(a, _cross(b, c))), 1 + _dot(a, b) + _dot(b, c) + _dot(c, a))


def v1_angle(a, b):
    return np.arctan2(_norm(_cross(a, b)), _dot(a, b))


def M(x):
    v = mp.matrix([mp.mpf(float(t)) for t in x])
    return v / mp.norm(v)


def cr(a, b):
    return mp.matrix([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def dot(a, b):
    return (a.T * b)[0]


def ref_cc(a, b, c):
    w = cr(b - a, c - a)
    w = w / mp.norm(w)
    return w if dot(w, a + b + c) > 0 else -w


def ref_area(a, b, c):
    return 2 * mp.atan2(abs(dot(a, cr(b, c))), 1 + dot(a, b) + dot(b, c) + dot(c, a))


def triangles(n, k=300, seed=0):
    V, F = icosa.icosphere(n)
    idx = np.random.default_rng(seed).choice(len(F), k, replace=False)
    a, b, c = V[F[idx, 0]], V[F[idx, 1]], V[F[idx, 2]]
    out = {}
    for name, fc, fa, fg in (('v1', v1_circumcenter, v1_area, v1_angle), ('V2', icosa.circumcenter, icosa.sph_tri_area, icosa.angle_between)):
        C, A, G = fc(a, b, c), fa(a, b, c), fg(a, b)
        ec, ea, eg = [], [], []
        for i in range(k):
            pa, pb, pc = M(a[i]), M(b[i]), M(c[i])
            rc, ra = ref_cc(pa, pb, pc), ref_area(pa, pb, pc)
            rg = mp.atan2(mp.norm(cr(pa, pb)), dot(pa, pb))
            ec.append(float(mp.norm(rc - mp.matrix(list(C[i])))))
            ea.append(float(abs(A[i] - ra) / ra))
            eg.append(float(abs(G[i] - rg) / rg))
        out[name] = (max(ec), max(ea), max(eg))
    return out


def cell_areas_at_v1_positions(k=150):
    import v1ref
    from insilica.grid.icosa import build_grid
    p1, nb, dg, a1 = (v1ref.load('n5', x) for x in ('pos', 'nbr', 'deg', 'area'))
    orig = icosa.lloyd_scvt
    icosa.lloyd_scvt = lambda V, ring, deg, xp=np, **kw: (p1.copy(), 0, 0.0)
    try:
        g2 = build_grid(5)
    finally:
        icosa.lloyd_scvt = orig
    e1, e2 = [], []
    for i in np.random.default_rng(1).choice(len(p1), k, replace=False):
        m = int(dg[i])
        P, Q = M(p1[i]), [M(p1[j]) for j in nb[i, :m]]
        C = [ref_cc(P, Q[j], Q[(j + 1) % m]) for j in range(m)]
        A = sum(ref_area(P, C[j], C[(j + 1) % m]) for j in range(m)) * R_PLANET ** 2
        e1.append(float(abs(a1[i] - A) / A))
        e2.append(float(abs(g2['area'][i] - A) / A))
    return max(e1), max(e2)


if __name__ == '__main__':
    for n in (5, 8):
        r = triangles(n)
        print(f'n = {n} triangles: corner position (rad) v1 {r["v1"][0]:.1e}, V2 {r["V2"][0]:.1e}; '
              f'area (relative) v1 {r["v1"][1]:.1e}, V2 {r["V2"][1]:.1e}; arc (relative) v1 {r["v1"][2]:.1e}, V2 {r["V2"][2]:.1e}')
    e1, e2 = cell_areas_at_v1_positions()
    print(f'n = 5 cell areas from v1\'s positions (relative): v1 {e1:.1e}, V2 {e2:.1e}')
