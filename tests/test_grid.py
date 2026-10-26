"""GRID: one integrity check per mesh level (topology, geometry, TRiSK identities; the disk cache at n = 4)."""
import math

import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.constants import R_PLANET

SPHERE = 4 * math.pi * R_PLANET ** 2


def trisk_identities(g):
    """Weight antisymmetry l_e d_e w_ee' = −l_e' d_e' w_e'e, and the discrete curl of a gradient (relative)."""
    e = np.repeat(np.arange(g.E, dtype=np.int64), np.diff(g.eoeStart))
    e2 = g.eoe.astype(np.int64)
    lhs = g.le[e] * g.de[e] * g.w
    key, rkey = e * g.E + e2, e2 * g.E + e
    o = np.argsort(key)
    j = o[np.searchsorted(key[o], rkey)]
    assert np.array_equal(key[j], rkey)
    anti = float(np.max(np.abs(lhs + lhs[j])) / np.max(np.abs(lhs)))
    f = 1e3 * np.sin(2 * g.lat) * np.cos(3 * g.lon) + 50 * g.pos[:, 0]
    gr = (f[g.eJ] - f[g.eI]) / g.de
    t1, t2 = g.circ1 * g.de * gr, g.circ2 * g.de * gr
    curl, mag = np.zeros(g.NV), np.zeros(g.NV)
    np.add.at(curl, g.ev1, t1)
    np.add.at(curl, g.ev2, t2)
    np.add.at(mag, g.ev1, np.abs(t1))
    np.add.at(mag, g.ev2, np.abs(t2))
    return anti, float(np.max(np.abs(curl) / mag))


@pytest.mark.parametrize('n', [pytest.param(n, marks=pytest.mark.xdist_group('big-grids' if n >= 7 else f'grid{n}')) for n in range(9)])
def test_grid(record, n):
    """GRID Mesh integrity: topology, geometry, TRiSK identities (and the disk cache at n = 4)"""
    g = gridmod.get(n)
    # topology: counts, 12 pentagons, Euler, every edge in two cells and every vertex in three
    pent = int(np.sum(g.deg == 5))
    topo = bool(g.N == 10 * 4 ** n + 2 and g.E == 30 * 4 ** n and g.NV == 20 * 4 ** n and pent == 12
                and pent + int(np.sum(g.deg == 6)) == g.N and g.N - g.E + g.NV == 2
                and np.all(np.bincount(g.cellEdge[g.cellEdge >= 0], minlength=g.E) == 2)
                and np.all(np.bincount(g.cornerV[g.cornerV >= 0], minlength=g.NV) == 3))
    # geometry: the sphere's area three ways, each cell's kites = its polygon, dual edges ⟂ primal edges
    sums = max(abs(np.sum(a) - SPHERE) / SPHERE for a in (g.area, g.areaCell, g.areaV))
    kites = float(np.max(np.abs(g.areaCell - g.area) / g.area))
    t, d = g.vPos[g.ev2] - g.vPos[g.ev1], g.pos[g.eJ] - g.pos[g.eI]
    orth = float(np.max(np.abs(np.sum(t * d, axis=1)) / (np.linalg.norm(t, axis=1) * np.linalg.norm(d, axis=1))))
    ratio = float(g.area.max() / g.area.min())
    anti, curl = trisk_identities(g)
    worst = max(sums, kites, orth, anti, curl)
    cache = None
    if n == 4:          # a grid read back from the disk cache equals a freshly built one, bit for bit
        fresh, _ = gridmod.make(4, order='v1')
        gridmod.get(4, order='v1')
        gridmod._mem.pop((4, 'v1', 'converged'), None)
        cached = gridmod.get(4, order='v1')
        a, b = fresh.arrays(), cached.arrays()
        cache = not [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))]
    ok = topo and worst <= 1e-12 and (n != 4 or ratio < 1.5) and cache is not False
    record(f'GRID n={n}', f'N {g.N}, E {g.E}, V {g.NV}, {pent} pentagons: topology {"exact" if topo else "WRONG"}; '
           f'Σ areas {fmt(sums)}, cell vs kites {fmt(kites)}, orthogonality {fmt(orth)}, antisymmetry {fmt(anti)}, curl(grad) {fmt(curl)}; '
           f'area max/min {ratio:.3f}; Lloyd {g.lloydIters} its, last move {fmt(float(g.lloydMoved))}'
           + ('' if cache is None else f'; cache {"identical" if cache else "DIFFERS"}'),
           'topology exact; ≤ 1e-12; area ratio < 1.5 at n = 4; cache bit-identical', ok)
    assert ok
