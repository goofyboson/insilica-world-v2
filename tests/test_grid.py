"""G1–G3: the mesh and its TRiSK geometry."""
import math

import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.constants import R_PLANET

SPHERE = 4 * math.pi * R_PLANET ** 2
LEVELS = list(range(0, 9))


def _group(n):
    return pytest.mark.xdist_group('big-grids' if n >= 7 else f'grid{n}')


@pytest.mark.parametrize('n', [pytest.param(n, marks=_group(n)) for n in LEVELS])
def test_G1_topology(record, n):
    """G1 Topology: N = 10·4ⁿ+2, E = 30·4ⁿ, V = 20·4ⁿ, 12 pentagons, N − E + V = 2"""
    g = gridmod.get(n)
    p = int(np.sum(g.deg == 5))
    six = int(np.sum(g.deg == 6))
    # every edge appears in exactly two cells; every vertex in exactly three
    ce = g.cellEdge[g.cellEdge >= 0]
    edge_count = np.bincount(ce, minlength=g.E)
    cv = g.cornerV[g.cornerV >= 0]
    v_count = np.bincount(cv, minlength=g.NV)
    ok = (g.N == 10 * 4 ** n + 2 and g.E == 30 * 4 ** n and g.NV == 20 * 4 ** n and p == 12 and p + six == g.N
          and g.N - g.E + g.NV == 2 and np.all(edge_count == 2) and np.all(v_count == 3))
    record(f'G1 n={n}', f'N {g.N}, E {g.E}, V {g.NV}, pentagons {p}, Euler {g.N - g.E + g.NV}', 'exact', ok)
    assert ok


@pytest.mark.parametrize('n', [pytest.param(n, marks=_group(n)) for n in LEVELS])
def test_G2_geometry(record, n):
    """G2 Geometry: areas sum to 4πR², kites tile cells and triangles, edges orthogonal, area ratio"""
    g = gridmod.get(n)
    s_poly = abs(np.sum(g.area) - SPHERE) / SPHERE
    s_kite = abs(np.sum(g.areaCell) - SPHERE) / SPHERE
    s_tri = abs(np.sum(g.areaV) - SPHERE) / SPHERE
    cell_vs_kites = float(np.max(np.abs(g.areaCell - g.area) / g.area))
    # orthogonality: the dual edge (vertex v1 → v2) is perpendicular to the primal edge (cell i → j)
    t = g.vPos[g.ev2] - g.vPos[g.ev1]
    d = g.pos[g.eJ] - g.pos[g.eI]
    orth = float(np.max(np.abs(np.sum(t * d, axis=1)) / (np.linalg.norm(t, axis=1) * np.linalg.norm(d, axis=1))))
    ratio = float(g.area.max() / g.area.min())
    worst = max(s_poly, s_kite, s_tri, cell_vs_kites, orth)
    ok = worst <= 1e-12 and (n != 4 or ratio < 1.5)
    value = (f'Σ area {fmt(s_poly)}, Σ kites {fmt(s_kite)}, Σ triangles {fmt(s_tri)}, cell vs kites {fmt(cell_vs_kites)}, '
             f'orthogonality {fmt(orth)}; area max/min {ratio:.3f}; Lloyd {g.lloydIters} iterations, last move {fmt(float(g.lloydMoved))} rad')
    record(f'G2 n={n}', value, '≤ 1e-12; area ratio < 1.5 at n = 4 (others reported)', ok)
    assert ok


def perot(g, u):
    """v1's cellWinds: wind vectors at cells from edge normal winds (Perot), projected to the tangent plane."""
    sign = g.cellSign.astype(float)
    ce = np.where(g.cellEdge >= 0, g.cellEdge, 0)
    c = np.where(g.cellEdge >= 0, sign * u[ce] * g.le[ce] * R_PLANET, 0.0)
    v = np.sum(c[..., None] * (g.xe[ce] - g.pos[:, None, :]), axis=1) / g.areaCell[:, None]
    return v - np.sum(v * g.pos, axis=1)[:, None] * g.pos


@pytest.mark.parametrize('n', [pytest.param(n, marks=_group(n)) for n in (4, 5, 6)])
def test_G3_trisk_identities(record, n):
    """G3 TRiSK identities: weight antisymmetry, discrete curl(grad) = 0, Perot rebuild of solid-body rotation"""
    g = gridmod.get(n)
    # antisymmetry l_e d_e w_ee' = −l_e' d_e' w_e'e
    e = np.repeat(np.arange(g.E), np.diff(g.eoeStart))
    e2 = g.eoe.astype(np.int64)
    lhs = g.le[e] * g.de[e] * g.w
    key = e * g.E + e2
    rkey = e2 * g.E + e
    o = np.argsort(key)
    j = o[np.searchsorted(key[o], rkey)]
    assert np.array_equal(key[j], rkey)
    anti = float(np.max(np.abs(lhs + lhs[j])) / np.max(np.abs(lhs)))
    # curl(grad f) at vertices for a smooth field
    f = 1e3 * np.sin(2 * g.lat) * np.cos(3 * g.lon) + 50 * g.pos[:, 0]
    gr = (f[g.eJ] - f[g.eI]) / g.de
    terms1, terms2 = g.circ1 * g.de * gr, g.circ2 * g.de * gr
    curl, mag = np.zeros(g.NV), np.zeros(g.NV)
    np.add.at(curl, g.ev1, terms1)
    np.add.at(curl, g.ev2, terms2)
    np.add.at(mag, g.ev1, np.abs(terms1))
    np.add.at(mag, g.ev2, np.abs(terms2))
    cg = float(np.max(np.abs(curl) / mag))
    # solid-body rotation about a tilted axis, 1 m/s at most
    om = np.array([0.3, -0.2, 0.9])
    om /= np.linalg.norm(om) * R_PLANET
    V = lambda x: np.cross(om, x * R_PLANET)
    u = np.sum(V(g.xe) * g.ne, axis=1)
    rec = perot(g, u)
    exact = V(g.pos)
    perr = float(np.max(np.linalg.norm(rec - exact, axis=1)) / np.max(np.linalg.norm(exact, axis=1)))
    ok = anti <= 1e-12 and cg <= 1e-12
    record(f'G3 n={n}', f'antisymmetry {fmt(anti)}, curl(grad) {fmt(cg)}, Perot solid-body error {fmt(perr)}',
           '≤ 1e-12, ≤ 1e-12; Perot reported (must fall with n)', ok)
    assert ok



@pytest.mark.xdist_group('grid6')
def test_G3_perot_converges(record):
    """G3 Perot error falls with resolution (n = 4, 5, 6)"""
    mx, rms = [], []
    for n in (4, 5, 6):
        g = gridmod.get(n)
        om = np.array([0.3, -0.2, 0.9])
        om /= np.linalg.norm(om) * R_PLANET
        V = lambda x: np.cross(om, x * R_PLANET)
        u = np.sum(V(g.xe) * g.ne, axis=1)
        err = np.linalg.norm(perot(g, u) - V(g.pos), axis=1) / np.max(np.linalg.norm(V(g.pos), axis=1))
        mx.append(float(err.max()))
        rms.append(float(np.sqrt(np.mean(err ** 2))))
    ok = mx[0] > mx[1] > mx[2]
    record('G3 Perot', f'max {" → ".join(fmt(x) for x in mx)} (×{mx[0] / mx[1]:.2f}, ×{mx[1] / mx[2]:.2f} per level); '
           f'rms {" → ".join(fmt(x) for x in rms)} (×{rms[0] / rms[1]:.2f}, ×{rms[1] / rms[2]:.2f})', 'max decreasing with n', ok)
    assert ok


@pytest.mark.xdist_group('grid4')
def test_G_cache_identical(record):
    """Grid cache: a grid read back from the disk cache equals a freshly built one, bit for bit"""
    fresh, _ = gridmod.make(4)
    gridmod.get(4)                      # makes sure the cache file exists
    gridmod._mem.pop(4, None)
    cached = gridmod.get(4)             # read from disk
    a, b = fresh.arrays(), cached.arrays()
    diff = [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))]
    record('S1 grid cache', 'identical' if not diff else f'differ: {diff}', 'bit-identical', not diff)
    assert not diff
