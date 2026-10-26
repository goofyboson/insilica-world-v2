"""ACCURACY: the engine on standard benchmarks (v1's N1 and N6 thresholds) and the Perot wind rebuild's convergence."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.constants import R_PLANET
from insilica.gpu import DeviceGrid, dsum
from insilica.grid.trisk import cell_winds
from insilica.swe import SWE
from insilica.transport import Transport
from insilica.world import cosine_bell, initial_state

pytestmark = pytest.mark.xdist_group('grid6')


def l2rel(area, a, b):
    return float(np.sqrt(np.sum(area * (a - b) ** 2) / np.sum(area * b ** 2)))


def solid_body(g, alpha=math.pi / 2 - 0.05, period_days=12):
    """v1 N6's solid-body rotation: Ω vector and the edge normal winds, sampled at the Voronoi edge midpoints."""
    u0 = 2 * math.pi * R_PLANET / (period_days * 86400)
    om = np.array([-math.sin(alpha) * u0 / R_PLANET, 0, math.cos(alpha) * u0 / R_PLANET])
    return om, np.sum(np.cross(om, g.xm * R_PLANET) * g.ne, axis=1)


def bell(g, order):
    """Cosine bell carried once around (v1 N6). Returns (L2 error, minimum, tracer drift)."""
    dg = DeviceGrid(g)
    steps = 576 * 2 ** (g.n - 4)
    dt = 12 * 86400 / steps
    psi0 = cosine_bell(g)
    psi = cp.asarray(psi0)[None].copy()
    Fm = cp.asarray(solid_body(g)[1] * g.le * dt)[None].copy()
    M, M1 = cp.asarray(g.areaCell)[None].copy(), cp.zeros((1, g.N))
    tr = Transport(dg)
    m0, lo = dsum(M, psi), 0.0
    for _ in range(steps):
        tr.mass_after(M, Fm, M1)
        tr.step(M, M1, Fm, psi, order)
        M, M1 = M1, M
        lo = min(lo, float(psi.min()))
    return l2rel(g.areaCell, psi.get()[0], psi0), lo, (dsum(M, psi) - m0) / m0


def test_accuracy(record):
    """ACCURACY N1 Williamson 2 (5 days, n = 4); N6 cosine bell (n = 4); Perot rebuild convergence (n = 4–6)"""
    g = gridmod.get(4)
    h0, b, u0 = initial_state(g, 'w2')
    s = SWE(DeviceGrid(g), b=b)
    h, u = cp.asarray(h0)[None].copy(), cp.asarray(u0)[None].copy()
    for _ in range(5 * 86400 // 450):
        s.step(h, u, 450.0)
    n1 = l2rel(g.areaCell, h.get()[0], h0)
    up, hi = bell(g, 1), bell(g, 3)
    ratio = up[0] / hi[0]
    n6 = ratio >= 2 and hi[1] > -1e-9 and abs(hi[2]) < 1e-12
    mx, rms = [], []
    for n in (4, 5, 6):
        gn = gridmod.get(n)
        om, un = solid_body(gn)
        exact = np.cross(om, gn.pos * R_PLANET)
        err = np.linalg.norm(cell_winds(gn, un) - exact, axis=1) / np.max(np.linalg.norm(exact, axis=1))
        mx.append(float(err.max()))
        rms.append(float(np.sqrt(np.mean(err ** 2))))
    fm, fr = min(mx[0] / mx[1], mx[1] / mx[2]), min(rms[0] / rms[1], rms[1] / rms[2])
    ok_n1, ok_p = n1 < 1e-3, fm >= 1.8 and fr >= 3.5
    record('ACCURACY N1', f'L2 {fmt(n1)}', '< 1e-3', ok_n1)
    record('ACCURACY N6', f'L2 upwind {up[0]:.3f}, FCT3 {hi[0]:.3f} (×{ratio:.2f}); min {fmt(hi[1])}; tracer drift {fmt(hi[2])}',
           '≥ 2× better than upwind, min > −1e-9, |drift| < 1e-12', n6)
    record('ACCURACY Perot', f'max {" → ".join(fmt(x) for x in mx)}, rms {" → ".join(fmt(x) for x in rms)} '
           f'(slowest fall per level: max ×{fm:.2f}, rms ×{fr:.2f})', 'max ≥ ×1.8, rms ≥ ×3.5 per level', ok_p)
    assert ok_n1 and n6 and ok_p
