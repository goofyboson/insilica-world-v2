"""K2 (v1's N1 and N6 on the GPU) and C1 (exact conservation)."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.constants import R_PLANET
from insilica.gpu import DeviceGrid, dsum
from insilica.swe import SWE
from insilica.transport import Transport
from insilica.world import World, cosine_bell, initial_state


def l2rel(area, a, b):
    return float(np.sqrt(np.sum(area * (a - b) ** 2) / np.sum(area * b ** 2)))


@pytest.mark.xdist_group('n4-numerics')
def test_K2_N1_williamson2(record):
    """K2 N1 Williamson 2: steady balanced zonal flow, height L2 error after 5 days (n = 4)"""
    g = gridmod.get(4)
    h0, b, u0 = initial_state(g, 'w2')
    dg = DeviceGrid(g)
    s = SWE(dg, b=b)
    h, u = cp.asarray(h0)[None].copy(), cp.asarray(u0)[None].copy()
    for _ in range(5 * 86400 // 450):
        s.step(h, u, 450.0)
    err = l2rel(g.areaCell, h.get()[0], h0)
    record('K2 N1', f'L2 {fmt(err)} (v1: 2.0e-4)', '< 1e-3', err < 1e-3)
    assert err < 1e-3


def solid_body_fm(g, dt, alpha=math.pi / 2 - 0.05):
    """Edge volume fluxes (m³ per step) of v1 N6's solid-body rotation, over a unit-thickness layer."""
    a = R_PLANET
    u0 = 2 * math.pi * a / (12 * 86400)
    om = np.array([-math.sin(alpha) * u0 / a, 0, math.cos(alpha) * u0 / a])
    V = np.cross(om, g.xe * a)
    return np.sum(V * g.ne, axis=1) * g.le * dt


def bell_run(n, order, revolutions=1.0):
    g = gridmod.get(n)
    dg = DeviceGrid(g)
    steps = int(round(576 * 2 ** (n - 4) * revolutions))
    dt = 12 * 86400 * revolutions / steps
    psi0 = cosine_bell(g)
    psi = cp.asarray(psi0)[None].copy()
    Fm = cp.asarray(solid_body_fm(g, dt))[None].copy()
    M = cp.asarray(g.areaCell)[None].copy()
    M1 = cp.zeros_like(M)
    tr = Transport(dg)
    m0, v0 = dsum(M, psi), dsum(M)
    lo = 0.0
    for _ in range(steps):
        tr.mass_after(M, Fm, M1)
        tr.step(M, M1, Fm, psi, order)
        M, M1 = M1, M
        lo = min(lo, float(psi.min()))
    m1, v1 = dsum(M, psi), dsum(M)
    return dict(l2=l2rel(g.areaCell, psi.get()[0], psi0), lo=lo, drift=(m1 - m0) / m0, vdrift=(v1 - v0) / v0, steps=steps)


@pytest.mark.xdist_group('n4-numerics')
def test_K2_N6_transport(record):
    """K2 N6 Transport: cosine bell once around (Williamson 1, n = 4), FCT3 against upwind"""
    up, hi = bell_run(4, 1), bell_run(4, 3)
    ratio = up['l2'] / hi['l2']
    ok = ratio >= 2 and hi['lo'] > -1e-9 and abs(hi['drift']) < 1e-12
    record('K2 N6', f'L2 upwind {up["l2"]:.3f}, FCT3 {hi["l2"]:.3f} (×{ratio:.2f}; v1 ×2.5); min {fmt(hi["lo"])}; tracer drift {fmt(hi["drift"])}',
           '≥ 2× better, no negatives (> −1e-9), mass |drift| < 1e-12', ok)
    assert ok


@pytest.mark.parametrize('n', [pytest.param(4, marks=pytest.mark.xdist_group('n4-numerics')),
                               pytest.param(6, marks=pytest.mark.xdist_group('n6'))])
def test_C1_tracer_conservation(record, n):
    """C1 Exact conservation: cosine bell once around, total tracer and volume"""
    r = bell_run(n, 3)
    ok = abs(r['drift']) <= 1e-13 and abs(r['vdrift']) <= 1e-13
    record(f'C1 bell n={n}', f'{r["steps"]} steps: tracer {fmt(r["drift"])}, volume {fmt(r["vdrift"])}; min {fmt(r["lo"])}', '|change| ≤ 1e-13', ok)
    assert ok


@pytest.mark.parametrize('n,days', [pytest.param(4, 5, marks=pytest.mark.xdist_group('n4-numerics')),
                                    pytest.param(6, 1, marks=pytest.mark.xdist_group('n6'))])
def test_C1_world_conservation(record, n, days):
    """C1 Exact conservation in the coupled world (Williamson 5 + bell): Σh·A, ΣM and ΣM·psi"""
    w = World(n=n, preset='w5')
    t0 = w.totals()
    w.step(int(days * 86400 / w.dt))
    t1 = w.totals()
    d = {k: (t1[k] - t0[k]) / t0[k] for k in t0}
    ok = all(abs(v) <= 1e-13 for v in d.values())
    record(f'C1 world n={n}', f'{w.tick} steps ({days} d): volume {fmt(d["volume"])}, M {fmt(d["M"])}, tracer {fmt(d["tracer"])}',
           '|change| ≤ 1e-13', ok)
    assert ok
