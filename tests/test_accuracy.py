"""ACCURACY: the layered core (time-split RK3, one layer) on the standard shallow-water benchmarks with v1's
thresholds, the cosine bell, and the Perot wind rebuild's convergence."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.constants import OMEGA, R_PLANET
from insilica.gpu import DeviceGrid, dsum
from insilica.grid.trisk import cell_winds
from insilica.transport import Transport
from insilica.world import World, cosine_bell

pytestmark = pytest.mark.xdist_group('grid6')
DEG = 180 / math.pi


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


def vertex_vorticity(g, u):
    z = np.zeros(g.NV)
    np.add.at(z, g.ev1, g.circ1 * g.de * u)
    np.add.at(z, g.ev2, g.circ2 * g.de * u)
    return z / g.areaV


def test_accuracy(record):
    """ACCURACY Williamson 2/5/6, Galewsky, a 30-day balanced flow at n = 5 (layered core); bell; Perot"""
    rows, ok = [], True

    # N1 Williamson 2, 5 days, at n = 4, 5, 6: error and its fall per level
    err = []
    for n in (4, 5, 6):
        w = World(n, 'w2')
        h0 = w.h.get()[0].copy()
        w.step(int(5 * 86400 / w.dt))
        err.append(l2rel(w.grid.areaCell, w.h.get()[0], h0))
    fall = min(err[0] / err[1], err[1] / err[2])
    rows.append(('ACCURACY N1', f'L2 {" → ".join(fmt(e) for e in err)} at n = 4, 5, 6 (×{err[0] / err[1]:.2f}, ×{err[1] / err[2]:.2f})',
                 '< 1e-3 at n = 4; falls ≥ ×3 per level', err[0] < 1e-3 and fall >= 3))

    # W5 mountain, 15 days at n = 4: energy drift, bounded
    w = World(4, 'w5')
    e0 = w.energy()
    w.step(int(15 * 86400 / w.dt))
    drift = (w.energy() - e0) / e0
    umax = float(cp.max(cp.abs(w.u)))
    rows.append(('ACCURACY W5', f'energy drift {fmt(drift)}, max |u| {umax:.1f} m/s', '|drift| < 1e-4, finite', abs(drift) < 1e-4 and math.isfinite(umax)))

    # W6 Rossby–Haurwitz wave 4, 7 days at n = 4: westward drift against theory (v1 N4's procedure)
    w = World(4, 'rh4')
    g = w.grid
    band = (g.lat * DEG >= 30) & (g.lat * DEG <= 50)
    per_day = int(round(86400 / w.dt))
    ph, tt = [], []
    for d in range(8):
        hh = w.h.get()[0]
        cs, sn = np.sum((hh * np.cos(4 * g.lon) * g.area)[band]), np.sum((hh * np.sin(4 * g.lon) * g.area)[band])
        ph.append(math.atan2(sn, cs) / 4)
        tt.append(w.tick * w.dt)
        if d < 7:
            w.step(per_day)
    for k in range(1, len(ph)):
        while ph[k] - ph[k - 1] > math.pi / 4:
            ph[k] -= math.pi / 2
        while ph[k] - ph[k - 1] < -math.pi / 4:
            ph[k] += math.pi / 2
    nu = np.polyfit(tt, ph, 1)[0]
    om = 7.848e-6
    theory = (4 * (3 + 4) * om - 2 * OMEGA) / ((1 + 4) * (2 + 4))
    rows.append(('ACCURACY W6', f'phase speed {nu:.3e} rad/s vs theory {theory:.3e} ({(nu / theory - 1) * 100:+.1f}%)', 'within ±10%',
                 abs(nu / theory - 1) <= 0.1))

    # Galewsky at n = 5 and 6: the bump grows into a breakdown by day 6 (v1 N3's metric, against an unperturbed twin)
    for n in (5, 6):
        g = gridmod.get(n)
        vlat = np.arcsin(np.clip(g.vPos[:, 2], -1, 1)) * DEG
        sel = (vlat >= 20) & (vlat <= 70)
        pert, base = World(n, 'galewsky'), World(n, 'galewsky')
        hb = base.h.get()
        bump = 120 * np.cos(g.lat) * np.exp(-((g.lon / (1 / 3)) ** 2)) * np.exp(-(((math.pi / 4 - g.lat) / (1 / 15)) ** 2))
        base.h[...] = cp.asarray(hb - bump[None])
        r = []
        for days in (1, 5):
            pert.step(int(days * 86400 / pert.dt))
            base.step(int(days * 86400 / base.dt))
            zp, zb = vertex_vorticity(g, pert.u.get()[0]), vertex_vorticity(g, base.u.get()[0])
            r.append(math.sqrt(np.sum((g.areaV * (zp - zb) ** 2)[sel]) / np.sum((g.areaV * zb ** 2)[sel])))
        umx = float(max(cp.max(cp.abs(pert.u)), cp.max(cp.abs(base.u))))
        rows.append((f'ACCURACY Galewsky n={n}', f'vorticity difference / jet day 1 {r[0]:.4f}, day 6 {r[1]:.3f} (×{r[1] / r[0]:.0f}); max |u| {umx:.0f} m/s',
                     'day 6 > 0.3, growth > ×20, max |u| < 200', r[1] > 0.3 and r[1] / r[0] > 20 and umx < 200))

    # 30 days of balanced flow at n = 5 (Williamson 2): wind error everywhere and at the pentagons (v1 N5)
    w = World(5, 'w2')
    g = w.grid
    u0 = w.u.get()[0].copy()
    w.step(int(30 * 86400 / w.dt))
    d = np.linalg.norm(cell_winds(g, w.u.get()[0]) - cell_winds(g, u0), axis=1)
    rows.append(('ACCURACY balanced', f'30 days at n = 5: max wind error {d.max():.3f} m/s (pentagons {d[g.deg == 5].max():.3f})', '< 1 m/s', d.max() < 1))

    # N6 cosine bell, n = 4
    g = gridmod.get(4)
    up, hi = bell(g, 1), bell(g, 3)
    ratio = up[0] / hi[0]
    rows.append(('ACCURACY N6', f'L2 upwind {up[0]:.3f}, FCT3 {hi[0]:.3f} (×{ratio:.2f}); min {fmt(hi[1])}; tracer drift {fmt(hi[2])}',
                 '≥ 2× better than upwind, min > −1e-9, |drift| < 1e-12', ratio >= 2 and hi[1] > -1e-9 and abs(hi[2]) < 1e-12))

    # Perot rebuild of solid-body rotation, n = 4, 5, 6
    mx, rms = [], []
    for n in (4, 5, 6):
        gn = gridmod.get(n)
        om, un = solid_body(gn)
        exact = np.cross(om, gn.pos * R_PLANET)
        e = np.linalg.norm(cell_winds(gn, un) - exact, axis=1) / np.max(np.linalg.norm(exact, axis=1))
        mx.append(float(e.max()))
        rms.append(float(np.sqrt(np.mean(e ** 2))))
    fm, fr = min(mx[0] / mx[1], mx[1] / mx[2]), min(rms[0] / rms[1], rms[1] / rms[2])
    rows.append(('ACCURACY Perot', f'max {" → ".join(fmt(x) for x in mx)}, rms {" → ".join(fmt(x) for x in rms)} '
                 f'(slowest fall per level: max ×{fm:.2f}, rms ×{fr:.2f})', 'max ≥ ×1.8, rms ≥ ×3.5 per level', fm >= 1.8 and fr >= 3.5))

    for rid, value, thr, passed in rows:
        record(rid, value, thr, passed)
        ok &= passed
    assert ok
