"""WAVES (fast and internal gravity waves at their true speeds through the split-explicit core) and BAROCLINIC
(Phillips two-layer instability: unstable above the critical shear, stable below)."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import cases
from insilica import grid as gridmod
from insilica.constants import GRAV
from insilica.world import World


def front(g, anom, r, bin_m=60e3):
    """Radius of the peak of the area-weighted radial profile of anom (parabolic interpolation between bins)."""
    idx = (r / bin_m).astype(int)
    nb = idx.max() + 1
    s = np.bincount(idx, weights=anom * g.areaCell, minlength=nb)
    w = np.bincount(idx, weights=g.areaCell, minlength=nb)
    prof = np.where(w > 0, s / np.maximum(w, 1e-30), -np.inf)
    k = int(np.argmax(prof[3:])) + 3
    y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
    return (k + 0.5 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * bin_m


def wave_speed(n, state, dt, t1_h, t2_h, layer, base, gprime=()):
    w = World(n, 'w2', params={'dt': dt, 'rotation': 0.0, 'gprime': list(gprime)}, state=state)
    r = cases.great_circle(w.grid, 0.0, 0.0)
    pos = {}
    for t in (t1_h, t2_h):
        w.step(int(round(t * 3600 / dt)) - w.tick)
        pos[t] = front(w.grid, w.h.get()[layer] - base, r)
    return (pos[t2_h] - pos[t1_h]) / ((t2_h - t1_h) * 3600), w.nfast


@pytest.mark.xdist_group('grid5')
def test_waves(record):
    """WAVES External gravity waves at √(gH) (air 300 m/s, ocean 200 m/s) and a two-layer internal wave (n = 5)"""
    out, ok = [], True
    for label, c0, dt in (('air', 300.0, 1800.0), ('ocean', 200.0, 3600.0)):
        g = World(5, 'w2').grid
        H = c0 ** 2 / GRAV
        c, nf = wave_speed(5, cases.gravity_wave(g, H), dt, 2, 8, 0, H)
        err = c / c0 - 1
        ok &= abs(err) <= 0.02
        out.append(f'{label} external {c:.1f} vs {c0:.0f} m/s ({err * 100:+.2f}%, step {dt / 60:.0f} min, {nf} fast sub-steps)')
    gp, h1, h2 = 0.5, 5000.0, 5000.0
    g = World(5, 'w2').grid
    ci = cases.internal_wave_speed(gp, h1, h2)
    c, nf = wave_speed(5, cases.internal_wave(g, h1, h2), 1800.0, 12, 48, 1, h2, gprime=[gp])
    err = c / ci - 1
    ok &= abs(err) <= 0.03
    out.append(f'internal {c:.2f} vs {ci:.2f} m/s ({err * 100:+.2f}%)')
    record('WAVES', '; '.join(out), 'external within 2%, internal within 3%', ok)
    assert ok


def disturbance_energy(a, b):
    """Energy of the difference between two worlds: Σ_e l d Σ_k h_e,k Δu²/2 + Σ A (g Δη² + g' Δz_1²)/2."""
    g = a.grid
    ha, hb, ua, ub = a.h.get(), b.h.get(), a.u.get(), b.u.get()
    he = 0.5 * (ha[:, g.eI] + ha[:, g.eJ])
    ke = 0.5 * np.sum(g.le * g.de * np.sum(he * (ua - ub) ** 2, axis=0))
    deta = ha.sum(0) - hb.sum(0)
    dz1 = ha[1] - hb[1]
    return ke + 0.5 * np.sum(g.areaCell * (GRAV * deta ** 2 + a.gprime[0] * dz1 ** 2))


@pytest.mark.xdist_group('grid6')
def test_baroclinic(record):
    """BAROCLINIC Phillips two-layer instability at n = 6: grows at 3× the critical shear, not at 0.3×"""
    out, ok = [], True
    for factor, want in ((3.0, 'grow'), (0.3, 'stay')):
        h, b, u, gp, h_base = cases.phillips(gridmod.get(6), factor)
        params = {'gprime': [gp]}
        pert = World(6, 'phillips', params=params, state=(h, b, u))
        base = World(6, 'phillips', params=params, state=(h_base, b, u))
        E = []
        for days in (1, 19):
            k = int(days * 86400 / pert.dt)
            pert.step(k)
            base.step(k)
            E.append(disturbance_energy(pert, base))
        growth = E[1] / E[0]
        passed = growth >= 100 if want == 'grow' else growth <= 2
        ok &= passed and bool(cp.all(cp.isfinite(pert.u)))
        out.append(f'{factor}× critical shear ({h.shape[0]} layers, g\' {gp:.3f}, shear {np.max(np.abs(u[0])):.1f} m/s): '
                   f'disturbance energy day 1 → 20 ×{growth:.3g}')
    record('BAROCLINIC', '; '.join(out), '≥ ×100 at 3×; ≤ ×2 at 0.3×', ok)
    assert ok
