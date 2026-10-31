"""WAVES (fast and internal gravity waves at their true speeds through the split-explicit core) and BAROCLINIC
(Phillips two-layer instability: unstable above the critical shear, stable below)."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import cases
from insilica import grid as gridmod
from insilica.constants import GRAV, OMEGA
from insilica.grid.trisk import cell_winds
from insilica.gpu import DeviceGrid
from insilica.ocean import OCEAN, Ocean
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
    """WAVES External gravity waves at √(gH) (air 300 m/s, ocean 200 m/s) and a two-layer internal wave (n = 5); the
    ocean's equatorial Kelvin wave at its first baroclinic speed (n = 6)"""
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
    ck, c1 = ocean_kelvin(6, 5.0, 25.0)
    err = ck / c1 - 1
    ok &= abs(err) <= 0.02
    out.append(f'ocean Kelvin (n = 6) {ck:.3f} vs {c1:.3f} m/s ({err * 100:+.2f}%)')
    record('WAVES', '; '.join(out), 'external within 2%, internal within 3%, ocean Kelvin within 2%', ok)
    assert ok


def ocean_kelvin(n, t1_d, t2_d):
    """The ocean alone (its default layers, fixed g', friction and smoothing; no air) at rest with a first-baroclinic-mode
    bump at (0°, 0°): the speed of the mode's eastward peak along the equator (|lat| < 2°) between two times, and c1."""
    g = gridmod.get(n)
    c = OCEAN
    h0 = [c['h_mixed'], c['h_thermo'], c['depth'] - c['h_mixed'] - c['h_thermo']]
    speeds, V, W = cases.layer_modes(h0, c['gprime'])
    T = np.stack([np.full(g.N, t) for t in (c['T_thermo'], c['T_thermo'], c['T_deep'])])
    oc = Ocean(DeviceGrid(g), state=(cases.mode_bump(g, h0, V[:, 1]), np.zeros((len(h0), g.E)), T))
    eq = np.abs(g.lat) < math.radians(2.0)
    x = np.angle(np.exp(1j * g.lon[eq])) * cases.R_PLANET
    east, b = x > 0, 100e3
    idx = (x[east] / b).astype(int)
    w = np.bincount(idx)
    pos, steps = [], 0
    for t in (t1_d, t2_d):
        while steps < int(round(t * 86400 / c['dt'])):
            oc.step(mix=False)
            steps += 1
        a = W[1] @ (oc.h.get()[:, eq] - np.asarray(h0)[:, None])
        prof = np.where(w > 0, np.bincount(idx, weights=a[east]) / np.maximum(w, 1), np.nan)
        k = int(np.nanargmax(prof[2:-1])) + 2
        y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
        pos.append((k + 0.5 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * b)
    return (pos[1] - pos[0]) / ((t2_d - t1_d) * 86400), speeds[1]


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


@pytest.mark.xdist_group('grid5')
def test_gyre(record):
    """GYRE Western intensification (n = 5): the ocean alone in a closed basin (15–55°N, 60° wide) under a prescribed wind stress, 3 years, the last one's mean: full-depth transport at 35°N against Sverdrup"""
    g = gridmod.get(5)
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    land = ~((np.abs(lon) < 30) & (lat > 15) & (lat < 55))
    oc = Ocean(DeviceGrid(g), Tsurf=np.full(g.N, 293.15), land=land.astype(float))
    tau0, c = 0.1, oc.cfg
    elat = 0.5 * (lat[g.eI] + lat[g.eJ])
    F = cp.asarray(-tau0 * np.cos(np.pi * (np.clip(elat, 15, 55) - 15) / 40) * cases.edge_east_north(g)[2] * c['dt'] / c['rho'] * oc.wall)
    north = np.stack([-np.sin(g.lat) * np.cos(g.lon), -np.sin(g.lat) * np.sin(g.lon), np.cos(g.lat)], 1)
    steps, V, nv = int(3 * 365 * 86400 / c['dt']), np.zeros(g.N), 0
    for s in range(steps):
        oc.u[0] += F / (0.5 * (oc.h[0][g.eI] + oc.h[0][g.eJ]))
        oc.step(mix=False)
        if s >= steps - 365 * 24 and s % 24 == 0:
            h, u = oc.h.get(), oc.u.get()
            V += sum(h[k] * np.sum(cell_winds(g, u[k]) * north, 1) for k in range(oc.L))
            nv += 1
    V /= nv
    row = (np.abs(lat - 35) < 2.5) & ~land
    dx = math.radians(1.0) * cases.R_PLANET * math.cos(math.radians(35))
    x = lon + 30
    prof = [float(V[row & (x >= a) & (x < a + 10)].mean()) * 10 * dx / 1e6 for a in range(0, 60, 10)]    # Sv per 10° bin
    curl = -tau0 * (np.pi / math.radians(40)) / cases.R_PLANET * math.sin(np.pi / 2)
    sverdrup = curl / (c['rho'] * 2 * OMEGA * math.cos(math.radians(35)) / cases.R_PLANET) * 60 * dx / 1e6
    interior = sum(prof[1:])
    ratio = prof[0] / (abs(interior) / 5)
    finite = bool(cp.isfinite(oc.u).all())
    ok = finite and abs(interior / sverdrup - 1) <= 0.3 and prof[0] > 0 and ratio > 3
    record('GYRE', f'Sv by 10° from the west wall {" ".join(f"{v:+.1f}" for v in prof)}; interior {interior:+.1f} vs Sverdrup {sverdrup:+.1f} '
           f'({(interior / sverdrup - 1) * 100:+.0f}%); west bin ×{ratio:.1f} the mean interior bin; max |u| {float(cp.abs(oc.u).max()):.2f} m/s',
           'interior within 30% of Sverdrup; west bin northward, > 3× the mean interior bin', ok)
    assert ok
