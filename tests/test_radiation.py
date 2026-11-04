"""LONGWAVE (V2-6a, SPEC #109, #110): the model's k-table longwave, as an independent NumPy evaluation of its method,
against line-by-line fluxes (tools/lbl.py, saved in tests/data/lbl: four fitted and three unfitted AFGL columns), and the
GPU kernels against that evaluation on a world's own sub-columns, clouds included."""
from pathlib import Path

import cupy as cp
import numpy as np
import pytest

from insilica.radiation import DATA
from insilica.world import World

REF = Path(__file__).resolve().parent / 'data' / 'lbl'
COLUMNS = ['us_standard_400', 'us_standard_800', 'tropical_400', 'subarctic_winter_400', 'tropical_800',
           'subarctic_winter_400_h2o2', 'us_standard_400_h2o2_dT10']
GRAV, M_AIR, NA = 9.80665, 28.9647e-3, 6.02214076e23
SUB_HPA = (0, 1, 5, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1e4)   # model-like sub-levels (#110)


def lw_numpy(t, x, y, u, fh, Tt, Tb, Ts, groups, fc=None, tc=None):
    """The model's longwave (lw_column) in NumPy: sub-levels' table positions x (log p), y (T), amounts u (sub x 5),
    H2O's place fh, edge temperatures Tt, Tb; groups (lists of sub-levels, top first), each with cloud cover fc and its
    sub-levels' cloud optical depths tc. Returns up and down from the air (group edges), the surface's transmission
    (per g-point), and the surface's Planck radiance per g-point."""
    lk, pg, tp0 = t['logk'], t['planck'], float(t['T_pl'][0])
    NT, NG = lk.shape[2], lk.shape[3]
    x = np.clip(x, 0, lk.shape[1] - 1 - 1e-4); y = np.clip(y, 0, NT - 1 - 1e-4)
    ip, it = x.astype(int), y.astype(int)
    fp, ft = (x - ip)[:, None], (y - it)[:, None]
    v = lambda a, b: lk[:, ip + a, it + b, :]                 # comp x sub x g
    lkc = np.exp(v(0, 0) + (v(1, 0) - v(0, 0)) * fp + (v(0, 1) - v(0, 0)) * ft + (v(0, 0) - v(0, 1) - v(1, 0) + v(1, 1)) * fp * ft)
    tau = u[:, :1] * (lkc[0] + (lkc[1] - lkc[0]) * fh[:, None]) + sum(u[:, c:c + 1] * lkc[c + 1] for c in range(1, 5))
    def B(T):
        z = np.clip(np.asarray(T, float) - tp0, 0, pg.shape[0] - 1 - 1e-4)
        j = z.astype(int)
        return pg[j] + (pg[j + 1] - pg[j]) * (z - j)[..., None]
    Bt, Bb, Bs = B(Tt), B(Tb), B(Ts)
    tc = np.zeros(len(x)) if tc is None else tc
    fc = np.zeros(len(groups)) if fc is None else fc
    mu = 1 / 1.66
    def through(I, ta, Bn, Bf):
        z = ta / mu
        tr = np.exp(-z)
        lin = np.where(z > 1e-3, 1 - (1 - tr) / np.maximum(z, 1e-30), z * (0.5 - z / 6))
        return I * tr + Bf * (1 - tr) + (Bn - Bf) * lin
    nk = len(groups)
    dn, up, trs = np.zeros((nk + 1, NG)), np.zeros((nk + 1, NG)), np.ones((nk + 1, NG))
    I = np.zeros(NG)
    for k, sl in enumerate(groups):
        Ic, Id = I.copy(), I.copy()
        for s in sl:
            Ic = through(Ic, tau[s], Bb[s], Bt[s])
            Id = through(Id, tau[s] + tc[s] * mu, Bb[s], Bt[s])
        I = (1 - fc[k]) * Ic + fc[k] * Id
        dn[k + 1] = np.pi * I
    I, tr = np.zeros(NG), np.ones(NG)
    for k in range(nk - 1, -1, -1):
        Ic, Id, tcl, tcd = I.copy(), I.copy(), tr.copy(), tr.copy()
        for s in groups[k][::-1]:
            t1 = np.exp(-tau[s] / mu)
            Ic, tcl = through(Ic, tau[s], Bt[s], Bb[s]), tcl * t1
            Id, tcd = through(Id, tau[s] + tc[s] * mu, Bt[s], Bb[s]), tcd * t1 * np.exp(-tc[s])
        I, tr = (1 - fc[k]) * Ic + fc[k] * Id, (1 - fc[k]) * tcl + fc[k] * tcd
        up[k], trs[k] = np.pi * I, tr
    return up, dn, trs, Bs


def afgl_terms(t, r):
    """A saved line-by-line column's layers in the model's terms (each its own group, no clouds)."""
    p, T = r['p'], r['T']
    pm = (p[1:] - p[:-1]) / np.log(p[1:] / p[:-1])
    Tm = (T[1:] + T[:-1]) / 2
    air = np.diff(p) / GRAV / M_AIR * NA / 1e4
    mid = lambda g: (r['vmr_' + g][1:] + r['vmr_' + g][:-1]) / 2
    u = np.stack([mid(g) * air for g in ('H2O', 'CO2', 'O3', 'N2O', 'CH4')], 1)
    P, Tt = t['P'], t['T']
    x = (np.log(P[0]) - np.log(pm)) / ((np.log(P[0]) - np.log(P[-1])) / (len(P) - 1))
    y = (Tm - Tt[0]) / (Tt[1] - Tt[0])
    return x, y, u, mid('H2O') / 0.04, T[:-1], T[1:], T[-1], [[s] for s in range(len(pm))]


def heating(up, dn, p):
    net = up - dn
    return (net[1:] - net[:-1]) / np.diff(p) * GRAV / 1004.64 * 86400


@pytest.mark.xdist_group('grid4')
def test_longwave(record):
    """LONGWAVE k-table longwave: against line-by-line on seven AFGL columns (three not fitted), and the GPU kernels against it on a world's sub-columns"""
    t = dict(np.load(DATA))
    worst = dict(olr=0.0, down=0.0, heat=0.0, forcing=0.0)
    olr = {}
    for name in COLUMNS:
        r = np.load(REF / f'lbl_{name}.npz')
        up, dn, tr, Bs = lw_numpy(t, *afgl_terms(t, r))
        U = up.sum(1) + np.pi * (tr * Bs).sum(1)         # the surface black at the lowest level's temperature
        D = dn.sum(1)
        Ur, Dr = r['up'].sum(0), r['dn'].sum(0)
        olr[name] = (U[0], Ur[0], U - D, Ur - Dr, r['p'])
        worst['olr'] = max(worst['olr'], abs(U[0] / Ur[0] - 1))
        worst['down'] = max(worst['down'], abs(D[-1] / Dr[-1] - 1))
        sub = sorted({int(np.argmin(np.abs(r['p'] - q * 100))) for q in SUB_HPA})
        e = np.abs(heating(U[sub], D[sub], r['p'][sub]) - heating(Ur[sub], Dr[sub], r['p'][sub]))
        worst['heat'] = max(worst['heat'], e[1:].max())   # below 1 hPa (#110)
    for a, b, lev in (('us_standard_400', 'us_standard_800', 0), ('tropical_400', 'tropical_800', 0)):
        Fk, Fr = olr[a][2] - olr[b][2], olr[a][3] - olr[b][3]
        p = olr[a][4]
        for j in (0, int(np.argmin(np.abs(p - (22700 if 'us' in a else 9400))))):    # top and tropopause
            worst['forcing'] = max(worst['forcing'], abs(Fk[j] / Fr[j] - 1))

    # the GPU kernels against the NumPy method on a world's own sub-columns (cloudy after a few steps)
    w = World(4, 'earth', params=dict(longwave='ktables'))
    w.step(24, graph=False)
    w.physics.radiate(w.h, w.theta, w.Ts, w.time, w.rad_every * w.dt, q=w.q, cw=w.cw)
    lw, L, N = w.physics.lw, w.L, w.grid.N
    sub, radf, Ts = lw.sub.get().reshape(4 * L, 14, N), lw.radf.get(), w.Ts.get()
    RW = L + 1
    kern = 0.0
    cells = np.linspace(0, N - 1, 64).astype(int)
    for i in cells:
        S = sub[:, :, i].astype(float)
        groups = [list(range(4 * k, 4 * k + 4)) for k in range(L)]
        up, dn, tr, Bs = lw_numpy(t, S[:, 2], S[:, 3], S[:, 4:9], S[:, 9], S[:, 0], S[:, 1], Ts[i], groups,
                                  fc=S[::4, 11], tc=S[:, 10])
        ref = np.concatenate([up.sum(1), dn.sum(1), (tr * Bs).sum(1) / Bs.sum()])
        got = radf[:3 * RW, i]
        kern = max(kern, np.max(np.abs(got - ref)) / max(1.0, dn.sum(1)[-1]))
    clouds = float(np.mean(sub[::4, 11, :] > 0.01))
    ok = worst['olr'] <= 0.01 and worst['down'] <= 0.01 and worst['heat'] <= 0.2 and worst['forcing'] <= 0.10 and kern <= 1e-4
    record('LONGWAVE', f'seven AFGL columns (three unfitted): OLR {worst["olr"] * 100:.2f}%, surface down {worst["down"] * 100:.2f}%, '
           f'heating {worst["heat"]:.3f} K/day per sub-level below 1 hPa, 2×CO₂ forcing (top, tropopause) {worst["forcing"] * 100:.1f}%; '
           f'kernels against NumPy {kern:.1e} of the surface down (64 columns, {clouds * 100:.0f}% of layers cloudy)',
           'OLR, surface down ≤ 1%; heating ≤ 0.2 K/day; forcing ≤ 10%; kernels ≤ 1e-4', ok)
    assert ok


EARTH_SPIN_YEARS = 8                             # the k-tables' earth settles to within a few W/m² by 8 years (#112)
EARTH_BUDGET = dict(sun=240.0, olr=239.0, olrc=266.0, lwdn=342.0, sws=161.0)   # Wild et al. 2013, CERES EBAF (W/m²)


@pytest.mark.xdist_group('grid4')
def test_earth_budget(record):
    """EARTH earth's global energy budget with the k-tables, all-sky and clear-sky, against measured (V2-6a, SPEC #109, #112)"""
    from insilica import runcache
    w0 = World(4, 'earth')
    year = int(round(365 * 86400 / w0.dt))
    w = runcache.spun_up('earth-n4', lambda: World(4, 'earth'), EARTH_SPIN_YEARS * year)[0]
    L, RW = w.L, w.L + 1
    a = cp.asarray(w.grid.areaCell / w.grid.areaCell.sum())
    acc = {k: cp.zeros(()) for k in EARTH_BUDGET}
    day = int(round(86400 / w.dt))
    S = 365 * 4                                  # four samples a day over the ninth year
    for _ in range(S):
        w.step(day // 4)
        p, co = w.physics, w.physics.cloudo
        f = p.sw.swf
        acc['sun'] += a @ p.sw_abs
        acc['olr'] += a @ p.olr
        acc['olrc'] += a @ (p.olr + co[L + 2])
        acc['lwdn'] += a @ p.lw.radf[2 * RW - 1]
        acc['sws'] += a @ (p.sw_abs * f[L] / cp.maximum(f[:L + 1].sum(0), 1e-30))
    m = {k: float(v) / S for k, v in acc.items()}
    err = {k: m[k] / EARTH_BUDGET[k] - 1 for k in m}
    names = dict(sun='absorbed sunlight', olr='OLR', olrc='clear-sky OLR', lwdn='surface longwave down', sws='surface sunlight')
    ok = all(abs(e) <= 0.05 for e in err.values())
    # known failure (#112, owner): too little low cloud lets too much sunlight reach the surface until V2-6b's boundary layer
    record('EARTH', f'n = 4, year {EARTH_SPIN_YEARS + 1}: ' + '; '.join(f'{names[k]} {m[k]:.1f} against {EARTH_BUDGET[k]:.0f} ({err[k] * 100:+.1f}%)' for k in m)
           + f'; net {m["sun"] - m["olr"]:+.1f}' + ('' if ok else ' — known failure to V2-6b (#112)'),
           'each within 5%', ok)
    assert ok
