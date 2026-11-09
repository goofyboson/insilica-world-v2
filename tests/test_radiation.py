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


EARTH_SPIN_YEARS = 3                             # coupled years before the asynchronous cycles (#117)
EARTH_CYCLES = (5, 20)                           # (1 coupled year + 20 ocean-only years) × 5, then a coupled year: the thermocline at equilibrium (#107, #117)


def _earth_equilibrium():
    from insilica import spinup
    w = World(4, 'earth5')                     # the five-layer earth: its boundary layer a layer of its own (#125), its stratosphere two (#129)
    year = int(round(365 * 86400 / w.dt))
    w.step(EARTH_SPIN_YEARS * year)
    spinup.spin_up(w, *EARTH_CYCLES)
    return w.step(year)
EARTH_BUDGET = dict(sun=240.0, olr=239.0, olrc=266.0, lwdn=342.0, sws=161.0)   # Wild et al. 2013, CERES EBAF (W/m²)
# WINDS (SPEC #126): NCEP-NCAR Reanalysis 1, monthly long-term means 1991-2020 (NOAA PSL), annual zonal means in 10°
# bands centred on −85 … 85, cos-weighted; layers by θ from the reanalysis's temperature, mass-weighted: 0 above 380 K
# (to the 10 hPa top), 1 at 305–380 K; the mixed layer against the 10 m wind. (value m/s, band centre °)
WINDS_OBS = dict(trades=(-3.8, -2.9),                     # 10 m at 15°S, 15°N
                 west=((6.4, -55), (1.6, 45)),            # 10 m peak at 35–65°, S and N
                 jet=((24.9, -55), (16.2, 45)),           # 305–380 K peak at 25–65°
                 strat=(26.5, 9.9),                       # above 380 K, peak at 25–85°
                 strat_eq=-4.9)                           # above 380 K, 5°S–5°N
# (reported, #129: 380–500 K and above 500 K, at 5°S/N and their peaks at 25–85°, S and N)
# RAIN (V2-6c, SPEC #134): GPCP v2.3 1991–2020 annual mean (tools/gpcp_rain.py): global, 10° bands from 90°S, the 5°-band maximum's latitude
RAIN_OBS = dict(glob=2.69, bands=(0.36, 0.74, 1.9, 3.07, 2.91, 2.62, 2.01, 2.48, 3.91, 5.18, 2.72, 1.82, 2.5, 2.63, 2.42, 1.8, 0.92, 0.53), peak=7.5)
WINDS_SPLIT = dict(low=((-1.6, -0.6), (26.7, -55), (14.3, 45)), high=((-8.6, -8.1), (26.7, -65), (8.3, 65)))


def _winds(w, usum, hsum, S):
    """Annual zonal means by layer in the observed bands, from the year's summed edge winds and layer masses."""
    from insilica.grid.trisk import cell_winds
    g = w.grid
    lat = np.degrees(g.lat)
    band = np.clip(((lat + 90) // 10).astype(int), 0, 17)
    wsum = np.bincount(band, weights=g.areaCell, minlength=18)
    zm = lambda f: np.bincount(band, weights=f * g.areaCell, minlength=18) / wsum
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    u, h = usum.get() / S, hsum.get() / S
    U = np.array([zm(np.sum(cell_winds(g, u[k]) * east, 1)) for k in range(w.L)])
    H = np.array([zm(h[k]) for k in range(w.L)])
    return U, H


@pytest.mark.xdist_group('grid4')
def test_earth_budget(record):
    """EARTH earth5's global energy budget with the k-tables, all-sky and clear-sky, against measured (V2-6a, SPEC #109, #112; earth4 since #125, earth5 since #129)"""
    from insilica import runcache
    w = runcache.spun_up(f'earth5-n4-eq{EARTH_SPIN_YEARS}x{EARTH_CYCLES[0]}x{EARTH_CYCLES[1]}', _earth_equilibrium, 0)[0]
    L, RW = w.L, w.L + 1
    a = cp.asarray(w.grid.areaCell / w.grid.areaCell.sum())
    acc = {k: cp.zeros(()) for k in EARTH_BUDGET}
    day = int(round(86400 / w.dt))
    S = 365 * 4                                  # four samples a day over the measured year
    usum, hsum = cp.zeros_like(w.u), cp.zeros_like(w.h)
    rsum, ssum = cp.zeros(w.grid.N), cp.zeros(w.grid.N)
    for _ in range(S):
        w.step(day // 4)
        usum += w.u; hsum += w.h
        rsum += w.physics.rain; ssum += w.physics.rainp[3]
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
    record('EARTH', f'earth5, n = 4, at equilibrium ({EARTH_SPIN_YEARS} coupled years, {EARTH_CYCLES[0]} × (1 coupled + {EARTH_CYCLES[1]} ocean-only), 1 coupled; the next year): ' + '; '.join(f'{names[k]} {m[k]:.1f} against {EARTH_BUDGET[k]:.0f} ({err[k] * 100:+.1f}%)' for k in m)
           + f'; net {m["sun"] - m["olr"]:+.1f}',
           'each within 5%', ok)

    # WINDS (V2-6b, SPEC #126): the same year's zonal-mean east wind against the reanalysis
    U, H = _winds(w, usum, hsum, S)
    cen = np.arange(-85, 86, 10)
    b = lambda c: int((c + 85) // 10)
    pl = lambda p: f'{p[0]:.1f} at {abs(p[1])}°{"S" if p[1] < 0 else "N"}'

    def peak(k, lo, hi, south):
        s = [b(c) for c in (range(-hi, -lo + 1, 10) if south else range(lo, hi + 1, 10))]
        i = max(s, key=lambda j: U[k, j])
        return U[k, i], cen[i]
    o, bl, ek = WINDS_OBS, w.L - 1, int(w.physics.p['edge_k'])     # ek: the 305–380 K layer, the stratosphere's layers above it (#129)
    U = np.concatenate([(np.sum(U[:ek] * H[:ek], 0) / np.sum(H[:ek], 0))[None], U])      # row 0: the stratosphere, mass-weighted
    S0 = ek + 1                                                                          # (U's row of the 305–380 K layer)
    tr = (U[bl + 1, b(-15)], U[bl + 1, b(15)])
    we = (peak(bl + 1, 35, 65, True), peak(bl + 1, 35, 65, False))
    je = (peak(S0, 25, 65, True), peak(S0, 25, 65, False))
    st = (peak(0, 25, 85, True)[0], peak(0, 25, 85, False)[0])
    seq = 0.5 * (U[0, b(-5)] + U[0, b(5)])
    split = ''
    if ek == 2:
        sp = WINDS_SPLIT
        split = '; '.join(f'{nm} 5°S/N {U[r, b(-5)]:.1f}/{U[r, b(5)]:.1f} ({sp[key][0][0]}/{sp[key][0][1]}), peaks S {pl(peak(r, 25, 85, True))} '
                          f'({pl(sp[key][1])}), N {pl(peak(r, 25, 85, False))} ({pl(sp[key][2])})'
                          for nm, key, r in (('380–500 K', 'low', 2), ('above 500 K', 'high', 1))) + '; '
    checks = dict(trades=all(abs(m - x) <= 1.5 for m, x in zip(tr, o['trades'])),
                  west=all(abs(m[0] - x[0]) <= 3 and abs(m[1] - x[1]) <= 10 for m, x in zip(we, o['west'])),
                  jet=all(abs(m[0] / x[0] - 1) <= 0.3 and abs(m[1] - x[1]) <= 10 for m, x in zip(je, o['jet'])),
                  strat=all(abs(m - x) <= 15 for m, x in zip(st, o['strat'])) and abs(seq - o['strat_eq']) <= 7)
    okw = all(checks.values())
    hpa = H.mean(1) * 9.80665 / 100
    record('WINDS', f'the same year, annual zonal means (m/s; observed NCEP 1991–2020 in brackets): trades 15°S/N {tr[0]:.1f}/{tr[1]:.1f} '
           f'({o["trades"][0]}/{o["trades"][1]}); surface westerlies S {pl(we[0])} ({pl(o["west"][0])}), N {pl(we[1])} ({pl(o["west"][1])}); '
           f'jet 305–380 K S {pl(je[0])} ({pl(o["jet"][0])}), N {pl(je[1])} ({pl(o["jet"][1])}); above 380 K peak S/N {st[0]:.1f}/{st[1]:.1f} '
           f'({o["strat"][0]}/{o["strat"][1]}), 5°S–5°N {seq:.1f} ({o["strat_eq"]}); unlimited: {split}305–380 K at 5°S/N {U[S0, b(-5)]:.1f}/{U[S0, b(5)]:.1f} (−2.7/−3.4), '
           f'cloud layer peak S/N {peak(S0 + 1, 25, 65, True)[0]:.1f}/{peak(S0 + 1, 25, 65, False)[0]:.1f} (15.3/8.8), layer masses ' + '/'.join(f'{x:.0f}' for x in hpa)
           + ' hPa (above 380 K 126, above 500 K 51); failing: ' + (', '.join(k for k, v in checks.items() if not v) or 'none'),
           'trades ±1.5 m/s; surface westerly peaks ±3 m/s, jet peaks ±30%, each within 10° of observed; above 380 K peaks ±15 m/s, 5°S–5°N ±7 m/s', okw,
           title='WINDS earth5\'s zonal winds against the reanalysis (V2-6b, SPEC #126; earth5 since #129)')
    # RAIN (V2-6c, #134): the same year's rain (and snow) against GPCP
    g, mm = w.grid, 86400.0 / S
    lat = np.degrees(g.lat)
    r, sn = rsum.get() * mm, ssum.get() * mm
    zmb = lambda x, width: np.bincount(np.clip(((lat + 90) // width).astype(int), 0, 180 // width - 1), weights=x * g.areaCell, minlength=180 // width)         / np.bincount(np.clip(((lat + 90) // width).astype(int), 0, 180 // width - 1), weights=g.areaCell, minlength=180 // width)
    glob = float(np.average(r, weights=g.areaCell))
    z10, z5 = zmb(r, 10), zmb(r, 5)
    pk = -90 + 5 * int(np.argmax(z5)) + 2.5
    ro = RAIN_OBS
    mid = range(4, 14)                                   # 50°S–50°N
    rb = {i: z10[i] / ro['bands'][i] - 1 for i in mid}
    okr = abs(glob / ro['glob'] - 1) <= 0.15 and abs(pk - ro['peak']) <= 10 and all(abs(x) <= 0.30 for x in rb.values())
    record('RAIN', f'the same year (rain and snow, mm/day; GPCP 1991–2020 in brackets): global {glob:.2f} ({ro["glob"]}, {100 * (glob / ro["glob"] - 1):+.0f}%); '
           f'5° maximum at {pk:+.1f}° ({ro["peak"]:+.1f}°); 10° bands 45°S–45°N ' + ', '.join(f'{z10[i]:.2f} ({ro["bands"][i]}, {100 * rb[i]:+.0f}%)' for i in mid)
           + f'; snow {float(np.average(sn, weights=g.areaCell)):.3f} globally, {float(np.average(sn, weights=g.areaCell * (np.abs(lat) > 50))):.2f} poleward of 50°',
           'global within 15%; maximum within 10° of latitude; each 10° band 50°S–50°N within 30%', okr,
           title="RAIN earth5's rain against GPCP (V2-6c, SPEC #134)")
    assert ok and okw and okr
