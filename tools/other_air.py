"""Radiation for other air (V2-6e, SPEC #226): the wide k-tables, collision-induced absorption per g-point, and the
line-by-line columns in other air that LONGWAVE checks them on.

    python tools/other_air.py ref NAME     a column of COLUMNS by line-by-line (tests/data/lbl/lbl_NAME.npz)
    python tools/other_air.py widemap      the longwave's g-points for other air (cache/ktables/map_wide.npz)
    python tools/other_air.py cia          collision-induced absorption per g-point (src/insilica/data/cia_ktables.npz)
    python tools/other_air.py wide         the longwave's wide tables, fitted (src/insilica/data/lw_ktables_wide.npz)
    python tools/other_air.py swwide       the shortwave's (src/insilica/data/sw_ktables_wide.npz)
    python tools/other_air.py swwidemap    the shortwave's g-points for other air (cache/ktables/swmap_wide.npz)
    python tools/other_air.py swref NAME   a column (COLUMNS, MAP_COLUMNS) by the shortwave's line-by-line (cache/ktables/sw_air_NAME.npz)
    python tools/other_air.py swcheck      the wide shortwave tables against them

The wide tables keep the weak lines that columns of up to about a bar of CO2 and 1% of CH4 need (lbl.UMAX_WIDE); Earth's
tables leave them out. The longwave's g-points are grouped again (make_ktables' k-means, the same g-points per band) on
Earth's three columns and MAP_COLUMNS in other air, so that CO2's bands too weak to matter on Earth (820-1390 cm-1) are
split by how they absorb at large amounts: on Earth's grouping a g-point's mean k mixed them, and 10% CO2 lost 10 W/m2 of
OLR in 980-1180 cm-1 (the shortwave's likewise, 5-25 W/m2 too much absorbed in CO2's near-infrared bands; it is grouped
and fitted again the same way, make_swtables'). Its k is the g-point's mean, then fitted as Earth's (make_ktables.fit: a factor per g-point smooth
in log p) on Earth's fitting columns and MAP_COLUMNS (their line-by-line in cache/ktables/, collision-induced absorption
added and not scaled); LONGWAVE's other-air COLUMNS stay unfitted. Collision-induced absorption (HITRAN's CIA files, cache/
hitran-cia/) is each g-point's mean binary coefficient over its wavenumbers (the longwave's plain, the shortwave's weighted
by the sunlight), tabulated in temperature on the tables' grids, times 1e40 (cm^5 / molecule^2).
"""
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import lbl
import make_ktables as mk
import make_swtables as ms

DATA = lbl.ROOT / 'src' / 'insilica' / 'data'
MASS = {'n2': 28.0134, 'o2': 31.9988, 'ar': 39.948, 'co2': 44.0095, 'ch4': 16.0425, 'h2': 2.01588, 'he': 4.002602,
        'ne': 20.1797}                                                  # g/mol, as constants.GASES
# the checking columns: an AFGL column's temperatures and vapor in another dry air (mole fractions), at its own 1013 hPa
COLUMNS = {'early_n2co2': ('us_standard', dict(n2=0.899, co2=0.1, ch4=1e-3)),     # early Earth: no O2, so no ozone
           'co2_half': ('tropical', dict(n2=0.5, co2=0.5))}
# the other-air columns the wide g-points are grouped on (not those LONGWAVE checks)
MAP_COLUMNS = {'map_early': ('subarctic_winter', dict(n2=0.98, co2=0.02, ch4=2e-3)), 'map_co2': ('us_standard', dict(n2=0.2, co2=0.8))}
PAIRS = list(lbl.CIA_FILES)
CIA_SCALE = 1e40


def m_air(air):
    return sum(x * MASS[g] for g, x in air.items()) / sum(air.values()) * 1e-3


def ref(name):
    if name in MAP_COLUMNS:
        base, air = MAP_COLUMNS[name]
        lbl.reference_air(name, base, air, m_air(air), o3=0.0, path=lbl.REF / f'lbl_{name}.npz')
    else:
        base, air = COLUMNS[name]
        lbl.reference_air(name, base, air, m_air(air), o3=0.0)


def air_column(base, air, o3=0.0):
    """An AFGL column's levels in another dry air: (P, vmr, m_air), as lbl.reference_air builds them."""
    import afgl
    P = afgl.profile(base)
    nl = len(P['p'])
    vmr = {'H2O': P['vmr']['H2O'], 'CO2': np.full(nl, air.get('co2', 0.0)), 'O3': P['vmr']['O3'] * o3,
           'N2O': np.zeros(nl), 'CH4': np.full(nl, air.get('ch4', 0.0))}
    return P, vmr, m_air(air)


def widemap():
    """make_ktables.build_map's grouping on Earth's columns and MAP_COLUMNS (their lines broadened by their air, with
    collision-induced absorption), into map_wide.npz."""
    import afgl
    lbl.set_wide()
    cols = []
    for c, x in mk.COLUMNS:
        P = afgl.profile(c, co2=x)
        cols.append((P, lbl.layers(P['p'], P['T'], P['vmr']), None, None))
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        cols.append((P, lbl.layers(P['p'], P['T'], vmr, m), air, np.diff(P['p']) / lbl.GRAV / m * lbl.NA / 1e4))
    out = {}
    for b in range(len(mk.NG)):
        t = time.time()
        nu0, d, n, nu = mk.band_grid(b)
        feats = []
        for P, L, air, ua in cols:
            lbl.AIR['co2'] = None if air is None else air.get('co2', 0.0)
            pm, Tm, x, u = L
            tau = lbl.optical_depth(pm, Tm, x, u, nu0, d, n, gases=[g for g in x if np.any(x[g] > 0)])
            if air is not None:
                tau += lbl.cia_tau(pm, Tm, ua, air, nu)
            feats.append(mk.cumulative_features(tau, P['p']))
            del tau
        lbl.AIR['co2'] = None
        lab = mk.kmeans(cp.concatenate(feats), mk.NG[b])
        out[f'lab{b}'] = lab.astype(cp.int8).get()
        frac = np.stack([cp.bincount(lab, weights=lbl.planck(nu, T), minlength=mk.NG[b]).get() for T in mk.T_PL])
        out[f'pl{b}'] = frac / frac.sum(axis=1, keepdims=True)
        out[f'w{b}'] = cp.bincount(lab, minlength=mk.NG[b]).get() / n
        print(f'band {mk.BANDS[b]}-{mk.BANDS[b + 1]}: g-point widths {np.round(out[f"w{b}"], 3)}, {time.time() - t:.0f}s', flush=True)
    np.savez(lbl.REF / 'map_wide.npz', **out)


def cia():
    m, sm = np.load(lbl.REF / 'map_wide.npz'), np.load(lbl.REF / 'swmap_wide.npz')
    lw = np.zeros((len(PAIRS), len(mk.T_TAB), sum(mk.NG)))
    for b in range(len(mk.NG)):
        t = time.time()
        _, _, _, nu = mk.band_grid(b)
        nu = cp.asnumpy(nu)
        lab = m[f'lab{b}'].astype(np.int64)
        cnt = np.bincount(lab, minlength=mk.NG[b])
        g0 = sum(mk.NG[:b])
        for c, pair in enumerate(PAIRS):
            for j, T in enumerate(mk.T_TAB):
                lw[c, j, g0:g0 + mk.NG[b]] = np.bincount(lab, weights=lbl.cia_k(pair, nu, T), minlength=mk.NG[b]) / cnt
        print(f'longwave band {mk.BANDS[b]}-{mk.BANDS[b + 1]}: {time.time() - t:.0f}s', flush=True)
    sw = np.zeros((len(PAIRS), len(ms.T_TAB), sum(ms.NG)))
    for b in range(len(ms.NG)):
        t = time.time()
        nu, w = ms.band_points(b)
        lab = sm[f'lab{b}'].astype(np.int64)
        W = np.maximum(np.bincount(lab, weights=w, minlength=ms.NG[b]), 1e-30)
        g0 = sum(ms.NG[:b])
        for c, pair in enumerate(PAIRS):
            for j, T in enumerate(ms.T_TAB):
                sw[c, j, g0:g0 + ms.NG[b]] = np.bincount(lab, weights=lbl.cia_k(pair, nu, T) * w, minlength=ms.NG[b]) / W
        print(f'shortwave band {b}: {time.time() - t:.0f}s', flush=True)
    path = DATA / 'cia_ktables.npz'
    np.savez_compressed(path, pairs=np.array(PAIRS), lw=(lw * CIA_SCALE).astype(np.float32), sw=(sw * CIA_SCALE).astype(np.float32),
                        T_lw=mk.T_TAB, T_sw=ms.T_TAB, scale=CIA_SCALE)
    for c, pair in enumerate(PAIRS):
        print(f'{pair}: longwave max {lw[c].max():.2e}, shortwave max {sw[c].max():.2e} cm^5/molecule^2')
    print(path, path.stat().st_size, 'bytes')


def _fit_factors(src, P):
    """Earth's tables' fitted factors (g x P) on the table pressures P."""
    A = np.load(lbl.REF / src)['A']
    return np.exp(A @ mk.node_weights(np.log(P)).T)


def _fit_columns(t):
    """MAP_COLUMNS as make_ktables.fit's further columns: their tables' optical depths in their air, the collision-induced
    absorption (from cia_ktables.npz's longwave, as the model adds it) apart."""
    from insilica.constants import air_radiation
    cz = np.load(DATA / 'cia_ktables.npz')
    cols = []
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        r = np.load(lbl.REF / f'lbl_{name}.npz')
        rad = air_radiation({k: v / sum(air.values()) for k, v in air.items()})
        tau, B, lp = mk.column_terms(t, P, vmr=vmr, m_air=m, broad=rad['broad'])
        p, T = P['p'], P['T']
        pm, Tm = (p[1:] - p[:-1]) / np.log(p[1:] / p[:-1]), (T[1:] + T[:-1]) / 2
        wc = np.diff(p) / lbl.GRAV / m * lbl.NA / 1e4 * 1e-20 * (pm / (1.380649e-23 * Tm) * 1e-6) * 1e-20
        C = np.tensordot(rad['cia'], cz['lw'].astype(float), axes=1)          # T x g
        y = np.clip(np.interp(Tm, mk.T_TAB, np.arange(len(mk.T_TAB))), 0, len(mk.T_TAB) - 1 - 1e-9)
        j = y.astype(int)
        add = wc[:, None] * (C[j] + (C[j + 1] - C[j]) * (y - j)[:, None])
        cols.append((tau, B, mk.node_weights(lp), p, r, np.where(p[1:] > 100.0, 1.0, 0.2), add))
    return cols


def wide():
    lbl.set_wide()
    mk.build_tables(out='lw_wide.npz', mapname='map_wide.npz')
    mk.fit('lw_wide.npz', extra=_fit_columns)
    mk.export('lw_wide.npz', DATA / 'lw_ktables_wide.npz')


def swwidemap():
    """make_swtables.build_map's grouping on Earth's three columns and MAP_COLUMNS (their air's lines, CIA, no O2), into
    swmap_wide.npz."""
    import swlbl
    lbl.set_wide()
    cols = [(ms.column(*c), None) for c in ms.COLUMNS[:3]]
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
        cols.append(((P, vmr, m), air))
    out = {}
    for b in range(len(ms.BANDS)):
        t = time.time()
        X, nu, w = [], None, None
        for P, air in cols:
            if air is None:
                tau, tr, th, nu, w = ms.column_tau(P, b)
                X.append(ms.features(tau, P))
            else:
                tau, tr, th, nu, w = sw_column_tau(P[0], P[1], P[2], air, b)
                X.append(ms.features(tau, P[0]))
            del tau, tr, th
        sr = swlbl.rayleigh(nu)
        X.append(cp.asarray(np.log10(sr / sr.max())[None, :] * 4, cp.float32))
        lab = mk.kmeans(cp.concatenate(X), ms.NG[b])
        labn = lab.get()
        ww = np.bincount(labn, weights=w, minlength=ms.NG[b])
        out[f'lab{b}'] = labn.astype(np.int8)
        out[f'sun{b}'] = ww
        out[f'ray{b}'] = np.bincount(labn, weights=w * sr, minlength=ms.NG[b]) / np.maximum(ww, 1e-30)
        out[f'haze{b}'] = (np.bincount(labn, weights=w * (1e7 / nu / 550.0) ** -swlbl.HAZE['angstrom'], minlength=ms.NG[b]) /
                           np.maximum(ww, 1e-30))
        print(f'band {b}: sunlight per g-point {np.round(ww, 1)} W/m2, {time.time() - t:.0f}s', flush=True)
    np.savez(lbl.REF / 'swmap_wide.npz', **out)


def _sw_fit_columns(t):
    """MAP_COLUMNS as make_swtables.fit's further columns, the collision-induced absorption apart."""
    cz = np.load(DATA / 'cia_ktables.npz')
    cols = []
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
        r = np.load(lbl.REF / f'sw_air_{name}.npz')
        t0 = dict(t)
        tau, tr, th = sw_terms(t0, P, vmr, m, air, cz)
        t0['cia_off'] = True
        tau_gas, _, _ = sw_terms(t0, P, vmr, m, air, None)
        pm = (P['p'][1:] - P['p'][:-1]) / np.log(P['p'][1:] / P['p'][:-1])
        cols.append((tau_gas, tr, th, mk.node_weights(np.log(pm * _sw_air(air)['broad'])), P['p'], r, tau - tau_gas))
    return cols


def swwide():
    lbl.set_wide()
    ms.build_tables(out='sw_wide.npz', mapname='swmap_wide.npz')
    ms.fit('sw_wide.npz', extra=_sw_fit_columns)
    ms.export('sw_wide.npz', 'sw_ktables_wide.npz')


# ---------------------------------------------------------------- the shortwave in other air

def _sw_air(air):
    from insilica.constants import air_radiation
    return air_radiation({k: v / sum(air.values()) for k, v in air.items()})


def sw_column_tau(P, vmr, m, air, b):
    """make_swtables.column_tau in another air: lines broadened by it, its O2 (vmr's), collision-induced absorption, Rayleigh
    by its molecules, its molar mass."""
    import swlbl
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], vmr, m)
    nu, w = ms.band_points(b)
    seg = ms.segments(b)
    tau = cp.zeros((len(pm), len(nu)), cp.float32)
    lbl.AIR['co2'] = air.get('co2', 0.0)
    for g in ms.GASES:
        if not np.any(x[g] > 0):
            continue
        for i in range(len(pm)):
            tau[i] += cp.float32(u[g][i]) * ms.gas_k(g, seg, Tm[i], pm[i], x['H2O'][i] if g == 'H2O' else 0.0)
    lbl.AIR['co2'] = None
    ua = np.diff(P['p']) / lbl.GRAV / m * lbl.NA / 1e4
    tau += lbl.cia_tau(pm, Tm, ua, air, nu)
    tr = cp.outer(cp.asarray(ua * _sw_air(air)['ray'], cp.float32), cp.asarray(swlbl.rayleigh(nu), cp.float32))
    share = np.array([swlbl.haze_tau(550.0e-7 ** -1, P['p'][i], P['p'][i + 1], P['p'][-1]) for i in range(len(pm))])
    th = cp.outer(cp.asarray(share, cp.float32), cp.asarray((1e7 / nu / 550.0) ** -swlbl.HAZE['angstrom'], cp.float32))
    return tau, tr, th, nu, w


def sw_ref(name):
    lbl.set_wide()
    base, air = COLUMNS[name] if name in COLUMNS else MAP_COLUMNS[name]
    P, vmr, m = air_column(base, air)
    vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
    out = {k: np.zeros((len(ms.MU0), len(ms.BANDS), len(P['p']))) for k in ('dir', 'dif', 'up')}
    for b in range(len(ms.BANDS)):
        t = time.time()
        tau, tr, th, nu, w = sw_column_tau(P, vmr, m, air, b)
        for j, mu0 in enumerate(ms.MU0):
            out['dir'][j, b], out['dif'][j, b], out['up'][j, b] = ms.fluxes(tau, tr, th, w, mu0, ms.ALB)
        print(f'{name} band {b}: at mu0 0.5 reflected {out["up"][1, b, 0]:.4f}, surface down '
              f'{out["dir"][1, b, -1] + out["dif"][1, b, -1]:.4f}, {time.time() - t:.0f}s', flush=True)
        del tau, tr, th
    np.savez(lbl.REF / f'sw_air_{name}.npz', p=P['p'], T=P['T'], mu0=np.array(ms.MU0), alb=ms.ALB, **out)


def sw_terms(t, P, vmr, m, air, cz):
    """make_swtables.column_terms in another air (the model's method): the tables read at the pressure times the air's
    broadening, Rayleigh by its molecules, collision-induced absorption from cia_ktables.npz's shortwave."""
    rad = _sw_air(air)
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], vmr, m)
    L = np.log(np.maximum(t['K'], 1e-40))
    nl, ng = len(pm), int(t['ng'].sum())
    tau = np.zeros((nl, ng))
    for i in range(nl):
        k = np.exp(ms.interp_logk(L, pm[i] * rad['broad'], Tm[i]))
        f = x['H2O'][i] / ms.X_H2O[1]
        tau[i] = u['H2O'][i] * (k[0] * (1 - f) + k[1] * f) + sum(u[g][i] * k[1 + c] for c, g in enumerate(ms.GASES) if c)
    ua = np.diff(P['p']) / lbl.GRAV / m * lbl.NA / 1e4
    if cz is not None:
        C = np.tensordot(rad['cia'], cz['sw'].astype(float), axes=1) / CIA_SCALE      # T x g
        y = np.clip(np.interp(Tm, ms.T_TAB, np.arange(len(ms.T_TAB))), 0, len(ms.T_TAB) - 1 - 1e-9)
        j = y.astype(int)
        n = pm / (1.380649e-23 * Tm) * 1e-6
        tau += (ua * n)[:, None] * (C[j] + (C[j + 1] - C[j]) * (y - j)[:, None])
    tr = np.outer(ua * rad['ray'], t['ray'])
    import swlbl
    e = 8000.0 / swlbl.HAZE['scale']
    share = (P['p'][1:] / P['p'][-1]) ** e - (P['p'][:-1] / P['p'][-1]) ** e
    th = swlbl.HAZE['tau550'] * np.outer(share, t['haze'])
    return tau, tr, th


def sw_check(src=None):
    """The wide shortwave tables against sw_ref's columns: reflected, surface down, the air's absorption (W/m2, S0 1361),
    heating (K/day per LONGWAVE's sub-levels below 1 hPa), per mu0."""
    t = dict(np.load(src or lbl.REF / 'sw_wide.npz'))
    cz = np.load(DATA / 'cia_ktables.npz')
    worst = dict(flux=0.0, heat=0.0)
    for name in COLUMNS:
        r = np.load(lbl.REF / f'sw_air_{name}.npz')
        base, air = COLUMNS[name]
        P, vmr, m = air_column(base, air)
        vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
        tau, tr, th = sw_terms(t, P, vmr, m, air, cz)
        for j, mu0 in enumerate(ms.MU0):
            Fd, Ff, Fu = (ms.band_sum(t, F) for F in ms.g_fluxes(tau, tr, th, mu0, ms.ALB))
            S = 1361.0 * mu0
            net, rnet = (Fd + Ff - Fu), (r['dir'][j] + r['dif'][j] - r['up'][j])
            up, upr = Fu.sum(0)[0] * S, r['up'][j].sum(0)[0] * S
            dn, dnr = (Fd + Ff).sum(0)[-1] * S, (r['dir'][j] + r['dif'][j]).sum(0)[-1] * S
            air_a, air_r = (net.sum(0)[0] - net.sum(0)[-1]) * S, (rnet.sum(0)[0] - rnet.sum(0)[-1]) * S
            sub = sorted({int(np.argmin(np.abs(P['p'] - q * 100))) for q in (0, 1, 5, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1e4)})
            es = np.abs(ms.heat(net.sum(0)[sub], P['p'][sub], mu0) - ms.heat(rnet.sum(0)[sub], P['p'][sub], mu0))
            worst['flux'] = max(worst['flux'], abs(up - upr), abs(dn - dnr), abs(air_a - air_r))
            worst['heat'] = max(worst['heat'], es[1:].max())
            bands = ' '.join(f'{(a - b_) * S:+.1f}' for a, b_ in zip((net[:, 0] - net[:, -1]), (rnet[:, 0] - rnet[:, -1])))
            print(f'{name} mu0 {mu0}: reflected {up:.2f}/{upr:.2f}, surface down {dn:.2f}/{dnr:.2f}, air absorbs {air_a:.2f}/{air_r:.2f} '
                  f'W/m2; heating {es[1:].max():.3f} K/day; air absorption by band {bands}')
    print('worst', worst)
    return worst


if __name__ == '__main__':
    if sys.argv[1] == 'swref':
        sw_ref(sys.argv[2])
    elif sys.argv[1] == 'swcheck':
        sw_check()
    elif sys.argv[1] == 'ref':
        ref(sys.argv[2])
    else:
        {'widemap': widemap, 'swwidemap': swwidemap, 'cia': cia, 'wide': wide, 'swwide': swwide}[sys.argv[1]]()
