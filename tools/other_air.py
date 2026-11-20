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
    python tools/other_air.py mars STEP    (V2-7) STEP for Mars's own tables (widemap ... swcheck): grouped and fitted on Mars's
                                           columns alone, 100-340 K, written with the suffix _mars (lw_ktables_mars.npz, ...)

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


# (V2-7, Mars) Mars's air and columns: levels from 1 Pa to the surface, the temperature by height z = 10.8 km ln(ps / p) (K, km);
# `mars` a mean mid-latitude day's (LONGWAVE), `mars_polar` the winter pole's, cold near the ground and warmed aloft by the descending
# branch (LONGWAVE); `map_mars` a warm equatorial day's and `map_mars_cold` a polar night's colder still (the g-points' grouping and fit)
MARS_AIR = dict(co2=0.951, n2=0.0259, ar=0.0194, o2=0.0016)
MARS_GRAV = 3.721
MARS_T = {'mars': (610.0, lambda z: np.maximum(140.0, 215.0 - 2.2 * z)),
          'mars_polar': (700.0, lambda z: np.where(z < 20, 150.0 - 0.75 * z, np.where(z < 40, 135.0 + 1.5 * (z - 20), 165.0 - 1.25 * (z - 40)))),
          'map_mars': (500.0, lambda z: np.maximum(150.0, 245.0 - 3.0 * z)),
          'map_mars_cold': (800.0, lambda z: np.maximum(110.0, 145.0 - 0.6 * z)),
          'map_mars_inv': (650.0, lambda z: np.where(z < 15, 160.0 - z, np.where(z < 35, 145.0 + 1.5 * (z - 15), np.maximum(130.0, 175.0 - 1.5 * (z - 35)))))}
COLUMNS.update(mars=('mars', MARS_AIR), mars_polar=('mars_polar', MARS_AIR))                   # (V2-7)
MARS_MAP = dict(map_mars=('map_mars', MARS_AIR), map_mars_cold=('map_mars_cold', MARS_AIR), map_mars_inv=('map_mars_inv', MARS_AIR))
# the table set being made: 'wide' (#226: Earth's columns and MAP_COLUMNS), or 'mars' (V2-7: MARS_MAP alone, down to 100 K)
SET = dict(name='wide', suf='', base=True)
# (V2-7) band: (N, S): after the grouping, its N g-points with the worst top-layer heating on MAP_COLUMNS split S ways by the top
# layer's own optical depth (mk.NG counts them: the k-means makes NG - N (S - 1)); see top_split
TOP_SPLIT = {}


def set_mars():
    """(V2-7) Make Mars's own tables: grouped and fitted on MARS_MAP alone (one set's pressure-smooth fit cannot serve Earth's
    stratosphere and Mars's lower air, which share pressures 50-100 K and ~2000x the CO2 apart), 100-340 K; files with _mars."""
    set_cold()
    MAP_COLUMNS.clear()
    MAP_COLUMNS.update(MARS_MAP)
    SET.update(name='mars', suf='_mars', base=False)
    # the g-points where Mars's air absorbs: CO2's 15 um band (500-630, 630-700, 700-820 cm-1) at the cost of the bands it hardly
    # touches (138 in all, the kernel's most 144); the grouping's features at Mars's own pressures, 2-610 Pa (scaled by the column's surface pressure)
    mk.NG = [4, 4, 22, 38, 30, 8, 8, 8, 8, 8]
    # the top split in CO2's three bands (16, 32 and 24 grouped, 6 more each: 138 in all); see top_split
    TOP_SPLIT.update({2: (2, 4), 3: (2, 4), 4: (2, 4)})
    # the fit's factor linear in temperature as well: the grouping (on two cold columns and one warm) over-cools warm air in
    # CO2's band wings by a few per cent, where hot bands grow; local in temperature, unlike the top's error, so a factor mends it
    mk.T_FIT = (170.0, 50.0)
    mk.P_FEAT = np.array([2.0, 5.0, 15.0, 40.0, 80.0, 150.0, 250.0, 400.0, 610.0]) / 610.0 * 1.0e5


def checks():
    """The set's unfitted check columns (COLUMNS): Mars's for Mars's tables, the others for the wide ones."""
    return {k: v for k, v in COLUMNS.items() if (v[0] in MARS_T) == (SET['name'] == 'mars')}


def mars_levels(name, n=41):
    """A Mars column's levels as afgl.profile's (p, T, vmr, from the top), with its gravity."""
    ps, T = MARS_T[name]
    p = np.geomspace(1.0, ps, n)
    z = 10.8 * np.log(ps / p)
    zero = np.zeros(n)
    return dict(p=p, T=np.asarray(T(z), float), z=z, grav=MARS_GRAV,
                vmr={'H2O': zero, 'CO2': np.full(n, MARS_AIR['co2']), 'O3': zero, 'N2O': zero, 'CH4': zero})


def set_cold():
    """(V2-7) The wide tables down to 100 K: the longwave's temperatures 100-340 K (from 150), its Planck table 90-350 K (from 140),
    the shortwave's 90-330 K (from 150); lines judged down to 100 K."""
    lbl.set_cold()
    mk.T_TAB = np.arange(100.0, 341.0, 10.0)
    mk.T_PL = np.arange(90.0, 351.0, 1.0)
    ms.T_TAB = np.arange(90.0, 341.0, 20.0)


def use_grav(P):
    """Set lbl's gravity for column P (Mars's own; Earth's otherwise)."""
    lbl.GRAV = P.get('grav', 9.80665)


def m_air(air):
    return sum(x * MASS[g] for g, x in air.items()) / sum(air.values()) * 1e-3


def ref(name):
    if name in MARS_MAP or name in COLUMNS and COLUMNS[name][0] in MARS_T:
        set_mars()
    if name in MAP_COLUMNS:
        base, air = MAP_COLUMNS[name]
        lbl.reference_air(name, mars_levels(base) if base in MARS_T else base, air, m_air(air), o3=0.0, path=lbl.REF / f'lbl_{name}.npz')
    else:
        base, air = COLUMNS[name]
        lbl.reference_air(name, mars_levels(base) if base in MARS_T else base, air, m_air(air), o3=0.0)


def air_column(base, air, o3=0.0):
    """An AFGL column's levels in another dry air: (P, vmr, m_air), as lbl.reference_air builds them."""
    import afgl
    P = mars_levels(base) if base in MARS_T else afgl.profile(base)
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
    for c, x in (mk.COLUMNS if SET['base'] else []):
        P = afgl.profile(c, co2=x)
        cols.append((P, lbl.layers(P['p'], P['T'], P['vmr']), None, None))
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        use_grav(P)
        cols.append((P, lbl.layers(P['p'], P['T'], vmr, m), air, np.diff(P['p']) / lbl.GRAV / m * lbl.NA / 1e4))
    lbl.GRAV = 9.80665
    out = {}
    for b in range(len(mk.NG)):
        t = time.time()
        nu0, d, n, nu = mk.band_grid(b)
        feats, taus = [], []
        for P, L, air, ua in cols:
            lbl.AIR['co2'] = None if air is None else air.get('co2', 0.0)
            pm, Tm, x, u = L
            tau = lbl.optical_depth(pm, Tm, x, u, nu0, d, n, gases=[g for g in x if np.any(x[g] > 0)])
            if air is not None:
                tau += lbl.cia_tau(pm, Tm, ua, air, nu)
            feats.append(mk.cumulative_features(tau, P['p']))
            taus.append(tau if b in TOP_SPLIT else None)
            del tau
        lbl.AIR['co2'] = None
        N, S = TOP_SPLIT.get(b, (0, 1))
        lab = mk.kmeans(cp.concatenate(feats), mk.NG[b] - N * (S - 1))
        if N:
            lab = top_split(lab, mk.NG[b] - N * (S - 1), N, S, [(P, L[0], tau) for (P, L, _, _), tau in zip(cols, taus)], nu, d)
        del taus
        out[f'lab{b}'] = lab.astype(cp.int8).get()
        frac = np.stack([cp.bincount(lab, weights=lbl.planck(nu, T), minlength=mk.NG[b]).get() for T in mk.T_PL])
        out[f'pl{b}'] = frac / frac.sum(axis=1, keepdims=True)
        out[f'w{b}'] = cp.bincount(lab, minlength=mk.NG[b]).get() / n
        print(f'band {mk.BANDS[b]}-{mk.BANDS[b + 1]}: g-point widths {np.round(out[f"w{b}"], 3)}, {time.time() - t:.0f}s', flush=True)
    np.savez(lbl.REF / f'map_wide{SET["suf"]}.npz', **out)


def _rt(tau, B, mu=1 / 1.66):
    """Up and down fluxes (levels x points, W/m2 per point) with the diffusivity angle, the source linear in optical depth."""
    def through(I, t, Bn, Bf):
        x = t / mu
        tr = cp.exp(-x)
        lin = cp.where(x > 1e-3, 1 - (1 - tr) / cp.maximum(x, 1e-30), x / 2 - x * x / 6)
        return I * tr + Bf * (1 - tr) + (Bn - Bf) * lin
    nl = tau.shape[0]
    up, dn = cp.zeros((nl + 1, tau.shape[1])), cp.zeros((nl + 1, tau.shape[1]))
    I = cp.zeros(tau.shape[1])
    for i in range(nl):
        I = through(I, tau[i], B[i + 1], B[i])
        dn[i + 1] = np.pi * I
    I = B[-1].copy()
    up[nl] = np.pi * I
    for i in range(nl - 1, -1, -1):
        I = through(I, tau[i], B[i], B[i + 1])
        up[i] = np.pi * I
    return up, dn


def top_split(lab, ng, N, S, cols, nu, d):
    """(V2-7) Split the N g-points of a band whose top two sub-layers' heating (Mars's 0.01-0.05 and 0.05-0.2 hPa scaled by the
    surface pressure) the grouping misjudges most on the map columns (cols: (P, layer pressures, optical depths)), each into S
    by the top layer's own log optical depth (near 7 Pa at 610 Pa, mean over the columns), appended as g-points ng, ng + 1, ...
    Where molecular speed, not collisions, sets the lines' widths (all Mars's air above ~400 Pa), a wing that is thick below
    is thin at the top: wavenumbers grouped together lower down part at the top, where the group's mean optical depth puts
    its opacity in the gaps between the line centres and lets the top absorb the warmer radiation from below there."""
    score = cp.zeros(ng)
    ktop = 0.0
    for P, pm, tau in cols:
        tau = tau.astype(cp.float64)
        p = P['p']
        sub = sorted({int(np.argmin(np.abs(p - q * p[-1] / 1.0e3))) for q in (0, 1, 5, 20)})
        B = cp.stack([lbl.planck(nu, T) * d for T in P['T']])
        cnt = cp.maximum(cp.bincount(lab, minlength=ng), 1)
        group = lambda F: cp.stack([cp.bincount(lab, weights=F[i], minlength=ng) for i in sub])
        U, D = _rt(tau, B)
        net = group(U) - group(D)
        Ug, Dg = _rt(cp.stack([cp.bincount(lab, weights=t, minlength=ng) / cnt for t in tau]), cp.stack([cp.bincount(lab, weights=Bi, minlength=ng) for Bi in B]))
        netg = Ug[sub] - Dg[sub]
        dp = cp.asarray(np.diff(p[sub]))[:, None]
        score += cp.abs((cp.diff(netg - net, axis=0) / dp)[1:3]).sum(0)
        ktop = ktop + cp.log(cp.maximum(tau[int(np.argmin(np.abs(pm - 7.0 * p[-1] / 610.0)))], 1e-30)) / len(cols)
    new = ng
    for q in cp.argsort(-score)[:N].get():
        sel = lab == int(q)
        edges = cp.quantile(ktop[sel], cp.linspace(0, 1, S + 1)[1:-1])
        part = cp.searchsorted(edges, ktop)
        for k in range(1, S):
            lab = cp.where(sel & (part == k), new, lab)
            new += 1
    return lab

def cia():
    m, sm = np.load(lbl.REF / f'map_wide{SET["suf"]}.npz'), np.load(lbl.REF / f'swmap_wide{SET["suf"]}.npz')
    lw = np.zeros((len(PAIRS), len(mk.T_TAB), sum(mk.NG)))
    for b in range(len(mk.NG)):
        t = time.time()
        _, _, _, nu = mk.band_grid(b)
        nu = cp.asnumpy(nu)
        lab = m[f'lab{b}'].astype(np.int64)
        cnt = np.maximum(np.bincount(lab, minlength=mk.NG[b]), 1)        # (V2-7: an empty g-point, 0)
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
    path = DATA / f'cia_ktables{SET["suf"]}.npz'
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
    cz = np.load(DATA / f'cia_ktables{SET["suf"]}.npz')
    cols = []
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        use_grav(P)
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
        cols.append((tau, B, mk.node_weights(lp), p, r, np.where(p[1:] > 100.0, 1.0, 0.2), add))   # (V2-7: on Mars too: scaled to its surface pressure, the fit came out worse)
        lbl.GRAV = 9.80665
    return cols


def wide():
    lbl.set_wide()
    sf = SET['suf']
    mk.build_tables(out=f'lw_wide{sf}.npz', mapname=f'map_wide{sf}.npz')
    mk.fit(f'lw_wide{sf}.npz', extra=_fit_columns, base=SET['base'])
    mk.export(f'lw_wide{sf}.npz', DATA / f'lw_ktables{"_wide" if not sf else sf}.npz')


def swwidemap():
    """make_swtables.build_map's grouping on Earth's three columns and MAP_COLUMNS (their air's lines, CIA, no O2), into
    swmap_wide.npz."""
    import swlbl
    lbl.set_wide()
    cols = [(ms.column(*c), None) for c in (ms.COLUMNS[:3] if SET['base'] else [])]
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
    np.savez(lbl.REF / f'swmap_wide{SET["suf"]}.npz', **out)


def _sw_fit_columns(t):
    """MAP_COLUMNS as make_swtables.fit's further columns, the collision-induced absorption apart."""
    cz = np.load(DATA / f'cia_ktables{SET["suf"]}.npz')
    cols = []
    for name, (base, air) in MAP_COLUMNS.items():
        P, vmr, m = air_column(base, air)
        vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
        use_grav(P)
        r = np.load(lbl.REF / f'sw_air_{name}.npz')
        t0 = dict(t)
        tau, tr, th = sw_terms(t0, P, vmr, m, air, cz)
        t0['cia_off'] = True
        tau_gas, _, _ = sw_terms(t0, P, vmr, m, air, None)
        pm = (P['p'][1:] - P['p'][:-1]) / np.log(P['p'][1:] / P['p'][:-1])
        cols.append((tau_gas, tr, th, mk.node_weights(np.log(pm * _sw_air(air)['broad'])), P['p'], r, tau - tau_gas))
        lbl.GRAV = 9.80665
    return cols


def swwide():
    lbl.set_wide()
    sf = SET['suf']
    ms.build_tables(out=f'sw_wide{sf}.npz', mapname=f'swmap_wide{sf}.npz')
    if SET['name'] == 'mars':
        # (V2-7) Mars's shortwave unfitted: on Mars's columns alone the fit ran away (factors e^-48 to e^37 within Mars's pressures)
        # and doubled the flux error (2.1 against 1.1 W/m2 unfitted)
        ms.export(f'sw_wide{sf}.npz', f'sw_ktables{sf}.npz')
        return
    ms.fit(f'sw_wide{sf}.npz', extra=_sw_fit_columns, base=SET['base'])
    ms.export(f'sw_wide{sf}.npz', f'sw_ktables{"_wide" if not sf else sf}.npz')


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
    if name in MARS_MAP or name in COLUMNS and COLUMNS[name][0] in MARS_T:
        set_mars()
    lbl.set_wide()
    base, air = COLUMNS[name] if name in COLUMNS else MAP_COLUMNS[name]   # (Mars's map columns are in MAP_COLUMNS after set_mars)
    P, vmr, m = air_column(base, air)
    vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
    use_grav(P)
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
    t = dict(np.load(src or lbl.REF / f'sw_wide{SET["suf"]}.npz'))
    cz = np.load(DATA / f'cia_ktables{SET["suf"]}.npz')
    worst = dict(flux=0.0, heat=0.0)
    for name, (base, air) in checks().items():
        r = np.load(lbl.REF / f'sw_air_{name}.npz')
        P, vmr, m = air_column(base, air)
        vmr['O2'] = np.full(len(P['p']), air.get('o2', 0.0))
        use_grav(P)
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
    if sys.argv[1] == 'mars':
        set_mars()
        del sys.argv[1]
    if sys.argv[1] == 'swref':
        sw_ref(sys.argv[2])
    elif sys.argv[1] == 'swcheck':
        sw_check()
    elif sys.argv[1] == 'ref':
        ref(sys.argv[2])
    else:
        {'widemap': widemap, 'swwidemap': swwidemap, 'cia': cia, 'wide': wide, 'swwide': swwide}[sys.argv[1]]()
