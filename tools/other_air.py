"""Radiation for other air (V2-6e, SPEC #226): the wide k-tables, collision-induced absorption per g-point, and the
line-by-line columns in other air that LONGWAVE checks them on.

    python tools/other_air.py ref NAME     a column of COLUMNS by line-by-line (tests/data/lbl/lbl_NAME.npz)
    python tools/other_air.py widemap      the longwave's g-points for other air (cache/ktables/map_wide.npz)
    python tools/other_air.py cia          collision-induced absorption per g-point (src/insilica/data/cia_ktables.npz)
    python tools/other_air.py wide         the longwave's wide tables, fitted (src/insilica/data/lw_ktables_wide.npz)
    python tools/other_air.py swwide       the shortwave's (src/insilica/data/sw_ktables_wide.npz)

The wide tables keep the weak lines that columns of up to about a bar of CO2 and 1% of CH4 need (lbl.UMAX_WIDE); Earth's
tables leave them out. The longwave's g-points are grouped again (make_ktables' k-means, the same g-points per band) on
Earth's three columns and MAP_COLUMNS in other air, so that CO2's bands too weak to matter on Earth (820-1390 cm-1) are
split by how they absorb at large amounts: on Earth's grouping a g-point's mean k mixed them, and 10% CO2 lost 10 W/m2 of
OLR in 980-1180 cm-1. Its k is the g-point's mean, then fitted as Earth's (make_ktables.fit: a factor per g-point smooth
in log p) on Earth's fitting columns and MAP_COLUMNS (their line-by-line in cache/ktables/, collision-induced absorption
added and not scaled); LONGWAVE's other-air COLUMNS stay unfitted. The shortwave's wide tables keep Earth's grouping and fitted
factors (make_swtables.fit) on their own raw k (not checked against line-by-line in other air). Collision-induced absorption (HITRAN's CIA files, cache/
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
    m, sm = np.load(lbl.REF / 'map_wide.npz'), np.load(lbl.REF / 'swmap.npz')
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


def swwide():
    lbl.set_wide()
    ms.build_tables(out='sw_wide.npz')
    t = dict(np.load(lbl.REF / 'sw_wide.npz'))
    t['K_raw'] = t['K']
    t['K'] = t['K'] * _fit_factors('sw.npz', mk.P_TAB)[None, :, :, None]
    np.savez(lbl.REF / 'sw_wide.npz', **t)
    ms.export('sw_wide.npz', 'sw_ktables_wide.npz')


if __name__ == '__main__':
    if sys.argv[1] == 'ref':
        ref(sys.argv[2])
    else:
        {'widemap': widemap, 'cia': cia, 'wide': wide, 'swwide': swwide}[sys.argv[1]]()
