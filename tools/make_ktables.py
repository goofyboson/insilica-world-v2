"""Longwave k-tables from line-by-line spectra (V2-6a, SPEC #109; tools/lbl.py).

Each band's spectral points are grouped into g-points by how they absorb through reference columns (k-means on the
logarithm of the optical depth from the top down to a set of pressures, in each reference column), so a g-point holds
wavenumbers that cool from the same heights in every column. A g-point's absorption coefficient for each gas is the mean
of the gas's k over its wavenumbers, tabulated in pressure and temperature (H2O at two vapor fractions, between which
self-broadening and the self continuum are linear); a layer's optical depth is then linear in each gas's amount (CO2
doubling, vapor) and no overlap rule is needed. Each g-point's share of the band's Planck emission is tabulated in
temperature.

    python tools/make_ktables.py map       group the spectral points (cache/ktables/map.npz)
    python tools/make_ktables.py tables    the k-tables (cache/ktables/lw.npz)
    python tools/make_ktables.py check     k-tables against line-by-line on the reference columns
"""
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afgl
import lbl

OUT = lbl.REF
BANDS = lbl.BANDS
NG = [4, 3, 4, 4, 3, 2, 3, 3, 3, 3]                  # g-points per band
DNU = lbl.DNU
COLUMNS = [('us_standard', 400e-6), ('tropical', 400e-6), ('subarctic_winter', 400e-6)]
P_TAB = np.exp(np.linspace(np.log(110000.0), np.log(1.0), 31))      # Pa, 1100 hPa to 0.01 hPa, top last
T_TAB = np.arange(150.0, 341.0, 10.0)
T_PL = np.arange(140.0, 351.0, 1.0)
X_H2O = (0.0, 0.04)                                  # H2O vmr of its two tables
P_FEAT = np.array([10.0, 100.0, 1000.0, 5000.0, 10000.0, 20000.0, 40000.0, 70000.0, 100000.0])  # Pa
GASES = ('H2O', 'CO2', 'O3', 'N2O', 'CH4')


def cumulative_features(tau, pl):
    """log10 of the optical depth from the top down to each of P_FEAT (interpolated in the levels' pressures pl, top
    first), clipped to [-4, 2] (transparent / opaque), per wavenumber: (len(P_FEAT) x n)."""
    cum = cp.concatenate([cp.zeros((1, tau.shape[1]), cp.float32), cp.cumsum(tau, axis=0)])
    lp = np.log(pl)
    rows = []
    for p in np.minimum(P_FEAT, pl[-1]):
        j = int(np.clip(np.searchsorted(lp, np.log(p)) - 1, 0, len(pl) - 2))
        w = (np.log(p) - lp[j]) / (lp[j + 1] - lp[j])
        rows.append(cp.log10(cp.maximum(cum[j] * (1 - w) + cum[j + 1] * w, 1e-4)).clip(-4, 2))
    return cp.stack(rows)


def kmeans(X, k, iters=60):
    """k-means over the columns of X (features x n), started from quantiles of the features' sum."""
    o = cp.argsort(X.sum(axis=0))
    lab = cp.empty(X.shape[1], cp.int32)
    lab[o] = (cp.arange(X.shape[1]) * k // X.shape[1]).astype(cp.int32)
    for _ in range(iters):
        cnt = cp.maximum(cp.bincount(lab, minlength=k), 1)
        C = (cp.stack([cp.bincount(lab, weights=X[f], minlength=k) for f in range(X.shape[0])]) / cnt).astype(cp.float32)
        new = cp.argmin(((X[:, None, :] - C[:, :, None]) ** 2).sum(axis=0), axis=0).astype(cp.int32)
        if bool((new == lab).all()):
            break
        lab = new
    return lab


def band_grid(b):
    nu0, d, n = lbl.grid(BANDS[b], BANDS[b + 1], DNU)
    return nu0 + d / 2, d, n, nu0 + d * (cp.arange(n, dtype=cp.float64) + 0.5)


def build_map():
    cols = [afgl.profile(c, co2=x) for c, x in COLUMNS]
    lays = [lbl.layers(P['p'], P['T'], P['vmr']) for P in cols]
    out = {}
    for b in range(len(NG)):
        t = time.time()
        nu0, d, n, nu = band_grid(b)
        X = cp.concatenate([cumulative_features(lbl.optical_depth(*L, nu0, d, n), P['p']) for P, L in zip(cols, lays)])
        lab = kmeans(X, NG[b])
        out[f'lab{b}'] = lab.astype(cp.int8).get()
        frac = np.stack([cp.bincount(lab, weights=lbl.planck(nu, T), minlength=NG[b]).get() for T in T_PL])
        out[f'pl{b}'] = frac / frac.sum(axis=1, keepdims=True)
        out[f'w{b}'] = cp.bincount(lab, minlength=NG[b]).get() / n
        print(f'band {BANDS[b]}-{BANDS[b + 1]}: g-point widths {np.round(out[f"w{b}"], 3)}, {time.time() - t:.0f}s', flush=True)
    np.savez(OUT / 'map.npz', **out)


def build_tables():
    m = np.load(OUT / 'map.npz')
    ng = sum(NG)
    comps = [('H2O', X_H2O[0]), ('H2O', X_H2O[1])] + [(g, 0.0) for g in GASES[1:]]
    K = np.zeros((len(comps), ng, len(P_TAB), len(T_TAB)))
    lines = {g: lbl.read_lines(g) for g in GASES}
    for b in range(len(NG)):
        t = time.time()
        nu0, d, n, nu = band_grid(b)
        lab = cp.asarray(m[f'lab{b}']).astype(cp.int32)
        cnt = cp.bincount(lab, minlength=NG[b])
        g0 = sum(NG[:b])
        for c, (g, x) in enumerate(comps):
            for i, p in enumerate(P_TAB):
                for j, T in enumerate(T_TAB):
                    k = lbl.absorption(lines[g], T, p, x, nu0, d, n)
                    K[c, g0:g0 + NG[b], i, j] = (cp.bincount(lab, weights=k, minlength=NG[b]) / cnt).get()
        print(f'band {BANDS[b]}-{BANDS[b + 1]}: {time.time() - t:.0f}s', flush=True)
    np.savez(OUT / 'lw.npz', K=K, P=P_TAB, T=T_TAB, T_pl=T_PL, x_h2o=np.array(X_H2O), bands=np.array(BANDS),
             ng=np.array(NG), comps=np.array([f'{g}{x}' for g, x in comps]),
             pl=np.concatenate([m[f'pl{b}'] for b in range(len(NG))], axis=1),
             w=np.concatenate([m[f'w{b}'] for b in range(len(NG))]))


# ---------------------------------------------------------------- using the tables (the model's method, in NumPy)

def interp_k(tab, p, T):
    """Bilinear in (log p, T) of log k: tab (... x P x T) at one p, T."""
    lp = -np.log(P_TAB)                               # increasing
    x = np.clip(np.interp(-np.log(p), lp, np.arange(len(lp))), 0, len(lp) - 1 - 1e-9)
    y = np.clip(np.interp(T, T_TAB, np.arange(len(T_TAB))), 0, len(T_TAB) - 1 - 1e-9)
    i, j = int(x), int(y)
    fx, fy = x - i, y - j
    L = np.log(np.maximum(tab, 1e-40))
    v = (L[..., i, j] * (1 - fx) * (1 - fy) + L[..., i + 1, j] * fx * (1 - fy) + L[..., i, j + 1] * (1 - fx) * fy +
         L[..., i + 1, j + 1] * fx * fy)
    return np.exp(v)


def band_planck(T, bands=BANDS):
    """Band-integrated Planck radiance (W m-2 sr-1) per band at temperature T."""
    out = []
    for b in range(len(bands) - 1):
        nu = np.linspace(bands[b], bands[b + 1], 20001)
        B = lbl.C1 * nu ** 3 / np.expm1(lbl.C2 * nu / T)
        out.append(np.trapezoid(B, nu))
    return np.array(out)


def k_fluxes(t, P, angles=None):
    """Up and down fluxes per band (band x level) through the column P with the k-tables t."""
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], P['vmr'])
    ng, nl = int(t['ng'].sum()), len(pm)
    tau = np.zeros((nl, ng))
    for i in range(nl):
        k = interp_k(t['K'], pm[i], Tm[i])           # comps x g
        f = x['H2O'][i] / X_H2O[1]
        tau[i] = u['H2O'][i] * (k[0] * (1 - f) + k[1] * f) + sum(u[g][i] * k[2 + c] for c, g in enumerate(GASES[1:]))
    gb = np.repeat(np.arange(len(t['ng'])), t['ng'])
    def src(T):
        fr = np.array([np.interp(T, t['T_pl'], t['pl'][:, g]) for g in range(ng)])
        return band_planck(T)[gb] * fr
    B = [src(T) for T in P['T']]
    Bs = src(P['T'][-1])
    mus, ws = angles or (lbl.MU, lbl.WMU)
    up, dn = np.zeros((ng, nl + 1)), np.zeros((ng, nl + 1))
    for mu, w in zip(mus, ws):
        I = np.zeros(ng)
        for i in range(nl):
            I = through(I, tau[i], B[i + 1], B[i], mu)
            dn[:, i + 1] += 2 * np.pi * w * mu * I
        I = Bs.copy()
        up[:, nl] += 2 * np.pi * w * mu * I
        for i in range(nl - 1, -1, -1):
            I = through(I, tau[i], B[i], B[i + 1], mu)
            up[:, i] += 2 * np.pi * w * mu * I
    sb = lambda F: np.stack([F[gb == b].sum(0) for b in range(len(t['ng']))])
    return sb(up), sb(dn)


def through(I, tau, Bnear, Bfar, mu):
    x = tau / mu
    tr = np.exp(-x)
    lin = np.where(x > 1e-3, 1 - (1 - tr) / np.maximum(x, 1e-30), x / 2 - x * x / 6)
    return I * tr + Bfar * (1 - tr) + (Bnear - Bfar) * lin


def check():
    t = dict(np.load(OUT / 'lw.npz'))
    for name, co2 in COLUMNS + [('us_standard', 800e-6)]:
        r = np.load(OUT / f'lbl_{name}_{co2 * 1e6:.0f}.npz')
        P = afgl.profile(name, co2=co2)
        up, dn = k_fluxes(t, P)
        hr, hk = lbl.heating(r['up'].sum(0), r['dn'].sum(0), P['p']), lbl.heating(up.sum(0), dn.sum(0), P['p'])
        trop = P['p'] > 10000.0
        print(f'{name} {co2 * 1e6:.0f} ppm: OLR lbl {r["up"][:, 0].sum():.2f} k {up[:, 0].sum():.2f} '
              f'({(up[:, 0].sum() / r["up"][:, 0].sum() - 1) * 100:+.2f}%), surface down lbl {r["dn"][:, -1].sum():.2f} '
              f'k {dn[:, -1].sum():.2f} ({(dn[:, -1].sum() / r["dn"][:, -1].sum() - 1) * 100:+.2f}%), heating max |err| '
              f'{np.abs(hk - hr).max():.3f} K/day (below 100 hPa {np.abs(hk - hr)[trop[1:]].max():.3f})')
        for b in range(len(t['ng'])):
            print(f'   band {BANDS[b]:4d}-{BANDS[b + 1]:4d}: OLR {r["up"][b, 0]:7.2f} {up[b, 0]:7.2f}  down '
                  f'{r["dn"][b, -1]:7.2f} {dn[b, -1]:7.2f}')


if __name__ == '__main__':
    {'map': build_map, 'tables': build_tables, 'check': check}[sys.argv[1]]()
