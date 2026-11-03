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
NG = [16, 8, 16, 16, 16, 8, 16, 16, 8, 16]          # g-points per band (the sweep: grouping errors against number)
DNU = lbl.DNU
COLUMNS = [('us_standard', 400e-6), ('tropical', 400e-6), ('subarctic_winter', 400e-6)]
P_TAB = np.exp(np.linspace(np.log(110000.0), np.log(1.0), 31))      # Pa, 1100 hPa to 0.01 hPa, top last
T_TAB = np.arange(150.0, 341.0, 10.0)
T_PL = np.arange(140.0, 351.0, 1.0)
X_H2O = (0.0, 0.04)                                  # H2O vmr of its two tables
P_FEAT = np.array([10.0, 100.0, 1000.0, 5000.0, 10000.0, 20000.0, 40000.0, 70000.0, 100000.0])  # Pa, from the top
P_FEAT_UP = np.array([90000.0, 75000.0, 55000.0, 35000.0])                                     # Pa, from the surface
GASES = ('H2O', 'CO2', 'O3', 'N2O', 'CH4')


def cumulative_features(tau, pl):
    """log10 of the optical depth from the top down to each of P_FEAT and from the surface up to each of P_FEAT_UP
    (interpolated in the levels' pressures pl, top first), clipped to [-4, 2] (transparent / opaque), per wavenumber."""
    cum = cp.concatenate([cp.zeros((1, tau.shape[1]), cp.float32), cp.cumsum(tau, axis=0)])
    lp = np.log(pl)
    rows = []
    for p in np.minimum(P_FEAT, pl[-1]):
        j = int(np.clip(np.searchsorted(lp, np.log(p)) - 1, 0, len(pl) - 2))
        w = (np.log(p) - lp[j]) / (lp[j + 1] - lp[j])
        rows.append(cp.log10(cp.maximum(cum[j] * (1 - w) + cum[j + 1] * w, 1e-4)).clip(-4, 2))
    for p in P_FEAT_UP:                               # from the surface up, which the surface's downward flux sees
        j = int(np.clip(np.searchsorted(lp, np.log(p)) - 1, 0, len(pl) - 2))
        w = (np.log(p) - lp[j]) / (lp[j + 1] - lp[j])
        rows.append(cp.log10(cp.maximum(cum[-1] - (cum[j] * (1 - w) + cum[j + 1] * w), 1e-4)).clip(-4, 2))
    return cp.stack(rows)


def kmeans(X, k, iters=60):
    """k-means over the columns of X (features x n), started from quantiles of the features' sum."""
    o = cp.argsort(X.sum(axis=0))
    lab = cp.empty(X.shape[1], cp.int32)
    lab[o] = (cp.arange(X.shape[1]) * k // X.shape[1]).astype(cp.int32)
    for _ in range(iters):
        cnt = cp.maximum(cp.bincount(lab, minlength=k), 1)
        C = (cp.stack([cp.bincount(lab, weights=X[f], minlength=k) for f in range(X.shape[0])]) / cnt).astype(cp.float32)
        new = cp.argmin(cp.stack([((X - C[:, c:c + 1]) ** 2).sum(axis=0) for c in range(k)]), axis=0).astype(cp.int32)
        if bool((new == lab).all()):
            break
        lab = new
    return lab


def rt(tau, B, Bs, mus=None, ws=None):
    """Up and down fluxes (levels) summed over spectral points or g-points: tau (layers x m), B (levels x m) and Bs (m)
    the Planck radiance times width (W m-2 sr-1) at the levels and the surface; cupy or numpy."""
    xp = cp.get_array_module(tau)
    mus, ws = (lbl.MU, lbl.WMU) if mus is None else (mus, ws)
    nl = tau.shape[0]
    up, dn = np.zeros(nl + 1), np.zeros(nl + 1)
    for mu, w in zip(mus, ws):
        I = xp.zeros(tau.shape[1])
        for i in range(nl):
            I = lbl.layer_through(I, tau[i], B[i + 1], B[i], mu) if xp is cp else through(I, tau[i], B[i + 1], B[i], mu)
            dn[i + 1] += 2 * np.pi * w * mu * float(I.sum(dtype=xp.float64))
        I = Bs
        up[nl] += 2 * np.pi * w * mu * float(I.sum(dtype=xp.float64))
        for i in range(nl - 1, -1, -1):
            I = lbl.layer_through(I, tau[i], B[i], B[i + 1], mu) if xp is cp else through(I, tau[i], B[i], B[i + 1], mu)
            up[i] += 2 * np.pi * w * mu * float(I.sum(dtype=xp.float64))
    return up, dn


def sweep(choices=(4, 8, 12, 16)):
    """The grouping's own error (cluster-mean optical depths of the reference columns themselves, no tables) for each
    number of g-points per band: OLR, surface down (W/m2) and the largest heating error (K/day) below 1 hPa."""
    cols = [afgl.profile(c, co2=x) for c, x in COLUMNS]
    lays = [lbl.layers(P['p'], P['T'], P['vmr']) for P in cols]
    for b in range(len(NG)):
        nu0, d, n, nu = band_grid(b)
        taus = [lbl.optical_depth(*L, nu0, d, n) for L in lays]
        X = cp.concatenate([cumulative_features(t, P['p']) for t, P in zip(taus, cols)])
        Bl = [[(lbl.planck(nu, float(T)) * d).astype(cp.float32) for T in P['T']] for P in cols]
        ref = [rt(t, B, B[-1]) for t, B in zip(taus, Bl)]
        for k in choices:
            lab = kmeans(X, k)
            cnt = cp.bincount(lab, minlength=k)
            row = []
            for t, B, P, (u0, d0) in zip(taus, Bl, cols, ref):
                tc = cp.stack([cp.bincount(lab, weights=t[i], minlength=k) / cnt for i in range(t.shape[0])])
                Bc = [cp.bincount(lab, weights=x, minlength=k) for x in B]
                u1, d1 = rt(tc, Bc, Bc[-1])
                hr, hk = lbl.heating(u0, d0, P['p']), lbl.heating(u1, d1, P['p'])
                ok = P['p'][1:] > 100.0
                row.append(f'{u1[0] - u0[0]:+6.2f} {d1[-1] - d0[-1]:+6.2f} {np.abs(hk - hr)[ok].max():5.2f}')
            print(f'band {BANDS[b]:4d}-{BANDS[b + 1]:4d} g {k:2d}:  ' + '   '.join(row), flush=True)
        del taus, X


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


DIFFUSIVITY = ([1 / 1.66], [1.66 / 2])               # the model's angle (flux = pi I at mu = 1/1.66)


def column_terms(t, P):
    """The column's layer optical depths per g-point from the tables (layers x g), Planck sources at the levels
    (levels x g, W m-2 sr-1), and its layers' log-pressures."""
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], P['vmr'])
    ng, nl = int(t['ng'].sum()), len(pm)
    tau = np.zeros((nl, ng))
    for i in range(nl):
        k = interp_k(t['K'], pm[i], Tm[i])           # comps x g
        f = x['H2O'][i] / X_H2O[1]
        tau[i] = u['H2O'][i] * (k[0] * (1 - f) + k[1] * f) + sum(u[g][i] * k[2 + c] for c, g in enumerate(GASES[1:]))
    gb = np.repeat(np.arange(len(t['ng'])), t['ng'])
    B = np.stack([band_planck(T)[gb] * np.array([np.interp(T, t['T_pl'], t['pl'][:, g]) for g in range(ng)])
                  for T in P['T']])
    return tau, B, np.log(pm)


def g_fluxes(tau, B, angles=None):
    """Up and down fluxes per g-point (g x level); the surface emits B[-1]."""
    nl, ng = tau.shape
    mus, ws = angles or DIFFUSIVITY
    up, dn = np.zeros((ng, nl + 1)), np.zeros((ng, nl + 1))
    for mu, w in zip(mus, ws):
        I = np.zeros(ng)
        for i in range(nl):
            I = through(I, tau[i], B[i + 1], B[i], mu)
            dn[:, i + 1] += 2 * np.pi * w * mu * I
        I = B[-1].copy()
        up[:, nl] += 2 * np.pi * w * mu * I
        for i in range(nl - 1, -1, -1):
            I = through(I, tau[i], B[i], B[i + 1], mu)
            up[:, i] += 2 * np.pi * w * mu * I
    return up, dn


def k_fluxes(t, P, angles=None):
    """Up and down fluxes per band (band x level) through the column P with the k-tables t."""
    tau, B, _ = column_terms(t, P)
    up, dn = g_fluxes(tau, B, angles)
    gb = np.repeat(np.arange(len(t['ng'])), t['ng'])
    sb = lambda F: np.stack([F[gb == b].sum(0) for b in range(len(t['ng']))])
    return sb(up), sb(dn)


# ---------------------------------------------------------------- fitting the tables to line-by-line

FIT_COLUMNS = COLUMNS + [('us_standard', 800e-6)]
CHECK_COLUMNS = [('tropical', 800e-6, {}), ('subarctic_winter', 400e-6, {'h2o': 2.0}),
                 ('us_standard', 400e-6, {'h2o': 2.0, 'dT': 10.0})]
NODES = np.log(np.array([1.0, 10.0, 50.0, 200.0, 1000.0, 3000.0, 10000.0, 20000.0, 40000.0, 60000.0, 80000.0, 110000.0]))
SIG_F, SIG_H, SIG_SMOOTH, SIG_A = 0.1, 0.02, 0.5, 1.5     # W/m2 per band; K/day; per node step; prior on log factor


def ref_name(name, co2, kw=None):
    return OUT / ('lbl_' + name + f'_{co2 * 1e6:.0f}' + ''.join(f'_{k}{v:g}' for k, v in (kw or {}).items()) + '.npz')


def node_weights(lp):
    """Linear interpolation weights in log p from NODES to the log-pressures lp (len(lp) x nodes), constant beyond."""
    W = np.zeros((len(lp), len(NODES)))
    x = np.clip(np.interp(lp, NODES, np.arange(len(NODES))), 0, len(NODES) - 1 - 1e-9)
    i = x.astype(int)
    W[np.arange(len(lp)), i] = 1 - (x - i)
    W[np.arange(len(lp)), i + 1] = x - i
    return W


def fit():
    """A factor on every gas's k per g-point, smooth in log pressure (linear between NODES), fitted by Levenberg-
    Marquardt so the diffusivity fluxes match line-by-line (three angles) on FIT_COLUMNS: each band's net flux at
    every level and heating in every layer (above 1 hPa at a fifth of the weight). Saves lw.npz's K fitted (K_raw kept)."""
    t = dict(np.load(OUT / 'lw.npz'))
    t['K'] = t.get('K_raw', t['K'])
    cols = []
    for name, co2 in FIT_COLUMNS:
        P = afgl.profile(name, co2=co2)
        r = np.load(ref_name(name, co2))
        tau, B, lp = column_terms(t, P)
        wH = np.where(P['p'][1:] > 100.0, 1.0, 0.2)
        cols.append((tau, B, node_weights(lp), P['p'], r, wH))
    nn = len(NODES)
    A = np.zeros((int(t['ng'].sum()), nn))
    for b in range(len(NG)):
        g = slice(sum(NG[:b]), sum(NG[:b + 1]))
        a = np.zeros((NG[b], nn))

        def resid(a, which=None):
            """Residuals (data then priors), and per node the change from perturbing all g-points' node."""
            out = []
            for tau, B, W, p, r, wH in cols:
                tg = tau[:, g] * np.exp(W @ a.T)
                up, dn = g_fluxes(tg, B[:, g])
                out.append((up, dn))
            return out

        def stack(fl, a):
            res = []
            for (up, dn), (tau, B, W, p, r, wH) in zip(fl, cols):
                net = up.sum(0) - dn.sum(0)
                rnet = r['up'][b] - r['dn'][b]
                res.append((net - rnet) / SIG_F)
                res.append((up.sum(0)[0] - r['up'][b][0]) / SIG_F * np.ones(1))
                res.append((dn.sum(0)[-1] - r['dn'][b][-1]) / SIG_F * np.ones(1))
                res.append(wH * (lbl.heating(up.sum(0), dn.sum(0), p) - lbl.heating(r['up'][b], r['dn'][b], p)) / SIG_H)
            res.append((np.diff(a, axis=1) / SIG_SMOOTH).ravel())
            res.append((a / SIG_A).ravel())
            return np.concatenate(res)

        def jac(a, fl0):
            """Each g-point's fluxes depend only on its own factors, so one perturbation of node j for all g-points at
            once gives that node's column for every g-point."""
            r0 = stack(fl0, a)
            J = np.zeros((len(r0), a.size))
            eps = 1e-4
            for j in range(nn):
                ap = a.copy()
                ap[:, j] += eps
                flp = resid(ap)
                for q in range(NG[b]):
                    fq = [(up0.copy(), dn0.copy()) for up0, dn0 in fl0]
                    for c in range(len(cols)):
                        fq[c][0][q] = flp[c][0][q]
                        fq[c][1][q] = flp[c][1][q]
                    aq = a.copy()
                    aq[q, j] += eps
                    J[:, q * nn + j] = (stack(fq, aq) - r0) / eps
            return r0, J

        mu = 1e-2
        fl = resid(a)
        r0 = stack(fl, a)
        cost0 = cost = float(r0 @ r0)
        for it in range(30):
            r0, J = jac(a, fl)
            JTJ, g_ = J.T @ J, J.T @ r0
            while True:
                step = -np.linalg.solve(JTJ + mu * np.diag(np.diag(JTJ) + 1e-12), g_)
                an = a + step.reshape(a.shape)
                fln = resid(an)
                c = float(stack(fln, an) @ stack(fln, an))
                if c < cost:
                    a, fl, mu = an, fln, mu / 3
                    break
                mu *= 4
                if mu > 1e8:
                    break
            if mu > 1e8 or cost - c < 1e-4 * cost:
                cost = min(cost, c)
                break
            cost = c
        A[g] = a
        print(f'band {BANDS[b]}-{BANDS[b + 1]}: cost {cost0:.1f} -> {cost:.1f} after {it + 1} steps, factors '
              f'{np.exp(a).min():.2f}-{np.exp(a).max():.2f}', flush=True)
    # apply: log k at table pressure P_TAB gets the node interpolation of A
    WT = node_weights(np.log(P_TAB))                  # P x nodes
    F = np.exp(A @ WT.T)                              # g x P
    t['K_raw'] = t['K']
    t['K'] = t['K'] * F[None, :, :, None]
    t['A'] = A
    np.savez(OUT / 'lw.npz', **t)


def through(I, tau, Bnear, Bfar, mu):
    x = tau / mu
    tr = np.exp(-x)
    lin = np.where(x > 1e-3, 1 - (1 - tr) / np.maximum(x, 1e-30), x / 2 - x * x / 6)
    return I * tr + Bfar * (1 - tr) + (Bnear - Bfar) * lin


def check():
    t = dict(np.load(OUT / 'lw.npz'))
    for name, co2, kw in [(n, c, {}) for n, c in FIT_COLUMNS] + CHECK_COLUMNS:
        r = np.load(ref_name(name, co2, kw))
        P = afgl.profile(name, co2=co2, **kw)
        up, dn = k_fluxes(t, P)
        name = name + ''.join(f' {k}={v:g}' for k, v in kw.items())
        hr, hk = lbl.heating(r['up'].sum(0), r['dn'].sum(0), P['p']), lbl.heating(up.sum(0), dn.sum(0), P['p'])
        trop = P['p'] > 10000.0
        # heating over model-like sub-levels (boundaries at the AFGL levels nearest these pressures)
        sub = sorted({int(np.argmin(np.abs(P['p'] - q * 100))) for q in (0, 1, 5, 20, 50, 100, 150, 200, 300, 400, 500,
                                                                           600, 700, 800, 900, 1e4)})
        hs = lambda U, D: lbl.heating(U.sum(0)[sub], D.sum(0)[sub], P['p'][sub])
        es = np.abs(hs(up, dn) - hs(r['up'], r['dn']))
        print(f'{name} {co2 * 1e6:.0f} ppm: OLR lbl {r["up"][:, 0].sum():.2f} k {up[:, 0].sum():.2f} '
              f'({(up[:, 0].sum() / r["up"][:, 0].sum() - 1) * 100:+.2f}%), surface down lbl {r["dn"][:, -1].sum():.2f} '
              f'k {dn[:, -1].sum():.2f} ({(dn[:, -1].sum() / r["dn"][:, -1].sum() - 1) * 100:+.2f}%), heating max |err| '
              f'{np.abs(hk - hr).max():.3f} K/day (below 100 hPa {np.abs(hk - hr)[trop[1:]].max():.3f}); over sub-levels max '
              f'{es.max():.3f} K/day, below 1 hPa {es[1:].max():.3f}: ' + ' '.join(f'{e:.2f}' for e in es))
        for b in range(len(t['ng'])):
            print(f'   band {BANDS[b]:4d}-{BANDS[b + 1]:4d}: OLR {r["up"][b, 0]:7.2f} {up[b, 0]:7.2f}  down '
                  f'{r["dn"][b, -1]:7.2f} {dn[b, -1]:7.2f}')


if __name__ == '__main__':
    {'map': build_map, 'tables': build_tables, 'fit': fit, 'check': check, 'sweep': sweep}[sys.argv[1]]()
