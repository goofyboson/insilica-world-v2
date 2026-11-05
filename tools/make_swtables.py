"""Shortwave k-tables from line-by-line spectra (V2-6a, SPEC #109; tools/swlbl.py), as the longwave's (make_ktables.py):
each band's spectral points grouped into g-points by k-means on the logarithm of the gases' optical depth from the top
down in three AFGL columns (and, where it varies, the Rayleigh cross section's), each gas's mean k per g-point (weighted
by the sunlight) tabulated in (p, T), the g-point's share of the sunlight, its mean Rayleigh cross section and haze
factor; then a factor per g-point smooth in log p fitted to line-by-line fluxes and heating.

    python tools/make_swtables.py ref NAME CO2PPM [h2o=..] [dT=..]   line-by-line references (cache/ktables/sw_*.npz)
    python tools/make_swtables.py map | tables | fit | check | export
"""
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afgl
import lbl
import swlbl
from make_ktables import kmeans, node_weights, NODES, P_TAB, X_H2O

OUT = lbl.REF
# bands (cm-1) as segments of (from, to, step); lines up to 20000 cm-1, beyond only O3 and Rayleigh
BANDS = [[(2000, 2600, 0.002)], [(2600, 4000, 0.002)], [(4000, 5250, 0.002)], [(5250, 7700, 0.002)],
         [(7700, 9350, 0.002)], [(9350, 12850, 0.002)], [(12850, 16000, 0.002)], [(16000, 20000, 0.002), (20000, 29000, 0.5)],
         [(29000, 38000, 0.5)], [(38000, 50000, 0.5)]]
NG = [8] * len(BANDS)
T_TAB = np.arange(150.0, 341.0, 20.0)
MU0 = (0.25, 0.5, 0.87)
ALB = 0.15
GASES = ('H2O', 'CO2', 'O3', 'N2O', 'CH4', 'O2')
VMR_O2 = 0.2095
COLUMNS = [('us_standard', 400e-6, {}), ('tropical', 400e-6, {}), ('subarctic_winter', 400e-6, {}),
           ('us_standard', 800e-6, {})]
CHECK = [('tropical', 800e-6, {}), ('subarctic_winter', 400e-6, {'h2o': 2.0}), ('us_standard', 400e-6, {'h2o': 2.0, 'dT': 10.0})]
P_FEAT = np.array([10.0, 100.0, 1000.0, 5000.0, 10000.0, 20000.0, 40000.0, 70000.0, 100000.0])
CHUNK = 400000
SUN = float(np.trapezoid(swlbl.solar(np.linspace(100, 50000, 500000)), np.linspace(100, 50000, 500000)))  # W/m2, the whole spectrum
_LINES = {}


def lines(g):
    if g not in _LINES:
        _LINES[g] = lbl.read_lines(g, 1975.0, 20025.0) if g != 'O3' else lbl.read_lines('O3', 1975.0, 3000.0)
    return _LINES[g]


def band_points(b):
    """The band's wavenumbers (cupy) and their sunlight (W/m2 per point), segment by segment."""
    nus, ws = [], []
    for lo, hi, d in BANDS[b]:
        n = int(round((hi - lo) / d))
        nu = lo + d * (np.arange(n) + 0.5)
        nus.append(nu)
        ws.append(swlbl.solar(nu) * d)
    return np.concatenate(nus), np.concatenate(ws)


def gas_k(g, nu, T, p, x):
    """k (cm2/molecule, cupy float32) of gas g at the band's wavenumbers nu (NumPy), segment by segment."""
    out = []
    i = 0
    for seg in nu:
        lo, d, n = seg
        k = cp.zeros(n, cp.float32)
        if g == 'O3':
            k += cp.asarray(swlbl.o3_xs(lo + d * (np.arange(n)), T), cp.float32)
        if lo < 20000 and not (g == 'O3' and lo > 3025):
            L = lines(g)
            if len(L['nu']):
                k += lbl.absorption(L, T, p, x, lo, d, n)
        out.append(k)
    return cp.concatenate(out)


def segments(b):
    return [(lo + d / 2, d, int(round((hi - lo) / d))) for lo, hi, d in BANDS[b]]


def column(name, co2, kw):
    P = afgl.profile(name, co2=co2, **kw)
    P['vmr']['O2'] = np.full(len(P['p']), VMR_O2)
    return P


def column_tau(P, b):
    """Per layer: the gases' optical depth (cupy, layers x points), Rayleigh's and the haze's."""
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], P['vmr'])
    nu, w = band_points(b)
    seg = segments(b)
    tau = cp.zeros((len(pm), len(nu)), cp.float32)
    for g in GASES:
        for i in range(len(pm)):
            tau[i] += cp.float32(u[g][i]) * gas_k(g, seg, Tm[i], pm[i], x['H2O'][i] if g == 'H2O' else 0.0)
    air = np.diff(P['p']) / lbl.GRAV / lbl.M_AIR * lbl.NA / 1e4
    tr = cp.outer(cp.asarray(air, cp.float32), cp.asarray(swlbl.rayleigh(nu), cp.float32))
    share = np.array([swlbl.haze_tau(550.0e-7 ** -1, P['p'][i], P['p'][i + 1], P['p'][-1]) for i in range(len(pm))])
    th = cp.outer(cp.asarray(share, cp.float32), cp.asarray((1e7 / nu / 550.0) ** -swlbl.HAZE['angstrom'], cp.float32))
    return tau, tr, th, nu, w


def fluxes(tau, tr, th, w, mu0, alb, xp=cp):
    """Band fluxes at the levels (down direct, down diffuse, up; W/m2 per W/m2 of the whole sun on the horizontal at the
    top), weighted by the points' sunlight w, in chunks of points."""
    nl, n = tau.shape
    Fd, Ff, Fu = np.zeros(nl + 1), np.zeros(nl + 1), np.zeros(nl + 1)
    S = SUN
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        tg, trr, tha = (xp.asarray(a[:, sl], dtype=xp.float64) for a in (tau, tr, th))
        ssa_a, g_a = swlbl.HAZE['ssa'], swlbl.HAZE['g']
        t = tg + trr + tha
        sc = trr + ssa_a * tha
        ssa = xp.where(t > 0, sc / xp.maximum(t, 1e-30), 0)
        g = xp.where(sc > 0, g_a * ssa_a * tha / xp.maximum(sc, 1e-30), 0)
        Fdir, Fdif, Fup = swlbl.adding(xp, *swlbl.layer_rt(xp, t, xp.minimum(ssa, 1 - 1e-9), g, mu0), alb)
        ww = xp.asarray(w[sl] / S)
        Fd += (Fdir @ ww).get() if xp is cp else Fdir @ ww
        Ff += (Fdif @ ww).get() if xp is cp else Fdif @ ww
        Fu += (Fup @ ww).get() if xp is cp else Fup @ ww
    return Fd, Ff, Fu


def ref_name(name, co2, kw):
    return OUT / ('sw_' + name + f'_{co2 * 1e6:.0f}' + ''.join(f'_{k}{v:g}' for k, v in kw.items()) + '.npz')


def reference(name, co2, kw):
    P = column(name, co2, kw)
    nb = len(BANDS)
    out = {k: np.zeros((len(MU0), nb, len(P['p']))) for k in ('dir', 'dif', 'up')}
    for b in range(nb):
        t = time.time()
        tau, tr, th, nu, w = column_tau(P, b)
        for m, mu0 in enumerate(MU0):
            out['dir'][m, b], out['dif'][m, b], out['up'][m, b] = fluxes(tau, tr, th, w, mu0, ALB)
        print(f'{name} band {b}: sunlight {w.sum():.1f} W/m2, at mu0 0.5 reflected {out["up"][1, b, 0]:.4f}, surface down '
              f'{out["dir"][1, b, -1] + out["dif"][1, b, -1]:.4f}, {time.time() - t:.0f}s', flush=True)
        del tau, tr, th
    np.savez(ref_name(name, co2, kw), p=P['p'], T=P['T'], mu0=np.array(MU0), alb=ALB, **out)
    tot = lambda k: out[k][1].sum(0)
    print(f'{name} {co2 * 1e6:.0f} {kw}: mu0 0.5 reflected {tot("up")[0]:.4f}, absorbed in the air '
          f'{1 - tot("up")[0] - (tot("dir")[-1] + tot("dif")[-1] - tot("up")[-1]):.4f}, at the surface '
          f'{tot("dir")[-1] + tot("dif")[-1] - tot("up")[-1]:.4f} of the sunlight')


def features(tau, P):
    """log10 of the gases' optical depth from the top to each of P_FEAT, clipped to [-4, 2]."""
    cum = cp.concatenate([cp.zeros((1, tau.shape[1]), cp.float32), cp.cumsum(tau, axis=0)])
    lp = np.log(P['p'])
    rows = []
    for p in np.minimum(P_FEAT, P['p'][-1]):
        j = int(np.clip(np.searchsorted(lp, np.log(p)) - 1, 0, len(lp) - 2))
        f = (np.log(p) - lp[j]) / (lp[j + 1] - lp[j])
        rows.append(cp.log10(cp.maximum(cum[j] * (1 - f) + cum[j + 1] * f, 1e-4)).clip(-4, 2))
    return cp.stack(rows)


def build_map():
    cols = [column(*c) for c in COLUMNS[:3]]
    out = {}
    for b in range(len(BANDS)):
        t = time.time()
        X, nu, w = [], None, None
        for P in cols:
            tau, tr, th, nu, w = column_tau(P, b)
            X.append(features(tau, P))
            del tau, tr, th
        sr = swlbl.rayleigh(nu)
        X.append(cp.asarray(np.log10(sr / sr.max())[None, :] * 4, cp.float32))     # Rayleigh, on a comparable scale
        lab = kmeans(cp.concatenate(X), NG[b])
        labn = lab.get()
        ww = np.bincount(labn, weights=w, minlength=NG[b])
        out[f'lab{b}'] = labn.astype(np.int8)
        out[f'sun{b}'] = ww
        out[f'ray{b}'] = np.bincount(labn, weights=w * sr, minlength=NG[b]) / np.maximum(ww, 1e-30)
        out[f'haze{b}'] = (np.bincount(labn, weights=w * (1e7 / nu / 550.0) ** -swlbl.HAZE['angstrom'], minlength=NG[b]) /
                           np.maximum(ww, 1e-30))
        print(f'band {b}: sunlight per g-point {np.round(ww, 1)} W/m2, {time.time() - t:.0f}s', flush=True)
    np.savez(OUT / 'swmap.npz', **out)


def has_lines(g, b):
    if g in ('H2O', 'O3'):
        return True
    nu = lines(g)['nu']
    return bool(((nu > BANDS[b][0][0] - 25) & (nu < min(BANDS[b][-1][1], 20000) + 25)).any())


def build_tables():
    m = np.load(OUT / 'swmap.npz')
    comps = [('H2O', X_H2O[0]), ('H2O', X_H2O[1])] + [(g, 0.0) for g in GASES[1:]]
    ng = sum(NG)
    K = np.zeros((len(comps), ng, len(P_TAB), len(T_TAB)))
    for b in range(len(BANDS)):
        t = time.time()
        nu, w = band_points(b)
        seg = segments(b)
        lab = cp.asarray(m[f'lab{b}']).astype(cp.int32)
        wc = cp.asarray(w, cp.float64)
        W = cp.maximum(cp.bincount(lab, weights=wc, minlength=NG[b]), 1e-30)
        g0 = sum(NG[:b])
        for c, (g, x) in enumerate(comps):
            if not has_lines(g, b):
                continue
            for i, p in enumerate(P_TAB):
                for j, T in enumerate(T_TAB):
                    k = gas_k(g, seg, T, p, x)
                    K[c, g0:g0 + NG[b], i, j] = (cp.bincount(lab, weights=k.astype(cp.float64) * wc, minlength=NG[b]) / W).get()
        print(f'band {b}: {time.time() - t:.0f}s', flush=True)
    np.savez(OUT / 'sw.npz', K=K, P=P_TAB, T=T_TAB, x_h2o=np.array(X_H2O), ng=np.array(NG),
             comps=np.array([f'{g}{x}' for g, x in comps]),
             sun=np.concatenate([m[f'sun{b}'] for b in range(len(BANDS))]) / SUN,
             ray=np.concatenate([m[f'ray{b}'] for b in range(len(BANDS))]),
             haze=np.concatenate([m[f'haze{b}'] for b in range(len(BANDS))]))


# ---------------------------------------------------------------- the model's method in NumPy, and the fit

def interp_logk(L, p, T):
    lp = -np.log(P_TAB)
    x = np.clip(np.interp(-np.log(p), lp, np.arange(len(lp))), 0, len(lp) - 1 - 1e-9)
    y = np.clip(np.interp(T, T_TAB, np.arange(len(T_TAB))), 0, len(T_TAB) - 1 - 1e-9)
    i, j = int(x), int(y)
    fx, fy = x - i, y - j
    return (L[..., i, j] * (1 - fx) * (1 - fy) + L[..., i + 1, j] * fx * (1 - fy) + L[..., i, j + 1] * (1 - fx) * fy +
            L[..., i + 1, j + 1] * fx * fy)


def column_terms(t, P):
    """Per layer and g-point: the gases' optical depth from the tables, Rayleigh's and the haze's."""
    pm, Tm, x, u = lbl.layers(P['p'], P['T'], P['vmr'])
    L = np.log(np.maximum(t['K'], 1e-40))
    nl, ng = len(pm), int(t['ng'].sum())
    tau = np.zeros((nl, ng))
    for i in range(nl):
        k = np.exp(interp_logk(L, pm[i], Tm[i]))
        f = x['H2O'][i] / X_H2O[1]
        tau[i] = u['H2O'][i] * (k[0] * (1 - f) + k[1] * f) + sum(u[g][i] * k[1 + c] for c, g in enumerate(GASES) if c)
    air = np.diff(P['p']) / lbl.GRAV / lbl.M_AIR * lbl.NA / 1e4
    tr = np.outer(air, t['ray'])
    e = 8000.0 / swlbl.HAZE['scale']
    share = (P['p'][1:] / P['p'][-1]) ** e - (P['p'][:-1] / P['p'][-1]) ** e
    th = swlbl.HAZE['tau550'] * np.outer(share, t['haze'])
    return tau, tr, th, np.log(pm)


def g_fluxes(tau, tr, th, mu0, alb):
    """Per g-point fluxes (levels x g): down direct, down diffuse, up, per unit of the g-point's sunlight."""
    ssa_a, g_a = swlbl.HAZE['ssa'], swlbl.HAZE['g']
    T = tau + tr + th
    sc = tr + ssa_a * th
    ssa = np.where(T > 0, sc / np.maximum(T, 1e-30), 0)
    g = np.where(sc > 0, g_a * ssa_a * th / np.maximum(sc, 1e-30), 0)
    return swlbl.adding(np, *swlbl.layer_rt(np, T, np.minimum(ssa, 1 - 1e-9), g, mu0), alb)


def band_sum(t, F):
    """Per band (band x levels) from per g-point fluxes (levels x g), weighted by the g-points' sunlight."""
    gb = np.repeat(np.arange(len(t['ng'])), t['ng'])
    return np.stack([(F[:, gb == b] * t['sun'][gb == b]).sum(1) for b in range(len(t['ng']))])


def heat(net, p, mu0):
    """Layer heating (K/day) from the net down flux (fraction of the sunlight on the horizontal), S0 1361 at mu0."""
    return -(net[1:] - net[:-1]) / np.diff(p) * lbl.GRAV / 1004.64 * 86400 * 1361.0 * mu0


SIG_F, SIG_H, SIG_SMOOTH, SIG_A = 0.1 / 1361.0, 0.02, 0.5, 1.5


def fit():
    """As the longwave's (make_ktables.fit): a factor on every gas's k per g-point, smooth in log p, fitted by
    Levenberg-Marquardt to line-by-line net fluxes and heating at three sun angles on the four columns (heating above
    1 hPa at a fifth of the weight, as the longwave's)."""
    t = dict(np.load(OUT / 'sw.npz'))
    t['K'] = t.get('K_raw', t['K'])
    cols = []
    for c in COLUMNS:
        P = column(*c)
        r = np.load(ref_name(*c))
        tau, tr, th, lp = column_terms(t, P)
        cols.append((tau, tr, th, node_weights(lp), P['p'], r))
    nn = len(NODES)
    A = np.zeros((int(t['ng'].sum()), nn))
    for b in range(len(NG)):
        g = slice(sum(NG[:b]), sum(NG[:b + 1]))
        sun = t['sun'][g]

        def run(a):
            out = []
            for tau, tr, th, W, p, r in cols:
                tg = tau[:, g] * np.exp(W @ a.T)
                out.append([g_fluxes(tg, tr[:, g], th[:, g], mu0, ALB) for mu0 in MU0])
            return out

        def stack(fl, a):
            res = []
            for fm, (tau, tr, th, W, p, r) in zip(fl, cols):
                for m, (Fd, Ff, Fu) in enumerate(fm):
                    net = ((Fd + Ff - Fu) * sun).sum(1)
                    rnet = r['dir'][m, b] + r['dif'][m, b] - r['up'][m, b]
                    res.append((net - rnet) * MU0[m] / SIG_F)
                    res.append(np.where(p[1:] > 100.0, 1.0, 0.2) * (heat(net, p, MU0[m]) - heat(rnet, p, MU0[m])) / SIG_H)
            res.append((np.diff(a, axis=1) / SIG_SMOOTH).ravel())
            res.append((a / SIG_A).ravel())
            return np.concatenate(res)

        def jac(a, fl0):
            """Each g-point's fluxes depend only on its own factors: one perturbation of a node for all g-points at
            once gives that node's column for every g-point."""
            r0 = stack(fl0, a)
            J = np.zeros((len(r0), a.size))
            eps = 1e-4
            for j in range(nn):
                ap = a.copy()
                ap[:, j] += eps
                flp = run(ap)
                for q in range(NG[b]):
                    fq = [[tuple(x.copy() for x in F) for F in fm] for fm in fl0]
                    for c in range(len(cols)):
                        for m in range(len(MU0)):
                            for z in range(3):
                                fq[c][m][z][:, q] = flp[c][m][z][:, q]
                    aq = a.copy()
                    aq[q, j] += eps
                    J[:, q * nn + j] = (stack(fq, aq) - r0) / eps
            return r0, J

        a = np.zeros((NG[b], nn))
        fl = run(a)
        r0 = stack(fl, a)
        cost0 = cost = float(r0 @ r0)
        lam = 1e-2
        for it in range(30):
            r0, J = jac(a, fl)
            JTJ, gr = J.T @ J, J.T @ r0
            while True:
                an = a - np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), gr).reshape(a.shape)
                fln = run(an)
                rn = stack(fln, an)
                c = float(rn @ rn)
                if c < cost:
                    a, fl, lam = an, fln, lam / 3
                    break
                lam *= 4
                if lam > 1e8:
                    break
            if lam > 1e8 or cost - c < 1e-4 * cost:
                cost = min(cost, c)
                break
            cost = c
        A[g] = a
        print(f'band {b}: cost {cost0:.1f} -> {cost:.1f} after {it + 1} steps, factors {np.exp(a).min():.2f}-{np.exp(a).max():.2f}',
              flush=True)
    WT = node_weights(np.log(P_TAB))
    t['K_raw'] = t['K']
    t['K'] = t['K'] * np.exp(A @ WT.T)[None, :, :, None]
    t['A'] = A
    np.savez(OUT / 'sw.npz', **t)


SUB_HPA = (0, 1, 5, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1e4)


def check():
    t = dict(np.load(OUT / 'sw.npz'))
    for c in COLUMNS + CHECK:
        try:
            r = np.load(ref_name(*c))
        except FileNotFoundError:
            continue
        P = column(*c)
        tau, tr, th, _ = column_terms(t, P)
        rows = []
        for m, mu0 in enumerate(MU0):
            Fd, Ff, Fu = (band_sum(t, F) for F in g_fluxes(tau, tr, th, mu0, ALB))
            S = 1361.0 * mu0
            up, upr = Fu.sum(0)[0] * S, r['up'][m].sum(0)[0] * S
            dn, dnr = (Fd + Ff).sum(0)[-1] * S, (r['dir'][m] + r['dif'][m]).sum(0)[-1] * S
            net, rnet = (Fd + Ff - Fu).sum(0), (r['dir'][m] + r['dif'][m] - r['up'][m]).sum(0)
            sub = sorted({int(np.argmin(np.abs(P['p'] - q * 100))) for q in SUB_HPA})
            es = np.abs(heat(net[sub], P['p'][sub], mu0) - heat(rnet[sub], P['p'][sub], mu0))
            air, airr = (net[0] - net[-1]) * S, (rnet[0] - rnet[-1]) * S
            rows.append(f'mu0 {mu0}: reflected {up:.2f}/{upr:.2f}, surface down {dn:.2f}/{dnr:.2f}, air absorbs {air:.2f}/{airr:.2f} '
                        f'W/m2; heating {es[1:].max():.3f} K/day below 1 hPa (top sub-level {es[0]:.2f})')
        print(c[0], f'{c[1] * 1e6:.0f}', c[2], '\n   ' + '\n   '.join(rows))


def export():
    """The fitted tables as the model reads them (src/insilica/data/sw_ktables.npz)."""
    t = dict(np.load(OUT / 'sw.npz'))
    path = lbl.ROOT / 'src' / 'insilica' / 'data' / 'sw_ktables.npz'
    np.savez_compressed(path, logk=np.log(np.maximum(t['K'], 1e-40)).transpose(0, 2, 3, 1).astype(np.float32),
                        P=P_TAB, T=T_TAB, sun=t['sun'], ray=t['ray'], haze=t['haze'], ng=t['ng'], comps=t['comps'],
                        haze_par=np.array([swlbl.HAZE[k] for k in ('tau550', 'ssa', 'g', 'scale')]))
    print(path, path.stat().st_size, 'bytes')


if __name__ == '__main__':
    if sys.argv[1] == 'ref':
        kw = {a.split('=')[0]: float(a.split('=')[1]) for a in sys.argv[4:]}
        reference(sys.argv[2], float(sys.argv[3]) * 1e-6, kw)
    else:
        {'map': build_map, 'tables': build_tables, 'fit': fit, 'check': check, 'export': export}[sys.argv[1]]()
