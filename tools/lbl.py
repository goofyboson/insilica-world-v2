"""Line-by-line longwave radiation (V2-6a, SPEC #109): the reference the k-tables are made from and checked against.

Absorption: HITRAN 2020 lines (tools/fetch_lines.py; Voigt profiles by Humlicek's W4, the Van Vleck-Huber radiation term,
a 25 cm-1 cut-off) and the MT_CKD 4.3 water-vapor continuum (self and foreign), whose convention the H2O lines follow: each
line minus its value at 25 cm-1 (the "plinth", which the continuum holds). Computed on the GPU in 32-bit (the profile
sums), from 64-bit line data. Fluxes: plane-parallel, no scattering, a source linear in optical depth in each layer,
three Gauss angles per hemisphere.
"""
import math
import struct
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HITRAN, MTCKD = ROOT / 'cache' / 'hitran', ROOT / 'cache' / 'mtckd' / 'absco-ref_wv-mt-ckd.nc'
C2 = 1.4387769                      # second radiation constant, cm K
KB, AMU, CLIGHT = 1.380649e-23, 1.66053907e-27, 2.99792458e8
H_PL = 6.62607015e-34
NA, M_AIR, GRAV = 6.02214076e23, 28.9647e-3, 9.80665
TREF, PREF = 296.0, 101325.0
CUT = 25.0
MOL = {'H2O': 1, 'CO2': 2, 'O3': 3, 'N2O': 4, 'CH4': 6}
# column amounts (molecules/cm2) well above any planned case, for dropping lines too weak to matter (S*u < WMIN cm-1)
UMAX = {'H2O': 3e23, 'CO2': 8e22, 'O3': 2e19, 'N2O': 2e19, 'CH4': 1e20}
WMIN = 1e-6


# ---------------------------------------------------------------- data

def read_nc(path):
    """Classic (CDF-1/2) netCDF: {name: array}."""
    b = open(path, 'rb').read()
    assert b[:3] == b'CDF'
    v, pos = b[3], [4]
    types = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}

    def i4():
        pos[0] += 4
        return struct.unpack('>i', b[pos[0] - 4:pos[0]])[0]

    def off():
        if v == 1:
            return i4()
        pos[0] += 8
        return struct.unpack('>q', b[pos[0] - 8:pos[0]])[0]

    def name():
        n = i4()
        pos[0] += (n + 3) // 4 * 4
        return b[pos[0] - (n + 3) // 4 * 4:][:n].decode()

    def attrs():
        i4()
        for _ in range(i4()):
            name()
            t, n = i4(), i4()
            pos[0] += (n * types[t][1] + 3) // 4 * 4

    nrec = i4()
    i4()
    dims = [(name(), i4()) for _ in range(i4())]
    attrs()
    i4()
    out = {}
    for _ in range(i4()):
        k = name()
        ids = [i4() for _ in range(i4())]
        attrs()
        t, _, begin = i4(), i4(), off()
        shape = [dims[i][1] or nrec for i in ids]
        dt, sz = types[t]
        n = int(np.prod(shape)) if shape else 1
        out[k] = np.frombuffer(b[begin:begin + n * sz], dtype=dt).reshape(shape).astype(dt.replace('>', '<'))
    return out


def iso_number(c):
    return 10 if c == '0' else (int(c) if c.isdigit() else 11 + ord(c) - ord('A'))


_Q = {}


def partition(m, i, T):
    """TIPS-2021 partition sum (HAPI), tabulated 100-400 K and interpolated."""
    if (m, i) not in _Q:
        import hapi
        g = np.arange(100.0, 401.0, 1.0)
        _Q[(m, i)] = (g, np.array([hapi.partitionSum(m, i, float(t)) for t in g]))
    g, q = _Q[(m, i)]
    return np.interp(T, g, q)


def read_lines(gas, numin=10.0, numax=3000.0):
    """The gas's lines in [numin - CUT, numax + CUT] strong enough to matter: dict of 64-bit arrays."""
    import hapi
    raw = np.frombuffer(open(HITRAN / f'{gas}.data', 'rb').read(), dtype='S161')
    rec = raw.view('S1').reshape(-1, 161)
    col = lambda a, b: np.array(rec[:, a:b].view(f'S{b - a}').ravel())
    nu = col(3, 15).astype(float)
    keep = (nu > numin - CUT) & (nu < numax + CUT)
    rec = rec[keep]
    nu = nu[keep]
    iso = np.array([iso_number(c.decode()) for c in col(2, 3)])
    d = dict(nu=nu, S=col(15, 25).astype(float), gair=col(35, 40).astype(float), gself=col(40, 45).astype(float),
             E=col(45, 55).astype(float), nair=col(55, 59).astype(float), dair=col(59, 67).astype(float), iso=iso)
    m = MOL[gas]
    d['mass'] = np.array([hapi.molecularMass(m, int(i)) for i in iso]) * AMU
    # strength's largest value between 180 and 320 K
    smax = d['S'].copy()
    for T in (180.0, 320.0):
        smax = np.maximum(smax, strength(d, T, m))
    keep = smax * UMAX[gas] > WMIN
    return {k: v[keep] for k, v in d.items()} | {'mol': m, 'gas': gas, 'nall': len(nu)}


def strength(d, T, m=None):
    m = m or d['mol']
    q = np.ones_like(d['nu'])
    for i in np.unique(d['iso']):
        s = d['iso'] == i
        q[s] = partition(m, int(i), TREF) / partition(m, int(i), T)
    nu, E = d['nu'], d['E']
    return d['S'] * q * np.exp(-C2 * E / T) / np.exp(-C2 * E / TREF) * \
        (1 - np.exp(-C2 * nu / T)) / (1 - np.exp(-C2 * nu / TREF))


_CKD = None


def continuum(nu, T, p, vmr):
    """MT_CKD 4.3 H2O continuum, cm2 per H2O molecule, at wavenumbers nu (cupy), temperature T, pressure p (Pa) and
    H2O volume mixing ratio vmr."""
    global _CKD
    if _CKD is None:
        c = read_nc(MTCKD)
        _CKD = {k: cp.asarray(c[k]) for k in ('wavenumbers', 'self_absco_ref', 'for_absco_ref', 'self_texp')}
    w = _CKD['wavenumbers']
    rho = p / 101300.0 * 296.0 / T          # MT_CKD: ref_press 1013 mb, 296 K
    sf = cp.interp(nu, w, _CKD['self_absco_ref']) * cp.exp(cp.interp(nu, w, _CKD['self_texp']) * math.log(296.0 / T))
    ff = cp.interp(nu, w, _CKD['for_absco_ref'])
    rad = nu * cp.tanh(C2 * nu / (2 * T))
    return (sf * vmr + ff * (1 - vmr)) * rho * rad


# ---------------------------------------------------------------- the line kernel

_SRC = r'''
__device__ __forceinline__ float voigt_re(float x, float y)
{   // Re w(x + iy), Humlicek (1982) W4; region IV in 64-bit (cancellation)
    float s = fabsf(x) + y;
    if (s >= 15.f) {                                  // t = y - ix; w = t 0.5641896 / (0.5 + t^2)
        float tr = y, ti = -x;
        float ur = 0.5f + tr*tr - ti*ti, ui = 2.f*tr*ti;
        return 0.5641896f * (tr*ur + ti*ui) / (ur*ur + ui*ui);
    }
    if (s >= 5.5f) {
        float tr = y, ti = -x;
        float ur = tr*tr - ti*ti, ui = 2.f*tr*ti;
        float nr = 1.410474f + 0.5641896f*ur, ni = 0.5641896f*ui;      // t (1.410474 + u 0.5641896)
        float ar = tr*nr - ti*ni, ai = tr*ni + ti*nr;
        float br = 0.75f + ur*(3.f + ur) - ui*ui, bi = ui*(3.f + ur) + ur*ui;  // 0.75 + u (3 + u)
        return (ar*br + ai*bi) / (br*br + bi*bi);
    }
    if (y >= 0.195f*fabsf(x) - 0.176f) {
        const float a[5] = {16.4955f, 20.20933f, 11.96482f, 3.778987f, 0.5642236f};
        const float b[6] = {16.4955f, 38.82363f, 39.27121f, 21.69274f, 6.699398f, 1.f};
        float tr = y, ti = -x;
        float nr = a[4], ni = 0.f;
        for (int k = 3; k >= 0; --k) { float r = nr*tr - ni*ti + a[k]; ni = nr*ti + ni*tr; nr = r; }
        float dr = b[5], di = 0.f;
        for (int k = 4; k >= 0; --k) { float r = dr*tr - di*ti + b[k]; di = dr*ti + di*tr; dr = r; }
        return (nr*dr + ni*di) / (dr*dr + di*di);
    }
    double tr = y, ti = -x;
    double ur = tr*tr - ti*ti, ui = 2.0*tr*ti;
    const double a[7] = {36183.31, -3321.9905, 1540.787, -219.0313, 35.76683, -1.320522, 0.56419};
    const double b[8] = {32066.6, -24322.84, 9022.228, -2186.181, 364.2191, -61.57037, 1.841439, -1.0};
    // numerator a0 - u(a1' ...) written as a polynomial in u with alternating signs
    double nr = a[6], ni = 0.0;
    for (int k = 5; k >= 0; --k) { double r = nr*ur - ni*ui + a[k]; ni = nr*ui + ni*ur; nr = r; }
    double dr = b[7], di = 0.0;
    for (int k = 6; k >= 0; --k) { double r = dr*ur - di*ui + b[k]; di = dr*ui + di*ur; dr = r; }
    // t * num / den
    double qr = (nr*dr + ni*di) / (dr*dr + di*di), qi = (ni*dr - nr*di) / (dr*dr + di*di);
    double er = exp(ur) * cos(ui);
    return (float)(er - (tr*qr - ti*qi));
}

extern "C" __global__ void lines(int npt, double nu0, double dnu, const int* lo, const int* hi,
                                 const double* nuc, const float* S, const float* gl, const float* gd, const float* pl,
                                 float cut, double T, float* k)
{   // k(nu) at one state: sum over lines of S * R(nu)/R(nu_c) * (Voigt(nu - nu_c) - plinth), |nu - nu_c| <= cut.
    // S here is S(T)/R(nu_c), gd the Doppler HWHM / sqrt(ln 2) and gl the Lorentz HWHM.
    __shared__ float sx[256], ss[256], sl[256], sd[256], sp[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double nb = nu0 + (double)(blockIdx.x * blockDim.x) * dnu;
    float xi = (float)(threadIdx.x * dnu);
    float acc = 0.f;
    for (int j0 = lo[blockIdx.x]; j0 < hi[blockIdx.x]; j0 += 256) {
        int j = j0 + threadIdx.x;
        if (j < hi[blockIdx.x]) {
            sx[threadIdx.x] = (float)(nuc[j] - nb); ss[threadIdx.x] = S[j];
            sl[threadIdx.x] = gl[j]; sd[threadIdx.x] = gd[j]; sp[threadIdx.x] = pl[j];
        }
        __syncthreads();
        int n = min(256, hi[blockIdx.x] - j0);
        for (int m = 0; m < n; ++m) {
            float dx = xi - sx[m];
            if (fabsf(dx) <= cut) {
                float rd = 1.f / sd[m];
                acc += ss[m] * (0.5641896f * rd * voigt_re(dx * rd, sl[m] * rd) - sp[m]);
            }
        }
        __syncthreads();
    }
    if (i < npt) {
        double nu = nu0 + i * dnu;
        k[i] = acc * (float)(nu * tanh(1.4387769 * nu / (2.0 * T)));
    }
}
'''
_K = None


def grid(numin, numax, dnu):
    n = int(round((numax - numin) / dnu))
    return numin, dnu, n


def absorption(L, T, p, vmr, nu0, dnu, n, h2o_vmr=0.0):
    """k(nu) of the gas whose lines L are (cm2/molecule; cupy float32 of n points from nu0 by dnu) at temperature T (K),
    pressure p (Pa); h2o_vmr sets self-broadening for H2O (the gas's own vmr) and the continuum is added for H2O."""
    global _K
    if _K is None:
        _K = cp.RawModule(code=_SRC, options=('-std=c++11',)).get_function('lines')
    patm = p / PREF
    pself = patm * (vmr if L['gas'] == 'H2O' else 0.0)
    nuc = L['nu'] + L['dair'] * patm
    S = strength(L, T)
    gl = (TREF / T) ** L['nair'] * (L['gair'] * (patm - pself) + L['gself'] * pself)
    gd = L['nu'] / CLIGHT * np.sqrt(2 * KB * T / L['mass'])        # Doppler 1/e half width (HWHM / sqrt(ln2))
    R0 = L['nu'] * np.tanh(C2 * L['nu'] / (2 * T))
    pl = gl / math.pi / (CUT ** 2 + gl ** 2) if L['gas'] == 'H2O' else np.zeros_like(gl)
    o = np.argsort(nuc)
    nuc, S, gl, gd, pl, R0 = nuc[o], S[o], gl[o], gd[o], pl[o], R0[o]
    nb = (n + 255) // 256
    edges = nu0 + np.arange(nb + 1) * 256 * dnu
    lo = np.searchsorted(nuc, edges[:-1] - CUT - 1e-6).astype(np.int32)
    hi = np.searchsorted(nuc, edges[1:] + CUT + 1e-6).astype(np.int32)
    k = cp.empty(n, cp.float32)
    f32 = lambda a: cp.asarray(a, cp.float32)
    _K((nb,), (256,), (np.int32(n), np.float64(nu0), np.float64(dnu), cp.asarray(lo), cp.asarray(hi), cp.asarray(nuc),
                       f32(S / R0), f32(gl), f32(gd), f32(pl), np.float32(CUT), np.float64(T), k))
    if L['gas'] == 'H2O':
        nu = nu0 + dnu * cp.arange(n, dtype=cp.float64)
        k += continuum(nu, T, p, vmr).astype(cp.float32)
    return k


# ---------------------------------------------------------------- fluxes

C1 = 1.191042e-8                    # 2hc^2, W m-2 sr-1 (cm-1)^-4
MU, WMU = np.polynomial.legendre.leggauss(3)
MU, WMU = (MU + 1) / 2, WMU / 2     # Gauss points on [0, 1]


def planck(nu, T):
    """Radiance B(nu, T), W m-2 sr-1 (cm-1)-1."""
    return C1 * nu ** 3 / cp.expm1(C2 * nu / T)


def layer_through(I, tau, Bnear, Bfar, mu):
    """Radiance leaving a layer of optical depth tau along cosine mu, entering with I, source linear in tau from Bfar
    (entry side) to Bnear (exit side)."""
    x = tau / mu
    t = cp.exp(-x)
    lin = cp.where(x > 1e-3, 1 - (1 - t) / cp.maximum(x, 1e-30), x / 2 - x * x / 6)
    return I * t + Bfar * (1 - t) + (Bnear - Bfar) * lin


def fluxes(tau, Tlev, Ts, nu, dnu, angles=None, eps=1.0):
    """Up and down longwave fluxes (W/m2) at the levels, from layer optical depths tau (nlay x nnu, top first; cupy),
    level temperatures Tlev (nlay + 1, top first) and surface temperature Ts; angles: (mu, weights) on [0, 1] (default
    three Gauss points), or a single diffusivity (mu = 1/1.66, weight 1/2)."""
    mus, ws = angles or (MU, WMU)
    nlay = tau.shape[0]
    B = [planck(nu, float(T)) for T in Tlev]
    Bs = planck(nu, float(Ts))
    up, dn = np.zeros(nlay + 1), np.zeros(nlay + 1)
    for mu, w in zip(mus, ws):
        I = cp.zeros_like(nu)
        dn[0] += 0.0
        for i in range(nlay):
            I = layer_through(I, tau[i], B[i + 1], B[i], mu)
            dn[i + 1] += 2 * math.pi * w * mu * float(I.sum()) * dnu
        I = eps * Bs + (1 - eps) * I
        up[nlay] += 2 * math.pi * w * mu * float(I.sum()) * dnu
        for i in range(nlay - 1, -1, -1):
            I = layer_through(I, tau[i], B[i], B[i + 1], mu)
            up[i] += 2 * math.pi * w * mu * float(I.sum()) * dnu
    return up, dn


def layers(plev, Tlev, vmr):
    """Layers between levels (top first): log-mean pressure, mean temperature and vmr, and each gas's column amount
    (molecules/cm2). vmr: {gas: level vmr}."""
    plev, Tlev = np.asarray(plev, float), np.asarray(Tlev, float)
    pm = (plev[1:] - plev[:-1]) / np.log(plev[1:] / plev[:-1])
    Tm = (Tlev[1:] + Tlev[:-1]) / 2
    air = np.diff(plev) / GRAV / M_AIR * NA / 1e4
    x = {g: (np.asarray(v)[1:] + np.asarray(v)[:-1]) / 2 for g, v in vmr.items()}
    return pm, Tm, x, {g: v * air for g, v in x.items()}


_LINES = {}


def optical_depth(pm, Tm, x, u, nu0, dnu, n, gases=None):
    """Layer optical depths (nlay x n, cupy float32) of the gases in x (all of them by default) over the grid."""
    tau = cp.zeros((len(pm), n), cp.float32)
    for g in gases or list(x):
        if g not in _LINES:
            _LINES[g] = read_lines(g)
        for i in range(len(pm)):
            tau[i] += cp.float32(u[g][i]) * absorption(_LINES[g], Tm[i], pm[i], x[g][i] if g == 'H2O' else 0.0, nu0, dnu, n)
    return tau


def heating(up, dn, plev, cp_air=1004.64):
    """Layer heating, K/day, from level fluxes (top first) and level pressures (Pa)."""
    net = np.asarray(up) - np.asarray(dn)
    return (net[1:] - net[:-1]) / np.diff(plev) * GRAV / cp_air * 86400.0
