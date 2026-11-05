"""Line-by-line shortwave (V2-6a, SPEC #109): the reference the shortwave k-tables are made from and checked against.

Sunlight: the TSIS-1 hybrid solar reference spectrum (Coddington et al. 2021, 2023; 1 nm, 202-2730 nm, cache/solar),
beyond 2730 nm a 5778 K Planck tail matched there. Absorption: HITRAN lines of H2O (with MT_CKD), CO2, CH4, N2O, O2 as in
tools/lbl.py; O3 by the Serdyuchenko-Gorshelev cross sections (213-1100 nm, 193-293 K; cache/o3xs). Scattering: Rayleigh
(Bodhaine et al. 1999) and a background haze. Fluxes: delta-Eddington two-stream in each layer, combined by adding over
a surface of given albedo, the same method the model uses.
"""
import math

import cupy as cp
import numpy as np

import lbl

SOLAR = lbl.ROOT / 'cache' / 'solar' / 'hsrs_1nm.csv'
O3XS = lbl.ROOT / 'cache' / 'o3xs' / 'SerdyuchenkoGorshelevVersionJuly2013.dat'
HAZE = dict(tau550=0.12, angstrom=1.0, ssa=0.95, g=0.70, scale=1500.0)   # background aerosol (global mean AOD about 0.12)
_SOL = _O3 = None


def solar(nu):
    """Solar spectral irradiance at 1 AU, W m-2 (cm-1)-1, at wavenumbers nu (NumPy)."""
    global _SOL
    if _SOL is None:
        a = np.loadtxt(SOLAR, delimiter=',', skiprows=1)
        lam, E = a[:, 0], a[:, 1]                                   # nm, W m-2 nm-1
        n = 1e7 / lam[::-1]
        _SOL = (n, (E * lam ** 2 / 1e7)[::-1])
    n, E = _SOL
    nu = np.asarray(nu, float)
    out = np.interp(nu, n, E, left=0.0, right=0.0)
    lo = nu < n[0]                                                  # beyond 2730 nm: Planck 5778 K, matched at the edge
    if lo.any():
        B = lambda x: x ** 3 / np.expm1(lbl.C2 * x / 5778.0)
        out[lo] = E[0] * B(nu[lo]) / B(n[0])
    return out


def o3_xs(nu, T):
    """O3 absorption cross section (cm2/molecule) at wavenumbers nu (NumPy) and temperature T: linear in T between the
    measured 193-293 K (held beyond), the 213 nm value held to shorter wavelengths, zero beyond 1100 nm."""
    global _O3
    if _O3 is None:
        rows = []
        for s in open(O3XS, encoding='latin-1'):
            p = s.split()
            if len(p) == 12:
                try:
                    rows.append([float(x.replace(',', '.')) for x in p])
                except ValueError:
                    pass
        a = np.array(rows)
        _O3 = (1e7 / a[::-1, 0], np.maximum(a[::-1, 1:], 0.0), np.arange(293.0, 192.0, -10.0))
    n, X, Ts = _O3
    nu = np.asarray(nu, float)
    j = np.clip(np.interp(T, Ts[::-1], np.arange(len(Ts))[::-1]), 0, len(Ts) - 1)
    i0 = int(np.floor(j))
    i1 = min(i0 + 1, len(Ts) - 1)
    f = j - i0
    x = (1 - f) * X[:, i0] + f * X[:, i1]
    out = np.interp(nu, n, x, left=0.0, right=x[-1])
    return out


def rayleigh(nu):
    """Rayleigh scattering cross section of air (cm2/molecule), Bodhaine et al. (1999) eq. 29."""
    l2 = (1e4 / np.asarray(nu, float)) ** 2                         # um^2
    return 1e-28 * (1.0455996 - 341.29061 / l2 - 0.90230850 * l2) / (1 + 0.0027059889 / l2 - 85.968563 * l2)


def haze_tau(nu, p_top, p_bot, ps):
    """The haze's optical depth between pressures p_top and p_bot over a surface at ps: tau550 (lambda / 550 nm)^-a, in
    height as exp(-z / scale) with the air's 8 km scale height (so as (p / ps)^(8000 / scale))."""
    e = 8000.0 / HAZE['scale']
    share = (p_bot / ps) ** e - (p_top / ps) ** e
    return HAZE['tau550'] * (1e7 / np.asarray(nu, float) / 550.0) ** -HAZE['angstrom'] * share


# ---------------------------------------------------------------- two-stream (delta-Eddington) and adding

def layer_rt(xp, tau, ssa, g, mu0):
    """Delta-Eddington reflectance and transmittance of homogeneous layers (Meador & Weaver 1980 in the stable form of
    Hogan & Bozzo 2018): diffuse R, T; for the direct beam (per unit of its flux on the horizontal) the reflected and the
    diffusely transmitted shares, and the direct transmittance exp(-tau / mu0)."""
    f = g * g
    tau = (1 - ssa * f) * tau
    ssa = (1 - f) * ssa / (1 - ssa * f)
    g = (g - f) / (1 - f)
    g1 = (7 - ssa * (4 + 3 * g)) / 4
    g2 = -(1 - ssa * (4 - 3 * g)) / 4
    g3 = (2 - 3 * g * mu0) / 4
    g4 = 1 - g3
    a1 = g1 * g4 + g2 * g3
    a2 = g1 * g3 + g2 * g4
    k = xp.sqrt(xp.maximum((g1 - g2) * (g1 + g2), 1e-12))
    e = xp.exp(-k * tau)
    e2 = e * e
    k2e = 2 * k * e
    rf = 1 / (k + g1 + (k - g1) * e2)
    Rd = g2 * (1 - e2) * rf
    Td = k2e * rf
    Tdd = xp.exp(-tau / mu0)
    km = k * mu0
    om = 1 - km * km
    om = xp.where(xp.abs(om) < 1e-6, xp.where(om < 0, -1e-6, 1e-6), om)
    rf = ssa * rf / om
    Rs = rf * ((1 - km) * (a2 + k * g3) - (1 + km) * (a2 - k * g3) * e2 - k2e * (g3 - a2 * mu0) * Tdd)
    Ts = rf * (k2e * (g4 + a1 * mu0) - Tdd * ((1 + km) * (a1 + k * g4) - (1 - km) * (a1 - k * g4) * e2))
    Rs = xp.clip(Rs, 0, 1 - Tdd)
    Ts = xp.clip(Ts, 0, 1 - Tdd - Rs)
    return Rd, Td, Rs, Ts, Tdd


def adding(xp, Rd, Td, Rs, Ts, Tdd, alb):
    """Fluxes at the levels (top first) per unit of the direct beam's flux on the horizontal at the top, over a surface of
    albedo alb (for both beams): down direct, down diffuse, up. Layers along axis 0."""
    nl = Rd.shape[0]
    shp = (nl + 1,) + Rd.shape[1:]
    Ad, As = xp.zeros(shp), xp.zeros(shp)
    Ad[nl] = alb
    As[nl] = alb
    for i in range(nl - 1, -1, -1):
        den = 1 / (1 - Ad[i + 1] * Rd[i])
        Ad[i] = Rd[i] + Td[i] * Td[i] * Ad[i + 1] * den
        As[i] = Rs[i] + (Tdd[i] * As[i + 1] + Ts[i] * Ad[i + 1]) * Td[i] * den
    Fdir, Fdif, Fup = xp.zeros(shp), xp.zeros(shp), xp.zeros(shp)
    Fdir[0] = 1
    Fup[0] = As[0]
    for i in range(nl):
        den = 1 / (1 - Ad[i + 1] * Rd[i])
        Fdir[i + 1] = Fdir[i] * Tdd[i]
        Fdif[i + 1] = (Td[i] * Fdif[i] + Rd[i] * As[i + 1] * Fdir[i + 1] + Ts[i] * Fdir[i]) * den
        Fup[i + 1] = Ad[i + 1] * Fdif[i + 1] + As[i + 1] * Fdir[i + 1]
    return Fdir, Fdif, Fup
