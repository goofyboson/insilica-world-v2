"""The AFGL 1986 reference atmospheres (Anderson et al., AFGL-TR-86-0110), from the joseki package's tables in
cache/afgl/ (z km, p hPa, T K, number density, then H2O, O3, N2O, CO, CH4 in ppmv)."""
import numpy as np

from lbl import ROOT

FILES = {'tropical': 'table_1a.csv', 'subarctic_winter': 'table_1e.csv', 'us_standard': 'table_1f.csv'}


def profile(name, ptop=1.0, co2=400e-6):
    """Levels from the top (pressure ptop Pa and above cut off) down to the surface: dict of p (Pa), T (K), z (km)
    and vmr {gas: levels}."""
    a = np.loadtxt(ROOT / 'cache' / 'afgl' / FILES[name], delimiter=',', skiprows=1)
    a = a[a[:, 1] * 100 >= ptop][::-1]
    vmr = {'H2O': a[:, 4] * 1e-6, 'CO2': np.full(len(a), co2), 'O3': a[:, 5] * 1e-6, 'N2O': a[:, 6] * 1e-6,
           'CH4': a[:, 8] * 1e-6}
    return dict(z=a[:, 0], p=a[:, 1] * 100, T=a[:, 2], vmr=vmr)
