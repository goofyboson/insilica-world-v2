"""OBSERVED SEA's cloud effect (V2-7b, SPEC #311): CERES EBAF Ed2.8 top-of-atmosphere reflected sunlight, all-sky and clear-sky
(obs4MIPs rsut and rsutcs, monthly 2000-03 to 2014-04, 1-degree grid; NASA NCCS ESGF, esgf.nccs.nasa.gov/thredds/catalog/esgcet/
legacy/obs4MIPs.NASA-LaRC.CERES-EBAF.atmos.mon.v20160610.html; Loeb et al. 2009), the cloud effect on sunlight rsutcs - rsut over
the whole years 2001-2013, reduced to the n = 4 cells as tools/woa_sst.py does (the cos-lat weighted mean within 2.5 degrees of each
cell's centre). The result is tests/data/ceres_swcre_n4.npz (swcre, W/m^2, per cell), read by OBSERVED SEA in tests/test_radiation.py.
Usage: python -I tools/ceres_cre.py cache/ceres"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncclassic  # noqa: E402

FILE = '{}_CERES-EBAF_L3B_Ed2-8_200003-201404.nc'


def field(d, v):
    _, _, var = ncclassic.read(os.path.join(d, FILE.format(v)))
    x = var[v].data().astype(float)
    assert x.shape == (170, 180, 360), 'unexpected grid'
    x[np.abs(x) > 1e10] = np.nan
    return x[10:166].mean(0), var['lat'].data().astype(float), var['lon'].data().astype(float)   # (Jan 2001 to Dec 2013: 13 years)


def main(d):
    from insilica import grid as gridmod
    g = gridmod.get(4)
    up, lat, lon = field(d, 'rsut')
    clr, _, _ = field(d, 'rsutcs')
    cre = clr - up
    assert np.isfinite(cre).all() and -150 < cre.min() and cre.max() < 40, 'out-of-range values'   # (+27 over the Gulf's bright coast, 25.5 N 50.5 E)
    LA, LO = np.meshgrid(np.radians(lat), np.radians(lon), indexing='ij')
    X = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    C, W = cre.reshape(-1), np.cos(LA).reshape(-1)
    c = np.stack([np.cos(g.lat) * np.cos(g.lon), np.cos(g.lat) * np.sin(g.lon), np.sin(g.lat)], -1)
    near = np.cos(np.radians(2.5))
    out = np.array([np.average(C[m], weights=W[m]) for m in (X @ c[i] > near for i in range(g.N))])
    p = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'ceres_swcre_n4.npz'
    np.savez(p, swcre=out)
    print(f'{p}: global {np.average(cre, weights=np.cos(LA)):.1f} W/m2 (1-degree), cells {out.min():.1f} to {out.max():.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
