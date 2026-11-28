"""SST observed values (V2-7b, SPEC #311): the World Ocean Atlas 2023 annual objectively analysed temperature at the surface,
1991-2020 decades, 1-degree grid (NOAA NCEI, woa23_decav91C0_t00_01.nc; Locarnini et al. 2024), reduced to the n = 4 cells: per
cell the cos-lat weighted mean of the 1-degree values within 2.5 degrees of its centre (NaN where none: land, sea ice shelves),
at least the sea's freezing point (-1.8 C). The result is tests/data/woa23_sst_n4.npz (sst, degrees C, per cell), read by EARTH's
SST check and the observed-sea year in tests/test_radiation.py. Needs h5py (the `dev` extra).
Usage: python -I tools/woa_sst.py cache/woa23"""
import sys
from pathlib import Path

import h5py
import numpy as np


def main(d):
    from insilica import grid as gridmod
    g = gridmod.get(4)
    f = h5py.File(d + '/woa23_decav91C0_t00_01.nc', 'r')
    t = f['t_an'][0, 0].astype(float)
    lat, lon = f['lat'][:].astype(float), f['lon'][:].astype(float)
    assert t.shape == (180, 360), 'unexpected grid'
    t[(t > 100) | (t < -10)] = np.nan                            # (the file's fill value)
    LA, LO = np.meshgrid(np.radians(lat), np.radians(lon), indexing='ij')
    x = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
    ok = np.isfinite(t)
    X, T, W = x[ok], t[ok], np.cos(LA[ok])
    c = np.stack([np.cos(g.lat) * np.cos(g.lon), np.cos(g.lat) * np.sin(g.lon), np.sin(g.lat)], -1)
    out = np.full(g.N, np.nan)
    near = np.cos(np.radians(2.5))
    for i in range(g.N):
        m = X @ c[i] > near
        if m.any():
            out[i] = np.average(T[m], weights=W[m])
    out = np.where(np.isfinite(out), np.maximum(out, -1.8), np.nan)
    p = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'woa23_sst_n4.npz'
    np.savez(p, sst=out)
    print(f'{p}: {int(np.isfinite(out).sum())} of {g.N} cells, {np.nanmin(out):.2f} to {np.nanmax(out):.2f} C')


if __name__ == '__main__':
    main(sys.argv[1])
