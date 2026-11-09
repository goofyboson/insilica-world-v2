"""RAIN observed values (V2-6c, SPEC #134): GPCP v2.3 monthly long-term mean precipitation 1991-2020 from NOAA PSL
(downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.ltm.1991-2020.nc; Adler et al. 2018), 2.5-degree grid, reduced to the annual
mean: global (cos-lat weighted), 10-degree bands from 90 S (as WINDS'), and the latitude of the zonal-mean maximum in 5-degree
bands. The results are RAIN_OBS in tests/test_radiation.py. Needs h5py (the `dev` extra).
Usage: python -I tools/gpcp_rain.py cache/gpcp-ltm"""
import sys

import h5py
import numpy as np


def main(d):
    f = h5py.File(d + '/precip.mon.ltm.1991-2020.nc', 'r')
    p, lat = f['precip'][:].astype(float), f['lat'][:].astype(float)
    assert p.shape == (12, 72, 144) and np.all(p > -1) and np.all(p < 100), 'missing or out-of-range values'
    zm = p.mean(0).mean(1)                                  # annual, zonal mean (mm/day)
    w = np.cos(np.radians(lat))
    glob = np.sum(zm * w) / np.sum(w)
    band = lambda width: np.clip(((lat + 90) // width).astype(int), 0, 180 // width - 1)
    b10, b5 = band(10), band(5)
    z10 = np.array([np.sum((zm * w)[b10 == i]) / np.sum(w[b10 == i]) for i in range(18)])
    z5 = np.array([np.sum((zm * w)[b5 == i]) / np.sum(w[b5 == i]) for i in range(36)])
    peak = -90 + 5 * int(np.argmax(z5)) + 2.5
    print(f'global {glob:.2f} mm/day; 10-degree bands from 85 S: {np.round(z10, 2).tolist()}; 5-degree peak at {peak:+.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
