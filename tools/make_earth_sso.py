"""Sub-grid orography sums for the mountain drag (V2-6b, SPEC #127; Lott & Miller 1997, statistics as Baines & Palmer 1990):
reduce NOAA's ETOPO1 ice-surface grid (1 arc-minute; Amante & Eakins 2009) to src/insilica/data/earth_sso_1deg.npz with, on 180 x 360
cells from 90 S and 180 W, per 1-arc-minute point h = max(z, 0) (the sea at its surface): `n` (points), `s1` (sum of h, m), `s2`
(sum of h^2), and the sums of the slope products `gxx`, `gyy`, `gxy` (dh/dx east, dh/dy north, by centred differences). A world
adds them over its cells (cases.earth_sso). Run once, from the repository:
    python tools/make_earth_sso.py cache/etopo/ETOPO1_Ice_g_gmt4.grd.gz
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_earth_topo import read_classic_netcdf_var

R = 6.371e6


def main(path, out='src/insilica/data/earth_sso_1deg.npz'):
    sums = {k: np.zeros((180, 360)) for k in ('n', 's1', 's2', 'gxx', 'gyy', 'gxy')}
    dy = R * np.radians(1 / 60)
    with gzip.open(path, 'rb') as f:
        shape, dt, offset = read_classic_netcdf_var(f, 'z')
        f.seek(offset)
        assert shape == [10801, 21601], shape                    # rows from 90 S, columns from 180 W (the last repeats the first)
        rd = lambda: np.maximum(np.frombuffer(f.read(shape[1] * dt.itemsize), dt).astype(float)[:-1], 0.0)
        prev, cur = None, rd()
        for j in range(shape[0]):
            nxt = rd() if j + 1 < shape[0] else None
            lat = -90 + j / 60
            jj = min(int(j // 60), 179)
            box = lambda a: a.reshape(360, 60).sum(1)
            sums['n'][jj] += 60
            sums['s1'][jj] += box(cur)
            sums['s2'][jj] += box(cur * cur)
            if prev is not None and nxt is not None:             # (not on the poles' rows)
                dx = dy * np.cos(np.radians(lat))
                gx = (np.roll(cur, -1) - np.roll(cur, 1)) / (2 * dx)
                gy = (nxt - prev) / (2 * dy)
                sums['gxx'][jj] += box(gx * gx); sums['gyy'][jj] += box(gy * gy); sums['gxy'][jj] += box(gx * gy)
            prev, cur = cur, nxt
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: v.astype(np.float64) for k, v in sums.items()})
    n = sums['n']
    mu = np.sqrt(np.maximum(sums['s2'] / n - (sums['s1'] / n) ** 2, 0))
    print(f'within-1-degree standard deviation: largest {mu.max():.0f} m; mean slope^2 {np.mean((sums["gxx"] + sums["gyy"]) / n):.4f}; '
          f'{os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8', errors='replace')
    main(sys.argv[1])
