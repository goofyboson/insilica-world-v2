"""The sea floor's small-scale roughness for the internal-tide drag (V2-6f, decision #228; Jayne & St Laurent 2001), from NOAA's
ETOPO1 (the same file as make_earth_topo.py): src/insilica/data/earth_rough_1deg.npz with `h2` (m^2) on 180 x 360 cells from 90 S
and 180 W: the mean, over the cell's four 0.5-degree boxes, of the variance of the 1-arc-minute depths about a plane fitted in
the box (the hills of ~2-50 km that make internal tides), counting a box only where all its points are deeper than 1000 m (the
deep ocean; the shelves' tides are braked by bottom friction instead). Run once, from the repository:
    python -I tools/make_earth_rough.py cache/etopo/ETOPO1_Ice_g_gmt4.grd.gz
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_earth_topo import read_classic_netcdf_var

BOX = 30                                    # 1-arc-minute points per 0.5-degree box
DEEP = 1000.0                               # m


def main(path, out='src/insilica/data/earth_rough_1deg.npz'):
    yy, xx = np.meshgrid(np.arange(BOX) - (BOX - 1) / 2, np.arange(BOX) - (BOX - 1) / 2, indexing='ij')
    X = np.stack([yy.ravel(), xx.ravel(), np.ones(BOX * BOX)], 1)
    proj = np.eye(BOX * BOX) - X @ np.linalg.pinv(X)            # residual about the box's plane
    h2 = np.zeros((360, 720))
    with gzip.open(path, 'rb') as f:
        shape, dt, offset = read_classic_netcdf_var(f, 'z')
        f.seek(offset)
        assert shape == [10801, 21601], shape
        for jb in range(360):
            rows = np.frombuffer(f.read(BOX * shape[1] * dt.itemsize), dt).astype(float).reshape(BOX, shape[1])[:, :-1]
            blk = rows.reshape(BOX, 720, BOX).transpose(1, 0, 2).reshape(720, BOX * BOX)     # box x points
            r = blk @ proj.T
            var = np.mean(r * r, axis=1)
            h2[jb] = np.where(np.all(blk < -DEEP, axis=1), var, 0.0)
        # (the last grid row, 90 N, is left out: 10801 = 360 x 30 + 1)
    h2_1 = h2.reshape(180, 2, 360, 2).mean(axis=(1, 3))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, h2=h2_1.astype(np.float32))
    cosw = np.cos(np.radians(np.arange(180) - 89.5))[:, None]
    deep = h2_1 > 0
    print(f'deep-ocean boxes {np.sum(deep * cosw) / (360 * cosw.sum()):.3f} of the area; rms roughness there '
          f'{np.sqrt(np.sum(h2_1 * cosw) / np.sum(deep * cosw)):.0f} m, largest 1-degree rms {np.sqrt(h2_1.max()):.0f} m; '
          f'{os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    main(sys.argv[1])
