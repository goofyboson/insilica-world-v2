"""Reduce NOAA's ETOPO1 ice-surface grid (ETOPO1_Ice_g_gmt4.grd.gz; 1 arc-minute, grid registered, public domain; Amante & Eakins
2009) to a 1-degree table for the `earth` preset: src/insilica/data/earth_topo_1deg.npz with, on 180 x 360 cells from 90 S and
180 W, `land` (the share of the cell above sea level) and `elev` (the mean height of that land, m), `depth` (the mean depth of its sea, m). Run once, from the repository:
    python tools/make_earth_topo.py cache/etopo/ETOPO1_Ice_g_gmt4.grd.gz
"""
import gzip
import os
import struct
import sys

import numpy as np


def read_classic_netcdf_var(f, want):
    """The header of a classic netCDF-3 file, and variable `want`'s (shape, numpy dtype, byte offset)."""
    TYPES = {1: ('b', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}
    magic = f.read(4)
    assert magic[:3] == b'CDF', magic
    u4 = lambda: struct.unpack('>I', f.read(4))[0]
    off = (lambda: struct.unpack('>Q', f.read(8))[0]) if magic[3] == 2 else u4
    def name():
        n = u4(); s = f.read(n).decode(); f.read(-n % 4)
        return s
    def attrs():
        u4(); c = u4()
        for _ in range(c):
            name(); t = u4(); n = u4(); b = n * TYPES[t][1]; f.read(b + (-b % 4))
    u4()                                                         # number of records
    u4(); dims = [(name(), u4()) for _ in range(u4())]
    attrs()
    u4()
    for _ in range(u4()):
        nm = name(); ids = [u4() for _ in range(u4())]; attrs(); t = u4(); u4(); o = off()
        if nm == want:
            return [dims[i][1] for i in ids], np.dtype(TYPES[t][0]), o
    raise KeyError(want)


def main(path, out='src/insilica/data/earth_topo_1deg.npz'):
    with gzip.open(path, 'rb') as f:
        shape, dt, offset = read_classic_netcdf_var(f, 'z')
        f.seek(offset)
        assert len(shape) == 2 and shape == [10801, 21601], shape     # rows from 90 S, columns from 180 W
        land, elev, depth, count = np.zeros((180, 360)), np.zeros((180, 360)), np.zeros((180, 360)), np.zeros((180, 360))
        for j in range(shape[0]):
            row = np.frombuffer(f.read(shape[1] * dt.itemsize), dt).astype(float)[:-1]
            jj = min(j // 60, 179)
            ii = np.arange(21600) // 60
            np.add.at(count[jj], ii, 1.0)
            np.add.at(land[jj], ii, (row > 0) * 1.0)
            np.add.at(elev[jj], ii, np.maximum(row, 0.0))
            np.add.at(depth[jj], ii, np.maximum(-row, 0.0))
    elev = np.where(land > 0, elev / np.maximum(land, 1.0), 0.0)
    depth = np.where(count > land, depth / np.maximum(count - land, 1.0), 0.0)
    land = land / count
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, land=land.astype(np.float32), elev=elev.astype(np.float32), depth=depth.astype(np.float32))
    cosw = np.cos(np.radians(np.arange(180) - 89.5))[:, None]
    print(f'land share {np.sum(land * cosw) / (360 * cosw.sum()):.4f} (Earth 0.29); mean land height {np.sum(elev * land * cosw) / np.sum(land * cosw):.0f} m; highest 1-degree mean {elev.max():.0f} m; {os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8', errors='replace')
    main(sys.argv[1])
