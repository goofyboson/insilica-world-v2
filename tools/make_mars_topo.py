"""Reduce the MGS MOLA topography (MEGDR megt90n000cb.img: 4 pixels per degree, 16-bit big-endian heights in m above the areoid,
lines from 90 N, samples from 0 E eastward; Smith et al. 2003, PDS MGS-M-MOLA-5-MEGDR-L3-V1.0) to a 1-degree table for the `mars`
preset (V2-7): src/insilica/data/mars_topo_1deg.npz with `elev` (the cell's mean height, m) and `sd` (the standard deviation of its
16 pixels, m) on 180 x 360 cells from 90 S and 180 W, as earth_topo_1deg.npz. Run once, from the repository:
    python -I tools/make_mars_topo.py cache/mars-mola/megt90n000cb.img
"""
import os
import sys

import numpy as np


def main(path, out='src/insilica/data/mars_topo_1deg.npz'):
    z = np.fromfile(path, dtype='>i2').astype(float)
    assert z.size == 720 * 1440, z.size
    z = z.reshape(720, 1440)[::-1]                              # from 90 S
    z = np.roll(z, 720, axis=1)                                 # from 180 W
    b = z.reshape(180, 4, 360, 4)
    elev, sd = b.mean(axis=(1, 3)), b.std(axis=(1, 3))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, elev=elev.astype(np.float32), sd=sd.astype(np.float32))
    w = np.cos(np.radians(np.arange(180) - 89.5))[:, None] * np.ones((1, 360))
    print(f'mean height {np.sum(w * elev) / w.sum():.0f} m; lowest {elev.min():.0f}, highest {elev.max():.0f} (1-degree cells); '
          f'north of 40 N {np.sum((w * elev)[130:]) / w[130:].sum():.0f}, south of 40 S {np.sum((w * elev)[:50]) / w[:50].sum():.0f}; '
          f'{os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    main(sys.argv[1])
