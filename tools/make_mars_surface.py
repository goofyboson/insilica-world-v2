"""Reduce MGS TES's bolometric albedo (global_albedo_8ppd.img: 8 pixels per degree, 32-bit little-endian reals; Christensen et al.
2001, PDS) and Putzig & Mellon's nightside thermal inertia (global_ti_night_2007.img: 20 pixels per degree, 16-bit big-endian
integers, J m^-2 K^-1 s^-1/2; Putzig & Mellon 2007, PDS MGS-M-TES-5-TIMAP-V1.0), both lines from 90 N and samples from 180 W, to
1-degree tables for the `mars` preset (V2-7, SPEC #238): src/insilica/data/mars_surface_1deg.npz with `albedo` and `ti` on 180 x 360
cells from 90 S and 180 W, as mars_topo_1deg.npz. The poles beyond 87 degrees, where the albedo map holds its fill value 0.06 and the
thermal inertia none, take their nearest latitude ring with data. Run once, from the repository:
    python -I tools/make_mars_surface.py cache/mars-tes/global_albedo_8ppd.img cache/mars-tes/global_ti_night_2007.img
"""
import os
import sys

import numpy as np


def reduce(x, ppd, bad):
    """1-degree means from 90 S and 180 W of a map from 90 N and 180 W, its bad pixels left out; rows with none valid filled from the
    nearest row with data."""
    x = x.astype(float)[::-1]                                   # from 90 S
    ok = ~bad[::-1]
    b = x.reshape(180, ppd, 360, ppd)
    w = ok.reshape(180, ppd, 360, ppd)
    n = w.sum(axis=(1, 3))
    out = np.where(n > 0, (b * w).sum(axis=(1, 3)) / np.maximum(n, 1), np.nan)
    rows = np.where(np.isfinite(out).all(axis=1))[0]
    for r in range(180):
        if not np.isfinite(out[r]).all():
            src = rows[np.argmin(np.abs(rows - r))]
            out[r] = np.where(np.isfinite(out[r]), out[r], out[src])
    return out


def main(alb_path, ti_path, out='src/insilica/data/mars_surface_1deg.npz'):
    a = np.fromfile(alb_path, dtype='<f4')
    assert a.size == 1440 * 2880, a.size
    a = a.reshape(1440, 2880)
    lat8 = 90 - (np.arange(1440) + 0.5) / 8
    alb = reduce(a, 8, (np.abs(lat8) > 87)[:, None] & np.ones((1, 2880), bool) | ~np.isfinite(a))
    t = np.fromfile(ti_path, dtype='>i2')
    assert t.size == 3600 * 7200, t.size
    t = t.reshape(3600, 7200)
    ti = reduce(t, 20, t <= 0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, albedo=alb.astype(np.float32), ti=ti.astype(np.float32))
    w = np.cos(np.radians(np.arange(180) - 89.5))[:, None] * np.ones((1, 360))
    m = lambda x, s=slice(None): np.sum((w * x)[s]) / w[s].sum()
    print(f'albedo: mean {m(alb):.3f}, {alb.min():.3f}-{alb.max():.3f}; thermal inertia: mean {m(ti):.0f}, {ti.min():.0f}-{ti.max():.0f}, '
          f'30S-30N {m(ti, slice(60, 120)):.0f}, north of 80N {m(ti, slice(170, 180)):.0f}; {os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
