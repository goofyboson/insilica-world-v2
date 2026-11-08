"""WINDS observed values (SPEC #126): NCEP-NCAR Reanalysis 1 monthly long-term means 1991-2020 from NOAA PSL
(downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived: pressure/uwnd.mon.ltm.1991-2020.nc, pressure/air.mon.ltm.1991-2020.nc,
surface_gauss/uwnd.10m.mon.ltm.1991-2020.nc), reduced to earth4's layers and 10-degree bands; the results are WINDS_OBS in
tests/test_radiation.py. Needs h5py (the `dev` extra).
Layers by potential temperature: 0 = theta > 380 K (to the 10 hPa top, its level standing for 0-15 hPa),
1 = 305-380 K, 2 = theta < 305 K above 900 hPa (cloud layer), 3 = 10 m wind (mixed layer).
Mass-weighted (dp) per column, then zonal mean, then cos(lat)-weighted over each band.
Usage: python -I tools/ncep_winds.py cache/ncep-ltm"""
import sys, h5py, numpy as np

d = sys.argv[1]
U = h5py.File(d + '/uwnd.mon.ltm.1991-2020.nc', 'r')
T = h5py.File(d + '/air.mon.ltm.1991-2020.nc', 'r')
S = h5py.File(d + '/uwnd.10m.mon.ltm.1991-2020.nc', 'r')
p = U['level'][:].astype(float)                         # 1000 .. 10 hPa
lat = U['lat'][:].astype(float)
u = U['uwnd'][:].astype(float)                          # (12, 17, 73, 144)
th = (T['air'][:].astype(float) + 273.15) * (1000.0 / p[None, :, None, None]) ** 0.2857
assert np.all(np.abs(u) < 1e3) and np.all(np.isfinite(th))
edges = np.concatenate([[1000.0], 0.5 * (p[1:] + p[:-1]), [0.0]])
dp = (edges[:-1] - edges[1:])[None, :, None, None]
masks = [th > 380, (th >= 305) & (th <= 380), (th < 305) & (p[None, :, None, None] <= 900)]
lat10 = S['lat'][:].astype(float); u10 = S['uwnd'][:].astype(float)
assert np.all(np.abs(u10) < 1e3)

centres = np.arange(-85, 86, 10)
def band(zm, la):                                       # zm(..., nlat) -> (..., 18)
    out = []
    for c in centres:
        s = (la >= c - 5) & (la <= c + 5)
        w = np.cos(np.radians(la[s]))
        out.append((zm[..., s] * w).sum(-1) / w.sum())
    return np.array(out).T

def layer_u(months):
    rows = []
    for m in masks:
        w = dp * m[months]
        num = (u[months] * w).sum(1); den = w.sum(1)       # (nm, 73, 144)
        num = num.mean(-1).mean(0); den = den.mean(-1).mean(0)   # mass-weighted zonal & time mean
        rows.append((band(num, lat), band(den, lat)))
    rows.append((band(u10[months].mean(-1).mean(0), lat10), None))
    return rows

np.set_printoptions(precision=1, suppress=True, linewidth=200)
print('band centres', centres)
for name, mo in [('annual', list(range(12))), ('DJF', [11, 0, 1]), ('JJA', [5, 6, 7])]:
    for k, (num, den) in enumerate(layer_u(mo)):
        if den is None:
            print(f'{name} layer 3 (10 m) u', num)
        else:
            print(f'{name} layer {k} ({den.mean():.0f} hPa mean) u', num / np.maximum(den, 1e-9))
zm = u.mean(-1)                                          # (12, 17, 73)
for lev in (200, 50, 10):
    i = int(np.where(p == lev)[0][0])
    print(f'annual {lev} hPa u', band(zm[:, i].mean(0), lat))
ann200 = zm[:, int(np.where(p == 200)[0][0])].mean(0)
for h, s in (('S', lat < 0), ('N', lat > 0)):
    j = np.argmax(np.where(s, ann200, -1e9)); print(f'200 hPa jet {h}: {ann200[j]:.1f} m/s at {lat[j]:.1f}')
