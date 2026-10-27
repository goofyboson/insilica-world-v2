"""RCE (radiative–convective equilibrium of the two air layers, no weather) and HELD-SUAREZ (the standard dry
benchmark, two layers) for stage V2-2a."""
import math

import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.grid.trisk import cell_winds
from insilica.world import World


@pytest.mark.xdist_group('grid4')
def test_rce(record):
    """RCE Radiative–convective equilibrium (n = 4, annual-mean sun, no dynamics): steady, neither layer drains, dry surface, TOA balance"""
    w = World(4, 'rce')
    A = w.grid.areaCell / w.grid.areaCell.sum()
    year = int(round(365 * 86400 / w.dt))
    w.step(19 * year)
    ts19 = float(np.sum(A * w.Ts.get()))
    w.step(year)
    h, Ts = w.h.get(), w.Ts.get()
    ts20 = float(np.sum(A * Ts))
    share = [float(np.sum(A * h[k]) / np.sum(A * h.sum(0))) for k in range(2)]
    toa = float(np.sum(A * (w.physics.sw_abs.get() - w.physics.olr.get())))
    lat = w.grid.lat
    ok = abs(ts20 - ts19) < 0.1 and min(share) >= 0.20 and abs(ts20 - 295) <= 5 and abs(toa) <= 0.5
    record('RCE', f'after 20 years: mean surface {ts20:.2f} K (equator {Ts[np.abs(lat) < 0.2].mean():.1f}, poles {Ts[np.abs(lat) > 1.3].mean():.1f}; '
           f'last year {ts20 - ts19:+.3f} K); layers {share[0] * 100:.1f}% / {share[1] * 100:.1f}% of the mass (upper / lower); '
           f'TOA {toa:+.3f} W/m²; θ* {w.physics.p["theta_star"]:.0f} K',
           'steady (< 0.1 K/yr); each layer ≥ 20%; surface 295 ± 5 K (dry, #41); |TOA| ≤ 0.5 W/m²', ok)
    assert ok


@pytest.mark.xdist_group('grid5')
def test_held_suarez(record):
    """HELD-SUAREZ Held & Suarez (1994) in two layers (n = 5, 400 days): steady, upper jet, surface westerlies and tropical easterlies, calm poles"""
    w = World(5, 'held_suarez')
    g = w.grid
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    lat = np.degrees(g.lat)
    per = int(round(86400 / w.dt))
    w.step(200 * per)
    acc, ke = 0.0, []
    for _ in range(200):
        w.step(per)
        u = w.u.get()
        acc = acc + np.stack([np.sum(cell_winds(g, u[k]) * east, 1) for k in range(2)])
        ke.append(float(np.sum(u ** 2)))
    zm = acc / 200
    band = lambda k, lo, hi: float(np.mean(zm[k][(np.abs(lat) >= lo) & (np.abs(lat) < hi)]))
    jet = max(float(np.mean(zm[0][(lat >= a) & (lat < a + 5)])) for a in range(30, 50, 5))
    jet_s = max(float(np.mean(zm[0][(lat >= -a - 5) & (lat < -a)])) for a in range(30, 50, 5))
    west, east_trop = band(1, 40, 55), band(1, 0, 15)
    # polar calm (#44): the zonal-mean wind beyond 80° in each hemisphere and layer (a split error once spun a vortex there)
    polar = max(abs(float(np.mean(zm[k][s * lat > 80]))) for k in range(2) for s in (1, -1))
    steady = np.mean(ke[100:]) / np.mean(ke[:100])
    h = w.h.get()
    finite = bool(np.isfinite(w.u.get()).all())
    ok = (finite and abs(steady - 1) <= 0.2 and 12 <= min(jet, jet_s) and max(jet, jet_s) <= 25 and west > 0 and east_trop < 0
          and polar < 5)
    record('HELD-SUAREZ', f'days 201–400: upper-layer jet {jet:.1f} m/s (north) / {jet_s:.1f} (south) at 30–50°; lower layer 40–55° {west:+.2f} m/s, '
           f'0–15° {east_trop:+.2f} m/s; poles (> 80°) worst {polar:.2f} m/s; kinetic energy days 301–400 / 201–300 ×{steady:.3f}; lower layer {h[1].sum() / h.sum() * 100:.1f}% of the mass; '
           f'step {w.dt:.0f} s', 'steady (±20%); upper jet 12–25 m/s (#41); westerlies 40–55°; easterlies 0–15°; poles < 5 m/s (#44)', ok)
    assert ok
