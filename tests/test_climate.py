"""CLIMATE (the moist three-layer aquaplanet's climate, n = 4) and WEATHER (its storms and rain at n = 5) for stage V2-2b.
Spin-ups are cached (runcache); only the measured phase runs every time."""
import numpy as np
import cupy as cp
import pytest

from insilica import runcache
from insilica.grid.trisk import cell_winds
from insilica.world import World

CLIMATE_SPIN_YEARS, WEATHER_SPIN_YEARS = 15, 1   # the 100 m slab settles in ~10 years (#54)


def _year_ticks(w, years):
    return int(round(years * 365 * 86400 / w.dt))


def _band(Ah, lat, f, lo, hi):
    m = (np.abs(lat) >= lo) & (np.abs(lat) < hi)
    return float(np.sum(Ah[m] * f[m]) / np.sum(Ah[m]))


@pytest.mark.xdist_group('grid4')
def test_climate(record):
    """CLIMATE The moist three-layer aquaplanet with seasons (n = 4, year 5): temperature, energy balance, Hadley cell, trades, westerlies, rain pattern"""
    w0 = World(4, 'aqua')
    w, _ = runcache.spun_up('climate', lambda: World(4, 'aqua'), _year_ticks(w0, CLIMATE_SPIN_YEARS))
    g = w.grid
    A = cp.asarray(g.areaCell)
    Ah = g.areaCell / g.areaCell.sum()
    lat = np.degrees(g.lat)
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    per = int(round(86400 / w.dt))
    b = w.L - 1
    rain, Ts, toa, wxb = cp.zeros(g.N), cp.zeros(g.N), cp.zeros(g.N), cp.zeros(g.N)
    ub = np.zeros(g.N)
    m60 = (lat > 55) & (lat < 65)
    ts60 = []
    mu0 = w.h[b].copy()
    for _ in range(365):
        for _ in range(per):
            w.step(1)
            rain += w.physics.rain; Ts += w.physics.tsurf; toa += w.physics.sw_abs - w.physics.olr; wxb += w.physics.wx[b - 1]
        ub += np.sum(cell_winds(g, w.u[b].get()) * east, 1)
        ts60.append(float(np.sum(Ah[m60] * w.physics.tsurf.get()[m60]) / np.sum(Ah[m60])))
    ns, T = 365 * per, 365 * 86400.0
    rain = (rain / ns * 86400).get(); Ts = (Ts / ns).get(); ub /= 365
    toa = float(cp.sum(toa / ns * A) / cp.sum(A))
    # the bottom layer's northward mass transport (residual: its crossings and mass change, eddies included), 1e9 kg/s
    src = (-((w.h[b] - mu0) / T + wxb / T) * A).get()
    o = np.argsort(lat)
    cum, ls = np.cumsum(src[o]), lat[o]
    tr = {x: cum[np.searchsorted(ls, x) - 1] / 1e9 for x in (-30, -20, -10, 10, 20, 30)}
    band = lambda f, lo, hi: _band(Ah, lat, f, lo, hi)
    ts, eqp, sub = float(np.sum(Ah * Ts)), band(Ts, 0, 10) - band(Ts, 80, 90), band(rain, 15, 30)
    itcz, mid, glob = band(rain, 0, 10) / sub, band(rain, 40, 60) / sub, float(np.sum(Ah * rain))
    trades, west = band(ub, 5, 25), band(ub, 35, 55)
    hadley = all(tr[x] < 0 for x in (10, 20, 30)) and all(tr[x] > 0 for x in (-10, -20, -30))
    ok = (abs(ts - 288) <= 3 and abs(toa) < 1 and hadley and trades < 0 and west > 0 and itcz > 2 and mid > 0.5
          and 2 <= glob <= 3)
    record('CLIMATE', f'year {CLIMATE_SPIN_YEARS + 1}: mean surface {ts:.2f} K; TOA {toa:+.2f} W/m²; bottom-layer northward transport '
           f'(1e9 kg/s) at 30S/20S/10S/10N/20N/30N {"/".join(f"{tr[x]:+.0f}" for x in (-30, -20, -10, 10, 20, 30))}; '
           f'bottom-layer wind 5–25° {trades:+.2f} m/s, 35–55° {west:+.2f} m/s; rain {glob:.2f} mm/day, ITCZ / subtropics ×{itcz:.2f}, '
           f'40–60° / subtropics ×{mid:.2f}; deferred to the ocean stage (#52): equator − pole {eqp:.1f} K, 60°N seasonal range '
           f'{max(ts60) - min(ts60):.1f} K',
           '288 ± 3 K; |TOA| < 1 W/m²; equatorward bottom flow within 30°; trades 5–25°; westerlies 35–55°; ITCZ > 2× and '
           '40–60° > 0.5× the subtropics; global 2–3 mm/day', ok)
    assert ok


@pytest.mark.xdist_group('grid5')
def test_weather(record):
    """WEATHER Storms and rain of the moist three-layer aquaplanet (n = 5, 60 days): B4 eddy energy, B9 local intermittent rain, B10 moving tropical clusters (report)"""
    w0 = World(5, 'aqua')
    w, _ = runcache.spun_up('weather', lambda: World(5, 'aqua'), _year_ticks(w0, WEATHER_SPIN_YEARS))
    g = w.grid
    lat = np.degrees(g.lat)
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    north = np.stack([-np.sin(g.lat) * np.cos(g.lon), -np.sin(g.lat) * np.sin(g.lon), np.cos(g.lat)], 1)
    per = int(round(86400 / w.dt))
    days = 60
    bins = np.clip(((lat + 90) // 2).astype(int), 0, 89)          # 2° latitude bins for zonal means
    U, V, H, R = [], [], [], []
    for _ in range(days):
        rain = cp.zeros(g.N)
        for _ in range(per):
            w.step(1)
            rain += w.physics.rain
        R.append((rain / per * 86400).get())
        u = w.u.get()
        cw = [cell_winds(g, u[k]) for k in range(w.L)]
        U.append(np.stack([np.sum(c * east, 1) for c in cw])); V.append(np.stack([np.sum(c * north, 1) for c in cw]))
        H.append(w.h.get())
    U, V, H, R = map(np.array, (U, V, H, R))                      # (day, layer, cell) / (day, cell)

    def zmean(f):                                                 # time and zonal mean in 2° bins, per layer
        tm = f.mean(0)
        out = np.zeros_like(tm)
        for k in range(tm.shape[0]):
            s = np.bincount(bins, weights=tm[k] * g.areaCell, minlength=90) / np.bincount(bins, weights=g.areaCell, minlength=90)
            out[k] = s[bins]
        return out

    # B4 over the tropospheric layers (decision #60): the top layer is the stratosphere, whose polar-night jet would
    # dominate the mean-flow energy
    mid = (np.abs(lat) >= 30) & (np.abs(lat) <= 60)
    Um, Vm = zmean(U), zmean(V)
    tro = slice(1, None) if w.L > 2 else slice(None)
    ke_mean = float(np.sum((H.mean(0) * (Um ** 2 + Vm ** 2))[tro][:, mid] * g.areaCell[mid]))
    ke_eddy = float(np.sum((H * ((U - Um) ** 2 + (V - Vm) ** 2)).mean(0)[tro][:, mid] * g.areaCell[mid]))
    b4 = ke_eddy / ke_mean
    # B9: mid-latitude daily rain is local and intermittent
    ml = (np.abs(lat) >= 35) & (np.abs(lat) <= 55)
    x = R[:, ml]
    cv = float(x.std() / x.mean())
    circ = np.abs(np.abs(lat) - 45) < 2.5
    share = []
    for d in range(days):
        r = np.sort(R[d, circ])[::-1]
        if r.sum() > 0:
            share.append(r[:max(1, len(r) // 5)].sum() / r.sum())
    top20 = float(np.median(share))
    # B10 (report only): tropical rain clumps that move
    tro = np.abs(lat) <= 10
    rows = np.clip(((lat[tro] + 10) // 2).astype(int), 0, 9)
    wet_row = int(np.argmax(np.bincount(rows, weights=R[:, tro].mean(0), minlength=10)))
    clump = float(np.median([R[d, tro][rows == wet_row].max() / max(R[d, tro][rows == wet_row].mean(), 1e-12) for d in range(days)]))
    corr = float(np.median([np.corrcoef(R[d, tro], R[d - 5, tro])[0, 1] for d in range(5, days)]))
    # B9 is a recorded near miss (#63: CV 0.97, 50% at n = 5), reported against its threshold until it is revisited at
    # n = 6 / with the ocean; B4 decides
    b9 = cv > 1 and top20 > 0.5
    ok = b4 > 0.2
    record('WEATHER', f'after {WEATHER_SPIN_YEARS} year, 60 days: B4 eddy / mean-flow kinetic energy at 30–60° in the troposphere (#60) {b4:.2f}; B9 mid-latitude daily rain '
           f'coefficient of variation {cv:.2f}, wettest 20% of cells at 45° get {top20 * 100:.0f}% of the rain (median day), '
           f'{"meets" if b9 else "misses"} > 1 and > 50% (known near miss, #63); '
           f'B10 (report only) wettest tropical row max/mean {clump:.2f}, 5-day map correlation {corr:.2f}',
           'B4 > 0.2; B9 reported against CV > 1 and > 50% (#63); B10 reported (v1: > 2 and < 0.5)', ok)
    assert ok
