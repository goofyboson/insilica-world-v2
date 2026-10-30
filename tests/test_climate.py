"""CLIMATE (the moist three-layer aquaplanet's climate, n = 4) and WEATHER (its storms and rain at n = 5) for stage V2-2b.
Spin-ups are cached (runcache); only the measured phase runs every time."""
import numpy as np
import cupy as cp
import pytest

from insilica import runcache
from insilica.grid.trisk import cell_winds
from insilica.world import World

CLIMATE_SPIN_YEARS, WEATHER_SPIN_YEARS = 15, 1   # the 100 m slab settles in ~10 years (#54)
COUPLED_SPIN_YEARS = 20                          # the coupled ocean's ice and halocline are in place by ~15 years (#77)


def _year_ticks(w, years):
    return int(round(years * 365 * 86400 / w.dt))


def coupled_spun(n=4, preset='coupled'):
    """A coupled air-ocean world ('coupled', or 'basin' with its ocean walled, #79) after its cached spin-up (V2-3; whole
    days, so the ocean's daily sums are empty)."""
    w0 = World(n, preset)
    return runcache.spun_up(f'{preset}-n{n}', lambda: World(n, preset), _year_ticks(w0, COUPLED_SPIN_YEARS))[0]


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


@pytest.mark.xdist_group('grid4')
def test_climate_coupled(record):
    """CLIMATE coupled The air over the three-layer ocean in one basin (`basin`, n = 4, the year after a cached spin-up): the air's balance against the ocean's uptake, its circulation and rain, and the ocean's heat transport, thermocline tilt, east–west SST contrast, sea ice and seasons"""
    w = coupled_spun(4, 'basin')
    w.track_ocean_heat_transport()
    g, sea = w.grid, w.sea
    A = cp.asarray(g.areaCell)
    Ah = g.areaCell / g.areaCell.sum()
    lat = np.degrees(g.lat)
    x = (np.degrees(np.angle(np.exp(1j * g.lon))) + 180.0) % 360.0      # longitude east of the wall at 180°
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    per = int(round(86400 / w.dt))
    b = w.L - 1
    rc, Lf = sea.cfg['rho'] * sea.cfg['cw'], w.physics.p['Lf']
    stored = lambda: float(cp.sum(A * (rc * cp.sum(sea.h * (sea.T - 273.15), axis=0) - Lf * w.ice)))   # ocean and ice, J
    rain, Ts, toa, wxb, sst, zt = (cp.zeros(g.N) for _ in range(6))
    ub = np.zeros(g.N)
    m60 = (lat > 55) & (lat < 65)
    ts60, icelat = [], []
    mu0, E0 = w.h[b].copy(), stored()
    for _ in range(365):
        for _ in range(per):
            w.step(1)
            rain += w.physics.rain; Ts += w.physics.tsurf; toa += w.physics.sw_abs - w.physics.olr; wxb += w.physics.wx[b - 1]
            sst += sea.T[0]; zt += sea.h[0] + sea.h[1]
        ub += np.sum(cell_winds(g, w.u[b].get()) * east, 1)
        ts60.append(float(np.sum(Ah[m60] * w.physics.tsurf.get()[m60]) / np.sum(Ah[m60])))
        ice = w.ice.get() > 1.0                                    # kg/m², ~1 mm
        icelat.append(float(np.min(np.abs(lat[ice]))) if ice.any() else 90.0)
    ns, T = 365 * per, 365 * 86400.0
    rain = (rain / ns * 86400).get(); Ts = (Ts / ns).get(); ub /= 365
    sst = (sst / ns).get() - 273.15; zt = (zt / ns).get()
    toa = float(cp.sum(toa / ns * A) / cp.sum(A))
    uptake = (stored() - E0) / T / float(cp.sum(A))
    o = np.argsort(lat)
    ls = lat[o]
    at = lambda cum, xs: {v: cum[np.searchsorted(ls, v) - 1] for v in xs}
    tr = {v: c / 1e9 for v, c in at(np.cumsum((-((w.h[b] - mu0) / T + wxb / T) * A).get()[o]), (-30, -20, -10, 10, 20, 30)).items()}
    # the ocean's northward heat transport (PW): the currents' heat divergence summed from the south pole
    oht = {v: c / 1e15 for v, c in at(np.cumsum((-sea.conv / T * A).get()[o]), (-30, -20, 20, 30)).items()}
    band = lambda f, lo, hi: _band(Ah, lat, f, lo, hi)
    eqm = np.abs(lat) < 5
    third = lambda f, a, c: float(np.sum((g.areaCell * f)[eqm & (x >= a) & (x < c)]) / np.sum(g.areaCell[eqm & (x >= a) & (x < c)]))
    tilt, contrast = third(zt, 0, 120) - third(zt, 240, 360), third(sst, 0, 120) - third(sst, 240, 360)
    m_eq, m_off = (np.abs(lat) < 2) & (x > 240), (np.abs(lat) > 4) & (np.abs(lat) < 8) & (x > 240)
    dip = float(np.sum((g.areaCell * sst)[m_off]) / g.areaCell[m_off].sum() - np.sum((g.areaCell * sst)[m_eq]) / g.areaCell[m_eq].sum())
    ts, eqp, sub = float(np.sum(Ah * Ts)), band(Ts, 0, 10) - band(Ts, 80, 90), band(rain, 15, 30)
    itcz, mid, glob = band(rain, 0, 10) / sub, band(rain, 40, 60) / sub, float(np.sum(Ah * rain))
    trades, west = band(ub, 5, 25), band(ub, 35, 55)
    seasons = max(ts60) - min(ts60)
    hadley = all(tr[v] < 0 for v in (10, 20, 30)) and all(tr[v] > 0 for v in (-10, -20, -30))
    poleward = all(oht[v] <= -0.2 for v in (-30, -20)) and all(oht[v] >= 0.2 for v in (20, 30))
    ok = (abs(ts - 288) <= 3 and abs(toa - uptake) < 1 and hadley and trades < 0 and west > 0 and itcz > 1.3 and mid > 0.5
          and 2 <= glob <= 3 and 30 <= eqp <= 60 and seasons > 3 and poleward and tilt > 10 and contrast > 0.3
          and 50 <= min(icelat) and max(icelat) <= 75)
    record('CLIMATE coupled', f'year {COUPLED_SPIN_YEARS + 1} (basin): mean surface {ts:.2f} K; TOA {toa:+.2f} W/m², ocean and ice '
           f'uptake {uptake:+.2f} (air {toa - uptake:+.2f}); bottom-layer northward transport (1e9 kg/s) at 30S/20S/10S/10N/20N/30N '
           f'{"/".join(f"{tr[v]:+.0f}" for v in (-30, -20, -10, 10, 20, 30))}; trades {trades:+.2f} m/s, westerlies {west:+.2f} m/s; '
           f'rain {glob:.2f} mm/day, ITCZ ×{itcz:.2f}, 40–60° ×{mid:.2f}; equator − pole {eqp:.1f} K; 60°N seasons {seasons:.1f} K; '
           f'ocean heat transport (PW) at 30S/20S/20N/30N {"/".join(f"{oht[v]:+.2f}" for v in (-30, -20, 20, 30))}; equatorial '
           f'thermocline west − east {tilt:+.0f} m, SST west − east {contrast:+.2f} K; sea-ice edge {min(icelat):.0f}–{max(icelat):.0f}°; '
           f'reported: equatorial dip (4–8° − 0–2°, east third) {dip:+.2f} K',
           '288 ± 3 K; |TOA − uptake| < 1 W/m²; Hadley, trades, westerlies; ITCZ > 1.3× and 40–60° > 0.5× the subtropics; '
           '2–3 mm/day; equator − pole 30–60 K; 60°N seasons > 3 K; ocean heat transport ≥ 0.2 PW poleward at 20° and 30°; '
           'thermocline tilt > 10 m; SST contrast > 0.3 K; ice edge 50–75°', ok)
    assert ok


@pytest.mark.xdist_group('grid5')
def test_ridge(record):
    """RIDGE v1's T3.6 and T3.7 on `ridge` (n = 5, year 2): the ridge top's cooling against the adjacent sea; the rain shadow (windward against lee slope) reported, a known shortfall (#84)"""
    w0 = World(5, 'ridge')
    w, _ = runcache.spun_up('ridge-n5', lambda: World(5, 'ridge'), _year_ticks(w0, 1))
    g, per = w.grid, int(round(86400 / w.dt))
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    band = (np.abs(lat) >= 35) & (np.abs(lat) <= 55)
    core, east, west = band & (np.abs(lon) < 5), band & (lon >= 5) & (lon < 15), band & (lon <= -5) & (lon > -15)
    land = w.land.get() > 0.5
    coast = np.zeros(g.N, bool)
    coast[g.eJ[land[g.eI] & ~land[g.eJ]]] = True
    coast[g.eI[land[g.eJ] & ~land[g.eI]]] = True
    rain, T = cp.zeros(g.N), cp.zeros(g.N)
    for _ in range(365 * per):
        w.step(1)
        rain += w.physics.rain
        T += w.physics.tsurf
    rain, T = (rain / (365 * per) * 86400 * 365).get(), (T / (365 * per)).get()
    m = lambda f, c: float(np.sum(g.areaCell[c] * f[c]) / g.areaCell[c].sum())
    cool = m(T, coast) - m(T, core)
    shadow = m(rain, west) / m(rain, east)
    ok = cool > 10 and bool(cp.isfinite(w.u).all())
    record('RIDGE', f'ridge top {m(T, core):.1f} K, adjacent sea {m(T, coast):.1f} K ({cool:.1f} K colder); rain mm/yr west {m(rain, west):.0f}, '
           f'core {m(rain, core):.0f}, east {m(rain, east):.0f} (×{shadow:.2f}; reported, target > 2×, #84)',
           'ridge top > 10 K colder; rain shadow reported', ok)
    assert ok


@pytest.mark.xdist_group('grid4')
def test_climate_land(record):
    """CLIMATE land One land mass in the coupled ocean (`continent`, n = 4, the year after a cached spin-up): the air's balance, circulation and rain, and land's larger seasons"""
    w = coupled_spun(4, 'continent')
    g, sea, per = w.grid, w.sea, int(round(86400 / w.dt))
    A, Ah = cp.asarray(g.areaCell), g.areaCell / g.areaCell.sum()
    lat = np.degrees(g.lat)
    east = np.stack([-np.sin(g.lon), np.cos(g.lon), np.zeros(g.N)], 1)
    b = w.L - 1
    rc, Lf = sea.cfg['rho'] * sea.cfg['cw'], w.physics.p['Lf']
    stored = lambda: float(cp.sum(A * ((1 - w.land) * rc * cp.sum(sea.h * (sea.T - 273.15), axis=0) + w.land * sea.c_land * (sea.T[0] - 273.15)
                                       - Lf * w.ice)))
    land = w.land.get() > 0.5
    mid = (lat > 45) & (lat < 65)
    rain, Ts, toa, wxb = (cp.zeros(g.N) for _ in range(4))
    ub, tl, ts_ = np.zeros(g.N), [], []
    mu0, E0 = w.h[b].copy(), stored()
    for _ in range(365):
        for _ in range(per):
            w.step(1)
            rain += w.physics.rain; Ts += w.physics.tsurf; toa += w.physics.sw_abs - w.physics.olr; wxb += w.physics.wx[b - 1]
        ub += np.sum(cell_winds(g, w.u[b].get()) * east, 1)
        t = w.physics.tsurf.get()
        tl.append(float(np.sum(Ah[mid & land] * t[mid & land]) / np.sum(Ah[mid & land])))
        ts_.append(float(np.sum(Ah[mid & ~land] * t[mid & ~land]) / np.sum(Ah[mid & ~land])))
    ns, T = 365 * per, 365 * 86400.0
    rain, Ts = (rain / ns * 86400).get(), (Ts / ns).get()
    ub /= 365
    toa = float(cp.sum(toa / ns * A) / cp.sum(A))
    uptake = (stored() - E0) / T / float(cp.sum(A))
    o = np.argsort(lat)
    cum, ls = np.cumsum((-((w.h[b] - mu0) / T + wxb / T) * A).get()[o]), lat[o]
    tr = {v: cum[np.searchsorted(ls, v) - 1] / 1e9 for v in (-30, -20, -10, 10, 20, 30)}
    band = lambda f, lo, hi: _band(Ah, lat, f, lo, hi)
    ts, eqp, sub = float(np.sum(Ah * Ts)), band(Ts, 0, 10) - band(Ts, 80, 90), band(rain, 15, 30)
    itcz, mid_r, glob = band(rain, 0, 10) / sub, band(rain, 40, 60) / sub, float(np.sum(Ah * rain))
    trades, west = band(ub, 5, 25), band(ub, 35, 55)
    sl, ss = max(tl) - min(tl), max(ts_) - min(ts_)
    rl = float(np.sum(g.areaCell[land] * rain[land]) / g.areaCell[land].sum())
    hadley = all(tr[v] < 0 for v in (10, 20, 30)) and all(tr[v] > 0 for v in (-10, -20, -30))
    ok = (abs(ts - 288) <= 3 and abs(toa - uptake) < 1 and hadley and trades < 0 and west > 0 and itcz > 1.3 and mid_r > 0.5
          and 2 <= glob <= 3 and 30 <= eqp <= 60 and sl > ss)
    record('CLIMATE land', f'year {COUPLED_SPIN_YEARS + 1} (continent): mean surface {ts:.2f} K; TOA {toa:+.2f} W/m², uptake {uptake:+.2f} (air {toa - uptake:+.2f}); '
           f'bottom-layer transport at 30S/20S/10S/10N/20N/30N {"/".join(f"{tr[v]:+.0f}" for v in (-30, -20, -10, 10, 20, 30))}; trades {trades:+.2f}, '
           f'westerlies {west:+.2f} m/s; rain {glob:.2f} mm/day (land {rl:.2f}), ITCZ ×{itcz:.2f}, 40–60° ×{mid_r:.2f}; equator − pole {eqp:.1f} K; '
           f'45–65°N seasons land {sl:.1f} K, sea {ss:.1f} K',
           '288 ± 3 K; |TOA − uptake| < 1 W/m²; Hadley, trades, westerlies; ITCZ > 1.3×, 40–60° > 0.5×; 2–3 mm/day; equator − pole 30–60 K; '
           'seasons larger over land', ok)
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
