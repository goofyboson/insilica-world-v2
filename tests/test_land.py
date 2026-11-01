"""WATER and ICE (stage V2-4b, #85, #86): water and ice over and through the land, without air where it can be, on any grid."""
import numpy as np
import cupy as cp
import pytest

from insilica import grid as gridmod
from insilica.gpu import DeviceGrid
from insilica.land import LandWater, RIVER_SPEED
from insilica.world import World


def _water(n, land, b, soilcap=400.0):
    g = gridmod.get(n)
    N = g.N
    pond, soil, ice = cp.zeros(N), cp.zeros(N), cp.zeros(N)
    return g, LandWater(DeviceGrid(g), cp.asarray(b, dtype=float), cp.asarray(land, dtype=float), cp.full(N, float(soilcap)), pond, soil, ice)


def _lonlat(g):
    return np.degrees(np.angle(np.exp(1j * g.lon))), np.degrees(g.lat)


def speed(n, days=60, dt=3600.0):
    """A shallow pulse on a steady slope falling eastward: its centre of mass, days 15 to `days`, m/s."""
    g0 = gridmod.get(n)
    lon, lat = _lonlat(g0)
    land = (np.abs(lat) < 20) & (np.abs(lon) < 90)
    g, lw = _water(n, land, np.where(land, 10.0 + 5.0 * (90.0 - lon), 0.0))
    lw.pond[cp.asarray(land & (lon > -85) & (lon < -70))] = 10.0
    A, R = g.areaCell, np.sqrt(g.areaCell.sum() / (4 * np.pi))
    t, x = [], []
    for s in range(int(days * 86400 / dt)):
        lw.step(dt)
        if s * dt >= 15 * 86400 and s % 24 == 0:
            p = lw.pond.get()
            t.append(s * dt); x.append(np.sum(A * p * lon) / np.sum(A * p))
    return np.polyfit(t, np.radians(x) * R, 1)[0]


@pytest.mark.xdist_group('grid4')
def test_water(record):
    """WATER Surface water and soil water (#85): a bowl fills to a level surface; a flow's speed on any grid; seepage down to the sea; water conserved; in the column, soaking in, frozen ground and a spring"""
    # a bowl: a cone 30° across, the water poured on one cell of its side
    g = gridmod.get(4)
    x = g.pos
    ang = np.degrees(np.arccos(np.clip(x[:, 0], -1, 1)))
    g, lw = _water(4, np.ones(g.N), 1.0 + 300.0 * ang / 30.0)
    A = g.areaCell
    side = int(np.argmin(np.abs(ang - 20.0)))
    lw.pond[side] = 2.0e6
    V0 = float(np.sum(A * lw.pond.get()))
    for _ in range(3 * 365 * 24):
        lw.step(3600.0)
    p = lw.pond.get()
    z = lw.b.get() + p / 1000.0
    wet = p > 1.0
    level = float(z[wet].max() - z[wet].min())
    bowl = abs(float(np.sum(A * p)) / V0 - 1)
    # speed on any grid
    sp = {n: speed(n) for n in (4, 5, 6)}
    # seepage: a saturated strip's soil reaches the sea, conserved
    lon, lat = _lonlat(g)
    land = (np.abs(lat) < 30) & (np.abs(lon) < 40)
    g, lw = _water(4, land, np.where(land, 100.0 + 12.0 * (40.0 - np.abs(lon)), 0.0))
    lw.soil[cp.asarray(land)] = 400.0
    W0 = float(np.sum(A * lw.soil.get()))
    for _ in range(365 * 24):
        lw.step(3600.0)
    S, M = float(np.sum(A * lw.soil.get())), float(np.sum(A * lw.mouth.get()))
    seep = abs((S + M) / W0 - 1)
    # the column's side: no routing, one step of the land cells of a fresh world
    w = World(4, 'ridge')
    w.routing = None
    wc = w.land.get() > 0.5
    idx = np.where(wc)[0]
    third = len(idx) // 3
    warm, cold, spring = idx[:third], idx[third:2 * third], idx[2 * third:]
    Ts, soil, pond = w.Ts.get(), w.soil.get(), w.pond.get()
    cap = w.soilcap.get()
    Ts[warm], soil[warm], pond[warm] = 290.0, 0.25 * cap[warm], 100.0
    Ts[cold], soil[cold], pond[cold] = 260.0, 0.25 * cap[cold], 40.0     # less than freezing warms the ground to 273.15 K
    Ts[spring], soil[spring], pond[spring] = 290.0, 1.5 * cap[spring], 100.0
    w.Ts[...] = cp.asarray(Ts); w.soil[...] = cp.asarray(soil); w.pond[...] = cp.asarray(pond)
    w.step(1, graph=False)
    s1, p1 = w.soil.get(), w.pond.get()
    assert abs(w.physics.p['k_inf'] * 3600 - 5.0) < 1e-12              # v1's 5 mm per hour
    soak = w.physics.p['k_inf'] * w.dt * 0.75
    soak_err = float(np.abs((s1[warm] - soil[warm]) / soak - 1).max())
    frozen = float(np.max(s1[cold] - soil[cold]).clip(0))     # it may evaporate; it takes none in
    sprung = float(np.min(p1[spring] - pond[spring]) / np.min(0.5 * cap[spring]))
    spread = max(sp.values()) / min(sp.values())
    ok = (level <= 1e-3 and bowl <= 1e-13 and all(abs(v / RIVER_SPEED - 1) <= 0.2 for v in sp.values()) and seep <= 1e-13 and M > 0
          and soak_err <= 1e-9 and frozen == 0 and s1[spring].max() <= cap[spring].max() and sprung > 0.9)
    record('WATER', f'bowl level to {level * 1e3:.3f} mm after 3 years, water {bowl:.1e}; flow speed at n = 4/5/6 ' + '/'.join(f'{sp[n]:.3f}' for n in (4, 5, 6)) +
           f' m/s against {RIVER_SPEED} (×{spread:.3f} across grids); seepage to the sea {M / W0 * 100:.1f}% of the soil in a year, water {seep:.1e}; soaking in {soak_err:.1e} off the rule; '
           f'frozen ground takes {frozen:.1f} kg/m²; a spring returns ×{sprung:.3f} of the excess', 'level ≤ 1 mm; water ≤ 1e-13; speed within 20% at each n; soaking ≤ 1e-9; frozen ground 0; spring > 0.9', ok)
    assert ok


YEAR = 365 * 86400.0
ICE_RATE, ICE_MARGIN = 0.3 / YEAR, np.radians(15.0)     # m of ice per second; the dome's edge, radians from the pole


def dome_exact(ang, R, gam):
    """The steady shallow-ice dome on a flat bed of the sphere (Vialov's, with the sphere's geometry): the flow out at angle t is
    a R tan(t/2) = gam H^5 |H'|^3, so that H^(5/3) dH = (a R tan(t/2) / gam)^(1/3) R dt, integrated in from the margin."""
    t = np.linspace(ICE_MARGIN, 0, 20001)
    f = (ICE_RATE * R * np.tan(t / 2) / gam) ** (1 / 3) * R
    H = (8 / 3 * np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * -np.diff(t))])) ** (3 / 8)
    return np.interp(ang, t[::-1], H[::-1], right=0.0)


def dome(n, dty, years=200000):
    """Grow the dome from nothing: the snow falls inside the margin, and what crosses the margin is taken away. Its height and volume against the exact ones."""
    g, lw = _water(n, np.ones(gridmod.get(n).N), np.zeros(gridmod.get(n).N))
    A, R = g.areaCell, np.sqrt(g.areaCell.sum() / (4 * np.pi))
    ang = np.arccos(np.clip(g.pos[:, 2], -1, 1))
    inside = cp.asarray((ang < ICE_MARGIN).astype(float))
    add = inside * ICE_RATE * dty * YEAR * lw.rho_ice
    for _ in range(int(years / dty)):
        lw.ice_step(dty * YEAR)
        lw.ice += add
        lw.ice *= inside
    H, E = lw.ice.get() / lw.rho_ice, dome_exact(ang, R, lw.gam)
    return H.max() / E.max() - 1, np.sum(A * H) / np.sum(A * E) - 1


def island(wall):
    """A kilometres-thick cap on an island, n = 4, for 20 000 years; a wall around it, or none. Returns the ice's error, what reached the sea, how far ice_prev is from that, and the thinnest ice."""
    g = gridmod.get(4)
    ang = np.degrees(np.arccos(np.clip(g.pos[:, 2], -1, 1)))
    land = ang < 20.0
    g, lw = _water(4, land, np.where(land, 100.0 + (5000.0 if wall else 0.0) * (np.abs(ang - 14.0) < 3.0), 0.0))
    lw.ice_prev = cp.zeros(g.N)
    lw.ice[cp.asarray(ang < (10.0 if wall else 18.0))] = 2.0e6
    A = cp.asarray(g.areaCell)
    I0 = float(cp.sum(A * lw.ice))
    for _ in range(1000):
        lw.ice_step(20.0 * YEAR)
    calved = float(cp.sum(A * lw.calved))
    return abs(float(cp.sum(A * lw.ice)) / I0 - 1), calved / I0, float(cp.abs(lw.ice_prev - lw.calved).max()), float(lw.ice.min())


def test_ice(record):
    """ICE Ice sheets (#86): a dome grown from nothing against the exact shallow-ice profile, closer as the grid refines; ice conserved, reaching the sea as sea ice the salt does not count, and not climbing a wall"""
    d5, v5 = dome(5, 10)
    d6, v6 = dome(6, 5)
    err, gave, prev, low = island(False)
    werr, wgave, _, _ = island(True)
    ok = (abs(d5) <= 0.1 and abs(v5) <= 0.1 and abs(d6) <= 0.1 and abs(v6) <= 0.1 and abs(d6) < abs(d5) and err <= 1e-13 and gave > 0 and prev <= 1e-9 and low >= 0
          and werr <= 1e-13 and wgave == 0)
    record('ICE', f'dome height against the exact {d5 * 100:+.1f}% at n = 5, {d6 * 100:+.1f}% at n = 6; volume {v5 * 100:+.1f}%, {v6 * 100:+.1f}%; island ice conserved to {err:.1e}, {gave * 100:.1f}% reached the sea in 20 000 years, '
           f'fresh-water count off by {prev:.1e} kg/m², thinnest ice {low:.1e}; behind a wall {wgave:.1e} reached it', 'height and volume within 10% at n = 5 and 6, height closer at n = 6; ice ≤ 1e-13; some reaches the sea; count ≤ 1e-9; ice ≥ 0; none past a wall', ok)
    assert ok
