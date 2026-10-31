"""WATER (stage V2-4b, #85): water over and through the land, without air where it can be, on any grid."""
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
