"""TIDES (V2-6f, SPEC #228): the Moon's and the Sun's tide on the ocean, v1's T1-T6 for a dynamic tide, and its dissipation."""
import math

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica.world import World

DAY = 86400.0


def moon_dir(t, tp):
    """The Moon's direction in the planet's frame at time t (s), from tides.Tides' parameter vector, in NumPy (T1's own copy
    of kernels/tides.cu's ephemeris)."""
    tilt, year_s, day_s, ecc, lam_peri = tp[:5]
    if ecc > 0:
        nu0 = -lam_peri
        E0 = 2 * math.atan(math.sqrt((1 - ecc) / (1 + ecc)) * math.tan(nu0 / 2))
        M = E0 - ecc * math.sin(E0) + 2 * math.pi * t / year_s
        E = M
        for _ in range(6):
            E -= (E - ecc * math.sin(E) - M) / (1 - ecc * math.cos(E))
        ls = 2 * math.atan2(math.sqrt(1 + ecc) * math.sin(E / 2), math.sqrt(1 - ecc) * math.cos(E / 2)) + lam_peri
    else:
        ls = 2 * math.pi * t / year_s
    ras = math.atan2(math.cos(tilt) * math.sin(ls), math.cos(ls))
    rot = 2 * math.pi * math.fmod(t, day_s) / day_s
    e, inc = tp[9], tp[10]
    peri, node = tp[16] + 2 * math.pi * t / tp[13], tp[15] + 2 * math.pi * t / tp[12]
    M = tp[14] - peri + 2 * math.pi * t / tp[11]
    E = M
    for _ in range(6):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    u = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2)) + peri - node
    bet = math.asin(math.sin(inc) * math.sin(u))
    lam = node + math.atan2(math.cos(inc) * math.sin(u), math.cos(u))
    dec = math.asin(math.sin(bet) * math.cos(tilt) + math.cos(bet) * math.sin(tilt) * math.sin(lam))
    ra = math.atan2(math.sin(lam) * math.cos(tilt) - math.tan(bet) * math.sin(tilt), math.cos(lam))
    lo = -rot + (ra - ras)
    return np.array([math.cos(dec) * math.cos(lo), math.cos(dec) * math.sin(lo), math.sin(dec)])


def sea_level(w, hours, cell):
    """Hourly sea level (m) at one cell over the given hours."""
    steps = int(round(w.sea.cfg['dt'] / w.dt))
    out = np.zeros(hours)
    for k in range(hours):
        w.step(steps)
        out[k] = float(w.sea.h[:, cell].sum())
    return out - out.mean()


def peak_period(x, lo, hi):
    """The period (in samples) of the spectrum's strongest peak between periods lo and hi, refined by a parabola."""
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0)
    band = np.where((f > 1 / hi) & (f < 1 / lo))[0]
    k = band[np.argmax(P[band])]
    a, b, c = np.log(P[k - 1:k + 2])
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return 1 / (f[k] + d * (f[1] - f[0]))


@pytest.mark.xdist_group('grid4')
def test_tides(record):
    """TIDES Moon and tides: the force's bulges, the semidiurnal period, spring and neap, amplitude against mass, no moon, tidal mixing and its ledger; the dissipation reported"""
    out, ok = [], True
    # T1: the force's two bulges, under the Moon and opposite it (the Sun's tide off), at two times
    w = World(4, 'coupled', params=dict(tides=dict(sun_gm=0.0)))
    g, pos = w.grid, w.grid.pos
    spacing = float(np.sqrt(np.mean(g.areaCell))) / w.planet['radius']
    t1 = 0.0
    for t in (3.3 * DAY, 17.8 * DAY):
        w.time[0] = t
        w.tides.force(w.time, 0.0)
        V = w.tides.V.get()
        m = moon_dir(t, w.tides.par.get())
        a1 = math.acos(np.clip(pos[np.argmax(np.where(pos @ m > 0, V, -np.inf))] @ m, -1, 1))
        a2 = math.acos(np.clip(pos[np.argmax(np.where(pos @ m < 0, V, -np.inf))] @ -m, -1, 1))
        t1 = max(t1, a1 / spacing, a2 / spacing)
    ok &= t1 <= 1.0
    out.append(f'T1 bulges from the Moon and its antipode {t1:.2f} cell spacings (2 times)')

    # T2, T3: an aquaplanet's sea level at a mid-latitude cell, 90 days with Moon and Sun: the semidiurnal period; the spring-neap
    # cycle as the beat of its lunar and solar semidiurnal lines (the daily range's or envelope's own period mixes in the other
    # lines' beats, the elliptic N2's at 27.55 days; and S2 cannot be told from the luni-solar K2, 0.033 h away, in under 182 days,
    # which pulls the measured S2 and the beat a little short)
    w = World(4, 'coupled', params=dict(tides=True))
    cell = int(np.argmin((np.degrees(w.grid.lat) - 30) ** 2 + (np.degrees(w.grid.lon) - 40) ** 2))
    x = sea_level(w, 92 * 24, cell)[48:]
    t2 = peak_period(x, 11.0, 13.0)
    ok &= abs(t2 - 12.42) <= 0.2
    m2, s2 = peak_period(x, 12.2, 12.6), peak_period(x, 11.9, 12.1)
    t3 = 1 / (1 / s2 - 1 / m2) / 24
    ok &= abs(t3 - 14.77) <= 0.5
    out.append(f'T2 dominant period {t2:.2f} h (12.42); T3 spring–neap {t3:.2f} days (14.77) from the lines at {m2:.4f} and {s2:.4f} h')

    # T4: twice the Moon's mass, twice the tide (the Sun's off; the bottom's quadratic drag off, so the response is linear)
    amp = []
    for gm in (1.0, 2.0):
        w = World(4, 'coupled', params=dict(tides=dict(sun_gm=0.0, cd=0.0, moon=dict(gm=4.9028e12 * gm))))
        amp.append(np.std(sea_level(w, 20 * 24, cell)[48:]))
    t4 = amp[1] / amp[0]
    ok &= abs(t4 / 2 - 1) <= 0.05
    out.append(f'T4 amplitude ratio at twice the mass {t4:.4f} (2 ± 5%)')

    # T5: no Moon, no lunar tide (the Sun's 12.00 h alone, nothing at 12.42 h); the water's volume kept
    w = World(4, 'coupled', params=dict(tides=dict(moon=None)))
    A = cp.asarray(w.grid.areaCell)
    v0 = float(cp.sum(w.sea.h * A))
    y = sea_level(w, 30 * 24, cell)[48:]
    P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0)
    near = lambda per: P[np.argmin(np.abs(f - 1 / per))]
    t5 = near(12.42) / near(12.00)
    vol = abs(float(cp.sum(w.sea.h * A)) - v0) / v0
    ok &= t5 <= 1e-2 and vol <= 1e-13
    out.append(f'T5 no Moon: power at 12.42 h against the Sun\'s 12.00 h {t5:.1e}; ocean volume {fmt(vol)}')

    # T6: tidal mixing on earth8's equilibrium (#248) (EARTH's, cached): 10 days with a Moon of 4x the tidal pull against none;
    # the mixing's heat and salt exact on every call; and Earth's own tide's dissipation over the same 10 days
    from insilica import runcache
    from test_radiation import EARTH_CYCLES, EARTH_SPIN_YEARS, _earth_equilibrium
    base = runcache.spun_up(f'earth8-n4-eq{EARTH_SPIN_YEARS}x{EARTH_CYCLES[0]}x{EARTH_CYCLES[1]}', _earth_equilibrium, 0)[0]
    import os, tempfile
    from insilica.io import save as savefile
    f0 = os.path.join(tempfile.gettempdir(), f'tides{os.getpid()}.iw2')
    base.save(f0)
    meta, arrays = savefile.read(f0)
    runs, worst, D = {}, 0.0, {}
    for name, td in (('none', dict(moon=None, sun_gm=0.0)), ('4x', dict(moon=dict(gm=4 * 4.9028e12))), ('earth', True)):
        prm = dict(meta['params'] or {}); prm['tides'] = td; meta['params'] = prm
        savefile.write(f0, meta, arrays)
        e = World.load(f0)
        sea, tid = e.sea, e.tides
        A = cp.asarray(e.grid.areaCell)

        def mix(dt, _m=tid.mix, s=sea):
            h0, s0 = float(cp.sum(A * cp.sum(s.h * s.T, axis=0))), float(cp.sum(A * cp.sum(s.h * s.S, axis=0)))
            _m(dt)
            nonlocal worst
            worst = max(worst, abs(float(cp.sum(A * cp.sum(s.h * s.T, axis=0))) - h0) / h0,
                        abs(float(cp.sum(A * cp.sum(s.h * s.S, axis=0))) - s0) / s0)
        tid.mix = mix
        e.step(int(round(10 * DAY / e.dt)), graph=False)
        runs[name] = (sea.T[0].get(), sea.h.sum(0).get())
        D[name] = float(cp.sum(tid.dsum * A)) / (10 * DAY) / 1e12
    os.remove(f0)
    land = e.land.get() > 0.5
    depth = runs['none'][1]
    shelf = (~land) & (depth < 200.0)
    cool = float(np.average(runs['4x'][0][shelf] - runs['none'][0][shelf], weights=e.grid.areaCell[shelf]))
    ok &= cool < 0 and worst <= 1e-13
    out.append(f'T6 shallow seas (< 200 m, {shelf.sum()} cells): mixed layer {cool:+.3f} K with a 4× Moon against none after 10 days; '
               f'the mixing\'s heat and salt {fmt(worst)}')
    out.append(f'dissipation (reported): Earth\'s Moon and Sun {D["earth"] - D["none"]:.2f} TW of tide (observed 3.7; the '
               f'bottom\'s friction without them {D["none"]:.2f} TW)')
    record('TIDES', '; '.join(out), 'T1 ≤ 1 cell; T2 12.42 ± 0.2 h; T3 14.77 ± 0.5 d; T4 2 ± 5%; T5 ≤ 1e-2, volume ≤ 1e-13; T6 cooler, ≤ 1e-13', ok)
    assert ok
