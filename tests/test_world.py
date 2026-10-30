"""CONSERVE (exact budgets) and REPRO (bit-reproducibility, save/load, independence of cell numbering)."""
import os
import tempfile

import numpy as np
import cupy as cp
import pytest

from conftest import fmt
from insilica import cases
from insilica import grid as gridmod
from insilica import runcache
from insilica import world as worldmod
from insilica.gpu import dsum
from insilica.world import World


@pytest.mark.parametrize('n,days', [pytest.param(4, 15, marks=pytest.mark.xdist_group('grid4')),
                                    pytest.param(6, 1, marks=pytest.mark.xdist_group('grid6'))])
def test_conserve(record, n, days):
    """CONSERVE Budgets on two ocean layers over a ridge (the lower outcrops, with bottom stress): per-layer volume,
    mass and tracer exact and thickness never below the floor; the exchange hook's budgets; and energy with no
    dissipation at all on a balanced two-layer jet"""
    w = World(n, 'outcrop')
    t0 = w.totals()
    low, steps = float(cp.min(w.h)), int(days * 86400 / w.dt)
    for _ in range(steps):
        w.step()
        low = min(low, float(cp.min(w.h)))
    t1 = w.totals()
    layer_keys = [k for k in t0 if k != 'momentum']
    worst = max(abs((t1[k] - t0[k]) / t0[k]) for k in layer_keys)
    finite = bool(cp.all(cp.isfinite(w.h)) and cp.all(cp.isfinite(w.u)))
    floor_ok = low >= 1.0 - 1e-9
    # exchange: move ±(up to 40% of the donor above the floor) between the layers, then check every budget
    rng = np.random.default_rng(n)
    hh = w.h.get()
    up, down = 0.4 * (hh[1] - 1.0), 0.4 * (hh[0] - 1.0)
    wx = np.where(rng.random(w.grid.N) < 0.5, up, -down) * rng.random(w.grid.N)
    x0 = w.totals()
    w.exchange(wx[None])
    x1 = w.totals()
    xch = {'volume': abs((x1['volume0'] + x1['volume1']) - (x0['volume0'] + x0['volume1'])) / (x0['volume0'] + x0['volume1']),
           'tracer': abs((x1['tracer0'] + x1['tracer1']) - (x0['tracer0'] + x0['tracer1'])) / (x0['tracer0'] + x0['tracer1']),
           'momentum': abs(x1['momentum'] - x0['momentum']) / abs(x0['momentum']),
           'M = h·A': max(float(cp.max(cp.abs(w.M[k] / cp.asarray(w.grid.areaCell) - w.h[k])) / cp.max(w.h[k])) for k in range(2))}
    moved = float(np.sum(np.abs(wx) * w.grid.areaCell) / x0['volume1'])
    # energy, no dissipation: the subcritical two-layer jet (BAROCLINIC's 0.3× case, unperturbed)
    h, b, u, gp, hb = cases.phillips(w.grid, 0.3)
    j = World(n, 'phillips', params={'gprime': [gp], 'visc4_U': 0.0}, state=(hb, b, u))
    e0 = j.energy()
    j.step(int(days * 86400 / j.dt))
    drift = (j.energy() - e0) / e0
    ok = worst <= 1e-13 and floor_ok and finite and max(xch.values()) <= 1e-13 and abs(drift) < 1e-4
    record(f'CONSERVE n={n}', f'outcrop {steps} steps ({days} d): per-layer volume, mass, tracer worst {fmt(worst)}; thinnest layer {low:.3f} m; '
           f'exchange of {moved:.1%} of the lower layer: ' + ', '.join(f'{k} {fmt(v)}' for k, v in xch.items())
           + f'; two-layer jet energy drift, no dissipation, {days} d: {fmt(drift)}',
           '≤ 1e-13; ≥ floor (1 m); exchange ≤ 1e-13; energy drift < 1e-4', ok)
    air_ok = cpl_ok = True
    if n == 4:
        air_ok = conserve_air(record)
        cpl_ok = conserve_coupled(record) and conserve_land(record)
    assert ok and air_ok and cpl_ok


def conserve_land(record, days=2):
    """The land's budgets (V2-4a, `continent` from its cached spin-up, every step): the column's energy ledger with land; the
    land's water (soil and ice change = rain − evaporation − runoff); the ocean's salt against its fresh water, runoff included;
    no water across a coast; total energy against TOA; and a save/load continuation."""
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from test_climate import coupled_spun
    w = coupled_spun(4, 'continent')
    g, sea = w.grid, w.sea
    A, At = cp.asarray(g.areaCell), float(np.sum(g.areaCell))
    rho, Sr, land = sea.cfg['rho'], sea.cfg['S_ref'], w.land
    water = lambda: float(cp.sum(A * land * (w.soil + w.ice)))
    icesea = lambda: float(cp.sum(A * (1 - land) * w.ice))
    salt = sea.salt
    W0, I0, S0, flux_l, fw, res, col = water(), icesea(), salt(), 0.0, 0.0, 0.0, 0.0
    for _ in range(int(days * 86400 / w.dt)):
        e0 = w.energy()
        w.step(1, graph=False)
        p = w.physics
        toa = float(cp.sum(A * (p.sw_abs - p.olr))) * w.dt
        res += w.energy() - e0 - toa
        col = max(col, abs(float(cp.sum(A * p.stored)) - toa) / abs(e0))
        flux_l += float(cp.sum(A * land * (p.rain - p.evap - p.runoff))) * w.dt
        fw += float(cp.sum(A * ((1 - land) * (p.rain - p.evap) + land * p.runoff))) * w.dt
    fw -= icesea() - I0
    wland = abs(water() - W0 - flux_l) / W0
    sflux = abs((salt() - S0) + Sr * fw / 1000.0) / S0
    coast = float(cp.abs(sea.u[:, cp.asarray(sea.wall == 0)]).max())
    energy = res / At / (days * 86400)
    f = os.path.join(tempfile.gettempdir(), f'land{os.getpid()}.iw2')
    w.save(f)
    r = World.load(f)
    w.step(48); r.step(48)
    same = w.state_hash() == r.state_hash()
    os.remove(f)
    ok = col <= 1e-13 and wland <= 1e-13 and sflux <= 1e-13 and coast == 0 and abs(energy) <= 0.1 and same
    record('CONSERVE land', f'continent, {days} d after the spin-up: column energy {fmt(col)} of the energy stored; land water {fmt(wland)}; salt against fresh water '
           f'and runoff {fmt(sflux)}; fastest water across a coast {coast:.1f} m/s; energy change − TOA {energy:+.4f} W/m²; reload continues {"identically" if same else "DIFFERENTLY"}',
           '≤ 1e-13; ≤ 1e-13; ≤ 1e-13; 0; |energy − TOA| ≤ 0.1 W/m²; identical', ok)
    return ok


def conserve_coupled(record, days=2):
    """The coupled world's budgets (V2-3, `basin` from its cached spin-up, every step): the ocean's own step keeps its
    heat, salt and volume (per layer outside the mixing); salt changes only by the virtual flux of rain, evaporation and
    sea ice; the wind stress conserves the air's and the ocean's momentum away from the walls; and the total energy
    changes by the TOA flux, up to the air core's own non-conservation (#81)."""
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from test_climate import coupled_spun
    w = coupled_spun(4, 'basin')
    g, sea = w.grid, w.sea
    A = cp.asarray(g.areaCell)
    At, lede = float(A.sum()), cp.asarray(g.le * g.de)
    rc, rho, Sr = sea.cfg['rho'] * sea.cfg['cw'], sea.cfg['rho'], sea.cfg['S_ref']
    heat = lambda: float(cp.sum(A * cp.sum(sea.h * (sea.T - 273.15), axis=0))) * rc
    vol = lambda: [float(cp.sum(A * sea.h[k])) for k in range(sea.L)]
    salt = lambda: float(cp.sum(A * cp.sum(sea.h * sea.S, axis=0)))
    worst = dict(heat=0.0, volume=0.0, salt=0.0, momentum=0.0)
    step0, couple0 = sea.step, sea.couple

    def step(mix=True):
        h0, v0, s0 = heat(), vol(), salt()
        step0(mix=mix)
        h1, v1, s1 = heat(), vol(), salt()
        dv = [abs(sum(v1) - sum(v0)) / sum(v0)] + ([] if mix else [abs(b - a) / a for a, b in zip(v0, v1)])
        worst.update(heat=max(worst['heat'], abs(h1 - h0) / abs(h0)), volume=max(worst['volume'], *dv),
                     salt=max(worst['salt'], abs(s1 - s0) / s0))
    open_ = cp.asarray(sea.core.wall)

    def momentum():
        b = w.L - 1
        ha, ho = 0.5 * (w.h[b][g.eI] + w.h[b][g.eJ]), rho * 0.5 * (sea.h[0][g.eI] + sea.h[0][g.eJ])
        return (float(cp.sum(open_ * lede * (ha * w.u[b] + ho * sea.u[0]))),
                float(cp.sum(open_ * lede * (ha * cp.abs(w.u[b]) + ho * cp.abs(sea.u[0])))))

    def couple(*a, **k):
        m0 = momentum()
        couple0(*a, **k)
        worst['momentum'] = max(worst['momentum'], abs(momentum()[0] - m0[0]) / m0[1])
    sea.step, sea.couple = step, couple
    res, fw, ice0, S0 = 0.0, 0.0, w.ice.copy(), salt()
    for _ in range(int(days * 86400 / w.dt)):
        e0 = w.energy()
        w.step(1, graph=False)                                # the wrapped parts run every step
        res += w.energy() - e0 - float(cp.sum(A * (w.physics.sw_abs - w.physics.olr))) * w.dt
        fw += float(cp.sum(A * (w.physics.rain - w.physics.evap))) * w.dt
    fw -= float(cp.sum(A * (w.ice - ice0)))
    saltflux = abs((salt() - S0) + Sr * fw / rho) / S0
    energy = res / At / (days * 86400)
    ok = max(worst.values()) <= 1e-13 and saltflux <= 1e-13 and abs(energy) <= 0.1
    record('CONSERVE coupled', f'basin, {days} d after the spin-up: ocean step heat {fmt(worst["heat"])}, volume {fmt(worst["volume"])}, '
           f'salt {fmt(worst["salt"])}; salt against fresh water {fmt(saltflux)}; wind-stress momentum {fmt(worst["momentum"])} of the '
           f'gross; energy change − TOA {energy:+.4f} W/m²',
           '≤ 1e-13; ≤ 1e-13; ≤ 1e-13; |energy − TOA| ≤ 0.1 W/m²', ok)
    return ok


def conserve_air(record):
    """The air's budgets (V2-2a, V2-2b): the column physics' energy ledger with latent heat and sea ice (sunlight absorbed
    − OLR = the change in enthalpy, vapor's latent energy, slab heat and ice, every step, crossings, condensation and
    convection included) and its water ledger (vapor change = evaporation − rain); air mass, the tracer and the vapor
    through dynamics, physics and crossings of the moist three-layer aquaplanet; and the energy of adiabatic air dynamics
    with the interfacial stress's dissipation booked as heat."""
    w = World(4, 'rce', params={'physics': {'moist': 1.0}}, state=cases.aqua(gridmod.get(4)))
    A = cp.asarray(w.grid.areaCell)
    worst = per_step = water = 0.0
    for _ in range(48):
        W0 = w.water()
        w.step(1)
        flux = float(cp.sum(A * (w.physics.sw_abs - w.physics.olr))) * w.dt
        stored = float(cp.sum(A * w.physics.stored))
        worst = max(worst, abs(stored - flux) / w.energy())          # relative to the energy stored (v1's ledger measure)
        per_step = max(per_step, abs(stored - flux) / (float(cp.sum(A * w.physics.sw_abs)) * w.dt))
        water = max(water, abs(w.water() - W0 - float(cp.sum(A * (w.physics.evap - w.physics.rain))) * w.dt) / w.water())
    a = World(4, 'aqua', params={'tracer': True})
    m0, W0, src = a.totals(), a.water(), 0.0
    for _ in range(int(10 * 86400 / a.dt)):
        a.step(1)
        src += float(cp.sum(A * (a.physics.evap - a.physics.rain))) * a.dt
    m1 = a.totals()
    vol = lambda m: sum(m[f'volume{k}'] for k in range(a.L))
    trc = lambda m: sum(m[f'tracer{k}'] for k in range(a.L))
    mass = abs(vol(m1) - vol(m0)) / vol(m0)
    tracer = abs(trc(m1) - trc(m0)) / trc(m0)
    vapor = abs(a.water() - W0 - src) / W0
    crossed = float(cp.sum(cp.abs(a.physics.wx) * A) / cp.sum(A))
    g = a.grid
    s2 = np.sin(g.lat) ** 2
    h = np.full((2, g.N), cases.MU_REF / 2)
    th = np.stack([320 - 15 * s2, 300 - 30 * s2])
    j = World(4, 'held_suarez', params=dict(physics={'theta_star': -1.0}, stress=[0.0, 0.0], visc4_U=0.0),
              state=(h, np.zeros(g.N), np.zeros((2, g.E)), dict(theta=th, Ts=np.zeros(g.N))))
    j.physics.hs[...] = 0
    e0 = j.energy()
    j.step(int(15 * 86400 / j.dt))
    drift = (j.energy() - e0) / e0
    ok = worst <= 1e-13 and water <= 1e-13 and mass <= 1e-13 and tracer <= 1e-13 and vapor <= 1e-13 and abs(drift) < 1e-4
    record('CONSERVE air', f'moist column ledgers, 48 steps: energy worst {fmt(worst)} of the energy stored ({fmt(per_step)} of one step of sunlight), '
           f'water {fmt(water)}; moist aqua 10 d (3 layers; dynamics, radiation, crossings ~{crossed:.2f} kg/m² per step, convection, rain): '
           f'air mass {fmt(mass)}, tracer {fmt(tracer)}, vapor against evaporation − rain {fmt(vapor)}; adiabatic air (two layers), '
           f'15 d: energy drift {fmt(drift)}', '≤ 1e-13; ≤ 1e-13; drift < 1e-4', ok)
    return ok


@pytest.mark.xdist_group('grid4')
def test_repro(record, tmp_path, monkeypatch):
    """REPRO Bit-reproducibility: repeat runs, graph vs direct, save/load, spin-up cache, layers; numbering independence"""
    P = {'visc4_U': 20.0, 'drag': 1e-6}
    out, ok = [], True

    # 1. Identical runs, and the CUDA-graph tick against direct ticks; repeated tree sums
    a = World(n=4, preset='aqua').step(200)              # the air world: dynamics, radiation, crossings, θ, slab, time
    b = World(n=4, preset='aqua').step(200)
    c = World(n=4, preset='aqua').step(200, graph=False)
    x = cp.random.default_rng(1).random(10 ** 6)
    same = a.state_hash() == b.state_hash() == c.state_hash() and a.totals() == b.totals() and len({dsum(x) for _ in range(3)}) == 1
    ok &= same
    out.append(f'repeat, graph vs direct, sums {"identical" if same else "DIFFER"}')

    # 2. Save at T, load, continue K = straight T + K; save → load → save gives identical bytes
    p1, p2 = tmp_path / 'a.iw2', tmp_path / 'b.iw2'
    a.save(p1)
    r = World.load(p1)
    r.save(p2)
    same_file = p1.read_bytes() == p2.read_bytes()
    r.step(100)
    a.step(100)
    same_run = r.state_hash() == a.state_hash() == World(n=4, preset='aqua').step(300).state_hash()
    ok &= same_file and same_run
    out.append(f'save/load file {"identical" if same_file else "DIFFERS"}, continuation {"identical" if same_run else "DIFFERS"}')

    # 3. Spin-up cache: a miss then a hit, same state
    monkeypatch.setattr(runcache, 'CACHE_DIR', tmp_path)
    s1, hit1 = runcache.spun_up('repro-w5', lambda: World(n=4, preset='w5'), 50)
    s2, hit2 = runcache.spun_up('repro-w5', lambda: World(n=4, preset='w5'), 50)
    cache_ok = (not hit1) and hit2 and s1.state_hash() == s2.state_hash()
    ok &= cache_ok
    out.append(f'spin-up cache {"miss→hit, identical" if cache_ok else "WRONG"}')

    # 4. Layers: a one-layer case split into two equal layers with g' = 0 moves exactly as one layer
    w2 = World(4, 'galewsky', layers=2, params=dict(P)).step(100)
    w1 = World(4, 'galewsky', params=dict(P)).step(100)
    hb, ub, pb = w2.h.get(), w2.u.get(), w2.psi.get()
    lay = (hb[0].tobytes() == hb[1].tobytes() and ub[0].tobytes() == ub[1].tobytes() and pb[0].tobytes() == pb[1].tobytes()
           and (2 * hb[0]).tobytes() == w1.h.get()[0].tobytes() and ub[0].tobytes() == w1.u.get()[0].tobytes()
           and pb[0].tobytes() == w1.psi.get()[0].tobytes())
    ok &= lay
    out.append(f'2 equal layers = 1 layer {"bit for bit" if lay else "DIFFER"}')

    # 5. Numbering: the Hilbert-ordered grid against v1's numbering, 960 steps (Williamson 5 + bell, FCT3)
    h = World(n=4, preset='w5', params=P)
    real_get = worldmod.gridmod.get
    monkeypatch.setattr(worldmod.gridmod, 'get', lambda n: real_get(n, order='v1'))
    v = World(n=4, preset='w5', params=P)
    monkeypatch.setattr(worldmod.gridmod, 'get', real_get)
    h.step(960)
    v.step(960)
    p = h.grid.perm.astype(np.int64)
    gv = v.grid
    kv = np.minimum(gv.eI, gv.eJ).astype(np.int64) * gv.N + np.maximum(gv.eI, gv.eJ)
    ea, eb = p[h.grid.eI], p[h.grid.eJ]
    kr = np.minimum(ea, eb) * gv.N + np.maximum(ea, eb)
    o = np.argsort(kv)
    em = o[np.searchsorted(kv[o], kr)]
    sg = np.where(ea < eb, 1.0, -1.0)
    rel = lambda x, y: float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
    dn = max(rel(h.h.get()[0], v.h.get()[0][p]), rel(h.u.get()[0], sg * v.u.get()[0][em]), rel(h.psi.get()[0], v.psi.get()[0][p]))
    ok &= dn <= 1e-12
    out.append(f'Hilbert vs v1 numbering after 960 steps {fmt(dn)}')

    record('REPRO', '; '.join(out), 'bit-identical; numbering ≤ 1e-12', ok)
    assert ok
