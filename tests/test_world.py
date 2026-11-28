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
from insilica import gpu as gpumod
from insilica.gpu import dsum
from insilica.ensemble import Ensemble
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
    land's water (soil, surface water, ice and what waits at the mouths change by rain − evaporation − what the sea took); the ocean's salt against its fresh water, the rivers included;
    no water across a coast; total energy against TOA; and a save/load continuation. The sea ice drifts (#185), so these budgets
    hold with it moving."""
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from test_climate import coupled_spun
    w = coupled_spun(4, 'continent')
    g, sea = w.grid, w.sea
    w.ice += cp.asarray(np.abs(g.lat) > np.radians(50.0)) * w.land * 3.0e6        # 3 km of ice on the far land, so that some flows into the sea (#86)
    w.enable_ice_drift(0.02)                                                    # sea-ice drift (V2-6e, #185)
    moved = 0.0
    A, At = cp.asarray(g.areaCell), float(np.sum(g.areaCell))
    rho, Sr, land = sea.cfg['rho'], sea.cfg['S_ref'], w.land
    calved = w.routing.calved
    water = lambda: float(cp.sum(A * (land * (w.soil + w.ice + w.pond) + w.mouth + calved)))
    icesea = lambda: float(cp.sum(A * (1 - land) * w.ice))
    salt = sea.salt
    W0, I0, S0, C0, flux_l, fw, res, col = water(), icesea(), salt(), float(cp.sum(A * calved)), 0.0, 0.0, 0.0, 0.0
    for _ in range(int(days * 86400 / w.dt)):
        e0 = w.energy()
        took = float(cp.sum(A * w.mouth))                      # the ocean takes what waits at the start of its tick
        w.step(1, graph=False)
        if w._kind(w.tick - 1):                                # what the drift carried this ocean step (it moves ice and ice_prev together)
            moved += float(cp.sum(cp.abs(w.sea._iF)))
        p = w.physics
        toa = float(cp.sum(A * (p.sw_abs - p.olr))) * w.dt
        res += w.energy() - e0 - toa
        col = max(col, abs(float(cp.sum(A * p.stored)) - toa) / abs(e0))
        flux_l += float(cp.sum(A * land * (p.rain - p.evap))) * w.dt - took
        fw += float(cp.sum(A * (1 - land) * (p.rain - p.evap))) * w.dt + took
    fw -= icesea() - I0 - (float(cp.sum(A * calved)) - C0)          # ice from the land is not water the sea froze
    wland = abs(water() - W0 - flux_l) / W0
    sflux = abs((salt() - S0) + Sr * fw / 1000.0) / S0
    coast = float(cp.abs(sea.u[:, cp.asarray(sea.wall == 0)]).max())
    gave = (float(cp.sum(A * calved)) - C0) / days / 86400.0 / float(cp.sum(A * land)) * 86400.0
    energy = res / At / (days * 86400)
    f = os.path.join(tempfile.gettempdir(), f'land{os.getpid()}.iw2')
    w.save(f)
    r = World.load(f)
    r.enable_ice_drift(0.02)
    w.step(48); r.step(48)
    same = w.state_hash() == r.state_hash()
    os.remove(f)
    ok = col <= 1e-13 and wland <= 1e-13 and sflux <= 1e-13 and coast == 0 and abs(energy) <= 0.1 and same and gave > 0
    record('CONSERVE land', f'continent, {days} d after the spin-up, with 3 km of ice on land beyond 50° (it gives the sea {gave:.1e} mm/day of the land), the sea ice drifting ({moved / days / float(cp.sum(A * (1 - land) * w.ice)):.1e} of it a day): column energy {fmt(col)} of the energy stored; land water {fmt(wland)}; salt against fresh water '
           f'and rivers {fmt(sflux)}; fastest water across a coast {coast:.1f} m/s; energy change − TOA {energy:+.4f} W/m²; reload continues {"identically" if same else "DIFFERENTLY"}',
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
    # the seven-layer earth's columns (V2-6b; earth6 since #142, earth7 since #153): the mixed layer, deep plumes taking in the cloud layer's air, their downdrafts, the land
    # (#216: in an air of its own, nine parts N₂ to one of CO₂, its constants compiled in: the ledgers close on the air's own cp)
    # (#217: carrying CO₂, N₂O and CH₄ as tracers, CO₂ with a surface source on 100 cells)
    # (#218: with earth7's land column under the ground's skin: its conduction keeps the land's heat, and it reloads)
    from insilica.constants import SOIL_DZ
    air = {'physics': {'gases': dict(n2=0.9, co2=0.1)}}
    # (#220: with geology's accumulators, each sum against the same kept here, the runoff against the water reaching the sea)
    # (#307: with V2-7b's moist mixed layer, its cloud from theta_l and q_t and its cloud top's cooling in its entrainment)
    e = World(4, 'earth7', params=dict(air, physics=dict(air['physics'], ml_moist=1.0), gas_tracers=1, geology=1))
    e.gas_src[0, :100] = 1e-9
    gamt = lambda: [float(cp.sum(e.M * e.gx[j])) for j in range(e.gx.shape[0])]
    g0, gsrc = gamt(), 0.0
    e4 = e4w = 0.0
    lede, ge = cp.asarray(e.grid.le * e.grid.de), e.grid
    he = lambda h: 0.5 * (h[:, ge.eI] + h[:, ge.eJ])
    ke = lambda h, u: float(cp.sum(lede * cp.sum(0.5 * he(h) * u * u, axis=0)))
    mom = lambda h, u: cp.sum(he(h) * u, axis=0)                 # each edge's column momentum per area
    drag = {n: dict(energy=0.0, gross=0.0, momentum=0.0) for n in ('oro', 'gw')}

    def ledger(name, ap):
        """The drags (#127, #128): kinetic energy removed = heat booked; the waves (#128) keep each column's momentum."""
        def apply(h, th, u, dt, diss):
            d = drag[name]
            k0, d0, m0 = ke(h, u), float(cp.sum(lede * diss)), mom(h, u)
            ap(h, th, u, dt, diss)
            removed, booked = k0 - ke(h, u), float(cp.sum(lede * diss)) - d0
            d['gross'] += abs(removed)
            d['energy'] = max(d['energy'], abs(removed - booked) / k0)
            dm = mom(h, u) - m0
            d['momentum'] = max(d['momentum'], float(cp.sum(lede * cp.abs(dm)) / cp.sum(lede * cp.sum(he(h) * cp.abs(u), axis=0))))
        return apply
    e.oro.apply, e.gw.apply = ledger('oro', e.oro.apply), ledger('gw', e.gw.apply)
    sea, lnd, soil = e.sea, e.land, {'worst': 0.0, 'moved': 0.0}
    dzc = cp.asarray(SOIL_DZ)[:, None] * sea.soil['c']

    def column_heat():
        """The land's skin and column (J), and its gross relative to freezing."""
        x = sea.cap * (sea.T[0] - 273.15) + cp.sum(dzc * (sea.Tg - 273.15), axis=0)
        return float(cp.sum(A * lnd * x)), float(cp.sum(A * lnd * (sea.cap * cp.abs(sea.T[0] - 273.15) + cp.sum(dzc * cp.abs(sea.Tg - 273.15), axis=0))))

    def ground(dt, _g=sea.ground):
        h0, gross = column_heat()
        t0 = sea.Tg.copy()
        _g(dt)
        soil['worst'] = max(soil['worst'], abs(column_heat()[0] - h0) / gross)
        soil['moved'] += float(cp.sum(A * lnd * cp.sum(dzc * cp.abs(sea.Tg - t0), axis=0))) / float(cp.sum(A * lnd)) / dt
    sea.ground = ground
    from insilica.geology import CRACK, EA_WEATHER, FT_BAND, R_GAS, ROWS, T0_WEATHER
    geo, lm = e.geology, lnd > 0.5
    ref = {k: cp.zeros(e.grid.N) for k in ('rain', 'evap', 'tsum', 'frozen', 'crack', 'ftc', 'rx1', 'weather')}
    gs = dict(fst=geo.row('fstate').copy(), bucket=cp.zeros(e.grid.N), ring=[], n=0, runoff=0.0, sent=0.0, below=0)

    def gcol(rain, evap, ice=None, _c=geo.column):
        T, dt, tf = geo.Ts.copy(), e.dt, geo.tf
        _c(rain, evap, ice)
        ref['rain'] += rain * dt; ref['evap'] += evap * dt; ref['tsum'] += T * dt
        ref['frozen'] += cp.where(lm & (T < tf), dt, 0.0)
        ref['crack'] += cp.where(lm & (T >= tf + CRACK[0]) & (T <= tf + CRACK[1]), dt, 0.0)
        thaw = lm & (gs['fst'] > 0.5) & (T > tf + FT_BAND)
        ref['ftc'] += thaw
        gs['fst'] = cp.where(thaw, 0.0, cp.where(lm & (gs['fst'] < 0.5) & (T < tf - FT_BAND), 1.0, gs['fst']))
        gs['bucket'] = gs['bucket'] + rain * dt
        gs['n'] += 1
        if gs['n'] % geo.slot == 0:
            gs['ring'] = (gs['ring'] + [gs['bucket']])[-geo.nslot:]
            gs['bucket'] = cp.zeros(e.grid.N)
            ref['rx1'] = cp.maximum(ref['rx1'], sum(gs['ring']))
    geo.column = gcol

    def route(dt, _r=e.routing.step):
        m0, a0, p0, i0 = e.mouth.copy(), geo.acc.copy(), e.pond.copy(), e.ice.copy()
        _r(dt)
        d = geo.acc - a0
        run = d[ROWS.index('runoff')]
        gs['runoff'] = max(gs['runoff'], abs(float(cp.sum(A * run)) - float(cp.sum(A * (e.mouth - m0)))) / float(cp.sum(A * cp.abs(run))))
        gs['sent'] += float(cp.sum(A * (e.mouth - m0)))
        ref['weather'] += run * cp.exp(EA_WEATHER / R_GAS * (1 / T0_WEATHER - 1 / e.Ts))
        # out at least its net loss, to rounding of the amounts and of the flows through the cell (#226: a nearly empty pond passing water on)
        dr, di = d[ROWS.index('river')], d[ROWS.index('iceflow')]
        gs['below'] += int(cp.sum(lm & (dr < (p0 - e.pond) - 1e-12 * (cp.abs(p0) + cp.abs(e.pond) + 2 * cp.abs(dr)))))
        gs['below'] += int(cp.sum(lm & (di < (i0 - e.ice) - 1e-12 * (cp.abs(i0) + cp.abs(e.ice) + 2 * cp.abs(di)))))
    e.routing.step = route
    snow = 0.0                                            # the snow reaching the ground (#134): its freezing heat given aloft
    for t in range(48):
        W0, E0 = e.water(), e.energy()
        e.step(1, graph=False)                            # (the drag's ledger reads the host every step)
        p = e.physics
        gsrc += float(cp.sum(A * e.gas_src[0])) * e.dt
        if t == 0:                                        # (#217) the k-tables reading the carried gases, all at the planet's values
            f = World(4, 'earth7', params=air)
            f.step(1, graph=False)
            olr_gx = abs(float(cp.sum(A * p.olr)) / float(cp.sum(A * f.physics.olr)) - 1)
        snow += float(cp.sum(A * p.rainp[3])) / float(cp.sum(A)) * e.dt
        e4 = max(e4, abs(float(cp.sum(A * p.stored)) - float(cp.sum(A * (p.sw_abs - p.olr))) * e.dt) / abs(E0))
        e4w = max(e4w, abs(e.water() - W0 - float(cp.sum(A * (p.evap - p.rain))) * e.dt) / e.water())
    g1 = gamt()
    gerr = max(float(cp.max(cp.abs(geo.row(k) - v)) / max(float(cp.max(cp.abs(v))), 1e-300)) for k, v in ref.items())
    geo_ok = (gerr <= 1e-13 and gs['runoff'] <= 1e-13 and gs['below'] == 0 and float(ref['rx1'].max()) > 0 and float(ref['ftc'].sum()) > 0
              and gs['sent'] > 0 and float(geo.row('iceflow').max()) > 0 and 'geo' in e.STATE)
    fs = os.path.join(tempfile.gettempdir(), f'soil{os.getpid()}.iw2')
    e.save(fs)
    reload_ok = World.load(fs).state_hash() == e.state_hash() and 'oTg' in e.STATE
    os.remove(fs)
    gbud = max(abs(g1[0] - g0[0] - gsrc) / g1[0], *(abs(g1[j] - g0[j]) / g0[j] for j in range(1, len(g0))))
    guni = max(float((e.gx[j].max() - e.gx[j].min()) / e.gx[j].mean()) for j in range(1, len(g0)))
    fr = mars_frost()
    ok = fr['ok'] and (worst <= 1e-13 and water <= 1e-13 and mass <= 1e-13 and tracer <= 1e-13 and vapor <= 1e-13 and abs(drift) < 1e-4 and e4 <= 1e-13
          and e4w <= 1e-13 and snow > 0 and all(d['energy'] <= 1e-13 and d['gross'] > 0 for d in drag.values()) and drag['gw']['momentum'] <= 1e-13
          and gbud <= 1e-13 and guni <= 1e-12 and olr_gx <= 1e-12 and soil['worst'] <= 1e-13 and soil['moved'] > 0 and reload_ok
          and geo_ok)
    record('CONSERVE air', f'moist column ledgers, 48 steps: energy worst {fmt(worst)} of the energy stored ({fmt(per_step)} of one step of sunlight), '
           f'water {fmt(water)}; moist aqua 10 d (3 layers; dynamics, radiation, crossings ~{crossed:.2f} kg/m² per step, convection, rain): '
           f'air mass {fmt(mass)}, tracer {fmt(tracer)}, vapor against evaporation − rain {fmt(vapor)}; adiabatic air (two layers), '
           f'15 d: energy drift {fmt(drift)}; seven-layer earth in N₂–CO₂ air (cp {e.gas["cp"]:.1f}, R {e.gas["rd"]:.1f}, #216), 48 steps (mixed layer, cumulus layer, plumes with downdrafts and entrainment, a shallow plume, ice in the air with its '
           f'freezing heat aloft, {snow * 86400 / (48 * e.dt):.3f} kg/m² of snow a day reaching the ground): '
           f'column energy {fmt(e4)} of the energy stored, water {fmt(e4w)}; kinetic energy removed against heat booked: mountain drag (#127) '
           f'{fmt(drag["oro"]["energy"])}, waves (#128) {fmt(drag["gw"]["energy"])} of the kinetic energy ({drag["oro"]["gross"] / 48:.2e}, '
           f'{drag["gw"]["gross"] / 48:.2e} J a step); the waves\' column momentum {fmt(drag["gw"]["momentum"])} of the gross; '
           f'gas tracers (#217): CO₂, N₂O, CH₄ against their sources (CO₂ {gsrc / g1[0]:.1e} of its amount from the surface) {fmt(gbud)}, '
           f'N₂O and CH₄ started uniform stay so to {fmt(guni)}, OLR reading them against the planet\'s values {fmt(olr_gx)}; '
           f'land column (#218, {len(SOIL_DZ)} layers to {sum(SOIL_DZ):.1f} m, {soil["moved"] / 48:.1f} W/m² changing its layers on land, mean over the steps): its heat {fmt(soil["worst"])} of its gross, '
           f'geology’s accumulators (#220; rain, evaporation, temperature, frost, freeze-thaw, the day’s largest rain, weathering) against the same kept here {fmt(gerr)}, '
           f'the land’s runoff against the water reaching the sea ({gs["sent"] / 1e12:.1f} Gt) {fmt(gs["runoff"])}, outflows below a cell’s net loss {gs["below"]}; '
           f'reload {"identical" if reload_ok else "DIFFERENT"}; ' + fr['text'],
           '≤ 1e-13; ≤ 1e-13; drift < 1e-4; uniform ≤ 1e-12; reload identical', ok)
    return ok


def mars_frost():
    """(V2-7, #236) Mars's CO2 frost: a start with cold polar ground (140 K poleward of 60°N), frost lying poleward of 60°S, and the
    second layer at 120 K over the north pole's cap, so frost forms on the ground and in the air and sublimates, in 48 steps. Each step:
    the column's energy with the frost's against sunlight − OLR, the air's mass with the frost's, the kinetic energy that leaves or
    arrives with the frost's air against the heat booked for it; (#238) the ground column's heat through its step (TES albedo, Putzig's
    thermal inertia); and a reload. (#239) The dust: each cell's column at the first radiation step against the maps read here at that
    season, and the sunlight the air absorbs with it against without."""
    from insilica.constants import MARS, air_props
    g = gridmod.get(4)
    h, b, u, extra = cases.mars_build(g, cases.mars_topo(g))
    north, south = g.lat > np.radians(60), g.lat < np.radians(-60)
    extra = dict(extra, frost=np.where(south, 50.0, 0.0))
    extra['Ts'] = np.where(north, 140.0, extra['Ts'])
    kap = air_props(MARS['gases'], t_cp=MARS['t_cp'])['kappa']
    k1 = h.shape[0] - 4                                                            # (the layer under theta* 1700 K)
    pm = np.cumsum(h, axis=0)[k1] * cases.GRAV - 0.5 * h[k1] * cases.GRAV          # (Earth-gravity pressures, as the builder's)
    extra['theta'][k1] = np.where(g.lat > np.radians(80), 120.0 * (cases.P0 / pm) ** kap, extra['theta'][k1])
    m = World(4, 'mars', state=(h, b, u, extra))
    A, ph = cp.asarray(m.grid.areaCell), m.physics
    lede, ge = cp.asarray(m.grid.le * m.grid.de), m.grid
    ke = lambda h_, u_: float(cp.sum(lede * cp.sum(0.25 * (h_[:, ge.eI] + h_[:, ge.eJ]) * u_ * u_, axis=0)))
    mass = lambda: float(cp.sum(A * (cp.sum(m.h, axis=0) + m.frost)))
    st = dict(e=0.0, ke=0.0, gross=0.0, ground=0.0, air=0.0, sub=0.0)

    def apply(h_, M_, u_, th_, _a=ph.frost_mass):
        _a(h_, M_, u_, th_)
        removed, booked = ke(ph.co2x, u_), float(cp.sum(ph.frost_ke))     # (the kinetic energy of the air moved; the mass budget checks h follows it)
        st['gross'] += abs(removed)
        st['ke'] = max(st['ke'], abs(removed - booked) / max(abs(removed), 1e-300))
        x = ph.co2x
        st['ground'] += float(cp.sum(A * cp.maximum(x[-1], 0))); st['sub'] += float(cp.sum(A * cp.maximum(-x[-1], 0)))
        st['air'] += float(cp.sum(A * x[:-1]))
    ph.frost_mass = apply
    gr = m.ground
    gheat = lambda: (float(cp.sum(A * (gr.cap * (m.Ts - 273.15) + gr.rho_c * cp.sum(gr.dz[:, None] * (gr.Tg - 273.15), axis=0)))),
                     float(cp.sum(A * (gr.cap * cp.abs(m.Ts - 273.15) + gr.rho_c * cp.sum(gr.dz[:, None] * cp.abs(gr.Tg - 273.15), axis=0)))))
    st['ground_heat'] = 0.0

    def gstep(Ts, dt, _g=gr.step):
        h0, gross = gheat()
        _g(Ts, dt)
        st['ground_heat'] = max(st['ground_heat'], abs(gheat()[0] - h0) / gross)
    gr.step = gstep
    m0, e0, worst = mass(), m.energy(), 0.0
    # (#239) the dust: its column at the first step against the map at the season (Ls from the orbit, as sun_lon), and the air's
    # sunlight with it against without (the shortwave run again on the same state with the dust left out)
    m.step(1, graph=False)
    du, pp = ph.dust, ph.p
    du.update(m.time, m.h)                                # (on this state and time: the kernel against the maps read here)
    e, nu0 = pp['ecc'], -pp['lam_peri']
    E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu0 / 2))
    M = E0 - e * np.sin(E0) + 2 * np.pi * float(m.time.get()[0]) / pp['year_s']
    Ek = M
    for _ in range(8):
        Ek -= (Ek - e * np.sin(Ek) - M) / (1 - e * np.cos(Ek))
    ls = np.degrees(2 * np.arctan2(np.sqrt(1 + e) * np.sin(Ek / 2), np.sqrt(1 - e) * np.cos(Ek / 2)) + pp['lam_peri']) % 360
    # the map read here from its file (not the model's cells' table): bilinear at each cell's latitude and longitude
    from pathlib import Path
    dm = np.load(Path(cases.__file__).parent / 'data' / 'mars_dust_my30.npz')
    lsb, la, lo, cmap = dm['ls'], dm['lat'], dm['lon'], dm['cdod610'].astype(float)
    pos = m.grid.pos
    clat, clon = np.degrees(np.arcsin(np.clip(pos[:, 2], -1, 1))), np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    yy = np.clip((clat - la[0]) / (la[1] - la[0]), 0, len(la) - 1 - 1e-9)
    xx = ((clon - lo[0]) / (lo[1] - lo[0])) % len(lo)
    y0, x0 = yy.astype(int), xx.astype(int)
    fy, fx, x1 = yy - y0, xx - x0, (xx.astype(int) + 1) % len(lo)
    tab = ((1 - fy) * ((1 - fx) * cmap[:, y0, x0] + fx * cmap[:, y0, x1]) + fy * ((1 - fx) * cmap[:, y0 + 1, x0] + fx * cmap[:, y0 + 1, x1]))
    j = int(np.searchsorted(lsb, ls)) - 1
    j0, j1 = j % len(lsb), (j + 1) % len(lsb)
    wgt = ((ls - lsb[j0]) % 360) / ((lsb[j1] - lsb[j0]) % 360)
    want = (tab[j0] * (1 - wgt) + tab[j1] * wgt) * m.grav * m.h.sum(axis=0).get() / 610
    dust_read = float(np.max(np.abs(du.dtau.get() - want)) / np.max(want))
    lit = ph.sw.swf[m.L].get() > 0
    air_dust = ph.sw.swf[:m.L].get().sum(0)
    ph.lw.dust = None
    ph.radiate(m.h, m.theta, m.Ts, m.time, m.rad_every * m.dt)
    air_none = ph.sw.swf[:m.L].get().sum(0)
    ph.lw.dust = du
    ph.radiate(m.h, m.theta, m.Ts, m.time, m.rad_every * m.dt)
    dust_air = (float(air_dust[lit].mean()), float(air_none[lit].mean()))
    m0, e0 = mass(), m.energy()
    for _ in range(48):
        m.step(1, graph=False)
        flux = float(cp.sum(A * (ph.sw_abs - ph.olr))) * m.dt
        st['e'] = max(st['e'], abs(float(cp.sum(A * ph.stored)) - flux) / abs(e0))
    worst = abs(mass() - m0) / m0
    fs = os.path.join(tempfile.gettempdir(), f'frost{os.getpid()}.iw2')
    m.save(fs)
    reload_ok = World.load(fs).state_hash() == m.state_hash() and 'frost' in m.STATE and 'gTg' in m.STATE
    os.remove(fs)
    ok = (st['e'] <= 1e-13 and worst <= 1e-13 and st['ke'] <= 1e-13 and st['gross'] > 0 and st['ground'] > 0 and st['sub'] > 0
          and st['air'] > 0 and st['ground_heat'] <= 1e-13 and reload_ok and dust_read <= 1e-6 and dust_air[0] > dust_air[1])
    tot = float(cp.sum(A)) * 48
    text = (f'Mars CO₂ frost (#236), 48 steps: column energy with the frost {fmt(st["e"])} of the energy, air + frost mass {fmt(worst)}, '
            f'kinetic energy with the frost\'s air against heat booked {fmt(st["ke"])} ({st["gross"] / 48:.2e} J a step); condensed on the ground '
            f'{st["ground"] / tot * 1e3:.2f}, in the air {st["air"] / tot * 1e3:.2f}, sublimated {st["sub"] / tot * 1e3:.2f} g/m² a step (planet mean); '
            f'the ground column (#238, {len(gr.dz)} layers to {float(gr.dz.sum()):.1f} m) its heat {fmt(st["ground_heat"])} of its gross; '
            f'dust (#239, Ls {ls:.1f}): the column against the maps {fmt(dust_read)} of its largest ({float(du.dtau.max()):.3f}), the air '
            f'absorbing {dust_air[0] * 100:.2f}% of the sunlight with it, {dust_air[1] * 100:.2f}% without (sunlit cells); '
            f'reload {"identical" if reload_ok else "DIFFERENT"}')
    return dict(ok=ok, text=text)


@pytest.mark.xdist_group('grid4')
def test_repro(record, tmp_path, monkeypatch):
    """REPRO Bit-reproducibility: repeat runs, graph vs direct, ensemble members, save/load, spin-up cache, layers; numbering independence"""
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
    # ... and as members of an ensemble running side by side (V2-5)
    e = Ensemble(World(n=4, preset='aqua') for _ in range(3)).step(200)
    ens = all(m.state_hash() == a.state_hash() for m in e)
    ok &= ens
    out.append(f'3 ensemble members {"identical" if ens else "DIFFER"}')

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

    # 5. Numbering: the Hilbert-ordered grid against v1's numbering, 960 steps (Williamson 5 + bell, FCT3), in the 64-bit
    # reference: it checks the renumbering's indexing, which mixed precision's 32-bit sums in another order would hide (#106)
    prec = gpumod.PRECISION
    gpumod.set_precision('double')
    h = World(n=4, preset='w5', params=P)
    real_get = worldmod.gridmod.get
    monkeypatch.setattr(worldmod.gridmod, 'get', lambda n: real_get(n, order='v1'))
    v = World(n=4, preset='w5', params=P)
    monkeypatch.setattr(worldmod.gridmod, 'get', real_get)
    gpumod.set_precision(prec)
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
