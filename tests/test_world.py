"""CONSERVE (exact budgets) and REPRO (bit-reproducibility, save/load, independence of cell numbering)."""
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
    assert ok


@pytest.mark.xdist_group('grid4')
def test_repro(record, tmp_path, monkeypatch):
    """REPRO Bit-reproducibility: repeat runs, graph vs direct, save/load, spin-up cache, layers; numbering independence"""
    P = {'visc4_U': 20.0, 'drag': 1e-6}
    out, ok = [], True

    # 1. Identical runs, and the CUDA-graph tick against direct ticks; repeated tree sums
    a = World(n=4, preset='galewsky', params=P).step(200)
    b = World(n=4, preset='galewsky', params=P).step(200)
    c = World(n=4, preset='galewsky', params=P).step(200, graph=False)
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
    same_run = r.state_hash() == a.state_hash() == World(n=4, preset='galewsky', params=P).step(300).state_hash()
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
