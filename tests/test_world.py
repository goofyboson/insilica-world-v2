"""CONSERVE (exact budgets) and REPRO (bit-reproducibility, save/load, independence of cell numbering)."""
import numpy as np
import cupy as cp
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica import runcache
from insilica import world as worldmod
from insilica.gpu import dsum
from insilica.world import World


@pytest.mark.parametrize('n,days', [pytest.param(4, 5, marks=pytest.mark.xdist_group('grid4')),
                                    pytest.param(6, 1, marks=pytest.mark.xdist_group('grid6'))])
def test_conserve(record, n, days):
    """CONSERVE Williamson 5 + cosine bell: total volume Σh·A, layer mass ΣM and tracer ΣM·psi"""
    w = World(n=n, preset='w5', params={'nu4': 1e14 * 16.0 ** (4 - n), 'drag': 1e-6})
    t0 = w.totals()
    w.step(int(days * 86400 / w.dt))
    t1 = w.totals()
    d = {k: (t1[k] - t0[k]) / t0[k] for k in t0}
    ok = all(abs(v) <= 1e-13 for v in d.values())
    record(f'CONSERVE n={n}', f'{w.tick} steps ({days} d): volume {fmt(d["volume"])}, M {fmt(d["M"])}, tracer {fmt(d["tracer"])}',
           '|relative change| ≤ 1e-13', ok)
    assert ok


@pytest.mark.xdist_group('grid4')
def test_repro(record, tmp_path, monkeypatch):
    """REPRO Bit-reproducibility: repeat runs, graph vs direct, save/load, spin-up cache, layers; numbering independence"""
    P = {'nu4': 1e14, 'drag': 1e-6}
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

    # 4. Layers: two identical layers stay bit-identical to each other and to a one-layer run
    w2 = World(n=4, preset='galewsky', layers=2, params=P).step(100)
    w1 = World(n=4, preset='galewsky', layers=1, params=P).step(100)
    lay = all(getattr(w2, k)[0].get().tobytes() == getattr(w2, k)[1].get().tobytes() == getattr(w1, k)[0].get().tobytes()
              for k in World.STATE)
    ok &= lay
    out.append(f'2 layers = 1 layer {"bit for bit" if lay else "DIFFER"}')

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
