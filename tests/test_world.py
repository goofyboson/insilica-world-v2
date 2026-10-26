"""D1 (determinism) and S1 (save/load, spin-up cache)."""
import hashlib

import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import runcache
from insilica.gpu import dsum
from insilica.world import World

pytestmark = pytest.mark.xdist_group('world')


def test_D1_determinism(record):
    """D1 Determinism: two identical runs (Galewsky, n = 5, 200 steps) give identical state hashes and totals"""
    a = World(n=5, preset='galewsky').step(200)
    b = World(n=5, preset='galewsky').step(200)
    ha, hb = a.state_hash(), b.state_hash()
    x = cp.random.default_rng(1).random(10 ** 6)
    sums = {dsum(x) for _ in range(5)}
    ok = ha == hb and a.totals() == b.totals() and len(sums) == 1
    record('D1', f'hashes {"identical" if ha == hb else "DIFFER"} ({ha[:12]}); totals identical {a.totals() == b.totals()}; '
           f'5 tree sums of 1e6 values: {len(sums)} distinct', 'identical', ok)
    assert ok


def test_S1_save_load(record, tmp_path):
    """S1 Save/load: save → load → save identical bytes; save at T then K steps = T + K straight (bit-identical)"""
    T, K = 150, 150
    w = World(n=4, preset='galewsky', params={'nu4': 1e14, 'drag': 1e-6}).step(T)
    p1, p2 = tmp_path / 'a.iw2', tmp_path / 'b.iw2'
    w.save(p1)
    r = World.load(p1)
    r.save(p2)
    same_file = p1.read_bytes() == p2.read_bytes()
    w.step(K)
    r.step(K)
    same_run = w.state_hash() == r.state_hash() and w.tick == r.tick == T + K
    straight = World(n=4, preset='galewsky', params={'nu4': 1e14, 'drag': 1e-6}).step(T + K)
    same_straight = straight.state_hash() == r.state_hash()
    size = p1.stat().st_size
    ok = same_file and same_run and same_straight
    record('S1', f'file re-save {"identical" if same_file else "DIFFERS"} ({size / 1e6:.2f} MB); reloaded run {"identical" if same_run else "DIFFERS"}; '
           f'vs straight run {"identical" if same_straight else "DIFFERS"}', 'bit-identical', ok)
    assert ok


def test_S1_spinup_cache(record, tmp_path, monkeypatch):
    """S1 Spin-up cache: a cached spin-up equals the one computed afresh"""
    monkeypatch.setattr(runcache, 'CACHE_DIR', tmp_path)
    make = lambda: World(n=4, preset='w5')
    a, hit_a = runcache.spun_up('test-w5', make, 100)
    b, hit_b = runcache.spun_up('test-w5', make, 100)
    ok = (not hit_a) and hit_b and a.state_hash() == b.state_hash() and a.tick == b.tick == 100
    record('S1 spin-up cache', f'first {"hit" if hit_a else "miss"}, second {"hit" if hit_b else "miss"}; states identical {a.state_hash() == b.state_hash()}',
           'miss then hit; identical', ok)
    assert ok
