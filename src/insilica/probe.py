"""Performance probe (test P1): the cost of the ported TRiSK shallow-water RK3 step on this GPU.

Reports ns per layer-edge per RK3 step for n = 4–8, L = 1, 2, 4 layers, in 64-bit and 32-bit, the per-launch
overhead from Python, and the effect of renumbering cells along a Morton curve (memory locality). One RK3 step
here is 27 kernel launches (7 per stage, 6 updates); a single layer (L = 1) is also what one sub-step of the
split-explicit fast mode will cost.

  python -m insilica.probe [--levels 4 5 6 7 8] [--json out.json]
"""
import argparse
import json
import time

import cupy as cp
import numpy as np

from . import grid as gridmod
from .gpu import DeviceGrid, module
from .grid.icosa import morton_order, reorder
from .grid.trisk import build_trisk
from .swe import SWE
from .world import default_dt, initial_state

ICOSA_KEYS = ('n', 'N', 'pos', 'lat', 'lon', 'area', 'east', 'north', 'nbr', 'deg', 'dist', 'edgeLen', 'edgeDir', 'corners',
              'totalArea', 'lloydIters', 'lloydMoved')


def morton_grid(g):
    d = {k: getattr(g, k) for k in ICOSA_KEYS}
    d = reorder(d, morton_order(g.pos))
    d.update(build_trisk(d))
    return gridmod.Grid(d)


def time_steps(g, layers, dtype, steps=None):
    dg = DeviceGrid(g, dtype=dtype)
    h0, b, u0 = initial_state(g, 'w2')
    s = SWE(dg, layers=layers, b=b)
    h = cp.asarray(np.tile(h0, (layers, 1)), dtype=dtype)
    u = cp.asarray(np.tile(u0, (layers, 1)), dtype=dtype)
    dt = default_dt(g.n)
    for _ in range(3):
        s.step(h, u, dt)
    if steps is None:
        steps = max(10, min(200, int(2e8 // (layers * g.E))))
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Stream.null.synchronize()
    w0 = time.perf_counter()
    start.record()
    for _ in range(steps):
        s.step(h, u, dt)
    stop.record()
    stop.synchronize()
    wall = (time.perf_counter() - w0) / steps
    gpu = cp.cuda.get_elapsed_time(start, stop) / 1e3 / steps
    ok = bool(cp.all(cp.isfinite(h)))
    return dict(n=g.n, L=layers, dtype=np.dtype(dtype).name, steps=steps, ms_per_step=gpu * 1e3, wall_ms_per_step=wall * 1e3,
                ns_per_layer_edge=gpu / (layers * g.E) * 1e9, finite=ok)


def launch_overhead(k=2000):
    """Wall time per launch of an empty kernel from Python (µs), and GPU time per launch."""
    f = cp.RawKernel('extern "C" __global__ void nop(int n) {}', 'nop')
    f((1,), (32,), (np.int32(0),))
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(k):
        f((1,), (32,), (np.int32(0),))
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / k * 1e6


def run(levels=(4, 5, 6, 7, 8), layers=(1, 2, 4), log=print):
    out = dict(launch_us=launch_overhead(), runs=[], morton=[])
    log(f'launch overhead: {out["launch_us"]:.1f} µs per kernel launch (27 launches per RK3 step)')
    for n in levels:
        g = gridmod.get(n)
        for L in layers:
            for dt in (np.float64, np.float32):
                r = time_steps(g, L, dt)
                out['runs'].append(r)
                log(f'n={n} L={L} {r["dtype"]}: {r["ms_per_step"]:.3f} ms/step (wall {r["wall_ms_per_step"]:.3f}), '
                    f'{r["ns_per_layer_edge"]:.2f} ns per layer-edge-step')
        if n >= 6:
            gm = morton_grid(g)
            for L in (1, 2):
                r = time_steps(gm, L, np.float64)
                base = next(x for x in out['runs'] if x['n'] == n and x['L'] == L and x['dtype'] == 'float64')
                r['speedup'] = base['ns_per_layer_edge'] / r['ns_per_layer_edge']
                out['morton'].append(r)
                log(f'n={n} L={L} float64 Morton order: {r["ns_per_layer_edge"]:.2f} ns (×{r["speedup"]:.2f})')
    return out


def summary(out):
    """64/32-bit ratios at each n (L = 2) and the gate value (n = 8, 64-bit, worst L)."""
    rows = {}
    for r in out['runs']:
        rows[(r['n'], r['L'], r['dtype'])] = r['ns_per_layer_edge']
    ratio = {n: rows[(n, 2, 'float64')] / rows[(n, 2, 'float32')] for (n, L, d) in rows if L == 2 and d == 'float64'}
    n8 = [rows[k] for k in rows if k[0] == 8 and k[2] == 'float64']
    return dict(ratio=ratio, n8_worst=max(n8) if n8 else None, n8_best=min(n8) if n8 else None)


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--levels', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    ap.add_argument('--json')
    a = ap.parse_args()
    module('trisk.cu', 'double')
    out = run(a.levels)
    s = summary(out)
    print('64/32-bit time ratio (L = 2):', ', '.join(f'n={n} ×{v:.2f}' for n, v in sorted(s['ratio'].items())))
    if s['n8_worst'] is not None:
        print(f'n = 8, 64-bit: {s["n8_best"]:.2f}–{s["n8_worst"]:.2f} ns per layer-edge-step (gate: 16 ns)')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(out, summary={k: (v if not isinstance(v, dict) else {str(x): y for x, y in v.items()}) for k, v in s.items()}), f, indent=1)


if __name__ == '__main__':
    cli()
