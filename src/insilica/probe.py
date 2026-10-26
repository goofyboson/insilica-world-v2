"""Performance probe (test PERF): the cost of the TRiSK shallow-water RK3 step on this GPU.

Engine configuration (fused kernels, one CUDA graph per step, Hilbert-ordered cells) at n = 4–8 for L = 1, 2, 4
layers in 64-bit and 32-bit, in ns per layer-edge per RK3 step; and, at L = 2 in 64-bit, what the CUDA graph and
the cell numbering each give: direct launches on v1's numbering, plus the graph, plus Hilbert or Morton numbering. A single layer (L = 1) is also what one split-explicit fast-mode sub-step will cost.

  python -m insilica.probe [--levels 4 5 6 7 8] [--json out.json]
"""
import argparse
import json
import time

import cupy as cp
import numpy as np

from . import grid as gridmod
from .gpu import DeviceGrid
from .swe import SWE
from .world import default_dt, initial_state

VARIANTS = (('direct, v1 order', 'v1', SWE, False),
            ('graph, v1 order', 'v1', SWE, True),
            ('graph, morton', 'morton', SWE, True),
            ('graph, hilbert', 'hilbert', SWE, True))


def time_steps(g, layers, dtype, cls=SWE, graph=True, steps=None):
    dg = DeviceGrid(g, dtype=dtype)
    h0, b, u0 = initial_state(g, 'w2')
    s = cls(dg, layers=layers, b=b)
    h = cp.asarray(np.tile(h0, (layers, 1)), dtype=dtype)
    u = cp.asarray(np.tile(u0, (layers, 1)), dtype=dtype)
    dt = default_dt(g.n)
    stream = cp.cuda.Stream(non_blocking=False)
    with stream:
        for _ in range(3):
            s.step(h, u, dt)
        run = lambda: s.step(h, u, dt)
        if graph:
            stream.begin_capture()
            s.step(h, u, dt)
            gr = stream.end_capture()
            run = lambda: gr.launch(stream)
        if steps is None:
            steps = max(20, min(400, int(4e8 // (layers * g.E))))
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        stream.synchronize()
        w0 = time.perf_counter()
        start.record(stream)
        for _ in range(steps):
            run()
        stop.record(stream)
        stop.synchronize()
    wall = (time.perf_counter() - w0) / steps
    gpu = cp.cuda.get_elapsed_time(start, stop) / 1e3 / steps
    return dict(n=g.n, L=layers, dtype=np.dtype(dtype).name, order=g.order, steps=steps, ms_per_step=gpu * 1e3,
                wall_ms_per_step=wall * 1e3, ns_per_layer_edge=gpu / (layers * g.E) * 1e9, finite=bool(cp.all(cp.isfinite(h))))


def launch_overhead(k=2000):
    """Wall time per launch of an empty kernel from Python (µs)."""
    f = cp.RawKernel('extern "C" __global__ void nop(int n) {}', 'nop')
    f((1,), (32,), (np.int32(0),))
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(k):
        f((1,), (32,), (np.int32(0),))
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / k * 1e6


def run(levels=(4, 5, 6, 7, 8), layers=(1, 2, 4), log=print):
    out = dict(launch_us=launch_overhead(), runs=[], variants=[])
    log(f'launch overhead: {out["launch_us"]:.1f} µs per kernel launch from Python')
    for n in levels:
        for name, order, cls, graph in VARIANTS:
            r = time_steps(gridmod.get(n, order=order), 2, np.float64, cls, graph)
            r['variant'] = name
            out['variants'].append(r)
            log(f'n={n} L=2 64-bit {name}: {r["ms_per_step"]:.3f} ms/step, {r["ns_per_layer_edge"]:.2f} ns per layer-edge-step')
        g = gridmod.get(n)
        for L in layers:
            for dt in (np.float64, np.float32):
                r = time_steps(g, L, dt)
                out['runs'].append(r)
                log(f'n={n} L={L} {r["dtype"]} engine: {r["ms_per_step"]:.3f} ms/step, {r["ns_per_layer_edge"]:.2f} ns per layer-edge-step')
    return out


def summary(out):
    """64/32-bit ratios (L = 2), the gate value (n = 8, 64-bit, worst L) and each variant's speed-up over direct
    launches on v1's numbering."""
    rows = {(r['n'], r['L'], r['dtype']): r['ns_per_layer_edge'] for r in out['runs']}
    ratio = {n: rows[(n, 2, 'float64')] / rows[(n, 2, 'float32')] for (n, L, d) in rows if L == 2 and d == 'float64'}
    n8 = [rows[k] for k in rows if k[0] == 8 and k[2] == 'float64']
    speedup = {}
    for r in out['variants']:
        base = next(x for x in out['variants'] if x['n'] == r['n'] and x['variant'] == VARIANTS[0][0])
        speedup[(r['n'], r['variant'])] = base['ms_per_step'] / r['ms_per_step']
    return dict(ratio=ratio, n8_worst=max(n8) if n8 else None, n8_best=min(n8) if n8 else None, speedup=speedup)


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--levels', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    ap.add_argument('--json')
    a = ap.parse_args()
    out = run(a.levels)
    s = summary(out)
    print('64/32-bit time ratio (L = 2):', ', '.join(f'n={n} ×{v:.2f}' for n, v in sorted(s['ratio'].items())))
    for n in a.levels:
        print(f'n={n} speed-up over direct launches, v1 order:', ', '.join(f'{v} ×{s["speedup"][(n, v)]:.2f}' for v, *_ in VARIANTS[1:]))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    cli()
