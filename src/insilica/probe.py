"""Performance probe (test PERF): what the layered engine costs on this GPU.

1. The slow step: ns per layer-edge per time-split RK3 step (one fast sub-step per stage), n = 4–8, L = 1, 2, 4,
   64-bit, CUDA graph, Hilbert-ordered cells.
2. Cost per model hour of whole world ticks (layers, fast sub-steps and tracer transport) for the plan's two
   showcase fluids: air at n = 6 (two layers, 10 km deep, the Phillips jet, 10-minute steps) and ocean at n = 8
   (two layers over a ridge, 1-hour steps with the external mode sub-stepped at ~160 m/s).

  python -m insilica.probe [--levels 4 5 6 7 8] [--json out.json]
"""
import argparse
import json
import time

import cupy as cp
import numpy as np

from . import grid as gridmod
from .gpu import DeviceGrid
from .layers import Layers
from .world import World, default_dt, initial_state

SHOWCASE = (('air', 6, 'phillips'), ('ocean', 8, 'outcrop'))


def time_steps(g, layers, steps=None):
    dg = DeviceGrid(g)
    h0, b, u0 = initial_state(g, 'w2')
    core = Layers(dg, layers, b=b, gprime=[0.1] * (layers - 1))
    h = cp.asarray(np.tile(h0 / layers, (layers, 1)))
    u = cp.asarray(np.tile(u0, (layers, 1)))
    dt = default_dt(g.n)
    stream = cp.cuda.Stream(non_blocking=False)
    with stream:
        core.step(h, u, dt, 1)
        stream.begin_capture()
        core.step(h, u, dt, 1)
        gr = stream.end_capture()
        if steps is None:
            steps = max(20, min(400, int(4e8 // (layers * g.E))))
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        stream.synchronize()
        start.record(stream)
        for _ in range(steps):
            gr.launch(stream)
        stop.record(stream)
        stop.synchronize()
    sec = cp.cuda.get_elapsed_time(start, stop) / 1e3 / steps
    return dict(n=g.n, L=layers, ms_per_step=sec * 1e3, ns_per_layer_edge=sec / (layers * g.E) * 1e9,
                finite=bool(cp.all(cp.isfinite(h))))


def model_hour(name, n, preset, hours=None):
    """Seconds of wall time per model hour for whole world ticks, measured over a few model hours."""
    w = World(n, preset)
    w.step(2)                                   # compile, capture the graph
    ticks = max(2, int(round((hours or (24 if n <= 6 else 6)) * 3600 / w.dt)))
    t0 = time.perf_counter()
    w.step(ticks)
    sec = (time.perf_counter() - t0) / (ticks * w.dt / 3600)
    return dict(name=name, n=n, preset=preset, layers=w.L, dt=w.dt, nfast=w.nfast, s_per_model_hour=sec,
                model_years_per_day=86400 / (sec * 24 * 365), finite=bool(cp.all(cp.isfinite(w.u))))


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
    out = dict(launch_us=launch_overhead(), runs=[], showcase=[])
    log(f'launch overhead: {out["launch_us"]:.1f} µs per kernel launch from Python')
    for n in levels:
        g = gridmod.get(n)
        for L in layers:
            r = time_steps(g, L)
            out['runs'].append(r)
            log(f'n={n} L={L}: {r["ms_per_step"]:.3f} ms per slow step, {r["ns_per_layer_edge"]:.2f} ns per layer-edge-step')
    for name, n, preset in SHOWCASE:
        if n in levels:
            r = model_hour(name, n, preset)
            out['showcase'].append(r)
            log(f'{name} n={n} ({preset}, {r["layers"]} layers, step {r["dt"]:.0f} s, {r["nfast"]} fast sub-steps): '
                f'{r["s_per_model_hour"] * 1e3:.1f} ms per model hour ({r["model_years_per_day"]:.1f} model years per day)')
    return out


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--levels', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    ap.add_argument('--json')
    a = ap.parse_args()
    out = run(a.levels)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    cli()
