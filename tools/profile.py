"""Where a world's time goes (V2-5): every kernel launch timed with CUDA events over one model day, grouped by part of the
model, against the step's real cost with its captured graph. Kernels launched through CuPy's own operations (copies, sums,
fills) and Python's launch overhead show as the difference between the direct run and the kernels' sum.

    python tools/profile.py earth 6            preset, n
    python tools/profile.py earth 6 --hours 6  a shorter window
"""
import argparse
import sys
import time
from collections import defaultdict

import cupy as cp

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from insilica.world import World  # noqa: E402

GROUPS = {'core': 'air dynamics', 'tr': 'transport', 'physics': 'air column physics', 'sea': 'ocean physics',
          'sea.core': 'ocean dynamics', 'sea.tr': 'ocean transport', 'routing': 'land water and ice'}


class Timed:
    """A kernel that records a CUDA event pair around every launch."""
    def __init__(self, k, label, log):
        self.k, self.label, self.log = k, label, log

    def __call__(self, *a, **kw):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        self.k(*a, **kw)
        e.record()
        self.log.append((self.label, s, e))


def instrument(w, log):
    for path in GROUPS:
        obj = w
        for part in path.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is None or not hasattr(obj, 'k'):
            continue
        for name, k in list(obj.k.items()):
            obj.k[name] = Timed(k, (GROUPS[path], name), log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset')
    ap.add_argument('n', type=int)
    ap.add_argument('--hours', type=float, default=24.0)
    a = ap.parse_args()
    w = World(a.n, a.preset)
    ticks = int(round(a.hours * 3600 / w.dt))
    w.step(ticks)                                          # warm: compile, capture graphs
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    w.step(ticks)
    cp.cuda.Device().synchronize()
    graph = (time.perf_counter() - t0) / ticks
    log = []
    instrument(w, log)
    t0 = time.perf_counter()
    w.step(ticks, graph=False)
    cp.cuda.Device().synchronize()
    direct = (time.perf_counter() - t0) / ticks
    per, launches = defaultdict(float), defaultdict(int)
    for label, s, e in log:
        per[label] += cp.cuda.get_elapsed_time(s, e) / 1e3 / ticks
        launches[label] += 1
    tot = sum(per.values())
    groups = defaultdict(float)
    for (g, _), t in per.items():
        groups[g] += t
    hour = 3600 / w.dt
    print(f'{a.preset} n = {a.n}: {w.grid.N} cells, {w.L} air layers, step {w.dt:.0f} s, {w.nfast} fast sub-steps; '
          f'{len(log) / ticks:.0f} kernel launches per step')
    print(f'per step: graph {graph * 1e3:.2f} ms, direct {direct * 1e3:.2f} ms, timed kernels {tot * 1e3:.2f} ms '
          f'(the rest of direct: CuPy operations and launch overhead {(direct - tot) * 1e3:.2f} ms)')
    print(f'per model hour (graph) {graph * hour * 1e3:.1f} ms = {w.dt / graph / 365:.0f} model years per day')
    print('by part (share of timed kernels):')
    for g, t in sorted(groups.items(), key=lambda x: -x[1]):
        print(f'  {g:22s} {t * 1e3:8.3f} ms  {t / tot * 100:5.1f}%')
    print('top kernels:')
    for (g, nm), t in sorted(per.items(), key=lambda x: -x[1])[:18]:
        print(f'  {g:22s} {nm:16s} {t * 1e3:8.3f} ms  {t / tot * 100:5.1f}%  {launches[(g, nm)] / ticks:6.1f} launches/step')


if __name__ == '__main__':
    main()
