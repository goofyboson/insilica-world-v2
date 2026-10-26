"""P1: performance probe. Marked `perf`: the runner (tools/run_tests.py) runs it alone after the parallel phase,
since other GPU tests running at the same time would distort its timings."""
import json

import pytest

from insilica import probe
from insilica.grid import CACHE_DIR

GATE_NS = 16.0      # n = 8, 64-bit: above this, stop and discuss (2× the plan's worst estimate of 8 ns)


@pytest.mark.perf
def test_P1_performance(record):
    """P1 Performance probe: ns per layer-edge per RK3 step, n = 4–8, L = 1/2/4, 64- and 32-bit; launch overhead"""
    out = probe.run(levels=(4, 5, 6, 7, 8), layers=(1, 2, 4), log=lambda *a: None)
    s = probe.summary(out)
    (CACHE_DIR / 'output').mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / 'output' / 'probe.json').write_text(json.dumps(out, indent=1))
    f64 = {(r['n'], r['L']): r for r in out['runs'] if r['dtype'] == 'float64'}
    table = '; '.join(f'n={n}: ' + ', '.join(f'L{L} {f64[(n, L)]["ns_per_layer_edge"]:.2f}' for L in (1, 2, 4)) for n in (4, 5, 6, 7, 8))
    ratio = ', '.join(f'n={n} ×{v:.2f}' for n, v in sorted(s['ratio'].items()))
    mort = ', '.join(f'n={r["n"]} L{r["L"]} ×{r["speedup"]:.2f}' for r in out['morton'])
    ms8 = f64[(8, 2)]['ms_per_step']
    ok = s['n8_worst'] <= GATE_NS and all(r['finite'] for r in out['runs'])
    record('P1', f'64-bit ns per layer-edge-step: {table}. 64/32-bit time (L2): {ratio}. Launch {out["launch_us"]:.1f} µs. '
           f'Morton order: {mort}. n = 8 two layers: {ms8:.1f} ms per RK3 step', f'report; gate n = 8 ≤ {GATE_NS:.0f} ns', ok)
    assert ok
