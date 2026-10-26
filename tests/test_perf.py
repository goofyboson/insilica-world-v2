"""P1: performance probe. Marked `perf`: the runner (tools/run_tests.py) runs it alone after the parallel phase,
since other GPU tests running at the same time would distort its timings."""
import json

import pytest

from insilica import probe
from insilica.grid import CACHE_DIR

GATE_NS = 16.0      # n = 8, 64-bit: above this, stop and discuss (2× the plan's worst estimate of 8 ns)


@pytest.mark.perf
def test_P1_performance(record):
    """P1 Performance probe: engine ns per layer-edge per RK3 step (n = 4–8, L = 1/2/4, 64/32-bit) and each V2-0b change"""
    out = probe.run(levels=(4, 5, 6, 7, 8), layers=(1, 2, 4), log=lambda *a: None)
    s = probe.summary(out)
    (CACHE_DIR / 'output').mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / 'output' / 'probe.json').write_text(json.dumps(out, indent=1))
    f64 = {(r['n'], r['L']): r for r in out['runs'] if r['dtype'] == 'float64'}
    table = '; '.join(f'n={n}: ' + ', '.join(f'L{L} {f64[(n, L)]["ns_per_layer_edge"]:.2f}' for L in (1, 2, 4)) for n in (4, 5, 6, 7, 8))
    ratio = ', '.join(f'n={n} ×{v:.2f}' for n, v in sorted(s['ratio'].items()))
    base = {r['n']: r['ms_per_step'] for r in out['variants'] if r['variant'] == probe.VARIANTS[0][0]}
    steps = '; '.join(f'n={n}: V2-0 {base[n]:.3f} ms → ' + ', '.join(f'{v.split(",")[0].replace("fused + graph", "+graph")}'
                      f'{"" if "v1 order" in v else " " + v.split(", ")[1]} ×{s["speedup"][(n, v)]:.2f}' for v, *_ in probe.VARIANTS[1:])
                      for n in (4, 5, 6, 7, 8))
    ms8 = f64[(8, 2)]['ms_per_step']
    ok = s['n8_worst'] <= GATE_NS and all(r['finite'] for r in out['runs'] + out['variants'])
    record('P1', f'engine 64-bit ns per layer-edge-step: {table}. 64/32-bit time (L2): {ratio}. Launch {out["launch_us"]:.1f} µs. '
           f'L2 64-bit speed-ups over V2-0: {steps}. n = 8 two layers: {ms8:.2f} ms per RK3 step', f'report; gate n = 8 ≤ {GATE_NS:.0f} ns', ok)
    assert ok
