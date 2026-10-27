"""PERF: performance probe. Marked `perf`: run with `tools/run_tests.py --perf` (at every stage close), alone after the
suite, since other GPU work at the same time would distort its timings."""
import json

import pytest

from insilica import probe
from insilica.grid import CACHE_DIR

GATE_NS = 16.0                                   # n = 8 slow step: 2× the plan's worst estimate of 8 ns
GATE_MODEL_HOUR = {'air': 0.030, 'ocean': 0.180}  # s per model hour: air with physics (#42), ocean the plan's estimate (§5)


@pytest.mark.perf
def test_perf(record):
    """PERF Engine speed: ns per layer-edge per slow step (n = 4–8, L = 1/2/4) and cost per model hour (air n = 6, ocean n = 8)"""
    out = probe.run(levels=(4, 5, 6, 7, 8), layers=(1, 2, 4), log=lambda *a: None)
    (CACHE_DIR / 'output').mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / 'output' / 'probe.json').write_text(json.dumps(out, indent=1))
    r = {(x['n'], x['L']): x for x in out['runs']}
    table = '; '.join(f'n={n}: ' + ', '.join(f'L{L} {r[(n, L)]["ns_per_layer_edge"]:.2f}' for L in (1, 2, 4)) for n in (4, 5, 6, 7, 8))
    worst8 = max(r[(8, L)]['ns_per_layer_edge'] for L in (1, 2, 4))
    show = '; '.join(f'{s["name"]} n={s["n"]} ({s["layers"]} layers, step {s["dt"]:.0f} s, {s["nfast"]} fast sub-steps): '
                     f'{s["s_per_model_hour"] * 1e3:.1f} ms per model hour ({s["model_years_per_day"]:.1f} model years per day)'
                     for s in out['showcase'])
    ok = (worst8 <= GATE_NS and all(s['s_per_model_hour'] <= GATE_MODEL_HOUR[s['name']] for s in out['showcase'])
          and all(x['finite'] for x in out['runs'] + out['showcase']))
    record('PERF', f'slow step, ns per layer-edge: {table}. Launch {out["launch_us"]:.1f} µs. {show}',
           f'n = 8 ≤ {GATE_NS:.0f} ns; air ≤ 30 ms, ocean ≤ 180 ms per model hour', ok)
    assert ok
