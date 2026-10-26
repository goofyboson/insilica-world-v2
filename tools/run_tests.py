"""Test runner: runs the suite with pytest-xdist in up to 5 processes (tests that share a grid or a run stay
together through xdist groups) and prints every measured value against its threshold, in suite order. The
performance probe (PERF) runs only with --perf, alone after the suite, at every stage close.

  python tools/run_tests.py              the suite
  python tools/run_tests.py --perf       the suite, then PERF
  python tools/run_tests.py --jobs 3     fewer processes
  python tools/run_tests.py --cold       a fresh, empty cache directory (grids, v1 dump, spin-ups rebuilt)
  python tools/run_tests.py -- -k grid   extra pytest arguments after --
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'cache' / 'output' / 'results'
ORDER = ["GRID", "V1-GRID", "V1-ENGINE", "ACCURACY", "WAVES", "BAROCLINIC", "CONSERVE", "REPRO", "VIEWER", "PERF"]


def sort_key(row):
    rid = row['id']
    head = rid.split(' ')[0]
    num = rid.split('=')[-1] if '=' in rid else ''
    return (ORDER.index(head) if head in ORDER else len(ORDER), int(num) if num.isdigit() else 0, rid)


def environment(py, env):
    code = ('import cupy as cp, numpy as np, sys; p = cp.cuda.runtime.getDeviceProperties(0); '
            'print(f"{p[\'name\'].decode()} (cc {p[\'major\']}.{p[\'minor\']}), CuPy {cp.__version__}, '
            'CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}, NumPy {np.__version__}, Python {sys.version.split()[0]}")')
    r = subprocess.run([py, '-c', code], cwd=REPO, env=env, capture_output=True, text=True)
    return r.stdout.strip() or f'GPU environment check failed: {r.stderr.strip().splitlines()[-1:]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--perf', action='store_true')
    ap.add_argument('rest', nargs='*')
    a = ap.parse_args()
    if a.jobs > 5:
        sys.exit('at most 5 processes (CLAUDE.md: the CPU has 6 physical cores)')
    env = dict(os.environ, PYTHONIOENCODING='utf8', PYTHONWARNINGS='ignore')
    if a.cold:
        env['INSILICA_CACHE'] = tempfile.mkdtemp(prefix='insilica-cold-')
    for f in RESULTS.glob('*.jsonl'):
        f.unlink()
    py = sys.executable
    print('Environment:', environment(py, env))
    t0 = time.perf_counter()
    jobs = ['-n', str(a.jobs), '--dist', 'loadgroup'] if a.jobs > 1 else []
    r1 = subprocess.run([py, '-m', 'pytest', '-q', '-m', 'not perf', *jobs, *a.rest], cwd=REPO, env=env)
    t1 = time.perf_counter()
    r2 = subprocess.CompletedProcess([], 0)
    if a.perf:
        r2 = subprocess.run([py, '-m', 'pytest', '-q', '-m', 'perf', *a.rest], cwd=REPO, env=env)
    t2 = time.perf_counter()
    rows = []
    for f in RESULTS.glob('*.jsonl'):
        rows += [json.loads(l) for l in f.read_text(encoding='utf8').splitlines() if l.strip()]
    rows.sort(key=sort_key)
    passed = sum(r['passed'] for r in rows)
    print()
    print(f'{"test":<16} {"pass":<5} value  [threshold]')
    for r in rows:
        print(f'{r["id"]:<16} {"yes" if r["passed"] else "NO":<5} {r["value"]}  [{r["threshold"]}]')
    cache = 'empty cache' if a.cold else 'warm cache'
    perf = f', PERF {t2 - t1:.0f} s' if a.perf else ' (PERF not run; --perf)'
    print(f'\n{passed}/{len(rows)} results passed: suite {t1 - t0:.0f} s with {a.jobs} processes ({cache}){perf}.')
    sys.exit(0 if r1.returncode == 0 and r2.returncode == 0 else 1)


if __name__ == '__main__':
    main()
