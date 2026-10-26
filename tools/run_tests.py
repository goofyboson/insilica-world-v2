"""Test runner (V2's counterpart of v1's run-parallel.mjs): runs the suite with pytest-xdist in up to 5 processes
(tests that share a run or a big grid stay together through xdist groups), then the performance probe alone, and
prints every measured value against its threshold in test-id order.

  python tools/run_tests.py              all tests
  python tools/run_tests.py --jobs 3     fewer processes
  python tools/run_tests.py --cold       a fresh, empty cache directory (grids, v1 dump, spin-ups rebuilt)
  python tools/run_tests.py --no-perf    skip P1
  python tools/run_tests.py -- -k G2     extra pytest arguments after --
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
ORDER = ['F0', 'G1', 'G2', 'G3', 'G4', 'S1 grid', 'K1', 'K2', 'C1', 'D1', 'S1', 'W1', 'P1', 'R1']


def sort_key(row):
    rid = row['id']
    best = max((p for p in ORDER if rid.startswith(p)), key=len, default=None)
    num = ''.join(c for c in rid.split('=')[-1] if c.isdigit())
    return (ORDER.index(best) if best else len(ORDER), rid.split(' ')[0], int(num) if num and '=' in rid else 0, rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--no-perf', action='store_true')
    ap.add_argument('rest', nargs='*')
    a = ap.parse_args()
    if a.jobs > 5:
        sys.exit('at most 5 processes (CLAUDE.md: the CPU has 6 physical cores)')
    env = dict(os.environ, PYTHONIOENCODING='utf8')
    if a.cold:
        env['INSILICA_CACHE'] = tempfile.mkdtemp(prefix='insilica-cold-')
    for f in RESULTS.glob('*.jsonl'):
        f.unlink()
    py = sys.executable
    t0 = time.perf_counter()
    jobs = ['-n', str(a.jobs), '--dist', 'loadgroup'] if a.jobs > 1 else []
    r1 = subprocess.run([py, '-m', 'pytest', '-q', '-m', 'not perf', *jobs, *a.rest], cwd=REPO, env=env)
    t1 = time.perf_counter()
    r2 = subprocess.CompletedProcess([], 0)
    if not a.no_perf:
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
    cache = 'cold (empty cache)' if a.cold else 'warm cache'
    print(f'\nR1: {passed}/{len(rows)} results passed; {t1 - t0:.0f} s for the suite with {a.jobs} processes ({cache}), '
          f'{t2 - t1:.0f} s for P1 alone; {t2 - t0:.0f} s in all.')
    sys.exit(0 if r1.returncode == 0 and r2.returncode == 0 else 1)


if __name__ == '__main__':
    main()
