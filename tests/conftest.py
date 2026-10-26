"""Shared test fixtures. Every test reports its measured value against its threshold with `record`; the runner
(tools/run_tests.py) prints them as one table in test-id order."""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'cache' / 'output' / 'results'
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def record(request):
    """record(test_id, value, threshold, passed): log one measured result (and fail the test if not passed)."""
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')

    def _record(tid, value, threshold, passed, title=None):
        OUT.mkdir(parents=True, exist_ok=True)
        row = dict(id=tid, title=title or (request.node.function.__doc__ or '').strip().split('\n')[0], value=value,
                   threshold=threshold, passed=bool(passed), node=request.node.nodeid)
        with open(OUT / f'{worker}.jsonl', 'a', encoding='utf8') as f:
            f.write(json.dumps(row) + '\n')
        return passed

    return _record


def fmt(x):
    return f'{x:.2e}' if isinstance(x, float) else str(x)
