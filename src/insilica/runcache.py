"""Spin-up cache (carried over from v1 SPEC §18 #52): a world after its spin-up phase is saved in
`cache/runs/`, keyed by a label, the tick count and a hash of every engine source file, and loaded instead of
re-run. Only spin-ups are cached, never the ticks a test measures. INSILICA_NO_CACHE=1 bypasses it."""
import os

from .grid import CACHE_DIR
from .gpu import source_hash
from .world import World


def spun_up(label, make, ticks):
    """make() → a fresh World; returns it after `ticks` steps, from the cache when possible."""
    f = CACHE_DIR / 'runs' / f'{label}-{ticks}-{source_hash()}.iw2'
    if not os.environ.get('INSILICA_NO_CACHE') and f.exists():
        return World.load(f), True
    w = make().step(ticks)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f'{f.name}.{os.getpid()}.tmp')
    w.save(tmp)
    os.replace(tmp, f)
    return w, False
