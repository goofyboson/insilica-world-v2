"""The planet's mesh: icosahedral Voronoi cells (Lloyd/SCVT) with TRiSK C-grid geometry, built once per level n
and cached on disk (`cache/grids/`, keyed by n and a hash of the generator code)."""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .icosa import build_grid
from .trisk import build_trisk

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
CACHE_DIR = Path(os.environ.get('INSILICA_CACHE', REPO / 'cache'))
GPU_LLOYD_FROM_N = 6          # n ≥ 6 run the Lloyd iteration on the GPU (CuPy); results differ only by rounding


def generator_hash():
    h = hashlib.sha1()
    for f in ('icosa.py', 'trisk.py', '__init__.py'):
        h.update((_HERE / f).read_bytes())
    h.update(str(GPU_LLOYD_FROM_N).encode())
    return h.hexdigest()[:12]


class Grid:
    """Host (NumPy) copy of one mesh level. Attributes are v1's array names (pos, area, nbr, eI, …, plus the
    TRiSK arrays cellEdge, ev1, le, w, …); see icosa.py and trisk.py."""

    def __init__(self, arrays):
        self._keys = sorted(arrays)
        for k, v in arrays.items():
            setattr(self, k, v)
        self.n, self.N, self.E, self.NV = int(self.n), int(self.N), int(self.E), int(self.NV)

    def arrays(self):
        return {k: getattr(self, k) for k in self._keys}

    def __repr__(self):
        return f'Grid(n={self.n}, N={self.N}, E={self.E}, NV={self.NV})'


def make(n, xp=None):
    """Build level n from scratch (no cache). Returns (Grid, seconds)."""
    if xp is None:
        if n >= GPU_LLOYD_FROM_N:
            import cupy as xp
        else:
            xp = np
    t0 = time.perf_counter()
    g = build_grid(n, xp=xp)
    g.update(build_trisk(g))
    return Grid(g), time.perf_counter() - t0


_mem = {}


def get(n, use_cache=True):
    """Level n, from memory, the disk cache, or built (and then cached)."""
    if n in _mem:
        return _mem[n]
    f = CACHE_DIR / 'grids' / f'n{n}-{generator_hash()}.npz'
    if use_cache and f.exists():
        with np.load(f) as z:
            g = Grid({k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files})
    else:
        g, _ = make(n)
        if use_cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(tmp, **g.arrays())
            os.replace(tmp, f)
    _mem[n] = g
    return g
