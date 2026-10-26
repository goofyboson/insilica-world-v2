"""The planet's mesh: icosahedral Voronoi cells (Lloyd/SCVT) with TRiSK C-grid geometry, built once per level n
and cached on disk (`cache/grids/`, keyed by n, the cell order and a hash of the generator code).

Cell order (SPEC §9 #19): grids are built in v1's numbering ('v1') and then, by default, renumbered along a
space-filling curve for memory locality ('hilbert' or 'morton'); a renumbered grid keeps `perm` (its cell k is
v1-order cell perm[k]). Edges follow by their lower cell, vertices by first appearance."""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from .icosa import build_grid, hilbert_order, morton_order, reorder
from .trisk import build_trisk

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
CACHE_DIR = Path(os.environ.get('INSILICA_CACHE', REPO / 'cache'))
GPU_LLOYD_FROM_N = 6          # n ≥ 6 run the Lloyd iteration on the GPU (CuPy); results differ only by rounding
ORDERS = {'hilbert': hilbert_order, 'morton': morton_order}
DEFAULT_ORDER = 'hilbert'
ICOSA_KEYS = ('n', 'N', 'pos', 'lat', 'lon', 'area', 'east', 'north', 'nbr', 'deg', 'dist', 'edgeLen', 'edgeDir', 'corners',
              'totalArea', 'lloydIters', 'lloydMoved')


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
        self.order = str(arrays.get('order', 'v1'))

    def arrays(self):
        return {k: getattr(self, k) for k in self._keys}

    def __repr__(self):
        return f'Grid(n={self.n}, N={self.N}, E={self.E}, NV={self.NV}, order={self.order})'


def make(n, xp=None, order=None):
    """Build level n from scratch (no cache), in v1's numbering or renumbered. Returns (Grid, seconds)."""
    order = DEFAULT_ORDER if order is None else order
    if xp is None:
        if n >= GPU_LLOYD_FROM_N:
            import cupy as xp
        else:
            xp = np
    t0 = time.perf_counter()
    g = build_grid(n, xp=xp)
    if order != 'v1':
        g = reorder({k: g[k] for k in ICOSA_KEYS}, ORDERS[order](g['pos']))
    g.update(build_trisk(g))
    g['order'] = order
    return Grid(g), time.perf_counter() - t0


_mem = {}


def get(n, use_cache=True, order=None):
    """Level n, from memory, the disk cache, or built (and then cached)."""
    order = DEFAULT_ORDER if order is None else order
    if (n, order) in _mem:
        return _mem[(n, order)]
    f = CACHE_DIR / 'grids' / f'n{n}-{order}-{generator_hash()}.npz'
    if use_cache and f.exists():
        with np.load(f) as z:
            g = Grid({k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files})
    else:
        g, _ = make(n, order=order)
        if use_cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(tmp, **g.arrays())
            os.replace(tmp, f)
    _mem[(n, order)] = g
    return g
