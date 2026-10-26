"""The planet's mesh: icosahedral Voronoi cells (Lloyd/SCVT) with TRiSK C-grid geometry, built once per level n
and cached on disk (`cache/grids/`, keyed by n, the cell order, the smoothing rule and GRID_VERSION).

GRID_VERSION is bumped by hand whenever a change alters the grids' numbers (SPEC §9 #31); comments and refactors
that leave the numbers alone keep the cache. Test GRID rebuilds n = 4 from scratch and requires it to equal the
cached copy bit for bit, which catches a forgotten bump.

Smoothing (SPEC §9 #30): the engine's grids iterate Lloyd to its tolerance (lloyd='converged'); lloyd='v1' keeps v1's
400-iteration rule for the grids compared with v1's own.

Cell order (SPEC §9 #19): grids are built in v1's numbering ('v1') and then, by default, renumbered along a
space-filling curve for memory locality ('hilbert' or 'morton'); a renumbered grid keeps `perm` (its cell k is
v1-order cell perm[k]). Edges follow by their lower cell, vertices by first appearance."""
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
LLOYD = {'converged': 200000, 'v1': 400}        # iteration caps; both stop at the 1e-11 rad tolerance
DEFAULT_ORDER = 'hilbert'
ICOSA_KEYS = ('n', 'N', 'pos', 'lat', 'lon', 'area', 'east', 'north', 'nbr', 'deg', 'dist', 'edgeLen', 'edgeDir', 'corners',
              'totalArea', 'lloydIters', 'lloydMoved')


GRID_VERSION = 1


def generator_hash():
    """The grids' cache key and save-file stamp: the hand-maintained GRID_VERSION (not a hash of the source)."""
    return f'g{GRID_VERSION}'


class Grid:
    """Host (NumPy) copy of one mesh level. Attributes are v1's array names (pos, area, nbr, eI, …, plus the
    TRiSK arrays cellEdge, ev1, le, w, …); see icosa.py and trisk.py."""

    def __init__(self, arrays):
        self._keys = sorted(arrays)
        for k, v in arrays.items():
            setattr(self, k, v)
        self.n, self.N, self.E, self.NV = int(self.n), int(self.N), int(self.E), int(self.NV)
        self.order = str(arrays.get('order', 'v1'))
        self.lloyd = str(arrays.get('lloyd', 'v1'))

    def arrays(self):
        return {k: getattr(self, k) for k in self._keys}

    def __repr__(self):
        return f'Grid(n={self.n}, N={self.N}, E={self.E}, NV={self.NV}, order={self.order})'


def make(n, xp=None, order=None, lloyd='converged'):
    """Build level n from scratch (no cache), in v1's numbering or renumbered. Returns (Grid, seconds). A renumbered
    grid starts from the (cached) v1-numbered grid with the same smoothing, so Lloyd runs once per level."""
    order = DEFAULT_ORDER if order is None else order
    t0 = time.perf_counter()
    if order != 'v1':
        base = get(n, order='v1', lloyd=lloyd)
        g = reorder({k: getattr(base, k) for k in ICOSA_KEYS}, ORDERS[order](base.pos))
    else:
        if xp is None:
            if n >= GPU_LLOYD_FROM_N:
                import cupy as xp
            else:
                xp = np
        g = build_grid(n, xp=xp, max_iters=LLOYD[lloyd])
    g.update(build_trisk(g))
    g['order'], g['lloyd'] = order, lloyd
    return Grid(g), time.perf_counter() - t0


_mem = {}


def get(n, use_cache=True, order=None, lloyd='converged'):
    """Level n, from memory, the disk cache, or built (and then cached)."""
    order = DEFAULT_ORDER if order is None else order
    key = (n, order, lloyd)
    if key in _mem:
        return _mem[key]
    f = CACHE_DIR / 'grids' / f'n{n}-{order}-{lloyd}-{generator_hash()}.npz'
    if use_cache and f.exists():
        with np.load(f) as z:
            g = Grid({k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files})
    else:
        g, _ = make(n, order=order, lloyd=lloyd)
        if use_cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f'.{os.getpid()}.tmp.npz')
            np.savez(tmp, **g.arrays())
            os.replace(tmp, f)
    _mem[key] = g
    return g
