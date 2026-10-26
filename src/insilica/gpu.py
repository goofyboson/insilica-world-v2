"""GPU plumbing: compiles the CUDA C kernels in `kernels/` at run time with NVRTC (through CuPy), uploads a
Grid to the device, and provides deterministic sums."""
import hashlib
from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

KERNEL_DIR = Path(__file__).resolve().parent / 'kernels'
BLOCK = 256
NULL = np.uint64(0)          # a null device pointer as a kernel argument
SUM_THREADS, SUM_CHUNK = 256, 8
# --fmad=false: no fused multiply-adds, so every product is rounded as in v1's JavaScript and as NumPy does.
NVRTC_OPTIONS = ('--std=c++17', '--fmad=false')


def kernel_source(name):
    return (KERNEL_DIR / name).read_text(encoding='utf8')


def source_hash():
    """Hash of all kernel and Python engine sources (keys the spin-up cache)."""
    h = hashlib.sha1()
    root = Path(__file__).resolve().parent
    for f in sorted(list(root.rglob('*.py')) + list(KERNEL_DIR.glob('*.cu'))):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


@lru_cache(maxsize=None)
def module(name, real='double'):
    """Compiled RawModule for kernels/<name>, with REAL = double (engine) or float (speed probe)."""
    return cp.RawModule(code=kernel_source(name), options=NVRTC_OPTIONS + (f'-DREAL={real}',))


def grid_dims(n, layers=1):
    return ((n + BLOCK - 1) // BLOCK, layers), (BLOCK,)


class DeviceGrid:
    """A Grid's arrays on the GPU: indices as int32/int8, geometry in REAL (float64 for the engine)."""

    INT32 = ('eI', 'eJ', 'nbr', 'cellEdge', 'cellEdgeAsc', 'ev1', 'ev2', 'vEdge', 'vCell', 'eoeStart', 'eoe')
    INT8 = ('cellSign', 'vEdgeSign', 'vSlot')
    REALS = ('le', 'de', 'areaCell', 'areaV', 'fV', 'kite', 'w')

    def __init__(self, grid, dtype=np.float64):
        self.host = grid
        self.n, self.N, self.E, self.NV = grid.n, grid.N, grid.E, grid.NV
        self.dtype = np.dtype(dtype)
        self.real = 'double' if self.dtype == np.float64 else 'float'
        for k in self.INT32:
            setattr(self, k, cp.asarray(np.ascontiguousarray(getattr(grid, k), dtype=np.int32)))
        for k in self.INT8:
            setattr(self, k, cp.asarray(np.ascontiguousarray(getattr(grid, k), dtype=np.int8)))
        for k in self.REALS:
            setattr(self, k, cp.asarray(np.ascontiguousarray(getattr(grid, k), dtype=self.dtype)))
        idx, wt = d2_stencil(grid)
        self.d2idx = cp.asarray(idx)
        self.d2wt = cp.asarray(wt.astype(self.dtype))

    def kernel(self, name, file='trisk.cu'):
        return module(file, self.real).get_function(name)


def d2_stencil(g):
    """v1's d2Stencil: per edge end, weights on four cells (c, j, o1, o2) for the second derivative along
    the line to the neighbor, scaled by d_e²/12 (3rd-order transport). Returns (idx int32 (E·8), wt (E·8))."""
    E = g.E
    nbr, deg, dist = g.nbr.astype(np.int64), g.deg.astype(np.int64), g.dist
    idx = np.zeros((E, 2, 4), dtype=np.int32)
    wt = np.zeros((E, 2, 4))
    for side, (c, k) in enumerate(((g.eI.astype(np.int64), g.eKi.astype(np.int64)), (g.eJ.astype(np.int64), g.eKj.astype(np.int64)))):
        m = deg[c]
        j = nbr[c, k]
        d1 = dist[c, k]
        hexa = m == 6
        ka = np.where(hexa, (k + 3) % 6, (k + 2) % m)
        kb = np.where(hexa, (k + 3) % 6, (k + 3) % m)
        o1, o2 = nbr[c, ka], nbr[c, kb]
        d2v = np.where(hexa, dist[c, (k + 3) % 6], 0.5 * (dist[c, ka] + dist[c, kb]))
        s = 2 / (d1 + d2v) * g.de * g.de / 12
        idx[:, side] = np.stack([c, j, o1, o2], axis=1)
        wt[:, side, 0] = -s * (1 / d1 + 1 / d2v)
        wt[:, side, 1] = s / d1
        wt[:, side, 2] = 0.5 * s / d2v
        wt[:, side, 3] = 0.5 * s / d2v
    return idx.reshape(-1), wt.reshape(-1)


def dsum(x, y=None):
    """Deterministic sum of x (or of x·y) on the GPU: a fixed-order tree reduction, bit-identical run to run.
    Returns a Python float."""
    x = cp.ascontiguousarray(x).ravel()
    real = 'double' if x.dtype == cp.float64 else 'float'
    k = module('trisk.cu', real).get_function('block_sum')
    if y is not None:
        y = cp.ascontiguousarray(y).ravel()
    while True:
        n = x.size
        blocks = (n + SUM_THREADS * SUM_CHUNK - 1) // SUM_THREADS // SUM_CHUNK
        out = cp.empty(blocks, dtype=x.dtype)
        k((blocks,), (SUM_THREADS,), (np.int64(n), x, y if y is not None else NULL, out))
        if blocks == 1:
            return float(out[0])
        x, y = out, None
