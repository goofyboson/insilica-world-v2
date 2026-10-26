"""v1 reference data for tests G4 and K1: runs tools/v1_dump.mjs against the read-only v1 checkout next to this
repository (once, cached in cache/v1/<v1 commit>/) and maps v1's edge and vertex numbers onto V2's.

Cells are numbered identically. Edges and vertices are matched by their cells (an edge by its sorted cell pair,
a vertex by its sorted cell triple): where a neighbor sits exactly due west of a cell (angle ±π, which happens on
the icosahedron's symmetry lines), rounding decides which slot comes first in v1 and in V2, so slot numbers and
hence edge numbers can differ there while the mesh is the same.
"""
import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
V1 = REPO.parent / 'insilica-world'


def v1_available():
    return (V1 / 'insilica-world.html').exists() and shutil.which('node') is not None


@lru_cache(maxsize=None)
def dump_dir():
    commit = subprocess.run(['git', '-C', str(V1), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'nogit'
    script = (REPO / 'tools' / 'v1_dump.mjs').read_bytes()
    import hashlib
    from insilica.grid import CACHE_DIR
    d = CACHE_DIR / 'v1' / f'{commit[:12]}-{hashlib.sha1(script).hexdigest()[:8]}'
    if not (d / 'meta.json').exists():
        tmp = d.with_name(d.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(['node', str(REPO / 'tools' / 'v1_dump.mjs'), str(tmp), str(V1)], check=True)
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
    return d


def meta():
    return json.loads((dump_dir() / 'meta.json').read_text())


def load(sub, name):
    return np.load(dump_dir() / sub / f'{name}.npy')


@lru_cache(maxsize=None)
def maps(n, grid):
    """(edge_map, vertex_map, edge_sign, cell_map): V2 index → v1 index, matched by cells. A renumbered grid
    (grid.perm) maps its cell k to v1 cell perm[k]; an edge whose lower cell is not v1's lower cell points the
    other way, so its normal quantities (u, fluxes) flip sign: v2 = edge_sign · v1[edge_map]."""
    N = grid.N
    cm = grid.perm.astype(np.int64) if hasattr(grid, 'perm') else np.arange(N)
    v1I, v1J = load(f'n{n}', 'eI').astype(np.int64), load(f'n{n}', 'eJ').astype(np.int64)
    key1 = v1I * N + v1J
    o1 = np.argsort(key1)
    a, b = cm[grid.eI], cm[grid.eJ]
    key2 = np.minimum(a, b) * N + np.maximum(a, b)
    pos = np.searchsorted(key1[o1], key2)
    edge_map = o1[np.minimum(pos, len(o1) - 1)]
    assert np.array_equal(key1[edge_map], key2), 'edge sets differ'
    edge_sign = np.where(a < b, 1.0, -1.0)

    def vkeys(nbr, deg, cornerV, cmap):
        nbr, deg = nbr.astype(np.int64), deg.astype(np.int64)
        k6 = np.arange(6)[None, :]
        valid = k6 < deg[:, None]
        nxt = np.where(valid, (k6 + 1) % deg[:, None], k6)
        nb = np.where(valid, nbr, 0)
        cells = np.broadcast_to(np.arange(N)[:, None], (N, 6))
        tri = np.sort(cmap[np.stack([cells, nb, np.take_along_axis(nb, nxt, axis=1)], -1)], -1)
        key = (tri[..., 0] * N + tri[..., 1]) * N + tri[..., 2]
        NV = int(cornerV.max()) + 1
        out = np.zeros(NV, dtype=np.int64)
        out[cornerV[valid]] = key[valid]
        return out

    k1 = vkeys(load(f'n{n}', 'nbr'), load(f'n{n}', 'deg'), load(f'n{n}', 'cornerV').astype(np.int64), np.arange(N))
    k2 = vkeys(grid.nbr, grid.deg, grid.cornerV.astype(np.int64), cm)
    o1 = np.argsort(k1)
    vertex_map = o1[np.searchsorted(k1[o1], k2)]
    assert np.array_equal(k1[vertex_map], k2), 'vertex sets differ'
    return edge_map, vertex_map, edge_sign, cm


def rel(a, b):
    """max |a − b| / max |b| (0 if both are zero)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    s = np.max(np.abs(b)) if b.size else 0.0
    return float(np.max(np.abs(a - b)) / s) if s > 0 else float(np.max(np.abs(a - b), initial=0.0))
