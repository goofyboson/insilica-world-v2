"""V1-GRID and V1-ENGINE: V2 against v1's own JavaScript results (dumped from the read-only v1 checkout)."""
import cupy as cp
import numpy as np
import pytest

import v1ref
from conftest import fmt
from insilica import grid as gridmod
from insilica.gpu import DeviceGrid
from insilica.swe import SWE
from insilica.transport import Transport

pytestmark = [pytest.mark.skipif(not v1ref.v1_available(), reason='v1 checkout or node not found'), pytest.mark.xdist_group('v1')]

CELL_REAL = ('pos', 'lat', 'area', 'east', 'north')
SLOT_REAL = ('dist', 'edgeLen', 'edgeDir', 'corners', 'kite')     # per (cell, slot): compared through the slot rotation
EDGE_REAL = ('de', 'le', 'xe', 'ne')
VERT_REAL = ('areaV', 'fV')
CELL_TRISK = ('areaCell',)


def slot_maps(n, g):
    """For each cell, V2 slot k → v1 slot (the same neighbor)."""
    nb1 = v1ref.load(f'n{n}', 'nbr').astype(np.int64)
    nb2 = g.nbr.astype(np.int64)
    eq = nb2[:, :, None] == nb1[:, None, :]
    eq &= nb2[:, :, None] >= 0
    has = eq.any(axis=2)
    return np.where(has, np.argmax(eq, axis=2), np.arange(6)[None, :]), has


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_v1_grid(record, n):
    """V1-GRID The grid (built in v1's numbering) is v1's mesh: same mesh; geometry within 1e-12 × 4^(n−4)"""
    g = gridmod.get(n, order='v1')
    sub = f'n{n}'
    em, vm, _, _ = v1ref.maps(n, g)                   # asserts identical edge and vertex sets
    sm, has = slot_maps(n, g)
    deg_same = np.array_equal(g.deg, v1ref.load(sub, 'deg'))
    # neighbor lists: identical as cycles; count cells whose slot numbering starts elsewhere
    nb1 = v1ref.load(sub, 'nbr')
    rotated = int(np.sum(np.any(g.nbr != nb1, axis=1)))
    cyc_ok = bool(np.all(has == (g.nbr >= 0)))
    for i in np.nonzero(np.any(g.nbr != nb1, axis=1))[0]:
        m = g.deg[i]
        a, b = list(g.nbr[i, :m]), list(nb1[i, :m])
        cyc_ok &= any(a == b[r:] + b[:r] for r in range(m))
    worst = {}
    for k in CELL_REAL:
        a, b = getattr(g, k), v1ref.load(sub, k).reshape(getattr(g, k).shape)
        worst[k] = v1ref.rel(a, b)
    lon1 = v1ref.load(sub, 'lon')
    worst['lon'] = float(np.max(np.abs(np.angle(np.exp(1j * (g.lon - lon1))))) / np.pi)      # modulo 2π
    rows = np.arange(g.N)[:, None]
    for k in SLOT_REAL:
        a = getattr(g, k)
        b = v1ref.load(sub, k).reshape(a.shape)[rows, sm]
        mask = g.nbr >= 0
        worst[k] = v1ref.rel(a[mask], b[mask])
    for k in EDGE_REAL:
        a = getattr(g, k)
        worst[k] = v1ref.rel(a, v1ref.load(sub, k).reshape(-1, *a.shape[1:])[em])
    for k in VERT_REAL:
        worst[k] = v1ref.rel(getattr(g, k), v1ref.load(sub, k)[vm])
    for k in CELL_TRISK:
        worst[k] = v1ref.rel(getattr(g, k), v1ref.load(sub, k))
    # tangential weights: for each edge the same (e', w) sequence once edges are mapped
    s1, e1, w1 = v1ref.load(sub, 'eoeStart'), v1ref.load(sub, 'eoe'), v1ref.load(sub, 'w')
    inv = np.empty_like(em)
    inv[em] = np.arange(g.E)
    eoe_same, wmax = True, 0.0
    cnt2, cnt1 = np.diff(g.eoeStart), np.diff(s1)[em]
    eoe_same &= np.array_equal(cnt2, cnt1)
    if eoe_same:
        idx1 = np.concatenate([np.arange(s1[e], s1[e + 1]) for e in em])
        eoe_same &= np.array_equal(g.eoe, inv[e1[idx1]])
        wmax = v1ref.rel(g.w, w1[idx1])
    worst['w'] = wmax
    # circulation signs on mapped edges
    sign_same = np.array_equal(g.circ1, v1ref.load(sub, 'circ1')[em]) and np.array_equal(g.circ2, v1ref.load(sub, 'circ2')[em])
    topo_ok = deg_same and cyc_ok and eoe_same and sign_same
    geo = max(worst.values())
    # v1's small-triangle rounding grows 4× per level (SPEC §9 #15, #16), so the bound does too.
    limit = 1e-12 * 4.0 ** (n - 4)
    ok = topo_ok and geo <= limit
    top = ', '.join(f'{k} {fmt(v)}' for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:4])
    record(f'V1-GRID n={n}', f'mesh {"identical" if topo_ok else "DIFFERS"} (edges, vertices, cycles, weight lists); '
           f'{rotated} of {g.N} cells start their slot cycle at another neighbor; geometry max {fmt(geo)} (largest: {top})',
           f'same mesh; geometry ≤ 1e-12·4^(n−4) = {limit:.2e}', ok)
    assert ok


def test_v1_engine(record):
    """V1-ENGINE The engine (fused kernels, CUDA graph, Hilbert-ordered grid) reproduces v1's shallow water and transport"""
    # Shallow water: Williamson 2 at n = 4 (450 s), and a steep mountain with ∇⁴, drag and the positivity limiter.
    # Transport: v1's upwind scheme on its cosine bell (the engine's 3rd-order flux differs from v1's on purpose,
    # SPEC §9 #20, and is checked by ACCURACY and REPRO). Arrays map through the cell permutation; normal
    # quantities flip sign where an edge points the other way.
    g = gridmod.get(4)
    dg = DeviceGrid(g)
    em, _, sg, cm = v1ref.maps(4, g)
    cell = lambda a: a[cm]
    edge = lambda a: sg * a[em]
    L = lambda k: v1ref.load('w2', k)
    stream = cp.cuda.Stream(non_blocking=False)

    def graphed(step, k):
        """Run step() k times as the engine does: the first directly, then as a replayed CUDA graph."""
        with stream:
            step()
            if k > 1:
                stream.begin_capture()
                step()
                gr = stream.end_capture()
                for _ in range(k - 1):
                    gr.launch(stream)
        stream.synchronize()

    h = cp.asarray(cell(L('h0')))[None].copy()
    u = cp.asarray(edge(L('u0')))[None].copy()
    s = SWE(dg)
    Fm = cp.zeros_like(u)
    graphed(lambda: s.step(h, u, 450.0, Fm), 1)
    r1 = max(v1ref.rel(h.get()[0], cell(L('h1'))), v1ref.rel(u.get()[0], edge(L('u1'))))
    rF = v1ref.rel(Fm.get()[0], edge(L('Fm1')))
    graphed(lambda: s.step(h, u, 450.0), 959)
    r960 = max(v1ref.rel(h.get()[0], cell(L('h960'))), v1ref.rel(u.get()[0], edge(L('u960'))))
    s2 = SWE(dg, b=cell(L('b2')), drag=1e-5, nu4=1e15)
    h = cp.asarray(cell(L('h2_0')))[None].copy()
    u = cp.asarray(edge(L('u2_0')))[None].copy()
    limited = 0
    for _ in range(10):
        graphed(lambda: s2.step(h, u, 450.0), 1)
        limited = max(limited, int(cp.sum(s2.scale < 1)))
    r10 = max(v1ref.rel(h.get()[0], cell(L('h2_10'))), v1ref.rel(u.get()[0], edge(L('u2_10'))))
    B = lambda k: v1ref.load('bell1', k)
    psi = cp.asarray(cell(B('psi0')))[None].copy()
    Fb = cp.asarray(edge(B('Fm')))[None].copy()
    M, M1 = cp.asarray(g.areaCell)[None].copy(), cp.zeros((1, g.N))
    tr = Transport(dg)

    def tstep():
        tr.mass_after(M, Fb, M1)
        tr.step(M, M1, Fb, psi, 1)
        cp.copyto(M, M1)

    graphed(tstep, 1)
    t1 = v1ref.rel(psi.get()[0], cell(B('psi1')))
    graphed(tstep, 575)
    t576 = v1ref.rel(psi.get()[0], cell(B('psi576')))
    ok = r1 <= 1e-12 and rF <= 1e-12 and r960 <= 1e-9 and r10 <= 1e-12 and t1 <= 1e-12 and t576 <= 1e-9 and limited > 0
    record('V1-ENGINE', f'{g.order} order, fused, graph: SWE 1 step {fmt(r1)} (flux {fmt(rF)}), 960 steps {fmt(r960)}; '
           f'mountain + ∇⁴ + drag + limiter (active in {limited} cells) 10 steps {fmt(r10)}; upwind transport 1 step {fmt(t1)}, '
           f'576 steps {fmt(t576)}', '1 step ≤ 1e-12; 576–960 steps ≤ 1e-9; limiter exercised', ok)
    assert ok
