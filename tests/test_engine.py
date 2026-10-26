"""E2–E4 (stage V2-0b): fused kernels and CUDA graphs against the one-loop-per-kernel reference, and the
locality-renumbered grid against v1's numbering."""
import cupy as cp
import numpy as np
import pytest

from conftest import fmt
from insilica import grid as gridmod
from insilica.gpu import DeviceGrid
from insilica.reference import SWEReference, TransportReference
from insilica.swe import SWE
from insilica.transport import Transport
from insilica.world import World, cosine_bell, initial_state

pytestmark = pytest.mark.xdist_group('engine')


def mountain_case(g):
    """A harsh case that exercises every optional branch: a steep mountain under a thin layer (positivity
    limiter), ∇⁴ hyperviscosity, drag, and three layers."""
    h, b, u = initial_state(g, 'w5')
    b2 = np.maximum(0, 5000 * (1 - np.arccos(np.clip(g.pos[:, 2] * 0.5 + g.pos[:, 0] * np.sqrt(0.75), -1, 1)) / 0.5))
    h2 = np.maximum(1.0, 5100 - b2 - 600 * g.pos[:, 2] ** 2)
    H = np.stack([h2, np.maximum(1.0, 5300 - b2 - 300 * g.pos[:, 2] ** 2), np.maximum(1.0, 5150 - b2 - 900 * g.pos[:, 2] ** 2)])
    U = np.stack([u, 3 * u, -u])
    return H, b2, U


def same(a, b):
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def test_E2_fused_matches_reference(record):
    """E2 Fused kernels and the CUDA-graph tick are bit-identical to the reference kernels"""
    g = gridmod.get(4)
    dg = DeviceGrid(g)
    swe_ok, finite, limited = True, True, 0
    # (a) the harsh mountain case for 30 steps (it is not a balanced state and breaks down by about step 50 in
    # either implementation); (b) a smooth three-layer Galewsky case for 100 steps
    hg, bg, ug = initial_state(g, 'galewsky')
    cases = [(mountain_case(g), 30), ((np.stack([hg, 0.6 * hg, 1.3 * hg]), bg, np.stack([ug, -0.5 * ug, 1.2 * ug])), 100)]
    for (H, b, U), steps in cases:
        kw = dict(layers=3, b=b, drag=[1e-5, 0.0, 2e-6], nu4=1e15)
        sf, sr = SWE(dg, **kw), SWEReference(dg, **kw)
        hf, uf = cp.asarray(H), cp.asarray(U)
        hr, ur = hf.copy(), uf.copy()
        Ff, Fr = cp.zeros_like(uf), cp.zeros_like(ur)
        for _ in range(steps):
            sf.step(hf, uf, 450.0, Ff)
            sr.step(hr, ur, 450.0, Fr)
            limited = max(limited, int(cp.sum(sf.scale < 1)))
        swe_ok &= same(hf.get(), hr.get()) and same(uf.get(), ur.get()) and same(Ff.get(), Fr.get())
        finite &= bool(cp.all(cp.isfinite(hf))) and bool(cp.all(cp.isfinite(uf)))
    # transport, both orders, with the fluxes of the run above
    tr_ok = True
    for order in (1, 3):
        tf, trf = Transport(dg, layers=3), TransportReference(dg, layers=3)
        psi = cp.asarray(np.stack([cosine_bell(g)] * 3))
        pf, pr = psi.copy(), psi.copy()
        Mf = cp.asarray(H) * cp.asarray(g.areaCell)[None, :]
        Mr, M1f, M1r = Mf.copy(), cp.zeros_like(Mf), cp.zeros_like(Mf)
        Fm = Ff * 0.1
        for _ in range(100):
            tf.mass_after(Mf, Fm, M1f)
            trf.mass_after(Mr, Fm, M1r)
            tf.step(Mf, M1f, Fm, pf, order)
            trf.step(Mr, M1r, Fm, pr, order)
            Mf, M1f = M1f, Mf
            Mr, M1r = M1r, Mr
        tr_ok &= same(pf.get(), pr.get()) and same(Mf.get(), Mr.get())
    # the graph-captured world tick against direct ticks
    a = World(n=4, preset='galewsky', params={'nu4': 1e14, 'drag': 1e-6}).step(50, graph=True)
    b_ = World(n=4, preset='galewsky', params={'nu4': 1e14, 'drag': 1e-6}).step(50, graph=False)
    graph_ok = a.state_hash() == b_.state_hash() and a._graph is not None
    ok = swe_ok and tr_ok and graph_ok and finite
    record('E2', f'SWE 3 layers (mountain 30 steps with the limiter active in up to {limited} cells; Galewsky 100 steps; ∇⁴, drag) {"identical" if swe_ok else "DIFFERS"}{"" if finite else ", NOT FINITE"}; '
           f'transport upwind + FCT {"identical" if tr_ok else "DIFFERS"}; graph tick vs direct {"identical" if graph_ok else "DIFFERS"}',
           'bit-identical', ok)
    assert ok


@pytest.mark.parametrize('n', [4, 5])
def test_E3_renumbered_same_mesh(record, n):
    """E3 The locality-renumbered grid is the same mesh as v1's numbering"""
    gr, gv = gridmod.get(n), gridmod.get(n, order='v1')
    p = gr.perm.astype(np.int64)
    inv = np.empty_like(p)
    inv[p] = np.arange(gr.N)
    # edges: by cell pair; vertices: by cell triple
    kv = np.minimum(gv.eI, gv.eJ).astype(np.int64) * gr.N + np.maximum(gv.eI, gv.eJ)
    a, b = p[gr.eI], p[gr.eJ]
    kr = np.minimum(a, b) * gr.N + np.maximum(a, b)
    o = np.argsort(kv)
    em = o[np.searchsorted(kv[o], kr)]
    topo = np.array_equal(kv[em], kr) and gr.NV == gv.NV and np.array_equal(gr.deg, gv.deg[p])
    nb_same = all(sorted(p[gr.nbr[i, :gr.deg[i]]]) == sorted(gv.nbr[p[i], :gv.deg[p[i]]]) for i in range(gr.N))
    sg = np.where(a < b, 1.0, -1.0)
    worst = {}
    for k in ('pos', 'area', 'areaCell'):
        worst[k] = float(np.max(np.abs(getattr(gr, k) - getattr(gv, k)[p])) / np.max(np.abs(getattr(gv, k))))
    for k in ('le', 'de'):
        worst[k] = float(np.max(np.abs(getattr(gr, k) - getattr(gv, k)[em])) / np.max(getattr(gv, k)))
    worst['ne'] = float(np.max(np.abs(gr.ne - sg[:, None] * gv.ne[em])))
    worst['areaV'] = float(abs(np.sort(gr.areaV) - np.sort(gv.areaV)).max() / gv.areaV.max())
    geo = max(worst.values())
    ok = topo and nb_same and geo <= 1e-15
    big = max(worst, key=worst.get)
    record(f'E3 n={n}', f'{gr.order} order: topology {"identical" if topo and nb_same else "DIFFERS"} through the permutation; '
           f'geometry max {fmt(geo)} ({big}); {int(np.sum(sg < 0))} of {gr.E} edges point the other way', 'exact; ≤ 1e-15', ok)
    assert ok


def test_E4_renumbered_run(record):
    """E4 A run on the renumbered grid equals one on v1's numbering (Williamson 5 + bell, n = 4, 960 steps)"""
    a = World(n=4, preset='w5', params={'nu4': 1e14, 'drag': 1e-6})
    gv = gridmod.get(4, order='v1')
    b = World.__new__(World)
    import insilica.world as wm
    real_get = wm.gridmod.get
    try:
        wm.gridmod.get = lambda n: real_get(n, order='v1')
        b = World(n=4, preset='w5', params={'nu4': 1e14, 'drag': 1e-6})
    finally:
        wm.gridmod.get = real_get
    a.step(960)
    b.step(960)
    p = a.grid.perm.astype(np.int64)
    kv = np.minimum(gv.eI, gv.eJ).astype(np.int64) * gv.N + np.maximum(gv.eI, gv.eJ)
    ea, eb = p[a.grid.eI], p[a.grid.eJ]
    kr = np.minimum(ea, eb) * gv.N + np.maximum(ea, eb)
    o = np.argsort(kv)
    em = o[np.searchsorted(kv[o], kr)]
    sg = np.where(ea < eb, 1.0, -1.0)
    rel = lambda x, y: float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
    r = {'h': rel(a.h.get()[0], b.h.get()[0][p]), 'u': rel(a.u.get()[0], sg * b.u.get()[0][em]),
         'psi': rel(a.psi.get()[0], b.psi.get()[0][p])}
    ok = max(r.values()) <= 1e-12
    record('E4', f'{a.grid.order} vs v1 order after 960 steps: h {fmt(r["h"])}, u {fmt(r["u"])}, tracer (FCT3) {fmt(r["psi"])}', '≤ 1e-12', ok)
    assert ok
