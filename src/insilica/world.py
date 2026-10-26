"""A world: a stack of layers on the layered core (layers.py) carrying one tracer per layer, on the GPU.

State (all 64-bit, layer-major): h (L, N) thickness, u (L, E) normal wind, M (L, N) layer volume per cell
(h·A, moved by the same fluxes as h so the tracer budget closes exactly), psi (L, N) tracer per volume.
A tick is one time-split RK3 step of the layers followed by one FCT transport step with that step's fluxes.
Cases and their settings are in cases.py; `params` overrides a case's settings:
  dt, gprime (list, L − 1), floor (m, default 1), drag (per layer, 1/s), stress (per layer, m/s: linear stress
  coefficient, deceleration c/h), nu4 (m⁴/s) or visc4_U (m/s: ν4 = U·d³/π⁴
  with d the mean cell spacing, the grid-scale wave then damps at U/d), rotation (multiplies f), beta (transport),
  div_damp (the fast mode's divergence damping, default 0.1), nfast (fast sub-steps per step; default from √(gH)).
"""
import hashlib
import math

import cupy as cp
import numpy as np

from . import cases
from . import grid as gridmod
from .cases import PRESETS, cosine_bell, default_dt, edge_east_north, initial_state  # noqa: F401  (re-exported)
from .gpu import DeviceGrid, dsum, source_hash
from .io import save as savefile
from .layers import DIV_DAMP, Layers
from .transport import BETA, Transport

FIELDS = ('h', 'z', 'eta', 'psi', 'speed', 'vort')      # z: height of the layer's top; eta: the free surface


def visc4(g, U):
    """ν4 = U·d³/π⁴ (d the mean cell spacing): a wave of length 2d damps at rate U/d, the grid's shortest modes
    (the Laplacian's largest eigenvalue, about 27/d²) at about 7.5·U/d (SPEC §9 #25)."""
    d = float(np.mean(g.de))
    return U * d ** 3 / math.pi ** 4


class World:
    def __init__(self, n=4, preset='w5', params=None, layers=None, state=None):
        """A world on level n from case `preset`. layers=K splits a one-layer case into K equal layers with
        g' = 0 (they then move as one: test REPRO). state=(h, b, u) starts from given arrays instead."""
        self.n, self.preset = n, preset
        self.grid = gridmod.get(n)
        self.dg = DeviceGrid(self.grid)
        spec = PRESETS[preset]
        self.params = dict(params or {})
        P = self.params
        h, b, u = state if state is not None else spec['build'](self.grid)
        h, u = np.asarray(h, dtype=float), np.asarray(u, dtype=float)
        if layers and layers != h.shape[0]:
            assert h.shape[0] == 1
            h, u = np.repeat(h / layers, layers, axis=0), np.repeat(u, layers, axis=0)
            P.setdefault('gprime', [0.0] * (layers - 1))
        self.L = h.shape[0]
        gp = P.get('gprime', spec.get('gprime', [0.0] * (self.L - 1)))
        gp = gp(self.grid) if callable(gp) else gp
        self.gprime = [float(x) for x in np.atleast_1d(gp)][:self.L - 1]
        dt = P.get('dt', spec['dt'](n))
        self.dt = float(dt)
        nu4 = P.get('nu4', visc4(self.grid, P['visc4_U']) if 'visc4_U' in P else visc4(self.grid, spec.get('visc4_U', 0.0)))
        self.b = cp.asarray(b, dtype=float)
        self.core = Layers(self.dg, self.L, b=self.b, gprime=self.gprime, drag=P.get('drag', spec.get('drag', 0.0)),
                           stress=P.get('stress', spec.get('stress', 0.0)),
                           nu4=nu4, floor=P.get('floor', 1.0), rotation=P.get('rotation', 1.0),
                           div_damp=P.get('div_damp', DIV_DAMP))
        if nu4:
            lam = self.core.lap_max()
            if nu4 * lam * lam * self.dt > 1.0:
                raise ValueError(f'nu4 {nu4:.3g} m^4/s is unstable at dt {self.dt:g} s on n = {n} (nu4*lambda^2*dt = '
                                 f'{nu4 * lam * lam * self.dt:.2f}; keep it <= 1: nu4 <= {1.0 / (lam * lam * self.dt):.3g})')
        self.tr = Transport(self.dg, layers=self.L, beta=P.get('beta', BETA))
        self.h, self.u = cp.asarray(h), cp.asarray(u)
        self.M = self.h * cp.asarray(self.grid.areaCell)[None, :]
        self.psi = cp.asarray(np.tile(cosine_bell(self.grid), (self.L, 1)))
        # fast sub-steps per step: from the deepest column, with 20% room for it to deepen
        self.nfast = int(P.get('nfast', math.ceil(1.2 * self.core.substeps(self.h, self.dt))))
        P['nfast'] = self.nfast                         # saved with the world, so a reload continues identically
        self.tick = 0
        self._Fm, self._M1 = cp.zeros_like(self.u), cp.zeros_like(self.M)
        # A blocking stream: work on it orders with the default stream, so reads after step() see the results.
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graph, self._tick0 = None, 0

    STATE = ('h', 'u', 'M', 'psi')

    def _tick(self):
        """One tick's kernels, on fixed buffers (no allocation), so the sequence can be captured as a graph."""
        self.core.step(self.h, self.u, self.dt, self.nfast, self._Fm)
        self.tr.mass_after(self.M, self._Fm, self._M1)
        self.tr.step(self.M, self._M1, self._Fm, self.psi, order=3)
        cp.copyto(self.M, self._M1)

    def step(self, k=1, graph=True):
        """k ticks. The first tick runs directly (and compiles the kernels); after that the tick is captured once
        as a CUDA graph and replayed with a single launch per tick (SPEC §9 #18). graph=False runs every tick
        directly; the results are bit-identical (test REPRO)."""
        for _ in range(k):
            if graph and self._graph is None and self.tick > self._tick0:
                with self._stream:
                    self._stream.begin_capture()
                    self._tick()
                    self._graph = self._stream.end_capture()
            if graph and self._graph is not None:
                self._graph.launch(self._stream)
            else:
                with self._stream:
                    self._tick()
            self.tick += 1
        self._stream.synchronize()
        return self

    def exchange(self, wx):
        """Move thickness wx (L − 1, N; m, positive up) between neighbouring layers with tracer and momentum."""
        with self._stream:
            self.core.exchange(cp.asarray(wx, dtype=float), self.h, self.u, self.M, self.psi)
        self._stream.synchronize()

    # ---- diagnostics ----
    def totals(self):
        """Deterministic totals per layer and summed: volume Σh·A, prognostic volume ΣM, tracer ΣM·psi; and the
        column's volume-weighted momentum Σ_e l_e d_e Σ_k h_e,k u_k."""
        A = cp.asarray(self.grid.areaCell)
        out = {}
        for k in range(self.L):
            out[f'volume{k}'] = dsum(self.h[k] * A)
            out[f'M{k}'] = dsum(self.M[k])
            out[f'tracer{k}'] = dsum(self.M[k], self.psi[k])
        g = self.grid
        he = 0.5 * (self.h[:, g.eI] + self.h[:, g.eJ])
        out['momentum'] = dsum(cp.sum(he * self.u, axis=0) * cp.asarray(g.le * g.de))
        return out

    def energy(self):
        return self.core.energy(self.h, self.u)

    def field(self, name, layer=0):
        """A cell field (CuPy, float64) for the viewer."""
        g = self.grid
        if name == 'h':
            return self.h[layer]
        if name in ('z', 'eta'):
            top = 0 if name == 'eta' else layer
            return self.b + cp.sum(self.h[top:], axis=0)
        if name == 'psi':
            return self.psi[layer]
        if name in ('speed', 'vort'):
            uh = self.u[layer].get()
            if name == 'speed':
                ce = np.where(g.cellEdge >= 0, g.cellEdge, 0)
                ke = np.sum(np.where(g.cellEdge >= 0, g.le[ce] * g.de[ce] * 0.25 * uh[ce] ** 2, 0.0), axis=1)
                return cp.asarray(np.sqrt(2 * ke / g.areaCell))
            z = np.zeros(g.NV)
            np.add.at(z, g.ev1, g.circ1 * g.de * uh)
            np.add.at(z, g.ev2, g.circ2 * g.de * uh)
            z /= g.areaV
            cv = np.where(g.cornerV >= 0, g.cornerV, 0)
            cell = np.sum(np.where(g.cornerV >= 0, z[cv] * g.kite, 0.0), axis=1)
            return cp.asarray(cell / g.areaCell)
        raise KeyError(name)

    def state_hash(self):
        h = hashlib.sha1()
        for k in self.STATE:
            h.update(getattr(self, k).get().tobytes())
        h.update(str(self.tick).encode())
        return h.hexdigest()

    # ---- save / load ----
    def save(self, path):
        meta = dict(kind='insilica-world-v2', n=self.n, preset=self.preset, layers=self.L, dt=self.dt, tick=self.tick,
                    params=self.params, order=self.grid.order, gridHash=gridmod.generator_hash(), sourceHash=source_hash())
        savefile.write(path, meta, {**{k: getattr(self, k).get() for k in self.STATE}, 'b': self.b.get()})

    @classmethod
    def load(cls, path):
        meta, arrays = savefile.read(path)
        if meta.get('gridHash') != gridmod.generator_hash() or meta.get('order') != gridmod.DEFAULT_ORDER:
            raise ValueError('save file was made with a different grid generator or cell order')
        w = cls(n=meta['n'], preset=meta['preset'], params=meta['params'],
                state=(arrays['h'], arrays['b'], arrays['u']))
        for k in cls.STATE:
            a = arrays[k]
            if a.shape != getattr(w, k).shape or a.dtype != np.float64:
                raise ValueError(f'save file array {k}: shape {a.shape} {a.dtype}, expected {getattr(w, k).shape} float64')
            getattr(w, k)[...] = cp.asarray(a)          # into the existing buffers a captured graph uses
        w.tick = w._tick0 = meta['tick']
        return w
