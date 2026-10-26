"""A V2-0 world: one shallow-water layer stack carrying one tracer per layer, on the GPU. It is the stand-in the
foundation's tools (save/load, viewer, tests, spin-up cache) work on until V2-1 brings the layered fluid core.

State (all 64-bit, layer-major): h (L, N) thickness, u (L, E) normal wind, M (L, N) layer volume per cell
(h·A, moved by the same fluxes as h so the tracer budget closes exactly), psi (L, N) tracer per volume.
Each tick is one RK3 shallow-water step followed by one FCT transport step with that step's fluxes.
"""
import math

import cupy as cp
import numpy as np

from . import grid as gridmod
from .constants import GRAV, OMEGA, R_PLANET
from .gpu import DeviceGrid, dsum, source_hash
from .io import save as savefile
from .swe import SWE
from .transport import BETA, Transport

PRESETS = ('w2', 'w5', 'galewsky')
FIELDS = ('h', 'hs', 'psi', 'speed', 'vort')     # hs: surface height h + b


def edge_east_north(g):
    """Latitude, longitude and the east/north components of each edge's normal, at the Voronoi edge midpoint xm,
    where analytic winds are sampled (SPEC §9 #17; v1's edgeInfo used xe)."""
    x = g.xm
    lat = np.arcsin(np.clip(x[:, 2], -1, 1))
    lon = np.arctan2(x[:, 1], x[:, 0])
    en = -np.sin(lon) * g.ne[:, 0] + np.cos(lon) * g.ne[:, 1]
    nn = (-np.sin(lat) * np.cos(lon) * g.ne[:, 0] - np.sin(lat) * np.sin(lon) * g.ne[:, 1] + np.cos(lat) * g.ne[:, 2])
    return lat, lon, en, nn


def initial_state(g, preset):
    """(h, b, u) as NumPy arrays for one layer: Williamson 2 (steady zonal flow), Williamson 5 (zonal flow
    over an isolated mountain) or the Galewsky jet (balanced jet with a small bump; it breaks into vortices)."""
    a, Om = R_PLANET, OMEGA
    lat_e, lon_e, en, _ = edge_east_north(g)
    b = np.zeros(g.N)
    if preset == 'w2':
        u0, gh0 = 2 * math.pi * a / (12 * 86400), 2.94e4
        s = np.sin(g.lat)
        h = (gh0 - (a * Om * u0 + u0 * u0 / 2) * s * s) / GRAV
        u = u0 * np.cos(lat_e) * en
    elif preset == 'w5':
        u0, h0, Rm, lc, pc = 20.0, 5960.0, math.pi / 9, -math.pi / 2, math.pi / 6
        s = np.sin(g.lat)
        dl = np.arctan2(np.sin(g.lon - lc), np.cos(g.lon - lc))
        b = 2000 * (1 - np.minimum(Rm, np.hypot(dl, g.lat - pc)) / Rm)
        h = h0 - (a * Om * u0 + u0 * u0 / 2) * s * s / GRAV - b
        u = u0 * np.cos(lat_e) * en
    elif preset == 'galewsky':
        umax, p0 = 80.0, math.pi / 7
        p1 = math.pi / 2 - p0
        e_n = math.exp(-4 / (p1 - p0) ** 2)

        def ujet(phi):
            phi = np.asarray(phi, dtype=float)
            inside = (phi > p0) & (phi < p1)
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                v = umax / e_n * np.exp(1 / ((phi - p0) * (phi - p1)))
            return np.where(inside, v, 0.0)

        M = 20000
        dphi = math.pi / M
        ph = -math.pi / 2 + (np.arange(1, M + 1) - 0.5) * dphi
        uu = ujet(ph)
        gh = np.concatenate([[0.0], np.cumsum(-a * uu * (2 * Om * np.sin(ph) + np.tan(ph) * uu / a) * dphi)])
        x = (g.lat + math.pi / 2) / dphi
        k = np.minimum(M - 1, np.floor(x).astype(int))
        h = (gh[k] + (x - k) * (gh[k + 1] - gh[k])) / GRAV
        h = h + 10000 - np.sum(h * g.areaCell) / np.sum(g.areaCell)
        h = h + 120 * np.cos(g.lat) * np.exp(-((g.lon / (1 / 3)) ** 2)) * np.exp(-(((math.pi / 4 - g.lat) / (1 / 15)) ** 2))
        u = ujet(lat_e) * en
    else:
        raise ValueError(f'unknown preset {preset!r} (choose from {PRESETS})')
    return h, b, u


def cosine_bell(g, lon_c=3 * math.pi / 2, lat_c=0.0, radius=R_PLANET / 3, height=1000.0):
    """Williamson 1's cosine bell (v1 N6): height/2·(1 + cos(π r / R)) within great-circle distance R."""
    cosd = np.sin(lat_c) * np.sin(g.lat) + np.cos(lat_c) * np.cos(g.lat) * np.cos(g.lon - lon_c)
    dist = R_PLANET * np.arccos(np.clip(cosd, -1, 1))
    return np.where(dist < radius, 0.5 * height * (1 + np.cos(math.pi * dist / radius)), 0.0)


def default_dt(n):
    """v1's numerics-test steps: 450 s at n = 4, halved per level."""
    return 450.0 * 2.0 ** (4 - n)


class World:
    def __init__(self, n=4, preset='w5', layers=1, dt=None, params=None):
        self.n, self.preset, self.L = n, preset, layers
        self.grid = gridmod.get(n)
        self.dg = DeviceGrid(self.grid)
        self.dt = float(dt if dt is not None else default_dt(n))
        self.params = dict(params or {})
        h, b, u = initial_state(self.grid, preset)
        self.b = cp.asarray(b)
        self.swe = SWE(self.dg, layers=layers, b=self.b, drag=self.params.get('drag', 0.0), nu4=self.params.get('nu4', 0.0))
        self.tr = Transport(self.dg, layers=layers, beta=self.params.get('beta', BETA))
        self.tick = 0
        L = layers
        self.h = cp.asarray(np.tile(h, (L, 1)))
        self.u = cp.asarray(np.tile(u, (L, 1)))
        self.M = self.h * cp.asarray(self.grid.areaCell)[None, :]
        self.psi = cp.asarray(np.tile(cosine_bell(self.grid), (L, 1)))
        self._Fm = cp.zeros_like(self.u)
        self._M1 = cp.zeros_like(self.M)
        # A blocking stream: work on it orders with the default stream, so reads after step() see the results.
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graph, self._tick0 = None, 0

    STATE = ('h', 'u', 'M', 'psi')

    def _tick(self):
        """One tick's kernels, on fixed buffers (no allocation), so the sequence can be captured as a graph."""
        self.swe.step(self.h, self.u, self.dt, self._Fm)
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

    # ---- diagnostics ----
    def totals(self):
        """Deterministic totals: volume Σ h·A, tracer Σ M·psi and the prognostic volume Σ M."""
        A = cp.asarray(self.grid.areaCell)
        return {'volume': dsum(self.h * A[None, :]), 'tracer': dsum(self.M, self.psi), 'M': dsum(self.M)}

    def field(self, name, layer=0):
        """A cell field (CuPy, float64) for the viewer."""
        g = self.dg
        if name == 'h':
            return self.h[layer]
        if name == 'hs':
            return self.h[layer] + self.b
        if name == 'psi':
            return self.psi[layer]
        if name in ('speed', 'vort'):
            uh = self.u[layer].get()
            if name == 'speed':
                ke = np.zeros(g.N)
                ce = self.grid.cellEdge
                for k in range(6):
                    e = ce[:, k]
                    ok = e >= 0
                    ee = np.where(ok, e, 0)
                    ke += np.where(ok, self.grid.le[ee] * self.grid.de[ee] * 0.25 * uh[ee] ** 2, 0.0)
                return cp.asarray(np.sqrt(2 * ke / self.grid.areaCell))
            z = np.zeros(g.NV)
            np.add.at(z, self.grid.ev1, self.grid.circ1 * self.grid.de * uh)
            np.add.at(z, self.grid.ev2, self.grid.circ2 * self.grid.de * uh)
            z /= self.grid.areaV
            cell = np.zeros(g.N)
            cv = self.grid.cornerV
            for k in range(6):
                v = cv[:, k]
                cell += np.where(v >= 0, z[np.maximum(v, 0)] * self.grid.kite[:, k], 0.0)
            return cp.asarray(cell / self.grid.areaCell)
        raise KeyError(name)

    def state_hash(self):
        import hashlib
        h = hashlib.sha1()
        for k in self.STATE:
            h.update(getattr(self, k).get().tobytes())
        h.update(str(self.tick).encode())
        return h.hexdigest()

    # ---- save / load ----
    def save(self, path):
        meta = dict(kind='insilica-world-v2', n=self.n, preset=self.preset, layers=self.L, dt=self.dt, tick=self.tick,
                    params=self.params, order=self.grid.order, gridHash=gridmod.generator_hash(), sourceHash=source_hash())
        savefile.write(path, meta, {k: getattr(self, k).get() for k in self.STATE})

    @classmethod
    def load(cls, path):
        meta, arrays = savefile.read(path)
        if meta.get('gridHash') != gridmod.generator_hash() or meta.get('order') != gridmod.DEFAULT_ORDER:
            raise ValueError('save file was made with a different grid generator or cell order')
        w = cls(n=meta['n'], preset=meta['preset'], layers=meta['layers'], dt=meta['dt'], params=meta['params'])
        for k in cls.STATE:
            a = arrays[k]
            if a.shape != getattr(w, k).shape or a.dtype != np.float64:
                raise ValueError(f'save file array {k}: shape {a.shape} {a.dtype}, expected {getattr(w, k).shape} float64')
            getattr(w, k)[...] = cp.asarray(a)          # into the existing buffers a captured graph uses
        w.tick = w._tick0 = meta['tick']
        return w
