"""A ground column under a planet's surface skin where there is no ocean (V2-7, SPEC #238; Mars): layers dz (m) of soil with volumetric
heat capacity rho_c (J/m^3/K) and each cell's conductivity k (W/m/K, from its thermal inertia: k = I^2 / rho_c), their temperatures Tg
(K, layers x N), under the skin, whose capacity rho_c * skin (J/m^2/K) the air's column physics heats (World.cap). Conduction through
the skin and the layers is implicit (ocean.cu land_column_k, as Earth's land column #218 with a conductivity per cell); the bottom is
insulated. The heat (J): sum A (cap Ts + rho_c sum dz Tg)."""
import cupy as cp
import numpy as np

from .gpu import grid_dims


class Ground:
    def __init__(self, dg, Ts, ti, dz, rho_c=1.2e6, skin=0.004):
        assert len(dz) <= 16, 'at most 16 ground layers'
        self.dg, self.rho_c = dg, float(rho_c)
        self.dz = cp.asarray([float(d) for d in dz])
        self.k = cp.asarray(np.asarray(ti, dtype=float) ** 2 / self.rho_c)
        self.cap = cp.full(dg.N, self.rho_c * skin)
        self.Tg = cp.repeat(cp.asarray(Ts, dtype=float)[None, :], len(dz), axis=0)   # starts at the surface's temperature
        self.kern = dg.kernel('land_column_k', 'ocean.cu')

    def step(self, Ts, dt):
        g, R = self.dg, self.dg.dtype.type
        self.kern(*grid_dims(g.N), (np.int32(g.N), np.int32(len(self.dz)), Ts, self.cap, self.Tg, self.dz, R(self.rho_c), self.k, R(dt)))

    def heat(self, Ts, A):
        return float(cp.sum(A * (self.cap * Ts + self.rho_c * cp.sum(self.dz[:, None] * self.Tg, axis=0))))
