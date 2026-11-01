"""Water moving over and through the land (stage V2-4b, decision #85): surface flow toward lower water surfaces at a set
speed, whatever the grid, lakes where it collects, seepage between soils, and the water reaching the sea. The columns' side
(rain on the pond, soaking in, springs, evaporation, freezing) is in the air column; see kernels/land.cu for the rules."""
import numpy as np
import cupy as cp

from .gpu import NULL, grid_dims

RIVER_SPEED = 1.0     # m/s
K_SEEP = 2.0e4 / 3600.0    # m²/s, v1's K_SEEP (2e4 m² per hour's tick)
RHO_W = 1000.0        # kg/m³
GLEN_A = 1.0e-16 / (365.0 * 86400.0)    # Pa⁻³ s⁻¹, Glen's flow law at about −10 °C, everywhere (#86)
GRAV = 9.81           # m/s²


class LandWater:
    def __init__(self, dg, b, land, soilcap, pond, soil, ice, rho_ice=917.0, uflow=RIVER_SPEED, kseep=K_SEEP):
        """b, land, soilcap: terrain (m), mask and capacity; pond, soil, ice: the world's buffers (kg/m²), changed in place."""
        self.dg, self.b, self.land, self.soilcap = dg, b, land, soilcap
        self.pond, self.soil, self.ice = pond, soil, ice
        self.rho_ice, self.uflow, self.kseep = rho_ice, uflow, kseep
        N = dg.N
        z = lambda: cp.zeros(N)
        self.mouth = z()                                  # kg/m² of the sea cell, waiting for the ocean
        self.z, self.zt, self.sp, self.ss, self.pond1, self.soil1 = z(), z(), z(), z(), z(), z()
        self.s, self.g2, self.si, self.ice1, self.calved = z(), z(), z(), z(), z()    # calved: ice given to each sea cell, summed (kg/m²)
        self.ice_prev = None                              # the ocean's, when there is one
        self.gam = 2.0 * GLEN_A * (rho_ice * GRAV) ** 3 / 5.0
        self.k = {n: dg.kernel(n, 'land.cu') for n in ('land_heights', 'land_scale', 'land_apply', 'ice_surface', 'ice_slope', 'ice_scale', 'ice_apply')}

    def step(self, dt):
        """One routing step of dt seconds."""
        g, R = self.dg, self.dg.dtype.type
        i32 = np.int32(g.N)
        u, ks, rw = R(self.uflow * dt), R(self.kseep * dt), R(RHO_W)
        self.k['land_heights'](*grid_dims(g.N), (i32, self.b, self.land, self.pond, self.ice, self.soil, self.soilcap, rw, R(self.rho_ice),
                                                 self.z, self.zt))
        flow = (self.land, self.z, self.pond, self.zt, self.soil, self.soilcap, g.cellEdge, g.eI, g.eJ, g.le, g.de, g.areaCell, rw, u, ks)
        self.k['land_scale'](*grid_dims(g.N), (i32,) + flow + (self.sp, self.ss))
        self.k['land_apply'](*grid_dims(g.N), (i32,) + flow + (self.sp, self.ss, self.pond1, self.soil1, self.mouth))
        cp.copyto(self.pond, self.pond1)
        cp.copyto(self.soil, self.soil1)
        self.ice_step(dt)

    def ice_step(self, dt):
        """The ice sheets' flow over dt seconds (#86)."""
        g, R = self.dg, self.dg.dtype.type
        i32, ri, gam = np.int32(g.N), R(self.rho_ice), R(self.gam * dt)
        self.k['ice_surface'](*grid_dims(g.N), (i32, self.b, self.land, self.ice, ri, self.s))
        self.k['ice_slope'](*grid_dims(g.N), (i32, self.land, self.s, g.cellEdge, g.eI, g.eJ, g.le, g.de, g.areaCell, self.g2))
        flow = (self.land, self.s, self.g2, self.ice, g.cellEdge, g.eI, g.eJ, g.le, g.de, g.areaCell, ri, gam)
        self.k['ice_scale'](*grid_dims(g.N), (i32,) + flow + (self.si,))
        self.k['ice_apply'](*grid_dims(g.N), (i32,) + flow + (self.si, self.ice1, self.ice_prev if self.ice_prev is not None else NULL, self.calved))
        cp.copyto(self.ice, self.ice1)
