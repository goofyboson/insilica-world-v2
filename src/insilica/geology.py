"""Geology's accumulators (V2-6e, decision #220): what the climate hands a later geology model (docs/v2-plan.md, V2-9),
summed per cell over a run: averaged rain and evaporation, river flow, rain extremes (the largest rain in one day), ice flow,
frost (time frozen, time in the frost-cracking window, freeze-thaw cycles) and weathering (runoff weighted by the
temperature factor of silicate dissolution). Off unless a world turns them on (`params['geology']`, World.enable_geology);
then they run inside the step graph and are saved with the world. Kernels in kernels/geo.cu."""
import numpy as np
import cupy as cp

from .gpu import NULL, grid_dims

ROWS = ('rain', 'evap', 'tsum', 'frozen', 'crack', 'ftc', 'fstate', 'rx1', 'bucket', 'ice', 'river', 'runoff', 'weather', 'iceflow')
FT_BAND = 0.5              # K: a cell freezes below Tf − 0.5 and thaws above Tf + 0.5, so noise around 0 °C is not counted as cycles
CRACK = (-8.0, -3.0)       # °C: the frost-cracking window (Anderson 1998; Hales & Roering 2007)
EA_WEATHER = 60.0e3        # J/mol: silicate dissolution's activation energy, mid-range of field estimates (~40–80 kJ/mol)
T0_WEATHER = 288.15        # K: the weathering factor is 1 at 15 °C
R_GAS = 8.314462618        # J/mol/K


class Geology:
    def __init__(self, dg, dt, day_s, Ts, tf, land=None):
        """dt: the air step (s); day_s: the planet's day (s), the rain extremes' window; Ts: the surface temperature buffer;
        tf: freezing (K); land: the land mask (or None: every cell counts as land)."""
        self.dg, self.dt, self.Ts, self.tf, self.land = dg, float(dt), Ts, float(tf), land
        N = dg.N
        self.acc = cp.zeros((len(ROWS), N))
        self.acc[ROWS.index('fstate')] = (Ts < tf - FT_BAND).astype(float)
        day = max(1, int(round(day_s / dt)))
        slot = max(1, int(round(3600.0 / dt)))
        self.slot = slot if day % slot == 0 else 1          # the day in slots of about an hour, when the day divides
        self.nslot = day // self.slot
        self.ring = cp.zeros((self.nslot, N))
        self.clock = cp.zeros(1)                            # ticks accumulated
        self.k = {n: dg.kernel(n, 'geo.cu') for n in ('geo_column', 'geo_tick', 'geo_runoff')}

    def row(self, name):
        return self.acc[ROWS.index(name)]

    def column(self, rain, evap, ice=None):
        """After the column physics, every tick."""
        g, R, i32 = self.dg, self.dg.dtype.type, np.int32
        self.k['geo_column'](*grid_dims(g.N), (i32(g.N), self.land if self.land is not None else NULL, rain, evap, self.Ts,
                                               ice if ice is not None else NULL, R(self.dt), R(self.tf), R(FT_BAND),
                                               R(self.tf + CRACK[0]), R(self.tf + CRACK[1]), i32(self.slot), i32(self.nslot),
                                               self.clock, self.acc, self.ring))
        self.k['geo_tick']((1,), (32,), (self.clock,))

    def runoff(self, land, pond, pond1, soil, soil1):
        """On each routing step, after the flows and before the new pond and soil replace the old (land.py)."""
        g, R = self.dg, self.dg.dtype.type
        self.k['geo_runoff'](*grid_dims(g.N), (np.int32(g.N), land, pond, pond1, soil, soil1, self.Ts,
                                               R(EA_WEATHER / R_GAS), R(T0_WEATHER), self.acc))

    def summary(self, area=None):
        """The handoff, per cell (NumPy), as means over the time accumulated: rain and evaporation and runoff (mm/day),
        weathering (mm/day of 15 °C-equivalent runoff), surface temperature (K), river and ice outflow (m³/s and kg/s
        through the cell), the frozen and frost-cracking fractions of time, freeze-thaw cycles per year (of 365.25 days),
        the largest one-day rain (mm), and the ice sheet's mean mass (kg/m²)."""
        t = float(self.clock[0]) * self.dt
        a = self.acc.get()
        A = self.dg.host.areaCell if area is None else area
        r = lambda n: a[ROWS.index(n)]
        day = 86400.0 / t if t else 0.0
        return dict(seconds=t, rain=r('rain') * day, evap=r('evap') * day, runoff=r('runoff') * day, weathering=r('weather') * day,
                    T=r('tsum') / t if t else r('tsum'), river=r('river') * A / t / 1000.0 if t else r('river'),
                    iceflow=r('iceflow') * A / t if t else r('iceflow'), frozen=r('frozen') / t if t else r('frozen'),
                    crack=r('crack') / t if t else r('crack'), ftc=r('ftc') * 365.25 * 86400.0 / t if t else r('ftc'),
                    rx1day=r('rx1'), ice=r('ice') / t if t else r('ice'))
