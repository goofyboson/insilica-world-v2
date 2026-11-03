"""The deep ocean's spin-up by asynchronous coupling (V2-5, lever 5; SPEC #107).

The deep ocean takes centuries to settle, the air days. So: a coupled year records what the air does to the ocean (the
Recorder: the column's heating of the mixed layer, its temperature, the mixing's daily sums of non-solar heat, sunlight,
fresh water with the rivers and wind stress², and the wind's momentum, in `nslot` slots of the year), then the ocean runs
alone on that forcing for years (OceanOnly), interpolated through the seasons, and the two couple again. The surface heat
keeps the air's feedback: Q = Q_rec + λ (T_rec − T₀) (Haney 1971; λ about the air's radiative, sensible and latent
response to the sea's temperature), so a mixed layer that drifts from the coupled year's is pulled back as the air would,
and when it matches, Q is the recorded flux exactly. Sea ice and the land are held as they were.

Changes the spin-up's path only: the climate after re-coupling is measured against a plain coupled spin-up."""
import cupy as cp
import numpy as np

from .gpu import grid_dims

LAMBDA = 10.0          # W/m²/K: the air's damping of a sea-temperature anomaly over years (the air warming with the sea): of 10 and 40 tried, 10 lagged a plain spin-up least (SPEC #107)
DAY = 86400.0


class Recorder:
    """Attach to a coupled world (World.set_recorder) for a year; then forcing() gives the replayable rates."""

    def __init__(self, world, nslot=365):
        sea, g = world.sea, world.dg
        assert sea is not None and world.air, 'a coupled world'
        self.w, self.sea, self.nslot = world, sea, nslot
        self.year = float(world.physics.p['year_s'])
        N, E = g.N, g.E
        z = lambda *s: cp.zeros(s)
        self.du, self.tau = z(nslot, E), z(nslot, E)
        self.Q, self.qns, self.sun, self.fw, self.T = (z(nslot, N) for _ in range(5))
        self.secs, self.days = z(nslot), z(nslot)
        self._uE, self._TN = z(E), z(N)
        self.k = {n: g.kernel(n, 'spinup.cu') for n in ('rec_du', 'rec_heat', 'rec_day')}

    def pre_couple(self):
        cp.copyto(self._uE, self.sea.u[0])

    def post_couple(self):
        g = self.w.dg
        self.k['rec_du'](*grid_dims(g.E), (np.int32(g.E), self.w.time, g.dtype.type(self.year), np.int32(self.nslot), self.sea.u[0],
                                           self._uE, self.du))

    def pre_column(self):
        cp.copyto(self._TN, self.sea.T[0])

    def post_column(self):
        g, R = self.w.dg, self.w.dg.dtype.type
        self.k['rec_heat'](*grid_dims(g.N), (np.int32(g.N), self.w.time, R(self.year), np.int32(self.nslot), self.sea.T[0], self._TN,
                                             self.sea.cap, R(self.w.dt), self.Q, self.secs))

    def pre_mix(self):
        g, s = self.w.dg, self.sea
        self.k['rec_day'](*grid_dims(max(g.N, g.E)), (np.int32(g.N), np.int32(g.E), self.w.time, g.dtype.type(0.5 * self.w.dt),
                                                      g.dtype.type(self.year),
                                                      np.int32(self.nslot), s.q_acc, s.sun_acc, s.fw_acc, s.tau_acc, s.T[0],
                                                      self.qns, self.sun, self.fw, self.tau, self.T, self.days))

    def forcing(self, balance_fw=True):
        """Per-second rates per slot (and the slot's mean mixed-layer temperature). balance_fw: take the year's global
        mean net fresh water over the sea out of every sea cell, keeping its pattern and seasons: replayed for years, one
        year's imbalance (what the land stored that year) would freshen or salt the whole ocean, which over a spin-up the
        closed water cycle does not."""
        secs, days = self.secs.get(), self.days.get()
        assert (secs > 0).all() and (days > 0).all(), 'record a whole year: every slot needs time and a mixing'
        s, d = cp.asarray(1.0 / secs)[:, None], cp.asarray(1.0 / (days * DAY))[:, None]
        f32 = lambda a: a.astype(cp.float32)
        fw = self.fw * d
        if balance_fw:
            g, sea = self.w.dg, (1 - self.sea.land) if self.sea.land is not None else cp.ones(self.w.dg.N)
            As = cp.asarray(g.host.areaCell) * sea
            mean = float(cp.sum(cp.asarray(days)[:, None] * fw * As[None, :]) / (float(days.sum()) * float(cp.sum(As))))
            fw = fw - mean * sea[None, :]
        return dict(du=f32(self.du * s), Q=f32(self.Q * s), tau=f32(self.tau * d), qns=f32(self.qns * d), sun=f32(self.sun * d),
                    fw=f32(fw), T=f32(self.T * cp.asarray(1.0 / days)[:, None]), nslot=self.nslot, year=self.year)


class OceanOnly:
    """The world's ocean stepped alone on recorded forcing, in place (the world's own buffers), on its own clock."""

    def __init__(self, world, forcing, lam=LAMBDA):
        self.w, self.sea, self.f, self.lam = world, world.sea, forcing, lam
        g = world.dg
        self.k = {n: g.kernel(n, 'spinup.cu') for n in ('play_edge', 'play_cell', 'play_time')}
        self.time = world.time.copy()
        self.n = (world.tick // world.ocean_every) % self.sea.mix_every     # the ocean steps since the last mixing
        self._stream = cp.cuda.Stream(non_blocking=False)
        self._graphs, self._warm = {}, {}                  # captured once each kind of step has run directly

    def _step(self, mix):
        w, s, f, g = self.w, self.sea, self.f, self.w.dg
        i32, R = np.int32, g.dtype.type
        c = s.cfg
        dt = c['dt']
        hdr = (self.time, R(f['year']), i32(f['nslot']))
        self.k['play_edge'](*grid_dims(g.E), (i32(g.E),) + hdr + (f['du'], f['tau'], R(dt), s.u[0], s.tau_acc))
        self.k['play_cell'](*grid_dims(g.N), (i32(g.N),) + hdr + (f['Q'], f['T'], f['qns'], f['sun'], f['fw'], R(self.lam),
                            R(c['S_ref']), R(c['rho']), R(dt), s.h, s.cap, s.T, s.S, s.qocean, s.qsw, s.fw_acc, s._land()))
        s.k['ocn_pen'](*grid_dims(g.N), (i32(g.N), i32(s.L), s.h, s.T, s.qocean, s.qsw, R(c['rho'] * c['cw']), R(c['sw_R']),
                                        R(c['sw_z1']), R(c['sw_z2']), R(dt), s.q_acc, s.sun_acc, s._land()))
        s.step(mix=mix)
        self.k['play_time']((1,), (1,), (self.time, R(dt)))

    def run(self, years):
        """years of ocean alone (whole ocean steps); returns self."""
        steps = int(round(years * self.f['year'] / self.sea.cfg['dt']))
        for _ in range(steps):
            mix = (self.n + 1) % self.sea.mix_every == 0
            gr = self._graphs.get(mix)
            if gr is None and self._warm.get(mix):
                with self._stream:
                    self._stream.begin_capture()
                    self._step(mix)
                    gr = self._graphs[mix] = self._stream.end_capture()
            if gr is not None:
                gr.launch(self._stream)
            else:
                with self._stream:
                    self._step(mix)
                self._warm[mix] = True
            self.n += 1
        self._stream.synchronize()
        return self


def spin_up(world, cycles, ocean_years, nslot=365, lam=LAMBDA, log=None):
    """cycles × (one coupled year, recorded; then ocean_years of ocean alone on it). The world ends just after the last
    ocean-only stretch, ready to continue coupled. log(cycle, recorder) is called after each coupled year."""
    year_ticks = int(round(float(world.physics.p['year_s']) / world.dt))
    for c in range(cycles):
        rec = Recorder(world, nslot)
        world.set_recorder(rec)
        world.step(year_ticks)
        world.set_recorder(None)
        if log:
            log(c, rec)
        oo = OceanOnly(world, rec.forcing(), lam)
        oo.run(ocean_years)
    return world
