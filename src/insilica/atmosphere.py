"""Column physics of the air layers (stages V2-2a and V2-2b, kernels/air.cu): radiation and surface exchange over a slab
ocean, heating as potential temperature, and mass crossing each isentropic interface θ*_j only when heated or cooled
across it (decisions #33–#35, #47). Two layers (Held–Suarez) or three (the aquaplanet, #45: θ* = 305 K and θ*₂ = 380 K).

Defaults are v1's where v1 had them (SPEC §16 of v1, as calibrated through its Stage 3c): S0 1361 W/m², 20% of the
sunlight absorbed in the air at a full column (ASW0), split between the layers like the vapor part (s⁴) of the
longwave profile (#51, #59; sw_tau 2; 1 splits like the whole profile, 0 by mass), surface emissivity 0.95, ocean albedo 0.06,
sensible heat 10·(1 + |u|/10) W/m²K, a 100 m mixed-layer slab; sensible heat goes into the bottom 100 hPa of the column
(mu_bl). The gray longwave's optical depth above pressure p is TAU0·[f·s + (1 − f)·s⁴], s = p/p_s (Frierson et al. 2006;
vapor-independent; #48): a vapor part (s⁴) and a well-mixed, CO₂-like part (f = 0.5, #59) that lets the top layer cool;
each layer's emissivity is 1 − exp(−its share).

Moisture (stage V2-2b, #46–#50; moist=1): vapor q in each layer, evaporation from open water
K_EVAP·(1 + |u|/10)·(q_sat(T_s) − q) kg/m²/s into the bottom 100 hPa (v1's form; K_EVAP 4e-3 and TAU0 0.9 calibrated
on the moist aquaplanet, #54), saturation adjustment in each layer
against its pressure-integrated capacity (the water rains out at once; #50), moist convection by plumes from the bottom
layer that detrain where they stop being buoyant (relaxed over τ_conv = 2 h; #46, #49), and thermodynamic sea ice on the
slab (v1's rules: freezing below 273.15 K, albedo 0.6 at full cover 0.05 m, a conducting skin over thicker ice).
"""
import math

import cupy as cp
import numpy as np

from .constants import CP_DRY, GRAV, KAPPA, P0
from .gpu import NULL, grid_dims

LMAX = 8
SIGMA = 5.670e-8
DAY_S, YEAR_S = 86400.0, 365 * 86400.0
TILT = math.radians(23.44)

DEFAULTS = dict(S0=1361.0, asw=0.2, tau0=0.9, mu_ref=P0 / GRAV, eps_s=0.95, h_sens=10.0, u0=10.0, alb=0.06,
                c_slab=1000.0 * 4186.0 * 100.0, mu_bl=1.0e4 / GRAV, theta_star=300.0, floor_=10.0, sigma=SIGMA,
                grav=GRAV, cp=CP_DRY, kappa=KAPPA, p0=P0, tilt=TILT, year_s=YEAR_S, day_s=DAY_S, tau_lin=0.5, theta_star2=380.0,
                moist=0.0, Lv=2.5e6, Lf=3.34e5, Tf=273.15, k_evap=4.0e-3, tau_conv=7200.0, alb_ice=0.7, rho_ice=917.0,
                ice_cover=0.05, k_ice=2.2, Rv=461.5, sat_layer=1.0, sw_tau=2.0, alb_dry=0.30, alb_wet=0.18, soil_beta=0.75,
                c_land=2.0e6, soil_base=150.0, soil_per_m=0.5, k_inf=5.0 / 3600.0, pond_cover=50.0, alb_snow=0.8,
                alb_ice_melt=0.50, alb_snow_melt=0.55, alb_glacier=0.35, snow_cover=30.0, alb_ramp=10.0, land_exch=1.0, plume_deep=0.0, lw_w0=0.0, c_dry=1.3e6, c_wet=0.0, sat_prof=0.0, land_rs=0.0, sat_base=0.0, bl_bot=0.0, cloud=0.0, rh_crit=0.8, re_liq=10e-6, re_ice=30e-6, cloud_g=0.85, k_lw_liq=130.0, k_lw_ice=65.0, c_auto=1e-3, q_auto=0.3e-3, v_ice=0.5, l_det=0.5e-3, tau_erode=1e5, k_revap=1e-5, mu_cloud=500.0, theta_rise=0.0, bl_mix=0.0, mu_night=250.0, bl_ent=0.2, bl_cap=0.0, bulk_c=0.0, bulk_gust=1.0, bulk_z=0.0, bulk_w=0.0, plume_ent=0.0, l_ice=0.1e-3, storm_area=1.0e8, graupel_v=5.0, ice_small=0.1, q_shape=0.0, edge_heat=0.0, freeze_dry=0.0, edge_vap=0.0, conv_revap=0.0, plume_root=0.0, bl_q=0.0, sfc=0.0, z0_land=0.1, downdraft=0.0, ent_cloud=0.0, c_deep=0.0, rs_max=0.0, rs_sun=180.0, q_sea=0.0, bl_reach=0.0, dry_edge=0.0, bl_lcl=0.0, edge_k=1.0, cover=0.0, ice_air=0.0, sfc_wind=0.0, revap=0.0, revap_rhs=0.9, revap_rhl=0.7, revap_area=0.05, root_q=0.0, ent_det=0.0, ent_ifs=0.0, shallow=0.0, cu_dth=0.0, cu_dp=0.0, buoy_v=0.0, cu_sub=0.0, bare_fx=0.0, det_all=0.0, cl_float=0.0, ml_clear=0.0, cu_cloud=0.0,
                oro=0.0, oro_g=0.3, oro_cd=1.0, oro_hnc=0.5, oro_ric=0.25,
                gw=0.0, gw_bt=0.004, gw_cw=40.0, gw_lambda=3.0e5, gw_psrc=31500.0, gw_cmax=99.6, gw_dc=1.2)
PAR_ORDER = ('S0', 'asw', 'tau0', 'mu_ref', 'eps_s', 'h_sens', 'u0', 'alb', 'c_slab', 'mu_bl', 'theta_star', 'floor_', 'sigma',
             'grav', 'cp', 'kappa', 'p0', 'tilt', 'year_s', 'day_s', 'tau_lin', 'theta_star2',
             'moist', 'Lv', 'Lf', 'Tf', 'k_evap', 'tau_conv', 'alb_ice', 'rho_ice', 'ice_cover', 'k_ice', 'Rv', 'sat_layer', 'sw_tau',
             'alb_dry', 'alb_wet', 'soil_beta', 'k_inf', 'pond_cover', 'alb_snow', 'alb_ice_melt', 'alb_snow_melt', 'alb_glacier', 'snow_cover', 'alb_ramp', 'land_exch', 'plume_deep', 'lw_w0', 'c_dry', 'c_wet', 'sat_prof', 'land_rs', 'sat_base', 'bl_bot', 'cloud', 'rh_crit', 're_liq', 're_ice', 'cloud_g', 'k_lw_liq', 'k_lw_ice', 'c_auto', 'q_auto', 'v_ice', 'l_det', 'tau_erode', 'k_revap', 'mu_cloud', 'theta_rise', 'bl_mix', 'mu_night', 'bl_ent', 'bl_cap', 'bulk_c', 'bulk_gust', 'bulk_z', 'bulk_w', 'plume_ent', 'l_ice', 'storm_area', 'graupel_v', 'ice_small', 'q_shape', 'edge_heat', 'freeze_dry', 'edge_vap', 'conv_revap', 'plume_root', 'bl_q', 'sfc', 'z0_land', 'downdraft', 'ent_cloud', 'c_deep', 'rs_max', 'rs_sun', 'q_sea', 'bl_reach', 'dry_edge', 'bl_lcl', 'edge_k', 'ice_air', 'revap', 'revap_rhs', 'revap_rhl', 'revap_area', 'root_q', 'ent_det', 'ent_ifs', 'shallow', 'cu_dth', 'cu_dp', 'buoy_v', 'cu_sub', 'bare_fx', 'det_all', 'cl_float', 'ml_clear', 'cu_cloud')


def annual_mean_insolation(lat, S0=1361.0, tilt=TILT, days=365):
    """Annual- and daily-mean insolation (W/m²) at latitudes lat (radians), averaged over `days` days."""
    lat = np.asarray(lat, dtype=float)
    out = np.zeros_like(lat)
    for d in range(days):
        dec = math.asin(math.sin(tilt) * math.sin(2 * math.pi * (d + 0.5) / days))
        x = np.clip(-np.tan(lat) * math.tan(dec), -1, 1)
        h0 = np.arccos(x)
        out += S0 / math.pi * (h0 * np.sin(lat) * math.sin(dec) + np.cos(lat) * math.cos(dec) * np.sin(h0))
    return out / days


class AirPhysics:
    def __init__(self, dg, layers, params=None, mean_sun=False, forcing='radiation', longwave='gray'):
        """forcing: 'radiation' (v1's sun, gray infrared and a slab surface) or 'held_suarez' (Newtonian relaxation of
        θ toward Held & Suarez's equilibrium; no surface). longwave: 'gray' (tau0's profile) or 'ktables' (V2-6a,
        radiation.Longwave, run by radiate())."""
        self.dg, self.L, self.forcing = dg, layers, forcing
        self.p = dict(DEFAULTS, **(params or {}))
        # theta_stars: the interfaces' theta* from the top down, any number (#101); else two layers' or three's
        stars = list(self.p.get('theta_stars') or ([self.p['theta_star2'], self.p['theta_star']] if layers > 2 else [self.p['theta_star']]))
        assert len(stars) == layers - 1, 'one theta* per interface'
        self.p['theta_star'] = stars[-1]
        self.par = cp.asarray([float(self.p[k]) for k in PAR_ORDER] + [float(x) for x in stars] + [0.0] * (LMAX - len(stars)))
        self.mode = 1 if mean_sun else 0
        N = dg.N
        self.insol = cp.asarray(annual_mean_insolation(np.arcsin(np.clip(dg.host.pos[:, 2], -1, 1)), self.p['S0'], self.p['tilt'])) \
            if mean_sun else cp.zeros(1)
        self.pos = cp.asarray(dg.host.pos)
        z = lambda *s: cp.zeros(s)
        self.uabs, self.wx, self.olr, self.sw_abs, self.stored = z(N), z(max(layers - 1, 1), N), z(N), z(N), z(N)
        # compiled for this many layers: the column's arrays then fit in registers, not local memory (V2-5)
        self.k = {n: dg.kernel(n, 'air.cu', (f'-DNL={layers}',)) for n in ('air_column', 'advance_time', 'cell_speed', 'cell_speed_perot', 'hs_column', 'surface_layer')}
        # the surface's wind (#135): 0 the bottom layer's edges' kinetic energy, 1 Perot's reconstructed cell wind
        self.perot = None
        if self.p['sfc_wind'] > 0:
            from .orodrag import operators
            self.perot = cp.asarray(operators(dg.host)[0].reshape(-1), dtype=dg.dtype)
        self.hs = cp.asarray([1.0 / (40 * DAY_S), 1.0 / (4 * DAY_S)])
        self.lat = cp.asarray(np.arcsin(np.clip(dg.host.pos[:, 2], -1, 1)))
        self.qnet = z(N)
        self.moist = bool(self.p['moist'])
        self.rain, self.evap, self.conv, self.tsurf = z(N), z(N), z(N), z(N)
        self.sfc = self.p['sfc'] > 0       # the surface layer's exchange (V2-6b, #114): rho C_D U and rho C_H U per cell
        self.cd, self.ch = (z(N), z(N)) if self.sfc else (None, None)
        self.dgo = None                     # optional diagnostics array (rows x N; air_column), set by a caller
        self.cover = None                   # land cover per cell (8 x N, cases.earth_cover; #132), set by World
        self.cloudo = z(4 * layers + 3, N)  # per layer cover, total cover, cloud effect on sunlight and longwave at the top (W/m²), per layer in-cloud longwave and shortwave optical depths (k-tables), and the liquid's share of the latter (#154)
        self.flash = z(N)                  # lightning, flashes/s/m² (#104)
        self.rainp = z(4, N)               # rain by source: condensation above the bottom layer, in it, convection; and the snow in all of it (#134)
        self._none = z(1)                                     # q and ice for a dry column (never read)
        self.lw = self.sw = None
        if longwave == 'ktables':                     # (with the k-tables' shortwave)
            from .radiation import Longwave, Shortwave
            self.lw = Longwave(dg, layers, self.par, **{k: self.p[k] for k in ('co2', 'n2o', 'ch4', 'tmode') if k in self.p})
            self.sw = Shortwave(dg, layers, self.par, self.lw, **({'inhom': self.p['cloud_inhom']} if 'cloud_inhom' in self.p else {}),
                                ice_re=self.p['re_ice'] if self.p.get('ice_sw', 0) > 0 else None)
            self.swalb = cp.full(N, self.p['alb'])

    def _speed(self, u):
        """The surface's wind speed at cells from the bottom layer's edge winds, into self.uabs."""
        g, i32 = self.dg, np.int32
        if self.perot is not None:
            self.k['cell_speed_perot'](*grid_dims(g.N), (i32(g.N), u[self.L - 1], g.cellEdge, self.perot, self.uabs))
        else:
            self.k['cell_speed'](*grid_dims(g.N), (i32(g.N), u[self.L - 1], g.cellEdge, g.le, g.de, g.areaCell, self.uabs))

    def radiate(self, mu, theta, Ts, t, interval, q=None, cw=None):
        """The k-tables' longwave and shortwave from the present state (the sun over the coming interval, s), saved for the
        column steps until the next call."""
        if self.lw is not None:
            co = self.cloudo if cw is not None else None
            self.lw(mu, theta, q, co, Ts)
            self.sw(self.mode, t, interval, self.pos, self.swalb, co)

    def surface(self, mu, theta, Ts, u, q=None, ice=None, land=None):
        """The surface layer's exchange coefficients from this step's bottom wind, theta and surface (V2-6b, #114): self.cd,
        self.ch = rho C_D U, rho C_H U (kg/m²/s), the drag's for the wind stress and the heat's and vapor's for the column."""
        g, k, i32 = self.dg, self.k, np.int32
        self._speed(u)
        k['surface_layer'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, self.uabs, mu, theta, q if q is not None else NULL, Ts,
                                             ice if ice is not None else NULL, land if land is not None else NULL, self.cd, self.ch,
                                             self.cover if self.cover is not None else NULL))

    def column(self, mu, theta, Ts, u, t, dt, q=None, ice=None, cap=None, qocean=None, qsw=None, land=None, soilcap=None, soil=None, pond=None, snow=None, cw=None, elev=None):
        """Radiation, surface exchange, heating, (moist: evaporation, condensation, sea ice, convection) and the θ*
        crossing for one step; the masses moved up go to self.wx (moved with momentum and tracers by Layers.exchange),
        rain and evaporation (kg/m²/s) to self.rain, self.evap. cap: the surface water's heat capacity per cell (J/m²/K;
        the ocean's mixed layer, V2-3), else the slab's; qocean receives the net flux into it (W/m²), qsw its sunlight;
        land, soilcap, soil, pond: the land mask, soil bucket and surface water (#84, #85)."""
        g, k, i32, R = self.dg, self.k, np.int32, self.dg.dtype.type
        if self.forcing == 'held_suarez':
            k['hs_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, self.hs, self.lat, mu, theta, R(dt), self.wx, self.qnet,
                                             self.stored))
            k['advance_time']((1,), (1,), (t, R(dt)))
            return
        self._speed(u)
        k['air_column'](*grid_dims(g.N), (i32(g.N), i32(self.L), self.par, i32(self.mode), t, self.insol, self.pos, self.uabs,
                                          mu, theta, q if q is not None else self._none, Ts, ice if ice is not None else self._none, R(dt),
                                          self.wx, self.olr, self.sw_abs, self.stored, self.rain, self.evap, self.conv,
                                          self.tsurf, self.rainp, self.dgo if self.dgo is not None else NULL,
                                          cap if cap is not None else NULL, qocean if qocean is not None else NULL,
                                          qsw if qsw is not None else NULL, land if land is not None else NULL,
                                          soilcap if soilcap is not None else NULL, soil if soil is not None else NULL, pond if pond is not None else NULL, snow if snow is not None else NULL, cw if cw is not None else NULL, self.cloudo if cw is not None else NULL, elev if elev is not None else NULL, self.flash,
                                          self.lw.radf if self.lw is not None else NULL,
                                          self.sw.swf if self.sw is not None else NULL, self.swalb if self.sw is not None else NULL,
                                          self.ch if self.sfc else NULL, self.cover if self.cover is not None else NULL))
        k['advance_time']((1,), (1,), (t, R(dt)))


def held_suarez_theta(p, lat):
    """Held & Suarez's equilibrium potential temperature at pressure p (Pa) and latitude lat (radians)."""
    s2 = np.sin(lat) ** 2
    T = np.maximum(200.0, (315.0 - 60.0 * s2 - 10.0 * np.log(p / P0) * (1 - s2)) * (p / P0) ** KAPPA)
    return T * (P0 / p) ** KAPPA
