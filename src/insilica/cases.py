"""Initial states. Each case gives (h (L, N), b (N), u (L, E)) as NumPy arrays and the layer settings it needs.

Single-layer benchmarks (Williamson et al. 1992; Galewsky et al. 2004) as in v1's numerics tests, and the
stacked-layer cases of V2-1: external and internal gravity waves, two-layer solid-body flows (Phillips baroclinic
instability, a balanced ocean), and an outcrop over a ridge.
"""
import math
from pathlib import Path

import numpy as np

from .constants import CP_DRY, GRAV, KAPPA, OMEGA, P0, R_PLANET


def edge_east_north(g):
    """Latitude, longitude and the east/north components of each edge's normal, at the Voronoi edge midpoint xm,
    where analytic winds are sampled (SPEC §9 #17; v1's edgeInfo used xe)."""
    x = g.xm
    lat = np.arcsin(np.clip(x[:, 2], -1, 1))
    lon = np.arctan2(x[:, 1], x[:, 0])
    en = -np.sin(lon) * g.ne[:, 0] + np.cos(lon) * g.ne[:, 1]
    nn = (-np.sin(lat) * np.cos(lon) * g.ne[:, 0] - np.sin(lat) * np.sin(lon) * g.ne[:, 1] + np.cos(lat) * g.ne[:, 2])
    return lat, lon, en, nn


def great_circle(g, lon_c, lat_c):
    c = np.array([math.cos(lat_c) * math.cos(lon_c), math.cos(lat_c) * math.sin(lon_c), math.sin(lat_c)])
    return R_PLANET * np.arccos(np.clip(g.pos @ c, -1, 1))


def cosine_bell(g, lon_c=3 * math.pi / 2, lat_c=0.0, radius=R_PLANET / 3, height=1000.0):
    """Williamson 1's cosine bell (v1 N6): height/2·(1 + cos(π r / R)) within great-circle distance R."""
    dist = great_circle(g, lon_c, lat_c)
    return np.where(dist < radius, 0.5 * height * (1 + np.cos(math.pi * dist / radius)), 0.0)


def default_dt(n):
    """v1's numerics-test steps: 450 s at n = 4, halved per level."""
    return 450.0 * 2.0 ** (4 - n)


def initial_state(g, preset):
    """One layer (h, b, u): Williamson 2 (steady zonal flow), Williamson 5 (zonal flow over an isolated mountain),
    Williamson 6 / 'rh4' (Rossby–Haurwitz wave 4) or the Galewsky jet (a balanced jet with a small bump)."""
    a, Om = R_PLANET, OMEGA
    lat_e, lon_e, en, nn = edge_east_north(g)
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
    elif preset == 'rh4':
        om = Kk = 7.848e-6
        R, h0 = 4, 8000.0
        c = np.cos(g.lat)
        A = om / 2 * (2 * Om + om) * c * c + 0.25 * Kk * Kk * c ** (2 * R) * ((R + 1) * c * c + (2 * R * R - R - 2) - 2 * R * R / (c * c))
        B = 2 * (Om + om) * Kk / ((R + 1) * (R + 2)) * c ** R * ((R * R + 2 * R + 2) - (R + 1) ** 2 * c * c)
        C = 0.25 * Kk * Kk * c ** (2 * R) * ((R + 1) * c * c - (R + 2))
        h = h0 + a * a * (A + B * np.cos(R * g.lon) + C * np.cos(2 * R * g.lon)) / GRAV
        ce, se = np.cos(lat_e), np.sin(lat_e)
        ue = a * om * ce + a * Kk * ce ** (R - 1) * (R * se * se - ce * ce) * np.cos(R * lon_e)
        vn = -a * Kk * R * ce ** (R - 1) * se * np.sin(R * lon_e)
        u = ue * en + vn * nn
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
        raise ValueError(f'unknown single-layer case {preset!r}')
    return h, b, u


def gravity_wave(g, H, amp=10.0, radius=1.0e6):
    """One layer of depth H at rest with a Gaussian bump of amplitude amp (m) at (0°, 0°)."""
    r = great_circle(g, 0.0, 0.0)
    return (H + amp * np.exp(-(r / radius) ** 2))[None], np.zeros(g.N), np.zeros((1, g.E))


def internal_wave(g, h1, h2, amp=50.0, radius=1.0e6):
    """Two layers at rest, flat free surface, the interface raised by a Gaussian bump (lower layer thicker)."""
    d = amp * np.exp(-(great_circle(g, 0.0, 0.0) / radius) ** 2)
    return np.stack([h1 - d, h2 + d]), np.zeros(g.N), np.zeros((2, g.E))


def internal_wave_speed(gprime, h1, h2, grav=GRAV):
    """Exact internal-mode speed of two Boussinesq layers with a free surface."""
    H = h1 + h2
    return math.sqrt(grav * H / 2 * (1 - math.sqrt(1 - 4 * gprime * h1 * h2 / (grav * H * H))))


def layer_modes(h, gprime, grav=GRAV):
    """Gravity-wave modes of Boussinesq layers h (top first) at rest under a free surface: speeds (fastest, the external
    mode, first), thickness structures V (columns) and W = V⁻¹ (rows project thickness anomalies onto the modes)."""
    L, gg = len(h), [grav] + list(gprime)
    G = np.array([[sum(gg[i] for i in range(k + 1) if j >= i) for j in range(L)] for k in range(L)])
    lam, V = np.linalg.eig(np.diag(np.asarray(h, dtype=float)) @ G)
    o = np.argsort(lam.real)[::-1]
    V = V.real[:, o]
    return np.sqrt(lam.real[o]), V, np.linalg.inv(V)


def mode_bump(g, h, v, amp=10.0, radius=5.0e5):
    """Layers h at rest with a Gaussian bump at (0°, 0°) in the thickness structure v (scaled so that no layer changes by
    more than amp: a linear wave)."""
    d = amp * np.exp(-(great_circle(g, 0.0, 0.0) / radius) ** 2)
    v = np.asarray(v, dtype=float)
    return np.asarray(h, dtype=float)[:, None] + (v / np.abs(v).max())[:, None] * d[None, :]


def solid_body_layers(g, H, U, gprime, b=None):
    """Two layers in steady solid-body zonal rotation, u_k = U_k cos φ, each in exact gradient-wind balance:
    M_k = C_k − (a Ω U_k + U_k²/2) sin²φ, with M_0 = gη and M_1 = gη + g' z_1. Layer area means are H."""
    a, s2 = R_PLANET, np.sin(g.lat) ** 2
    b = np.zeros(g.N) if b is None else b
    m = [-(a * OMEGA * Uk + Uk * Uk / 2) * s2 for Uk in U]
    A = g.areaCell / np.sum(g.areaCell)
    eta_v, z1_v = m[0] / GRAV, (m[1] - m[0]) / gprime
    eta = eta_v + H[0] + H[1] + np.sum(A * b) - np.sum(A * eta_v)
    z1 = z1_v + H[1] + np.sum(A * b) - np.sum(A * z1_v)
    lat_e, _, en, _ = edge_east_north(g)
    return np.stack([eta - z1, z1 - b]), b, np.stack([Uk * np.cos(lat_e) * en for Uk in U])


def zonal_jet_layers(g, H, ujets, gprime, M=20000):
    """Two layers with zonal winds u_k = ujets[k](φ), each in exact gradient-wind balance, integrated from the
    south pole: dM_k/dφ = −a u_k (f + u_k tan φ / a), with M_0 = gη and M_1 = gη + g' z_1. Layer area means are H."""
    a = R_PLANET
    dphi = math.pi / M
    ph = -math.pi / 2 + (np.arange(1, M + 1) - 0.5) * dphi
    Ms = []
    for uj in ujets:
        uu = uj(ph)
        Ms.append(np.concatenate([[0.0], np.cumsum(-a * uu * (2 * OMEGA * np.sin(ph) + np.tan(ph) * uu / a) * dphi)]))
    x = (g.lat + math.pi / 2) / dphi
    k = np.minimum(M - 1, np.floor(x).astype(int))
    at = lambda m: m[k] + (x - k) * (m[k + 1] - m[k])
    m0, m1 = at(Ms[0]), at(Ms[1])
    A = g.areaCell / np.sum(g.areaCell)
    eta_v, z1_v = m0 / GRAV, (m1 - m0) / gprime
    eta = eta_v + H[0] + H[1] - np.sum(A * eta_v)
    z1 = z1_v + H[1] - np.sum(A * z1_v)
    lat_e, _, en, _ = edge_east_north(g)
    return np.stack([eta - z1, z1]), np.zeros(g.N), np.stack([uj(lat_e) * en for uj in ujets])


def phillips(g, shear_factor, H=(5000.0, 5000.0), Ld=5.0e5, width_deg=7.0, amp=0.01):
    """Phillips two-layer baroclinic instability on a jet: equal layers, the lower at rest, the upper a Gaussian
    jet at 45°N (e-folding half-width width_deg) whose core speed is shear_factor × the critical shear 2 β L_d² (β, f
    at 45°; L_d the internal deformation radius there), each layer in gradient-wind balance, plus a wavenumber-6
    interface bump of relative size amp at 45°N. Returns (h, b, u, gprime, h_unperturbed)."""
    lat0 = math.pi / 4
    f0, beta = 2 * OMEGA * math.sin(lat0), 2 * OMEGA * math.cos(lat0) / R_PLANET
    He = H[0] * H[1] / (H[0] + H[1])
    gprime = (Ld * f0) ** 2 / He
    U0 = shear_factor * 2 * beta * Ld ** 2
    sig = math.radians(width_deg)
    jet = lambda phi: U0 * np.exp(-((np.asarray(phi) - lat0) / sig) ** 2)
    h, b, u = zonal_jet_layers(g, H, (jet, lambda phi: 0.0 * np.asarray(phi)), gprime)
    bump = amp * H[1] * np.cos(6 * g.lon) * np.exp(-((g.lat - lat0) / math.radians(10)) ** 2)
    hp = h.copy()
    hp[0] -= bump
    hp[1] += bump
    return hp, b, u, gprime, h


def outcrop(g, H=(2000.0, 500.0), U=0.5, ridge=900.0, gprime=0.02):
    """Two ocean-like layers flowing at U cos φ over a meridional ridge at 0° longitude whose crest is higher than
    the lower layer is deep: over the crest the lower layer starts at the floor and the upper layer meets the ridge."""
    b = ridge * np.exp(-(np.abs(np.angle(np.exp(1j * g.lon))) / math.radians(8)) ** 2) * np.cos(g.lat) ** 2
    h, _, u = solid_body_layers(g, H, (U, U), gprime)
    lower = np.maximum(h[1] - b, 1.0)
    upper = h[0] + h[1] - b - lower
    return np.stack([upper, lower]), b, u


# Cases a World can start from: name → (layers, builder(g) → (h, b, u), settings). dt is per level n.
def _single(name):
    return lambda g: tuple(np.asarray(x)[None] if i != 1 else x for i, x in enumerate(initial_state(g, name)))


PRESETS = {
    'w2': dict(build=_single('w2'), dt=default_dt),
    'w5': dict(build=_single('w5'), dt=default_dt),
    'rh4': dict(build=_single('rh4'), dt=lambda n: 300.0 * 2.0 ** (4 - n)),
    'galewsky': dict(build=_single('galewsky'), dt=default_dt),
    'phillips': dict(build=lambda g: phillips(g, 3.0)[:3], gprime=lambda g: phillips(g, 3.0)[3], dt=lambda n: 1200.0 * 2.0 ** (5 - n),
                     visc4_U=20.0),
    'outcrop': dict(build=lambda g: outcrop(g), gprime=0.02, dt=lambda n: 3600.0, stress=[0.0, 2.5e-4], visc4_U=0.5),
}


# ---- Air (stage V2-2a) ----
MU_REF = 1.0e5 / GRAV                       # kg/m², the column mass of 1000 hPa


AQUA_P = (0.22e5, 0.60e5)                  # Pa, the starting interfaces of the three air layers (decision #45)


def aqua(g):
    """Three air layers at rest over a slab aquaplanet (decision #45): interfaces at 220 and 600 hPa, the top layer's θ
    415 K and the middle's 342.5 K (between θ*₂ = 380 and θ* = 305 K), the bottom layer's 290 − 30 sin²φ K, and v1's
    starting surface temperature 288 − 40 (sin²φ − 1/3) K, for moist air no colder than 274 K (so the run doesn't start
    under metres of ice). Vapor: 10% / 50% / 70% of saturation in the top / middle / bottom layer at its mid-pressure."""
    s2 = np.sin(g.lat) ** 2
    edges = np.array([0.0, *AQUA_P, MU_REF * GRAV])
    h = np.repeat((np.diff(edges) / GRAV)[:, None], g.N, axis=1)
    theta = np.stack([np.full(g.N, 415.0), np.full(g.N, 342.5), 290.0 - 30.0 * s2])
    Ts = np.maximum(288.0 - 40.0 * (s2 - 1.0 / 3.0), 274.0)
    pm = (0.5 * (edges[:-1] + edges[1:]))[:, None]
    T = theta * (pm / P0) ** KAPPA
    qs = 0.622 * 611.0 * np.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm
    q = np.array([0.1, 0.5, 0.7])[:, None] * qs
    return h, np.zeros(g.N), np.zeros((3, g.E)), dict(theta=theta, Ts=Ts, q=q, ice=np.zeros(g.N))


def aqua_dry(g):
    """The aquaplanet's dry start: v1's surface temperature 288 − 40 (sin²φ − 1/3) K, no vapor."""
    h, b, u, x = aqua(g)
    return h, b, u, dict(theta=x['theta'], Ts=288.0 - 40.0 * (np.sin(g.lat) ** 2 - 1.0 / 3.0))


AIR = dict(t_ref=288.0)
AIR3 = dict(theta_star=305.0, theta_star2=380.0)             # the three-layer air's interfaces (decisions #45, #51)
# Surface drag on the bottom layer: Held & Suarez's k_f = 1/day over sigma > 0.7, integrated, c = 0.15 M / day ~ 0.017
# kg/m^2/s, close to an Earth-like rho C_D |U| (decision #57). No interfacial stress (#56).
SURFACE_DRAG = 0.15 * MU_REF / 86400.0
PRESETS['aqua'] = dict(build=aqua, dt=lambda n: 900.0 * 2.0 ** (5 - n), visc4_U=15.0, stress=[0.0, 0.0, SURFACE_DRAG],
                       tracer=False, air=dict(AIR, physics=dict(AIR3, moist=1.0)))
def aqua_layers(g, stars, edges, q_top=0.1, q_bot=0.7):
    """The aquaplanet's start in L = len(stars) + 1 layers (#101): stars, the interfaces' theta* from the top down; edges, their starting pressures (Pa). Each layer's theta midway between its edges, the top layer's above its edge and the bottom's the aquaplanet's; vapor rising from q_top to q_bot of saturation down the column."""
    L = len(stars) + 1
    s2 = np.sin(g.lat) ** 2
    e = np.array([0.0, *edges, MU_REF * GRAV])
    h = np.repeat((np.diff(e) / GRAV)[:, None], g.N, axis=1)
    th = [np.full(g.N, stars[0] + 35.0)] + [np.full(g.N, 0.5 * (stars[k - 1] + stars[k])) for k in range(1, L - 1)] + [290.0 - 30.0 * s2]
    theta = np.stack(th)
    Ts = np.maximum(288.0 - 40.0 * (s2 - 1.0 / 3.0), 274.0)
    pm = (0.5 * (e[:-1] + e[1:]))[:, None]
    T = theta * (pm / P0) ** KAPPA
    q = np.linspace(q_top, q_bot, L)[:, None] * (0.622 * 611.0 * np.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm)
    return h, np.zeros(g.N), np.zeros((L, g.E)), dict(theta=theta, Ts=Ts, q=q, ice=np.zeros(g.N))


def _aqua_n(stars, edges):
    return dict(PRESETS['aqua'], build=lambda g: aqua_layers(g, stars, edges), stress=[0.0] * len(stars) + [SURFACE_DRAG],
                air=dict(AIR, physics=dict(moist=1.0, theta_stars=stars)))


PRESETS['aqua_dry'] = dict(PRESETS['aqua'], build=aqua_dry, air=dict(AIR, physics=AIR3))
PRESETS['aqua5'] = _aqua_n([400.0, 350.0, 325.0, 305.0], [1.0e4, 2.5e4, 4.5e4, 8.0e4])
PRESETS['aqua7'] = _aqua_n([420.0, 370.0, 345.0, 330.0, 317.0, 305.0], [0.8e4, 1.8e4, 3.0e4, 4.5e4, 6.2e4, 8.0e4])
# V2-3: the moist three-layer air over the three-layer ocean (decisions #64–#79): the air's drag on the surface is the wind
# stress on the ocean (exact, in Ocean.couple), so the air's own bottom stress is off. Air steps must divide the hour.
# Evaporation recalibrated for the ocean's surface (K_EVAP 4.8e-3 against the slab's 4e-3: 2.0 → ~2.3 mm/day; #80).
PRESETS['coupled'] = dict(PRESETS['aqua'], stress=[0.0, 0.0, 0.0], ocean=dict(surface_drag=SURFACE_DRAG),
                          air=dict(AIR, physics=dict(AIR3, moist=1.0, k_evap=4.8e-3)))
# One ocean basin (decision #79): the coupled world with a thin pole-to-pole wall at 180° that the ocean cannot cross, so
# that gyres, western boundary currents and an east–west tilt of the equatorial thermocline can form; the air is unchanged.
PRESETS['basin'] = dict(PRESETS['coupled'], ocean=dict(surface_drag=SURFACE_DRAG, walls=(180.0,)))


def _over_terrain(g, b):
    """The aquaplanet's start over terrain b (m): upper interfaces flat, the bottom layer thinned so the air is at rest in
    hydrostatic balance (g b + θ π(p_s) unchanged), the surface colder by 6.5 K/km, the vapor as the aquaplanet's."""
    h, _, u, x = aqua(g)
    h = h.copy()
    ps0 = h.sum(0) * GRAV
    pi0 = (ps0 / P0) ** KAPPA
    ps = P0 * np.maximum(pi0 - GRAV * b / (CP_DRY * x['theta'][2]), 1e-3) ** (1 / KAPPA)
    h[2] -= (ps0 - ps) / GRAV
    # ground above the bottom layer's top: it keeps a thin 25 hPa, and the ground rises on through the middle layer's theta
    high = h[2] < 2500.0 / GRAV
    if high.any():
        pi_i = (AQUA_P[1] / P0) ** KAPPA
        z_i = CP_DRY * x['theta'][2] * (pi0 - pi_i) / GRAV
        pst = P0 * np.maximum(pi_i - GRAV * (b - z_i) / (CP_DRY * x['theta'][1]), 1e-3) ** (1 / KAPPA)
        ps_h = np.maximum(pst, h[0] * GRAV + 3000.0)
        h[2] = np.where(high, 2500.0 / GRAV, h[2])
        h[1] = np.where(high, (ps_h - 2500.0) / GRAV - h[0], h[1])
    return h, b, u, dict(x, Ts=x['Ts'] - 6.5e-3 * b)


def ridge_height(g):
    """v1's T3.6 ridge at 35–55° in both hemispheres: 2 km within 5° of longitude 0, 1 km out to 15° (m)."""
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    band = (np.abs(lat) >= 35) & (np.abs(lat) <= 55)
    return np.where(band & (np.abs(lon) < 5), 2000.0, np.where(band & (np.abs(lon) < 15), 1000.0, 0.0))


def continent_height(g):
    """One land mass within 30° of longitude 0 and 70° of the equator, 500 m high, open sea at the poles (m)."""
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    return np.where((np.abs(lat) < 70) & (np.abs(lon) < 30), 500.0, 0.0)


def relief_height(g):
    """V2-4b's land (#85): a continent falling from 1500 m in the east to 150 m at its west coast, with a closed basin near 20°N,
    and high polar land north of 68° rising to 1800 m at the pole (m)."""
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    main = (np.abs(lon) < 35) & (lat > -55) & (lat < 60)
    h = np.where(main, 150.0 + 1350.0 * (lon + 35.0) / 70.0, 0.0)
    r = np.hypot(lon + 5.0, lat - 20.0)
    h = np.where(main & (r < 12.0), np.minimum(h, 100.0 + 75.0 * r), h)
    return np.where(lat > 68.0, 300.0 + 1500.0 * (lat - 68.0) / 22.0, h)


# V2-4a (#84): land. `ridge`: mountains in the westerlies on the slab aquaplanet; `continent`: one land mass in the coupled ocean.
PRESETS['ridge'] = dict(PRESETS['aqua'], build=lambda g: _over_terrain(g, ridge_height(g)), land=lambda g: ridge_height(g) > 0)
PRESETS['continent'] = dict(PRESETS['coupled'], build=lambda g: _over_terrain(g, continent_height(g)),
                            land=lambda g: continent_height(g) > 0)
PRESETS['relief'] = dict(PRESETS['coupled'], build=lambda g: _over_terrain(g, relief_height(g)), land=lambda g: relief_height(g) > 0)


_EARTH = {}


def earth_land(g):
    """Modern Earth on grid g, from the 1-degree table of ETOPO1 (tools/make_earth_topo.py): each 1-degree cell goes to its nearest
    grid cell by area; land where over half the grid cell is land, at its land's mean height (m); the sea's mean depth elsewhere (m). Returns (land, height, depth)."""
    if g.N not in _EARTH:
        d = np.load(Path(__file__).parent / 'data' / 'earth_topo_1deg.npz')
        lat, lon = np.radians(np.arange(180) - 89.5), np.radians(np.arange(360) - 179.5)
        la, lo = np.meshgrid(lat, lon, indexing='ij')
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1).reshape(-1, 3)
        near = np.concatenate([np.argmax(xyz[i:i + 2000] @ g.pos.T, axis=1) for i in range(0, len(xyz), 2000)])
        w, f = np.cos(la).ravel(), d['land'].astype(float).ravel()
        A = np.bincount(near, w, g.N)
        fl = np.bincount(near, w * f, g.N)
        hl = np.bincount(near, w * f * d['elev'].astype(float).ravel(), g.N)
        land = fl > 0.5 * np.maximum(A, 1e-30)
        sea = np.bincount(near, w * (1 - f) * d['depth'].astype(float).ravel(), g.N)
        _EARTH[g.N] = land, np.where(land, hl / np.maximum(fl, 1e-30), 0.0), sea / np.maximum(A - fl, 1e-30)
    return _EARTH[g.N]


_SSO = {}


def earth_sso(g):
    """The sub-grid orography of each grid cell for the mountain drag (V2-6b, #127; Lott & Miller 1997 with Baines & Palmer 1990's
    statistics), from ETOPO1's 1-arc-minute heights (the sea at its surface; tools/make_earth_sso.py), each 1-degree cell going to its
    nearest grid cell as in earth_land: mu, the standard deviation of the heights about the cell's mean (m); sigma, the mean slope
    along the principal axis; gamma, the anisotropy (1 a round hill, 0 a long ridge); theta, the principal axis's angle from east
    (radians). Returns (mu, sigma, gamma, theta), each (N,)."""
    if g.N not in _SSO:
        d = np.load(Path(__file__).parent / 'data' / 'earth_sso_1deg.npz')
        lat, lon = np.radians(np.arange(180) - 89.5), np.radians(np.arange(360) - 179.5)
        la, lo = np.meshgrid(lat, lon, indexing='ij')
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1).reshape(-1, 3)
        near = np.concatenate([np.argmax(xyz[i:i + 2000] @ g.pos.T, axis=1) for i in range(0, len(xyz), 2000)])
        w = np.cos(la).ravel() / np.maximum(d['n'].ravel(), 1)       # each point weighted by its area
        s = {k: np.bincount(near, w * d[k].ravel(), g.N) for k in ('n', 's1', 's2', 'gxx', 'gyy', 'gxy')}
        n = np.maximum(s['n'], 1e-30)
        mu = np.sqrt(np.maximum(s['s2'] / n - (s['s1'] / n) ** 2, 0.0))
        K, Lg, M = s['gxx'] / n, s['gyy'] / n, s['gxy'] / n
        Hp, Kp, Lp = 0.5 * (K + Lg), 0.5 * (K - Lg), M
        r = np.sqrt(Kp ** 2 + Lp ** 2)
        sigma = np.sqrt(np.maximum(Hp + r, 0.0))
        gamma = np.sqrt(np.maximum(Hp - r, 0.0) / np.maximum(Hp + r, 1e-30))
        _SSO[g.N] = mu, sigma, gamma, 0.5 * np.arctan2(Lp, Kp)
    return _SSO[g.N]


# Land cover (V2-6b, #132): Noah's values (WRF's VEGPARM.TBL, MODIFIED_IGBP_MODIS_NOAH; mean of its seasonal minimum and maximum) for
# tools/make_earth_cover.COVER's classes; CLM's plant types as Noah's classes: needleleaf evergreen (1), broadleaf evergreen (2),
# needleleaf deciduous (3), broadleaf deciduous (4), shrub (closed shrubland, 6), tundra (CLM's boreal shrub and arctic grass:
# mixed tundra, 19), grass (10), crop (12); bare ground is Noah's barren (16) for roughness and snow, its albedo the soil's own.
# Per class: snow-free albedo, roughness z0 (m), minimum stomatal resistance RS (s/m), RGL (W/m²), deep snow's albedo MAXALB.
COVER = dict(alb=[0.38, 0.12, 0.12, 0.145, 0.165, 0.275, 0.175, 0.21, 0.20],
             z0=[0.01, 0.50, 0.50, 0.50, 0.50, 0.03, 0.15, 0.11, 0.10],
             rs=[0.0, 125.0, 150.0, 150.0, 100.0, 300.0, 150.0, 40.0, 40.0],
             rgl=[0.0, 30.0, 30.0, 30.0, 30.0, 100.0, 100.0, 100.0, 100.0],
             maxalb=[0.75, 0.52, 0.35, 0.54, 0.58, 0.60, 0.60, 0.70, 0.66])
_COVER = {}


def earth_cover(g):
    """Each land cell's surface from its land cover (V2-6b, #132; tools/make_earth_cover.py, NCAR's CLM data from MODIS), each 1-degree
    cell going to its nearest grid cell as in earth_land, by land area. Returns (8, N): roughness z0 (m, the cover-weighted mean of
    log z0); the bare soil's albedo dry and wet (broadband, CLM's soil colours); the bare share; the plants' snow-free albedo (over
    the plants' share); the share of deep snow's brightening the cover shows, v = (MAXALB - alb) / (MAXALB_bare - alb) by class
    (1 bare: snow covers it; under trees the snow shows only between them); the plants' minimum stomatal resistance (s/m, their
    conductances added by share) and rs_sun = RGL / 0.55 (W/m², Noah's F1 with LAI 2, as #118). Cells with no land data take the
    land's mean."""
    if g.N not in _COVER:
        d = np.load(Path(__file__).parent / 'data' / 'earth_cover_1deg.npz')
        lat, lon = np.radians(np.arange(180) - 89.5), np.radians(np.arange(360) - 179.5)
        la, lo = np.meshgrid(lat, lon, indexing='ij')
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1).reshape(-1, 3)
        near = np.concatenate([np.argmax(xyz[i:i + 2000] @ g.pos.T, axis=1) for i in range(0, len(xyz), 2000)])
        w = d['land'].astype(float).ravel()                              # land area per 1-degree cell
        A = np.bincount(near, w, g.N)
        sh = np.stack([np.bincount(near, w * s.astype(float).ravel(), g.N) for s in d['share']])
        dry, sat = (np.bincount(near, w * d[k].astype(float).ravel(), g.N) for k in ('dry', 'sat'))
        tot = w.sum()
        has = A > 0
        sh = np.where(has, sh / np.maximum(A, 1e-30), (sh.sum(1) / tot)[:, None])
        dry = np.where(has, dry / np.maximum(A, 1e-30), dry.sum() / tot)
        sat = np.where(has, sat / np.maximum(A, 1e-30), sat.sum() / tot)
        c = {k: np.asarray(v)[:, None] for k, v in COVER.items()}
        fb, veg = sh[0], np.maximum(1 - sh[0], 1e-12)
        z0 = np.exp(np.sum(sh * np.log(c['z0']), 0))
        alb_veg = np.where(sh[0] < 1, np.sum(sh[1:] * c['alb'][1:], 0) / veg, COVER['alb'][7])
        v = c['maxalb'] - c['alb']
        v = v / (COVER['maxalb'][0] - c['alb'])
        snow = np.sum(sh * v, 0) / np.sum(sh, 0)
        gs = np.sum(sh[1:] / c['rs'][1:], 0)
        rs = np.where(gs > 0, veg / np.maximum(gs, 1e-30), COVER['rs'][7])
        rs_sun = np.where(sh[0] < 1, np.sum(sh[1:] * c['rgl'][1:], 0) / veg, COVER['rgl'][7]) / 0.55
        _COVER[g.N] = np.stack([z0, dry, sat, fb, alb_veg, snow, rs, rs_sun])
    return _COVER[g.N]


def earth(g, rise=5.0e-3):
    """The start over Earth's terrain with the air bent over it: the top layer as the aquaplanet's and the rest of each column's
    air shared between the middle and bottom layers as at sea level, so both thin over high ground; the bottom layer's theta rises
    by `rise` (K/m, as theta_rise; decision #96) and the surface cools by 6.5 K/km. The ice sheets lie on Greenland and Antarctica (100 m;
    their height is already in the table)."""
    land, b, _ = earth_land(g)
    h, _, u, x = aqua(g)
    h, theta = h.copy(), x['theta'].copy()
    theta[-1] += rise * b
    ps0 = h.sum(0) * GRAV
    ps = P0 * np.maximum((ps0 / P0) ** KAPPA - GRAV * b / (CP_DRY * 0.5 * (theta[-2] + theta[-1])), 1e-3) ** (1 / KAPPA)
    h[1:] = h[1:] * (ps / GRAV - h[0]) / h[1:].sum(0)
    lat, lon = np.degrees(g.lat), np.degrees(np.angle(np.exp(1j * g.lon)))
    sheet = land & ((lat < -60) | ((lat > 60) & (lon > -75) & (lon < -10)))
    return h, b, u, dict(x, theta=theta, Ts=x['Ts'] - 6.5e-3 * b, ice=np.where(sheet, 100.0 * 917.0, 0.0))


def earth_bed(g):
    """The sea floor (m below sea level; land cells, walled off, keep the mean depth)."""
    land, _, depth = earth_land(g)
    return np.where(land, 4000.0, depth)


# Modern Earth's land and sea floor over the coupled ocean. Its physics is V2-4c's and V2-4e's (#88, #91, #94, #95), the rest of the worlds still without; land_drag 8 goes with land_exch 2.
# V2-6a (#109-#112): the k-tables' radiation, and the moist physics that put its troposphere on the observed profile (owner: option C):
# the surface air's vapor (q_shape), freeze-drying, edge vapor, plumes rooted at the surface, entraining 0.1 per 100 hPa, their rain evaporating.
PRESETS['earth'] = dict(PRESETS['coupled'], build=earth, land=lambda g: earth_land(g)[0], ocean=dict(surface_drag=SURFACE_DRAG, bed=earth_bed, land_drag=8.0),
                        air=dict(AIR, longwave='ktables', physics=dict(AIR3, moist=1.0, k_evap=4.8e-3, plume_deep=1.0, sat_prof=1.0, bl_bot=0.25, cloud=1.0, tau0=1.3, land_exch=2.0,
                                                                              q_shape=1.0, freeze_dry=1.0, edge_vap=1.0, plume_root=1.0, plume_ent=1e-5, conv_revap=0.05)))
def earth4(g, rise=5.0e-3, bl=1.0e4):
    """The four-layer start (V2-6b, #113): earth's three layers with its bottom layer split, its lowest bl (Pa) the mixed
    boundary layer (theta as the bottom layer's, the air above it 4 K warmer, below 305 K); vapor 80% of saturation in the
    boundary layer, 60% above it."""
    h3, b, u3, x = earth(g, rise)
    h = np.concatenate([h3[:2], np.maximum(h3[2:] - bl / GRAV, h3[2:] * 0.3), np.minimum(np.full_like(h3[2:], bl / GRAV), h3[2:] * 0.7)])
    th = np.concatenate([x['theta'][:2], np.minimum(x['theta'][2:] + 4.0, 304.0), x['theta'][2:]])
    th[2] = np.maximum(th[2], th[3] + 1.0)
    ps = h.sum(0) * GRAV
    pt = ps - 0.5 * h[3] * GRAV
    pm = np.stack([ps - h[3] * GRAV - 0.5 * h[2] * GRAV, pt])
    T = th[2:] * (pm / P0) ** KAPPA
    qs = 0.622 * 611.0 * np.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm
    q = np.concatenate([x['q'][:2], np.array([0.6, 0.8])[:, None] * qs])
    return h, b, np.concatenate([u3, u3[2:]]), dict(x, theta=th, q=q)


# V2-6b (#113): the boundary layer as a fourth air layer: its top's theta* follows its theta (bl_mix, base 200 K: wherever the
# air above is warmer), its depth set by its outflow, entrainment and night collapse; its vapor even, cloud at its top (bl_q);
# the surface layer's exchange (sfc, #114); convective downdrafts (0.3 of the plume's base mass) and the updraft's entrainment of the
# cloud layer's air (1e-4 per m), both Tiedtke (1989) (#116). The surface (#118): the ground a skin of the daily cycle's capacity,
# rho c D / 2 with D = sqrt(2 kappa / omega) = 0.12 m (rho c 2e6 J/m^3/K, kappa 5e-7 m^2/s), over the deep ground of SPEC #84's
# 2e6 (about the yearly cycle's, D 2.2 m); a surface resistance of 70 s/m in full sun (FAO's grass reference) closing to
# 5000 s/m in the dark (Jarvis 1976; Noah's F1 with LAI 2, so rs_sun 180 W/m^2); the sea's saturation 2% below fresh water's; the mixed layer
# takes in the air above it colder than itself, reaching through an emptied layer (#120), whose lower edge is then its own theta (#122);
# the 380 K edge is heated by radiation alone, the cloud layer's condensation happening far below it (#121); the mixed layer ends at its
# cloud base, its air above that leaving into the cloud layer (#124). The mountains' drag: blocked flow and gravity waves from the sub-grid
# orography (Lott & Miller 1997; #127); the gravity waves not tied to the ground (Alexander & Dunkerton 1999, as GFDL cg_drag; #128).
PRESETS['earth4'] = dict(PRESETS['earth'], build=earth4, stress=[0.0] * 4, shelf=100.0, sso=earth_sso,
                         air=dict(PRESETS['earth']['air'], physics=dict(PRESETS['earth']['air']['physics'], theta_stars=[380.0, 305.0, 200.0],
                                                                        bl_mix=1.0, bl_q=1.0, sfc=1.0, downdraft=0.3, ent_cloud=1e-4,
                                                                        c_land=1.2e5, c_deep=2.0e6, land_rs=70.0, rs_max=5000.0, rs_sun=180.0, q_sea=0.98, bl_reach=1.0, dry_edge=1.0, bl_lcl=1.0, oro=1.0, gw=1.0)))


def earth5(g, top=5000.0):
    """The five-layer start (V2-6b, #129): earth4's with its stratosphere split at 500 K, the air above it top (Pa; the 500 K surface
    lies at 46-55 hPa in the NCEP reanalysis) at 600 K, the rest at 440 K; their vapor the stratosphere's."""
    h4, b, u4, x = earth4(g)
    hu = np.minimum(np.full(g.N, top / GRAV), 0.5 * h4[0])
    h = np.concatenate([hu[None], (h4[0] - hu)[None], h4[1:]])
    th = np.concatenate([np.full((1, g.N), 600.0), np.full((1, g.N), 440.0), x['theta'][1:]])
    return h, b, np.concatenate([u4[:1], u4]), dict(x, theta=th, q=np.concatenate([x['q'][:1], x['q']]))


# V2-6b (#129): earth4 with its stratosphere in two layers, split at 500 K, so the waves can leave momentum in its lower part and its
# upper part apart; the layers above the 380 K edge (edge_k 2) are the stratosphere's: isothermal in the radiation, their vapor even.
# Land cover (#132): each land cell's roughness, snow-free albedo, snow under trees and stomata from its cover (earth_cover).
# Ice in the air (V2-6c, #134): ice saturation below freezing, snow and its freezing heat aloft, melting below, sublimation.
PRESETS['earth5'] = dict(PRESETS['earth4'], build=earth5, stress=[0.0] * 5, cover=earth_cover,
                         air=dict(PRESETS['earth4']['air'], physics=dict(PRESETS['earth4']['air']['physics'], theta_stars=[500.0, 380.0, 305.0, 200.0],
                                                                         edge_k=2.0, cover=1.0, ice_air=1.0, sfc_wind=1.0, revap=1.0, root_q=1.0, ent_det=1.0, ent_ifs=1.0, shallow=0.03)))


def earth6(g, cu=1.0e4):
    """The six-layer start (V2-6c, #142): earth5's with its cloud layer split, its lowest cu (Pa, at most 40% of it) the cumulus layer, its
    theta 2 K above the mixed layer's (half its cu_dth span) and below the cloud layer's, its vapor 75% of saturation."""
    h5, b, u5, x = earth5(g)
    th5, q5 = x['theta'], x['q']
    hc = np.minimum(np.full(g.N, cu / GRAV), 0.4 * h5[3])
    h = np.concatenate([h5[:3], (h5[3] - hc)[None], hc[None], h5[4:]])
    tc = np.minimum(th5[4] + 2.0, th5[3] - 1.0)
    ps = h.sum(0) * GRAV
    pm = ps - h[5] * GRAV - 0.5 * hc * GRAV
    T = tc * (pm / P0) ** KAPPA
    qs = 0.622 * 611.0 * np.exp(17.27 * (T - 273.15) / (T - 35.85)) / pm
    th = np.concatenate([th5[:4], tc[None], th5[4:]])
    q = np.concatenate([q5[:4], (0.75 * qs)[None], q5[4:]])
    return h, b, np.concatenate([u5[:4], u5[3:4], u5[4:]]), dict(x, theta=th, q=q)


# V2-6c (#142): earth5 with its cloud layer split: a cumulus layer above the mixed layer, its top following the mixed layer's theta plus
# cu_dth (4 K: BOMEX's cumulus layer), where the shallow plume leaves its air and the mixed layer takes air back in; the downdrafts start in
# the lower theta_e of the cumulus and cloud layers (Tiedtke's level of free sinking). theta* 204 is a placeholder: the interface follows the mixed layer.
# #144: the cumulus layer's top is where the mixed layer's air gets lifted saturated cu_dp (200 hPa, the IFS's shallow-cloud depth) above its
# cloud base, not a fixed 4 K above it: its span follows the moist adiabat (about 12 K in the tropics, 6-7 K in cold air).
# #145: the plumes' buoyancy in virtual theta (buoy_v); alone it drained the mixed layer (31-55 hPa over the sea) and warmed the planet.
# #146: the air the plumes take out of the mixed layer replaced by the cumulus layer's sinking between the clouds, up to cloud base (cu_sub).
# #148: bare soil evaporates from its drying top, as (wet / full)^2 (bare_fx, Noah's direct soil evaporation), the plants from the whole root zone.
PRESETS['earth6'] = dict(PRESETS['earth5'], build=earth6, stress=[0.0] * 6,
                         air=dict(PRESETS['earth5']['air'], physics=dict(PRESETS['earth5']['air']['physics'], theta_stars=[500.0, 380.0, 305.0, 204.0, 200.0],
                                                                         cu_dth=4.0, cu_dp=2.0e4, buoy_v=1.0, cu_sub=1.0, bare_fx=2.0)))


def earth7(g, lo=318.0, hi=350.0):
    """The seven-layer start (V2-6c, #153): earth6's with its 305-380 K layer split at 330 K, the lower part's theta lo and the upper
    part's hi, their masses so that the two keep the layer's mean theta (each at least a tenth of it), their vapor the layer's."""
    h6, b, u6, x = earth6(g)
    th6, q6 = x['theta'], x['q']
    f = np.clip((hi - th6[2]) / (hi - lo), 0.1, 0.9)
    h = np.concatenate([h6[:2], ((1 - f) * h6[2])[None], (f * h6[2])[None], h6[3:]])
    th = np.concatenate([th6[:2], np.full((1, g.N), hi), np.full((1, g.N), lo), th6[3:]])
    q = np.concatenate([q6[:3], q6[2:]])
    return h, b, np.concatenate([u6[:3], u6[2:]]), dict(x, theta=th, q=q)


# V2-6c (#153): earth6 with its 305-380 K layer split at 330 K. In the tropics that layer was the whole free troposphere above ~800 hPa
# (~700 hPa thick): one theta profile shape and one vapor amount held it ~15 K too cold in theta and at a third of its water (6.6 kg/m2
# against ~20), so the air absorbed too little sunlight (70 W/m2 against 79) and the surface evaporated the rest.
# V2-6c (#154): ice clouds with their own sunlight optics (Ebert & Curry 1992), not the drops'
PRESETS['earth7'] = dict(PRESETS['earth6'], build=earth7, stress=[0.0] * 7,
                         air=dict(PRESETS['earth6']['air'], physics=dict(PRESETS['earth6']['air']['physics'], theta_stars=[500.0, 380.0, 330.0, 305.0, 204.0, 200.0], ice_sw=1.0)))

PRESETS['rce'] = dict(build=aqua_dry, dt=lambda n: 3600.0, air=dict(AIR, mean_sun=True, physics=AIR3), dynamics=False)


def held_suarez(g, seed=1):
    """Held & Suarez (1994) in two air layers, started as their protocol starts: at rest and horizontally uniform (each
    layer at the area-mean equilibrium θ of its mass-mean pressure, the lower capped 1 K below θ*), with 0.1 K of
    seeded noise; the relaxation then builds the pole-to-equator gradient over weeks. (Starting from the equilibrium
    gradient itself at rest is far from balance: the adjustment swings the interface by hundreds of hPa.)"""
    from .atmosphere import held_suarez_theta
    h = np.full((2, g.N), MU_REF / 2)
    A = g.areaCell / np.sum(g.areaCell)
    p_mean = [0.25e5, 0.75e5]                                    # mass-mean pressures of the two halves (Pa), close enough
    rng = np.random.default_rng(seed)
    means = [float(np.sum(A * held_suarez_theta(p, g.lat))) for p in p_mean]
    means[1] = min(means[1], held_suarez_theta_star(g) - 1.0)
    theta = np.stack([np.full(g.N, m) for m in means]) + 0.1 * rng.standard_normal((2, g.N))
    return h, np.zeros(g.N), np.zeros((2, g.E)), dict(theta=theta, Ts=np.zeros(g.N))


def held_suarez_theta_star(g):
    from .atmosphere import held_suarez_theta
    return float(np.sum(g.areaCell * held_suarez_theta(0.5e5, g.lat)) / np.sum(g.areaCell))


# HS friction k_f = 1/day over sigma > 0.7, ramping from 0 to 1 at the ground: its column integral is k_f·0.15·M, the
# bottom stress c of the lower layer.
PRESETS['held_suarez'] = dict(build=held_suarez, dt=lambda n: 900.0 * 2.0 ** (5 - n), visc4_U=15.0,
                              stress=[0.0, SURFACE_DRAG], tracer=False,
                              air=dict(AIR, forcing='held_suarez', physics=dict(theta_star=held_suarez_theta_star)))
