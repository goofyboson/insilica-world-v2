"""Initial states. Each case gives (h (L, N), b (N), u (L, E)) as NumPy arrays and the layer settings it needs.

Single-layer benchmarks (Williamson et al. 1992; Galewsky et al. 2004) as in v1's numerics tests, and the
stacked-layer cases of V2-1: external and internal gravity waves, two-layer solid-body flows (Phillips baroclinic
instability, a balanced ocean), and an outcrop over a ridge.
"""
import math

import numpy as np

from .constants import GRAV, KAPPA, OMEGA, P0, R_PLANET


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
PRESETS['aqua_dry'] = dict(PRESETS['aqua'], build=aqua_dry, air=dict(AIR, physics=AIR3))
# V2-3: the moist three-layer air over a two-layer ocean (decisions #64–#69): the air's drag on the surface is the wind
# stress on the ocean (exact, in Ocean.couple), so the air's own bottom stress is off. Air steps must divide the hour.
PRESETS['coupled'] = dict(PRESETS['aqua'], stress=[0.0, 0.0, 0.0], ocean=dict(surface_drag=SURFACE_DRAG))
# One ocean basin (decision #79): the coupled world with a thin pole-to-pole wall at 180° that the ocean cannot cross, so
# that gyres, western boundary currents and an east–west tilt of the equatorial thermocline can form; the air is unchanged.
PRESETS['basin'] = dict(PRESETS['coupled'], ocean=dict(surface_drag=SURFACE_DRAG, walls=(180.0,)))
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
