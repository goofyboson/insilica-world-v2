"""Physical constants (SI units), carried over from v1's `src/sim/00-core.js` (constants K)."""
import math

R_PLANET = 6.371e6            # m
OMEGA = 2 * math.pi / 86400   # rad/s (v1 uses a 24-hour day)
GRAV = 9.80616                # m/s², the shallow-water test value (Williamson et al. 1992; v1's numerics tests)
T_REF = 273.15                # K, reference temperature for stored energies (v1 SPEC §18 #12)
R_DRY = 287.04                # J/kg/K, dry air
CP_DRY = 1004.6               # J/kg/K (v1 used 1004)
KAPPA = R_DRY / CP_DRY
# A world's planet (V2-6e, #186): radius, rotation and gravity, Earth's by default; its day, year, sun, orbit and gases are
# the air physics' settings (day_s, year_s, S0, tilt, ecc, lam_peri, co2, n2o, ch4), which a preset's `planet` dict may also set
PLANET = dict(radius=R_PLANET, omega=OMEGA, grav=GRAV)
P0 = 1.0e5                    # Pa, reference pressure for potential temperature
# The air's gases (V2-6e, #216): molar mass (g/mol) and molar heat capacity at constant pressure (J/mol/K, ideal gas at 300 K;
# NIST), for a world's dry air given by mole fraction (`gases`); held constant with temperature (an assumption: CO2's falls ~10%
# from 300 to 200 K)
GASES = dict(n2=(28.0134, 29.124), o2=(31.9988, 29.385), ar=(39.948, 20.786), co2=(44.0095, 37.135), ch4=(16.0425, 35.69),
             h2=(2.01588, 28.836), he=(4.002602, 20.786), ne=(20.1797, 20.786))
EARTH_AIR = dict(n2=0.78084, o2=0.20946, ar=0.00934, co2=0.00042)     # dry air's mole fractions (CO2 of ~2020)
R_GAS, M_WATER = 8.314462618, 18.01528                                  # J/mol/K, g/mol
# Radiation in another air (V2-6e, #226). Rayleigh scattering per molecule goes as the refractivity squared times the King
# factor: (n - 1 at STP, 1e-6; King factor) per gas (refractivities as tabulated in the CRC Handbook; King factors after Bates
# 1984 and Sneep & Ubachs 2005), relative to Earth's air of EARTH_AIR
RAYLEIGH = dict(n2=(298.0, 1.035), o2=(271.0, 1.10), ar=(281.0, 1.0), co2=(449.0, 1.15), ch4=(444.0, 1.0), h2=(139.0, 1.0),
                he=(35.0, 1.0), ne=(67.0, 1.0))
# how strongly each gas broadens lines relative to Earth's air (pressure broadening): CO2 1.33, its own lines' mean gamma_self /
# gamma_air in HITRAN 2020 (tools/lbl.E_CO2), taken for every absorber; the others as air (an assumption: no data at hand)
BROADEN = dict(co2=1.33)
# ozone's column against O2 (PAL: the O2 mole fraction over Earth's), Segura et al. (2003)'s 1-D photochemistry as tabulated by
# Kasting et al. (2023, R. Soc. Open Sci. 10, 230056): 311, 266, 124, 25.3, 1.2 DU at 1, 0.1, 0.01, 0.001, 0.0001 PAL; the
# profile's shape kept (at low O2 the layer also moves down: not modelled)
O3_O2 = ((1e-4, 1.2 / 311), (1e-3, 25.3 / 311), (1e-2, 124.0 / 311), (0.1, 266.0 / 311), (1.0, 1.0))
# collision-induced absorption's pairs, as tools/other_air.PAIRS and src/insilica/data/cia_ktables.npz
CIA_PAIRS = ('N2-N2', 'CO2-CO2', 'O2-O2', 'N2-H2', 'CO2-H2', 'CO2-CH4', 'N2-CH4')


def ozone_factor(pal):
    """Ozone's column relative to Earth's at pal (O3_O2): log-log between the points, 1 at and above 1 PAL, and below 1e-4 PAL
    the last segment's power law."""
    import numpy as np
    if pal >= 1.0:
        return 1.0
    if pal <= 0.0:
        return 0.0
    lx, ly = np.log10([a for a, _ in O3_O2]), np.log10([b for _, b in O3_O2])
    x = math.log10(pal)
    if x < lx[0]:
        return float(10 ** (ly[0] + (ly[1] - ly[0]) / (lx[1] - lx[0]) * (x - lx[0])))
    return float(10 ** np.interp(x, lx, ly))


def air_radiation(x):
    """The radiation's view of a dry air {gas: mole fraction, normalized} (#226): O2's mole fraction, Rayleigh scattering per
    molecule relative to Earth air's, the pressure broadening factor, ozone's factor, and each CIA pair's x_a x_b."""
    ray = lambda a: sum(v * (RAYLEIGH[k][0] ** 2 * RAYLEIGH[k][1]) for k, v in a.items())
    tot = sum(EARTH_AIR.values())
    earth = ray({k: v / tot for k, v in EARTH_AIR.items()})
    pairs = []
    for pr in CIA_PAIRS:
        a, b = (g.lower() for g in pr.split('-'))
        pairs.append(x.get(a, 0.0) * x.get(b, 0.0))
    o2 = x.get('o2', 0.0)
    return dict(o2=o2, ray=ray(x) / earth, broad=sum(v * BROADEN.get(k, 1.0) for k, v in x.items()),
                o3=ozone_factor(o2 / EARTH_AIR['o2']), cia=pairs)


def air_props(gases=None):
    """The dry air's constants from its gases' mole fractions (normalized): cp (J/kg/K), R (J/kg/K), kappa, molar mass M (g/mol),
    eps = M_water / M and the virtual-temperature factor 1/eps - 1, and the column kernels' compile-time defines. None: the
    constants the model has always used (cp 1004.6, R 287.04 in the dynamics and 287.0 in the column physics, eps 0.622,
    virtual factors 0.61 and 0.608, M 28.9647), bit for bit."""
    if not gases:
        return dict(cp=CP_DRY, kappa=KAPPA, rd=R_DRY, rd_col=287.0, eps=0.622, mair=28.9647, defs=())
    x = {k: float(v) for k, v in gases.items() if float(v) > 0}
    unknown = set(x) - set(GASES)
    assert not unknown, f'unknown gases {unknown}; known: {sorted(GASES)}'
    tot = sum(x.values())
    M = sum(v / tot * GASES[k][0] for k, v in x.items())
    cpm = sum(v / tot * GASES[k][1] for k, v in x.items())
    cp_, R = cpm / (M * 1e-3), R_GAS / (M * 1e-3)
    eps = M_WATER / M
    vf = 1 / eps - 1
    rad = air_radiation({k: v / tot for k, v in x.items()})
    defs = (f'-DAIR_RD={R!r}', f'-DAIR_EPS={eps!r}', f'-DAIR_VF={vf!r}', f'-DAIR_VFB={vf!r}', f'-DAIR_M={M!r}', f'-DAIR_MKG={M * 1e-3!r}',
            f'-DAIR_O2={rad["o2"]!r}', f'-DAIR_RAY={rad["ray"]!r}', f'-DAIR_BROAD={rad["broad"]!r}', f'-DAIR_O3={rad["o3"]!r}', '-DAIR_CIA')
    return dict(cp=cp_, kappa=R / cp_, rd=R, rd_col=R, eps=eps, mair=M, defs=defs, rad=rad)

# The land column (V2-6e, #218): moist mineral soil, volumetric heat capacity (J/m^3/K) and conductivity (W/m/K) (Hillel 1982; Oke
# 1987: 1.2-3.0e6 and 0.25-2.2 from dry sand to wet clay; damping depth sqrt(2 k / (c omega)) 2.2 m for the year, 0.12 m for the day),
# and its layers (m) under the skin, doubling to 6.3 m, three annual damping depths
C_SOIL, K_SOIL = 2.0e6, 1.0
SOIL_DZ = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
