"""Physical constants (SI units), carried over from v1's `src/sim/00-core.js` (constants K)."""
import math

R_PLANET = 6.371e6            # m
OMEGA = 2 * math.pi / 86400   # rad/s (v1 uses a 24-hour day)
GRAV = 9.80616                # m/s², the shallow-water test value (Williamson et al. 1992; v1's numerics tests)
T_REF = 273.15                # K, reference temperature for stored energies (v1 SPEC §18 #12)
R_DRY = 287.04                # J/kg/K, dry air
CP_DRY = 1004.6               # J/kg/K (v1 used 1004)
KAPPA = R_DRY / CP_DRY
P0 = 1.0e5                    # Pa, reference pressure for potential temperature
