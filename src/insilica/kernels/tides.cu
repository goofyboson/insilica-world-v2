// Moon and tides (V2-6f, decision #228). One thread per cell.
//
// tide_pot: the tide-generating potential of the Moon and the Sun at each cell, degree 2,
//   V = love * sum_b (GM_b R^2 / r_b^3) (3 cos^2 psi_b - 1) / 2,
// psi_b the angle between the cell and the body, love = 1 + k2 - h2 (the solid Earth's own tide). The Sun's direction follows
// the air column's (air.cu sun_lon: Kepler from the March equinox, declination from the tilt, its longitude -2 pi (t mod day) /
// day); the Moon's from its own Kepler orbit (eccentricity e, mean motion 2 pi / T, perigee advancing with period T_aps, an
// orbit inclined i to the ecliptic whose node regresses with period T_node), turned to the equator by the tilt and placed by
// its right ascension against the Sun's. The ocean's layers feel -grad(-V) through their Bernoulli function (layers.cu lay_bern).
//
// tide_mix: the bottom stress's dissipation D = rho c |u_b|^2 (W/m^2; c the bottom layer's whole stress coefficient: its linear
// one, the quadratic cd |u_b| and the internal tide's drag, |u_b| the cell's speed from its edges), a share gamma
// of it (the mixing efficiency) spent raising the column's potential energy by mixing heat and salt across its interfaces: the
// mixing the floor's dissipation drives fades with height above the floor, exp(-z / zeta), zeta about 500 m (St Laurent, Simmons
// & Jayne 2002), so an interface z above the floor gets gamma D dt exp(-z / zeta) (#229: a deep sea's thermocline, 3 km up, none;
// a shelf sea's interfaces, near the floor, nearly all). Two layers of thickness h1 over h2, buoyancy step db > 0, mixed a fraction
// f (their step falls to (1 - f) db) gain rho h1 h2 db f / 2, so f = E / (rho h1 h2 db / 2) up to 1. Heat and salt are exchanged
// in equal volumes v = f h1 h2 / (h1 + h2), so both are conserved exactly; the layers keep their water.

typedef REAL real;

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;
#define PI_ 3.141592653589793

// tp: tilt, year_s, day_s, ecc, lam_peri, sun_gm, sun_a, moon_gm, moon_a, moon_e, moon_i, moon_T, node_T, aps_T, moon_L0,
//     node0, aps0, love, radius
__device__ real kepler(real M, real e, real* rr) {
  real E = M;
  for (int it = 0; it < 6; it++) E -= (E - e * sin(E) - M) / (1 - e * cos(E));
  *rr = 1 - e * cos(E);
  return 2 * atan2(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2));
}

extern "C" __global__ void tide_pot(int N, const real* tp, const real* time, real back, const real* pos, real* V) {
  LOOP1(i, N)
  const real t = time[0] - back;
  const real tilt = tp[0], year_s = tp[1], day_s = tp[2], ecc = tp[3], lam_peri = tp[4];
  // the Sun: its ecliptic longitude and distance (as air.cu's sun_lon)
  real ls, rs = 1;
  if (!(ecc > 0)) ls = 2 * PI_ * t / year_s;
  else {
    const real nu0 = -lam_peri, E0 = 2 * atan(sqrt((1 - ecc) / (1 + ecc)) * tan(nu0 / 2));
    ls = kepler(E0 - ecc * sin(E0) + 2 * PI_ * t / year_s, ecc, &rs) + lam_peri;
  }
  const real ras = atan2(cos(tilt) * sin(ls), cos(ls)), decs = asin(sin(tilt) * sin(ls));
  const real rot = 2 * PI_ * fmod(t, day_s) / day_s;
  // the Moon: Kepler in its orbit, then the inclined orbit's ecliptic longitude and latitude, then the equator's
  const real e = tp[9], inc = tp[10];
  const real peri = tp[16] + 2 * PI_ * t / tp[13], node = tp[15] + 2 * PI_ * t / tp[12];
  real rm;
  const real lo = kepler(tp[14] - peri + 2 * PI_ * t / tp[11], e, &rm) + peri;    // true longitude in the orbit
  const real u = lo - node;
  const real bet = asin(sin(inc) * sin(u)), lam = node + atan2(cos(inc) * sin(u), cos(u));
  const real decm = asin(sin(bet) * cos(tilt) + cos(bet) * sin(tilt) * sin(lam));
  const real ram = atan2(sin(lam) * cos(tilt) - tan(bet) * sin(tilt), cos(lam));
  // directions in the planet's frame: the Sun at longitude -rot, the Moon its right ascension further east
  const real x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
  const real ls_e = -rot, lm_e = -rot + (ram - ras);
  const real cs = x * cos(decs) * cos(ls_e) + y * cos(decs) * sin(ls_e) + z * sin(decs);
  const real cm = x * cos(decm) * cos(lm_e) + y * cos(decm) * sin(lm_e) + z * sin(decm);
  const real R2 = tp[18] * tp[18];
  const real as = tp[6] * rs, am = tp[8] * rm;
  V[i] = tp[17] * (tp[5] * R2 / (as * as * as) * (3 * cs * cs - 1) / 2 + tp[7] * R2 / (am * am * am) * (3 * cm * cm - 1) / 2);
}

extern "C" __global__ void tide_mix(int N, int E, int L, const real* h, const real* u, real* T, real* S, const int* cellEdge,
                                    const real* le, const real* de, const real* area, const real* land, real c_bot, real rho,
                                    real gamma, real dt, real grav, real alpha, real alpha2, real beta, real* dsum, real cd,
                                    const real* cbot, real zeta) {
  LOOP1(i, N)
  if (land && land[i] > (real)0.5) return;
  const real* ub = u + (size_t)(L - 1) * E;
  real s = 0, cit = 0;
  int ne = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    s += le[e] * de[e] * ub[e] * ub[e];
    if (cbot) cit += cbot[e];
    ne++;
  }
  const real u2 = s / (2 * area[i]);                            // the cell's speed squared
  const real D = rho * (c_bot + cd * sqrt(u2) + (ne ? cit / ne : (real)0)) * u2;   // W/m^2
  dsum[i] += D * dt;
  const real En0 = gamma * D * dt;                               // J/m^2 for mixing, at the floor
  real z = 0;                                                    // the interface's height above the floor
  for (int k = L - 2; k >= 0; k--) {
    const size_t a = (size_t)k * N + i, b = (size_t)(k + 1) * N + i;
    const real h1 = h[a], h2 = h[b];
    z += h2;
    const real En = En0 * exp(-z / zeta);
    const real t1 = T[a] - (real)273.15, t2 = T[b] - (real)273.15;
    const real db = grav * (alpha * (t1 - t2) + (real)0.5 * alpha2 * (t1 * t1 - t2 * t2) - beta * (S[a] - S[b]));
    if (!(db > 0) || !(h1 > 0) || !(h2 > 0)) continue;
    const real full = rho * h1 * h2 * db / 2;
    const real f = En < full ? En / full : (real)1;
    const real v = f * h1 * h2 / (h1 + h2);
    const real dT = T[b] - T[a], dS = S[b] - S[a];
    T[a] += v / h1 * dT; T[b] -= v / h2 * dT;
    S[a] += v / h1 * dS; S[b] -= v / h2 * dS;
  }
}

// tide_itd: the internal tide's drag (Jayne & St Laurent 2001), tau / rho = kappa h^2 N u / 2 on the bottom layer, as a linear stress
// coefficient per edge c = kappa h^2 N / 2 (m/s): h^2 the sea floor's roughness (m^2), kappa its wavenumber, N the buoyancy frequency
// above the bottom: the case's (nbot, observed: the model's abyss is one mixed layer), else from the buoyancy step between the two
// lowest layers over the distance between their middles; the two cells' mean.
extern "C" __global__ void tide_itd(int N, int E, int L, const real* h, const real* T, const real* S, const real* h2, const int* eI,
                                    const int* eJ, real kappa, real grav, real alpha, real alpha2, real beta, real* cbot,
                                    const real* nbot) {
  LOOP1(e, E)
  real c = 0;
  for (int s2 = 0; s2 < 2; s2++) {
    const int i = s2 ? eJ[e] : eI[e];
    if (!(h2[i] > 0)) continue;
    if (nbot) { c += (real)0.5 * kappa * h2[i] * nbot[i]; continue; }
    const size_t a = (size_t)(L - 2) * N + i, b = (size_t)(L - 1) * N + i;
    const real t1 = T[a] - (real)273.15, t2 = T[b] - (real)273.15;
    const real db = grav * (alpha * (t1 - t2) + (real)0.5 * alpha2 * (t1 * t1 - t2 * t2) - beta * (S[a] - S[b]));
    const real dz = (real)0.5 * (h[a] + h[b]);
    if (db > 0 && dz > 0) c += (real)0.5 * kappa * h2[i] * sqrt(db / dz);
  }
  cbot[e] = (real)0.5 * c;
}
