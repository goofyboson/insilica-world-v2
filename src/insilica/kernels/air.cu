// Column physics of the two air layers over a slab surface (stage V2-2a, decisions #32-#35): v1's radiation and
// surface exchange, heating as potential temperature, and air crossing the isentropic interface theta* only when
// heated or cooled across it. One thread per cell; layers k = 0 (top) .. L-1 (bottom), mass mu_k (kg/m^2).
//
// Energy: a layer's enthalpy is mu_k pibar_k theta_k (pibar the mass-mean Exner function, cp T = pibar theta), the
// slab's C Ts. Heating Q (W/m^2) changes theta by Q dt / (mu pibar), so every flux moves energy exactly; the crossing
// books its rounding-level enthalpy residual into the receiving layer, so the column ledger closes to rounding.

typedef REAL real;
#define LMAX 8
#define PI_ 3.14159265358979323846

struct Par {
  real S0, asw, tau0, mu_ref, eps_s, h_sens, u0, alb, c_slab, mu_bl, theta_star, floor_, sigma, grav, cp, kappa, p0;
  real tilt, year_s, day_s, tw_top, tw_bottom;
};

// The parameters arrive as an array in this order (atmosphere.PAR_ORDER).
__device__ Par unpack(const real* a) {
  Par P;
  P.S0 = a[0]; P.asw = a[1]; P.tau0 = a[2]; P.mu_ref = a[3]; P.eps_s = a[4]; P.h_sens = a[5]; P.u0 = a[6]; P.alb = a[7];
  P.c_slab = a[8]; P.mu_bl = a[9]; P.theta_star = a[10]; P.floor_ = a[11]; P.sigma = a[12]; P.grav = a[13]; P.cp = a[14];
  P.kappa = a[15]; P.p0 = a[16]; P.tilt = a[17]; P.year_s = a[18]; P.day_s = a[19]; P.tw_top = a[20]; P.tw_bottom = a[21];
  return P;
}

__device__ __forceinline__ real exner(real p, const Par& P) { return P.cp * pow(p / P.p0, P.kappa); }

__device__ void layer_exner(int L, const real* mu, const Par& P, real* pb, real* pib) {
  real p = 0, xt = 0;
  for (int k = 0; k < L; k++) {
    const real pt = p;
    p = p + P.grav * mu[k];
    pb[k] = p;
    const real x = pow(p / P.p0, P.kappa);                       // one pow per interface
    pib[k] = P.cp * (p * x - pt * xt) / ((P.kappa + 1) * (p - pt));
    xt = x;
  }
}

// Crossing theta* (two layers, decision #34): a layer's theta spans [lo, hi] evenly about its mean, with theta* at
// the shared edge; heating moves the lower layer's warmest air past theta*, cooling the upper layer's coldest. The
// mass crossing is mu * |d theta| / (theta range) = mu |d theta| / (2 |theta* - theta|), theta the mean before this
// step's heating; a layer entirely on the wrong side of theta* crosses whole, down to the floor. Crossing air carries
// theta* (or its whole layer's theta). The enthalpy residual of re-dividing the column (rounding level) is booked
// into the receiving layer, so energy is exact. Updates mu, th, pib in place; returns the mass moved up (kg/m^2).
// dth_up: the theta change this step of the upper layer's air at theta* (its bottom). The lower layer is convectively
// mixed, so its own mean change drives its crossing; the upper layer is stably stratified, so the air at theta*
// responds to the heating at theta*, not to the layer's mean (decision #40). Callers without that information pass
// the upper layer's mean change.
__device__ real crossing(int L, const Par& P, real* mu, real* th, const real* dth, real dth_up, real* pib) {
  if (L != 2 || P.theta_star <= 0) return 0;          // theta_star <= 0: no crossing (diagnostics)
  const int top = 0, bot = 1;
  const real ts_ = P.theta_star;
  real dm = 0, thc = ts_;                                       // dm > 0: lower -> upper
  if (dth[bot] > 0) {
    if (th[bot] >= ts_) { dm = mu[bot] - P.floor_; thc = th[bot]; }
    else dm = mu[bot] * dth[bot] / (2 * (ts_ - (th[bot] - dth[bot])));
  }
  if (dth_up < 0) {
    real down;
    if (th[top] <= ts_) { down = mu[top] - P.floor_; thc = th[top]; }
    else down = mu[top] * (-dth_up) / (2 * ((th[top] - dth[top]) - ts_));
    dm -= down;
  }
  const real maxup = mu[bot] - P.floor_, maxdn = mu[top] - P.floor_;
  if (dm > maxup) dm = maxup > 0 ? maxup : (real)0;
  if (-dm > maxdn) dm = maxdn > 0 ? -maxdn : (real)0;
  if (dm != 0) {
    real Eb = 0, Ea = 0, mu2[LMAX], pb2[LMAX], pib2[LMAX], th2[LMAX];
    for (int k = 0; k < L; k++) { Eb += mu[k] * pib[k] * th[k]; mu2[k] = mu[k]; }
    mu2[bot] -= dm; mu2[top] += dm;
    layer_exner(L, mu2, P, pb2, pib2);
    th2[bot] = (mu[bot] * th[bot] - dm * thc) / mu2[bot];
    th2[top] = (mu[top] * th[top] + dm * thc) / mu2[top];
    for (int k = 0; k < L; k++) Ea += mu2[k] * pib2[k] * th2[k];
    const int recv = dm > 0 ? top : bot;
    th2[recv] += (Eb - Ea) / (mu2[recv] * pib2[recv]);
    for (int k = 0; k < L; k++) { th[k] = th2[k]; mu[k] = mu2[k]; pib[k] = pib2[k]; }
  }
  // A layer at its floor is a placeholder for air that is not there: its theta stays on its own side of theta* (the
  // lower at most theta*, the upper at least). Heat that would carry floor air across theta* belongs to the air that
  // is there, so it moves to the other layer, enthalpy for enthalpy (decision #34).
  const real tiny = P.floor_ * (real)1.000001;
  if (mu[bot] <= tiny && th[bot] > ts_) {
    th[top] += mu[bot] * pib[bot] * (th[bot] - ts_) / (mu[top] * pib[top]);
    th[bot] = ts_;
  }
  if (mu[top] <= tiny && th[top] < ts_) {
    th[bot] -= mu[top] * pib[top] * (ts_ - th[top]) / (mu[bot] * pib[bot]);
    th[top] = ts_;
  }
  return dm;
}

// mode 0: the sun at time t (diurnal and seasonal, v1's sunVector: year of year_s, day of day_s, tilt);
// mode 1: insol[i], a given mean insolation (W/m^2 on the ground's plane, before any absorption).
extern "C" __global__ void air_column(int N, int L, const real* par, int mode, const real* t, const real* insol, const real* pos,
                                      const real* uabs, real* mu_in, real* theta, real* Ts, real dt,
                                      real* wx, real* olr, real* sw_abs, real* stored) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const Par P = unpack(par);
  real mu[LMAX], th[LMAX], pb[LMAX], pib[LMAX], Q[LMAX], T[LMAX], eps[LMAX];
  real mt = 0;
  for (int k = 0; k < L; k++) { mu[k] = mu_in[(size_t)k * N + i]; th[k] = theta[(size_t)k * N + i]; mt += mu[k]; Q[k] = 0; }
  layer_exner(L, mu, P, pb, pib);
  real E0 = P.c_slab * Ts[i];
  for (int k = 0; k < L; k++) { T[k] = pib[k] * th[k] / P.cp; E0 += mu[k] * pib[k] * th[k]; }
  const int top = 0, bot = L - 1;

  // ---- shortwave ----
  real I;
  if (mode == 0) {
    const real ang = 2 * PI_ * t[0] / P.year_s;
    const real dec = asin(sin(P.tilt) * sin(ang));
    const real rot = 2 * PI_ * fmod(t[0], P.day_s) / P.day_s;
    const real sx = cos(dec) * cos(-rot), sy = cos(dec) * sin(-rot), sz = sin(dec);
    const real cz = pos[3 * i] * sx + pos[3 * i + 1] * sy + pos[3 * i + 2] * sz;
    I = P.S0 * (cz > 0 ? cz : (real)0);
  } else {
    I = insol[i];
  }
  const real toAir = P.asw * (mt / P.mu_ref) * I;              // absorbed in the air, split by mass (v1)
  for (int k = 0; k < L; k++) Q[k] += toAir * mu[k] / mt;
  const real swS = (I - toAir) * (1 - P.alb);
  real qS = swS;

  // ---- longwave, gray (v1's two-layer scheme): optical depth per layer in proportion to its mass, weighted toward
  // the lower layer (tw_bottom, tw_top; v1's 0.7 / 0.3 of TAU0 at an even split) ----
  for (int k = 0; k < L; k++) {
    const real w = (k == bot) ? P.tw_bottom : P.tw_top;
    eps[k] = 1 - exp(-P.tau0 * w * mu[k] / P.mu_ref);
  }
  const real ts = Ts[i];
  real up = P.eps_s * P.sigma * ts * ts * ts * ts;             // upward flux leaving the surface
  qS -= up;
  for (int k = bot; k >= 0; k--) {                              // upward beam, layer by layer
    const real a = eps[k] * up;
    Q[k] += a;
    const real em = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    Q[k] -= 2 * em;                                             // each layer emits up and down
    up = up - a + em;
  }
  olr[i] = up;
  for (int k = 0; k < L; k++) {                                 // each layer's downward emission, absorbed on the way down
    real dn = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int j = k + 1; j < L; j++) { const real a = eps[j] * dn; Q[j] += a; dn -= a; }
    qS += dn;
  }

  // ---- sensible heat into the bottom mu_bl of the column (the air touching the ground) ----
  real wbl[LMAX], thb = 0, mb = 0;
  {
    real need = P.mu_bl;
    for (int k = bot; k >= 0; k--) { const real m = mu[k] < need ? mu[k] : need; wbl[k] = m; need -= m; thb += m * th[k]; mb += m; }
  }
  thb /= mb;
  const real Tsa = thb * pow(pb[bot] / P.p0, P.kappa);
  const real H = P.h_sens * (1 + uabs[i] / P.u0) * (ts - Tsa);
  for (int k = 0; k < L; k++) Q[k] += H * wbl[k] / mb;
  qS -= H;

  // ---- heating as potential temperature; the slab ----
  real dth[LMAX];
  for (int k = 0; k < L; k++) { dth[k] = Q[k] * dt / (mu[k] * pib[k]); th[k] += dth[k]; }
  Ts[i] = ts + qS * dt / P.c_slab;
  sw_abs[i] = toAir + swS;

  wx[i] = crossing(L, P, mu, th, dth, dth[0], pib);
  real E1 = P.c_slab * Ts[i];
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; E1 += mu[k] * pib[k] * th[k]; }
  stored[i] = E1 - E0;                                          // the column's energy change, J/m^2
}

// Time for the sun, advanced inside the captured tick.
extern "C" __global__ void advance_time(real* t, real dt) { t[0] += dt; }

// Wind speed at cells (the lower layer's, from the edge winds' kinetic energy), for the surface exchange.
extern "C" __global__ void cell_speed(int N, const real* u, const int* cellEdge, const real* le, const real* de,
                                      const real* areaCell, real* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  real ke = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    ke += le[e] * de[e] * (real)0.25 * u[e] * u[e];
  }
  out[i] = sqrt(2 * ke / areaCell[i]);
}

// Held & Suarez (1994) forcing for two layers (decision #35): Newtonian relaxation of each layer's theta toward the
// equilibrium theta at the layer's mass-mean pressure, T_eq(p, phi) = max(200, [315 - 60 sin^2 phi - 10 ln(p/p0)
// cos^2 phi] (p/p0)^kappa), at the rate k_a + (k_s - k_a) f cos^4 phi, f the layer's mean of max(0, (sigma - 0.7)/0.3);
// k_a = 1/40 d, k_s = 1/4 d. The heating then crosses theta* like any other. (Friction is the bottom stress of the
// layered core.) hs[0..1] = k_a, k_s (1/s). stored = the column's enthalpy change (J/m^2); qnet = its heating (W/m^2).
extern "C" __global__ void hs_column(int N, int L, const real* par, const real* hs, const real* lat, real* mu_in, real* theta,
                                     real dt, real* wx, real* qnet, real* stored) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const Par P = unpack(par);
  real mu[LMAX], th[LMAX], pb[LMAX], pib[LMAX], dth[LMAX];
  real mt = 0;
  for (int k = 0; k < L; k++) { mu[k] = mu_in[(size_t)k * N + i]; th[k] = theta[(size_t)k * N + i]; mt += mu[k]; }
  layer_exner(L, mu, P, pb, pib);
  real E0 = 0, Q = 0;
  for (int k = 0; k < L; k++) E0 += mu[k] * pib[k] * th[k];
  const real s2 = sin(lat[i]) * sin(lat[i]), c2 = 1 - s2;
  const real ps = P.grav * mt;
  real pt = 0;
  for (int k = 0; k < L; k++) {
    const real pbar = pow(pib[k] / P.cp, 1 / P.kappa) * P.p0;   // the pressure whose Exner function is pibar
    real Teq = (315 - 60 * s2 - 10 * log(pbar / P.p0) * c2) * pow(pbar / P.p0, P.kappa);
    if (Teq < 200) Teq = 200;
    const real theq = Teq * P.cp / pib[k];
    const real st = pt / ps, sb = pb[k] / ps;                   // the layer's sigma range
    const real lo = st > 0.7 ? st : (real)0.7;
    const real f = sb > 0.7 ? ((sb - 0.7) * (sb - 0.7) - (lo - 0.7) * (lo - 0.7)) / (2 * 0.3 * (sb - st)) : (real)0;
    const real kt = hs[0] + (hs[1] - hs[0]) * f * c2 * c2;
    const real d = (theq - th[k]) * (1 - exp(-kt * dt));
    dth[k] = d;
    th[k] += d;
    Q += mu[k] * pib[k] * d / dt;
    pt = pb[k];
  }
  // the heating of the air at theta*, at the interface's pressure p_i (sigma_i = p_i / p_s)
  real dth_up = dth[0];
  if (L == 2) {
    const real pi_ = pb[0], si = pi_ / ps;
    real Ti = (315 - 60 * s2 - 10 * log(pi_ / P.p0) * c2) * pow(pi_ / P.p0, P.kappa);
    if (Ti < 200) Ti = 200;
    const real thi = Ti * pow(P.p0 / pi_, P.kappa);
    const real ki = hs[0] + (hs[1] - hs[0]) * (si > 0.7 ? (si - 0.7) / 0.3 : (real)0) * c2 * c2;
    dth_up = (thi - P.theta_star) * (1 - exp(-ki * dt));
  }
  wx[i] = crossing(L, P, mu, th, dth, dth_up, pib);
  real E1 = 0;
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; E1 += mu[k] * pib[k] * th[k]; }
  qnet[i] = Q;
  stored[i] = E1 - E0;
}
