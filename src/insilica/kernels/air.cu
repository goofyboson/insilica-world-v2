// Column physics of the air layers over a slab surface (stages V2-2a and V2-2b, decisions #32-#35, #40, #45-#50):
// radiation (v1's shortwave, a gray longwave with a fixed optical-depth profile in pressure) and surface exchange,
// heating as potential temperature, air crossing each isentropic interface theta*_j only when heated or cooled across
// it; with moisture (V2-2b) vapor in every layer, evaporation from open water, condensation and rain, moist convection
// (plumes from the bottom layer detraining where they stop being buoyant), and thermodynamic sea ice on the slab.
// One thread per cell; layers k = 0 (top) .. L-1 (bottom), L = 2 or 3, mass mu_k (kg/m^2), specific humidity q_k
// (kg/kg). Interface j lies between layers j and j+1 at pressure p_j = g (mu_0 + .. + mu_j); its theta* is theta_star
// for the lowest interface (j = L-2) and theta_star2 for the one above it (L = 3).
//
// Energy (J/m^2): a layer's enthalpy mu_k pibar_k theta_k (pibar the mass-mean Exner function, cp T = pibar theta), its
// latent energy Lv mu_k q_k, the slab's C Ts, and the ice's -Lf m_ice (relative to water at Tf). Every process moves
// energy between these exactly; re-dividing the column (crossing, convection) books its rounding-level residual into
// the receiving layer, so the ledger closes to rounding. Rain leaves with no energy (its latent heat was released on
// condensing); vapor taken from the slab costs the slab Lv per kilogram.

typedef REAL real;
#define LMAX 8
#define PI_ 3.14159265358979323846

struct Par {
  real S0, asw, tau0, mu_ref, eps_s, h_sens, u0, alb, c_slab, mu_bl, theta_star, floor_, sigma, grav, cp, kappa, p0;
  real tilt, year_s, day_s, tau_lin, theta_star2;
  real moist, Lv, Lf, Tf, k_evap, tau_conv, alb_ice, rho_ice, ice_cover, k_ice, Rv, sat_layer, sw_tau;
  real alb_dry, alb_wet, soil_beta, k_inf, pond_cover;
};

// The parameters arrive as an array in this order (atmosphere.PAR_ORDER).
__device__ Par unpack(const real* a) {
  Par P;
  P.S0 = a[0]; P.asw = a[1]; P.tau0 = a[2]; P.mu_ref = a[3]; P.eps_s = a[4]; P.h_sens = a[5]; P.u0 = a[6]; P.alb = a[7];
  P.c_slab = a[8]; P.mu_bl = a[9]; P.theta_star = a[10]; P.floor_ = a[11]; P.sigma = a[12]; P.grav = a[13]; P.cp = a[14];
  P.kappa = a[15]; P.p0 = a[16]; P.tilt = a[17]; P.year_s = a[18]; P.day_s = a[19]; P.tau_lin = a[20]; P.theta_star2 = a[21];
  P.moist = a[22]; P.Lv = a[23]; P.Lf = a[24]; P.Tf = a[25]; P.k_evap = a[26]; P.tau_conv = a[27]; P.alb_ice = a[28];
  P.rho_ice = a[29]; P.ice_cover = a[30]; P.k_ice = a[31]; P.Rv = a[32]; P.sat_layer = a[33]; P.sw_tau = a[34];
  P.alb_dry = a[35]; P.alb_wet = a[36]; P.soil_beta = a[37]; P.k_inf = a[38]; P.pond_cover = a[39];
  return P;
}

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

// Saturation specific humidity (v1's qsat: Tetens over water), at temperature T (K) and pressure p (Pa).
__device__ __forceinline__ real qsat(real T, real p) {
  return 0.622 * 611.0 * exp(17.27 * (T - 273.15) / (T - 35.85)) / p;
}

// The pressure whose Exner function is pib (a layer's representative pressure).
__device__ __forceinline__ real p_of(real pib, const Par& P) { return pow(pib / P.cp, 1 / P.kappa) * P.p0; }

// Saturation adjustment of air at Exner pib and pressure p in one Newton step: vapor beyond q_sat condenses and its
// latent heat warms the air. Updates theta and q; returns the condensate (kg/kg).
__device__ real condense(real& th, real& q, real pib, real p, const Par& P) {
  const real T = pib * th / P.cp, qs = qsat(T, p);
  if (!(q > qs)) return 0;
  const real dq = (q - qs) / (1 + P.Lv * P.Lv * qs / (P.cp * P.Rv * T * T));
  q -= dq;
  th += P.Lv * dq / pib;
  return dq;
}

// Saturation adjustment of a layer of uniform theta spanning pressures [pt, pb] (Pa), against the layer's capacity: q_sat
// averaged over its pressure range (4-point Gauss-Legendre; temperatures below 160 K hold no vapor worth counting), so
// a deep layer's warm bottom counts, as it does when its relative humidity is uniform with height. One Newton step with
// d(q_sat)/d(theta) from Clausius-Clapeyron at each node. Updates theta and q; returns the condensate (kg/kg).
__device__ real condense_layer(real& th, real& q, real pib, real pt, real pb, const Par& P) {
  const real x[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  const real wq[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  real qs = 0, dqs = 0;
  for (int j = 0; j < 4; j++) {
    const real p = (real)0.5 * (pt + pb) + (real)0.5 * (pb - pt) * x[j];
    const real ex = pow(p / P.p0, P.kappa), T = th * ex;
    if (T < 160) continue;
    const real s = qsat(T, p);
    qs += (real)0.5 * wq[j] * s;
    dqs += (real)0.5 * wq[j] * s * P.Lv / (P.Rv * T * T) * ex;
  }
  if (!(q > qs)) return 0;
  const real dq = (q - qs) / (1 + P.Lv / pib * dqs);
  q -= dq;
  th += P.Lv * dq / pib;
  return dq;
}

// Condensation in layer k of the column (mass mu): the layer-integrated capacity when sat_layer, else v1's rule at the
// layer's mass-mean pressure.
__device__ real condense_k(int k, const real* mu, real* th, real* q, const real* pib, const Par& P) {
  if (P.sat_layer > 0) {
    real pt = 0;
    for (int j = 0; j < k; j++) pt += P.grav * mu[j];
    return condense_layer(th[k], q[k], pib[k], pt, pt + P.grav * mu[k], P);
  }
  return condense(th[k], q[k], pib[k], p_of(pib[k], P), P);
}

// A parcel (theta, q) lifted to pressure p (Pa): its theta and vapor there after condensing what it can't hold.
__device__ void parcel(real th, real q, real p, const Par& P, real& thp, real& qp) {
  thp = th; qp = q;
  condense(thp, qp, P.cp * pow(p / P.p0, P.kappa), p, P);
}

__device__ real column_energy(int L, const real* mu, const real* pib, const real* th, const real* q, const Par& P) {
  real E = 0;
  for (int k = 0; k < L; k++) E += mu[k] * pib[k] * th[k] + P.Lv * mu[k] * q[k];
  return E;
}

// The theta* of interface j (between layers j and j+1).
__device__ __forceinline__ real tstar(int j, int L, const Par& P) { return j == L - 2 ? P.theta_star : P.theta_star2; }

// Re-divide the column after moving mass m (kg/m^2, > 0) from layer `from` to layer `to`. The air leaves with theta
// th_leave and vapor q_leave per kilogram and arrives with th_arrive and q_arrive (different when it condensed on the
// way). The column's energy is the same before and after up to rounding; that residual is booked into the receiving
// layer.
__device__ void move_mass(int L, const Par& P, real* mu, real* th, real* q, real* pib, int from, int to, real m,
                          real th_leave, real th_arrive, real q_leave, real q_arrive) {
  if (m == 0) return;
  real pb2[LMAX], pib2[LMAX], mu2[LMAX], th2[LMAX], q2[LMAX];
  const real Eb = column_energy(L, mu, pib, th, q, P);
  for (int k = 0; k < L; k++) { mu2[k] = mu[k]; th2[k] = th[k]; q2[k] = q[k]; }
  mu2[from] -= m; mu2[to] += m;
  layer_exner(L, mu2, P, pb2, pib2);
  th2[from] = (mu[from] * th[from] - m * th_leave) / mu2[from];
  th2[to] = (mu[to] * th[to] + m * th_arrive) / mu2[to];
  q2[from] = (mu[from] * q[from] - m * q_leave) / mu2[from];
  q2[to] = (mu[to] * q[to] + m * q_arrive) / mu2[to];
  const real Ea = column_energy(L, mu2, pib2, th2, q2, P);
  th2[to] += (Eb - Ea) / (mu2[to] * pib2[to]);
  for (int k = 0; k < L; k++) { th[k] = th2[k]; mu[k] = mu2[k]; pib[k] = pib2[k]; q[k] = q2[k]; }
}

// A middle layer's theta between its two edges [a, b] (decision #58): a truncated exponential density proportional to
// exp(s (theta - a) / (b - a)) with the layer's mean m. Its mean is a + (b - a) r(s), r(s) = 1 - 1/s + 1/(e^s - 1)
// (1/2 at s = 0, rising with s), solved for s by bisection; the share of the layer's mass per kelvin at the edges is then
// fa = s / (e^s - 1) / (b - a) and fb = s e^s / (e^s - 1) / (b - a). A layer whose mean sits near one edge holds most of
// its air there and little at the other.
__device__ void edge_density(real a, real b, real m, real& fa, real& fb) {
  const real W = b - a;
  real x = (m - a) / W;
  if (x < (real)1e-6) x = (real)1e-6;
  if (x > 1 - (real)1e-6) x = 1 - (real)1e-6;
  // safeguarded Newton on r(s) = x within a bracket (r rises with s; r'(s) = 1/s^2 - e^s/(e^s - 1)^2, 1/12 at s = 0)
  real lo = -700, hi = 700, s = 12 * (x - (real)0.5);
  for (int it = 0; it < 40; it++) {
    real r, dr;
    if (fabs(s) < (real)1e-4) { r = (real)0.5 + s / 12; dr = (real)1 / 12; }
    else {
      const real e1 = expm1(s);
      r = 1 - 1 / s + 1 / e1;
      dr = 1 / (s * s) - (e1 + 1) / (e1 * e1);
    }
    if (r < x) lo = s; else hi = s;
    real sn = s - (r - x) / dr;
    if (r == x) break;
    if (!(sn >= lo && sn <= hi) || !(dr > 0)) sn = (real)0.5 * (lo + hi);
    if (fabs(sn - s) < (real)1e-12 * (1 + fabs(s))) { s = sn; break; }
    s = sn;
  }
  if (fabs(s) < (real)1e-6) { fa = (1 - s / 2) / W; fb = (1 + s / 2) / W; return; }
  const real e1 = expm1(s);
  fa = s / e1 / W;
  fb = s * (e1 + 1) / e1 / W;
}

// Crossing the interfaces (decisions #34, #40, #47, #58). Heating carries air up across a layer's top edge, cooling
// carries it down across its bottom edge: the mass crossing is the layer's mass per kelvin at that edge times |d theta|,
// from the theta before the heating; a layer wholly beyond the edge crosses to its floor. The bottom and top layers
// have one known edge: their theta spans [lo, hi] evenly about the mean with theta*_j at the shared edge, so the mass
// crossing is mu |d theta| / (2 |theta* - theta|). A middle layer has both: its air is spread by edge_density. The
// convectively mixed bottom layer and the middle layer cross by their mean heating; the stratified top layer by the
// heating of its air at its bottom edge (dth_up; the radiative
// case passes its mean heating). All crossings are worked out from the state before any of them, then applied, lowest
// interface first; crossing air carries theta* (or its whole layer's theta) and its layer's vapor. Floor air stays on its
// own side of every interface. wxo[j] (j = 0..L-2) receives the mass moved up across interface j (kg/m^2); gross, if
// given, the up and down parts at each interface before netting and limiting (gross[2j], gross[2j+1]; diagnostics).
__device__ void crossing(int L, const Par& P, real* mu, real* th, real* q, const real* dth, real dth_up, real* pib,
                         real* wxo, real* gross = 0) {
  for (int j = 0; j + 1 < L; j++) wxo[j] = 0;
  if (P.theta_star <= 0) return;                           // theta_star <= 0: no crossing (diagnostics)
  real dmj[LMAX], thcj[LMAX];
  for (int j = 0; j + 1 < L; j++) {
    const int a = j, b = j + 1;                            // the layers above and below interface j
    const real ts_ = tstar(j, L, P);
    real dm = 0, thc = ts_;
    if (gross) { gross[2 * j] = 0; gross[2 * j + 1] = 0; }
    if (dth[b] > 0) {
      if (th[b] >= ts_) { dm = mu[b] - P.floor_; thc = th[b]; }
      else if (b == L - 1) dm = mu[b] * dth[b] / (2 * (ts_ - (th[b] - dth[b])));
      else {                                                // a middle layer's top edge
        real fa, fb;
        edge_density(tstar(b, L, P), ts_, th[b] - dth[b], fa, fb);
        dm = mu[b] * dth[b] * fb;
      }
    }
    const real da = a == 0 ? dth_up : dth[a];
    if (da < 0) {
      real down;
      if (th[a] <= ts_) { down = mu[a] - P.floor_; thc = th[a]; }
      else if (a == 0) down = mu[a] * (-da) / (2 * ((th[a] - dth[a]) - ts_));
      else {                                                // a middle layer's bottom edge
        real fa, fb;
        edge_density(ts_, tstar(a - 1, L, P), th[a] - dth[a], fa, fb);
        down = mu[a] * (-da) * fa;
      }
      dm -= down;
      if (gross) gross[2 * j + 1] = down;
    }
    if (gross) gross[2 * j] = dm + gross[2 * j + 1];
    dmj[j] = dm; thcj[j] = thc;
  }
  for (int j = L - 2; j >= 0; j--) {
    const int a = j, b = j + 1;
    real dm = dmj[j];
    const real maxup = mu[b] - P.floor_, maxdn = mu[a] - P.floor_;
    if (dm > maxup) dm = maxup > 0 ? maxup : (real)0;
    if (-dm > maxdn) dm = maxdn > 0 ? -maxdn : (real)0;
    if (dm > 0) move_mass(L, P, mu, th, q, pib, b, a, dm, thcj[j], thcj[j], q[b], q[b]);
    else if (dm < 0) move_mass(L, P, mu, th, q, pib, a, b, -dm, thcj[j], thcj[j], q[a], q[a]);
    wxo[j] = dm;
  }
  const real tiny = P.floor_ * (real)1.000001;
  for (int j = L - 2; j >= 0; j--) {
    const int a = j, b = j + 1;
    const real ts_ = tstar(j, L, P);
    if (mu[b] <= tiny && th[b] > ts_) {
      th[a] += mu[b] * pib[b] * (th[b] - ts_) / (mu[a] * pib[a]);
      th[b] = ts_;
    }
    if (mu[a] <= tiny && th[a] < ts_) {
      th[b] -= mu[a] * pib[a] * (ts_ - th[a]) / (mu[b] * pib[b]);
      th[a] = ts_;
    }
  }
}

// The longwave optical depth above pressure p in a column whose surface is at ps: tau0 [f s + (1 - f) s^4], s = p / ps
// (Frierson et al. 2006's gray profile: a linear part for well-mixed absorbers, a quartic one for vapor, which sits low;
// vapor-independent, decision #48).
__device__ __forceinline__ real tau_p(real p, real ps, const Par& P) {
  const real s = p / ps, s2 = s * s;
  return P.tau0 * (P.tau_lin * s + (1 - P.tau_lin) * s2 * s2);
}

// mode 0: the sun at time t (diurnal and seasonal, v1's sunVector); mode 1: insol[i], a given mean insolation.
// Outputs per cell: wx (L-1 x N: mass moved up across each interface, dry crossing + moist convection), olr, sw_abs
// (W/m^2), stored (the
// column's energy change, J/m^2), rain and evap (kg/m^2/s), conv (kg/m^2 moved up by moist convection this step),
// tsurf (K: the surface that radiates, the ice skin where the ice is thicker than ice_cover, else the slab), rainp
// (3 x N, kg/m^2/s: the rain from condensation in the layers above the bottom one, in the bottom one, and convection).
// qv and ice are read and written only when moist. On land (land[i] > 0.5, moist air; decision #84) the surface has the heat
// capacity cap and the albedo of its soil's wetness; it evaporates from the soil bucket (soil, capacity soilcap; kg/m^2),
// at the full rate above soil_beta of capacity (Manabe's bucket), and from surface water (pond, kg/m^2; decision #85), which
// covers pond / pond_cover of the cell at the sea's albedo and evaporates at the full rate. Rain and the ice's melt reach the
// pond, or rain falls as snow on ground below freezing (its latent heat to the ground) and lies as ice; a pond freezes on
// such ground; otherwise it soaks in at k_inf (1 - soil / soilcap). What the bucket cannot hold comes back to the pond.
extern "C" __global__ void air_column(int N, int L, const real* par, int mode, const real* t, const real* insol, const real* pos,
                                      const real* uabs, real* mu_in, real* theta, real* qv, real* Ts, real* ice, real dt,
                                      real* wx, real* olr, real* sw_abs, real* stored, real* rain, real* evap, real* conv,
                                      real* tsurf, real* rainp, real* dgo, const real* cap, real* qocean,
                                      real* qsw, const real* land, const real* soilcap, real* soil, real* pond) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const Par P = unpack(par);
  const bool moist = P.moist > 0;
  real mu[LMAX], th[LMAX], q[LMAX], pb[LMAX], pib[LMAX], Q[LMAX], T[LMAX], eps[LMAX];
  real mt = 0;
  for (int k = 0; k < L; k++) {
    mu[k] = mu_in[(size_t)k * N + i]; th[k] = theta[(size_t)k * N + i]; q[k] = moist ? qv[(size_t)k * N + i] : (real)0;
    mt += mu[k]; Q[k] = 0;
  }
  layer_exner(L, mu, P, pb, pib);
  real m_ice = moist ? ice[i] : (real)0;
  // the surface water's heat capacity (J/m^2/K): the ocean's mixed layer when coupled (cap), else the slab
  const real Cw = cap ? cap[i] : P.c_slab;
  const real E0 = column_energy(L, mu, pib, th, q, P) + Cw * Ts[i] - P.Lf * m_ice;
  for (int k = 0; k < L; k++) T[k] = pib[k] * th[k] / P.cp;
  const int top = 0, bot = L - 1;

  // ---- shortwave; the surface's albedo with the ice cover ----
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
  const real dice = m_ice / P.rho_ice;                           // ice thickness (m)
  const real fice = moist ? fmin((real)1, dice / P.ice_cover) : (real)0;
  const bool isl = moist && land && land[i] > (real)0.5;
  const real ice0 = m_ice;
  real wet = isl ? soil[i] : (real)0;
  real pw = isl ? pond[i] : (real)0;
  const real pcov = isl && pw > 0 ? fmin((real)1, pw / P.pond_cover) : (real)0;
  const real alb0 = isl ? pcov * P.alb + (1 - pcov) * (P.alb_dry + (P.alb_wet - P.alb_dry) * wet / soilcap[i]) : P.alb;
  const real alb = fice * P.alb_ice + (1 - fice) * alb0;
  const real toAir = P.asw * (mt / P.mu_ref) * I;
  if (P.sw_tau > 1) {                  // split like the vapor part (s^4) of the longwave profile (decision #59)
    const real ps_ = pb[bot];
    real st = 0;
    for (int k = 0; k < L; k++) {
      const real sb = pb[k] / ps_, sb4 = sb * sb * sb * sb;
      Q[k] += toAir * (sb4 - st);
      st = sb4;
    }
  } else if (P.sw_tau > 0)             // split like the longwave optical depth, where the vapor is (decision #51)
    for (int k = 0; k < L; k++)
      Q[k] += toAir * (tau_p(pb[k], pb[bot], P) - tau_p(k ? pb[k - 1] : (real)0, pb[bot], P)) / P.tau0;
  else
    for (int k = 0; k < L; k++) Q[k] += toAir * mu[k] / mt;
  const real swS = (I - toAir) * (1 - alb);

  // ---- longwave: each layer's emissivity from its share of the optical-depth profile; the downward longwave reaching
  // the surface ----
  for (int k = 0; k < L; k++) eps[k] = 1 - exp(-(tau_p(pb[k], pb[bot], P) - tau_p(k ? pb[k - 1] : (real)0, pb[bot], P)));
  real ldown = 0;
  for (int k = 0; k < L; k++) {
    real dn = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int j = k + 1; j < L; j++) dn -= eps[j] * dn;
    ldown += dn;
  }

  // ---- the air touching the ground: the bottom mu_bl of the column ----
  real wbl[LMAX], thb = 0, qb = 0, mb = 0;
  {
    real need = P.mu_bl;
    for (int k = bot; k >= 0; k--) { const real m = mu[k] < need ? mu[k] : need; wbl[k] = m; need -= m; thb += m * th[k]; qb += m * q[k]; mb += m; }
  }
  thb /= mb; qb /= mb;
  const real ps = pb[bot];
  const real Tsa = thb * pow(ps / P.p0, P.kappa);
  const real hs = P.h_sens * (1 + uabs[i] / P.u0);

  // ---- the surface: open water at Ts, or (ice thicker than ice_cover) a zero-heat-capacity skin at Tsk balancing
  // conduction through the ice against sunlight, longwave and sensible heat, capped at freezing (v1 SPEC 18 #11) ----
  const real ts = Ts[i];
  real tsk = ts, melt_top = 0, cond = 0;
  const bool skin = moist && dice > P.ice_cover;
  if (skin) {
    const real kc = P.k_ice / dice;
    real x = fmin(ts, P.Tf);
    for (int it = 0; it < 8; it++) {
      const real x3 = x * x * x;
      const real f = kc * (ts - x) - P.eps_s * P.sigma * x3 * x + swS + ldown - hs * (x - Tsa);
      x -= f / (-kc - 4 * P.eps_s * P.sigma * x3 - hs);
    }
    if (x > P.Tf) x = P.Tf;
    const real x4 = x * x * x * x;
    // what the skin can't pass on at its temperature melts ice from the top (zero when the balance was met)
    melt_top = kc * (ts - x) - P.eps_s * P.sigma * x4 + swS + ldown - hs * (x - Tsa);
    tsk = x;
    cond = kc * (ts - tsk);                                      // W/m^2 up from the slab into the skin
  }

  // ---- longwave from the emitting surface, up through the layers ----
  real up = P.eps_s * P.sigma * tsk * tsk * tsk * tsk;
  for (int k = bot; k >= 0; k--) {
    const real a = eps[k] * up;
    Q[k] += a;
    const real em = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    Q[k] -= 2 * em;
    up = up - a + em;
  }
  olr[i] = up;
  for (int k = 0; k < L; k++) {
    real dn = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int j = k + 1; j < L; j++) { const real a = eps[j] * dn; Q[j] += a; dn -= a; }
  }

  // ---- sensible heat and evaporation into the bottom air ----
  const real H = hs * (tsk - Tsa);
  for (int k = 0; k < L; k++) Q[k] += H * wbl[k] / mb;
  real E = 0;
  if (moist) {
    const real def = qsat(tsk, ps) - qb;
    E = def > 0 ? P.k_evap * (1 + uabs[i] / P.u0) * def * (1 - fice) : (real)0;   // kg/m^2/s, open water only
    if (isl) {
      const real full = P.soil_beta * soilcap[i];
      real Es = (1 - pcov) * E, Ep = pcov * E;
      if (wet < full) Es *= wet / full;
      if (Es * dt > wet) Es = wet / dt;
      if (Ep * dt > pw) Ep = pw / dt;
      E = Es + Ep;
      wet -= Es * dt;
      pw -= Ep * dt;
    }
    for (int k = 0; k < L; k++) q[k] += E * dt * wbl[k] / mb / mu[k];
  }
  evap[i] = E;
  tsurf[i] = tsk;

  // ---- heating as potential temperature ----
  real dth[LMAX];
  for (int k = 0; k < L; k++) { dth[k] = Q[k] * dt / (mu[k] * pib[k]); th[k] += dth[k]; }
  // diagnostics (dgo, optional, rows x N): per layer radiative + sensible heating (W/m^2), latent heating (W/m^2); per
  // interface the gross dry crossing up and down (kg/m^2 per step); per layer the convective mass delivered into it
  real dgl[LMAX], gross[2 * LMAX], cin[LMAX];
  for (int k = 0; k < L; k++) { dgl[k] = 0; cin[k] = 0; }

  // ---- the slab: open water takes the surface fluxes (evaporation costs it Lv per kg); under ice, the conduction up
  // into the skin. Freezing below Tf turns slab heat into ice; ice melts with slab heat above Tf and from the top
  // with the skin's surplus ----
  real qS;
  if (skin) qS = -cond;
  else qS = swS + ldown - P.eps_s * P.sigma * ts * ts * ts * ts - H - P.Lv * E;
  real Tn = ts + qS * dt / Cw;
  if (qocean) qocean[i] = qS;                                    // W/m^2 into the surface water (before ice)
  if (qsw) qsw[i] = skin ? (real)0 : swS;                      // its sunlight (open water), part of which passes below
  if (moist) {
    m_ice -= melt_top * dt / P.Lf;
    if (m_ice < 0) { Tn += -m_ice * P.Lf / Cw; m_ice = 0; }   // more top melt than ice: the rest warms the slab
    if (Tn < P.Tf) { if (!isl) { m_ice += (P.Tf - Tn) * Cw / P.Lf; Tn = P.Tf; } }
    else if (m_ice > 0) {
      const real m = (Tn - P.Tf) * Cw / P.Lf;
      if (m <= m_ice) { m_ice -= m; Tn = P.Tf; } else { Tn = P.Tf + (m - m_ice) * P.Lf / Cw; m_ice = 0; }
    }
    ice[i] = m_ice;
  }
  Ts[i] = Tn;
  sw_abs[i] = toAir + swS;

  // ---- condensation: one-step saturation adjustment per layer (v1 SPEC 18 #25); the latent heat warms the layer and
  // counts as heating for the crossing of theta*; the water rains out ----
  real R = 0, Rk[2] = {0, 0}, Rc = 0;            // Rk: condensation in the layers above the bottom, in the bottom
  if (moist)
    for (int k = 0; k < L; k++) {
      const real dq = condense_k(k, mu, th, q, pib, P);
      dth[k] += P.Lv * dq / pib[k];
      dgl[k] += P.Lv * mu[k] * dq / dt;
      R += mu[k] * dq;
      Rk[k == bot ? 1 : 0] += mu[k] * dq;
    }

  // ---- dry crossing of the interfaces ----
  real wxo[LMAX];
  crossing(L, P, mu, th, q, dth, dth[top], pib, wxo, gross);

  // ---- moist convection (decisions #46, #49): a plume of bottom-layer air lifted to the interface above it, condensing
  // on the way, compared with the air there, which is at theta*. Buoyant (theta_p > theta*), it rises; it keeps rising
  // through each further interface it is buoyant at and detrains into the layer above the last one, with its latent
  // heat; its condensate rains. Removing bottom air lowers the first interface, where the plume condenses less, so the
  // mass that would remove the buoyancy there is d / (-d(theta_p)/dp) / g; a share 1 - exp(-dt/tau_conv) of it moves per
  // step (a Betts-Miller-style relaxation; the one convective constant). ----
  real mc = 0;
  if (moist && P.theta_star > 0) {
    real pi_ = 0;
    for (int k = 0; k < bot; k++) pi_ += P.grav * mu[k];     // the bottom layer's top
    real thp, qc, thp2, qc2;
    parcel(th[bot], q[bot], pi_, P, thp, qc);
    const real d = thp - tstar(bot - 1, L, P);
    if (d > 0) {
      const real dp = 500;
      parcel(th[bot], q[bot], pi_ + dp, P, thp2, qc2);
      const real slope = (thp - thp2) / dp;                     // > 0: the buoyancy falls as the interface lowers
      const real maxup = mu[bot] - P.floor_;
      const real want = slope > 0 ? d / slope / P.grav : maxup;
      mc = fmin(want, maxup) * (1 - exp(-dt / P.tau_conv));
      if (mc > 0) {
        int kd = bot - 1;                                       // the layer it detrains into
        real pj = pi_;
        while (kd > 0) {                                        // buoyant at the next interface up too?
          pj -= P.grav * mu[kd];
          real t2, q2;
          parcel(th[bot], q[bot], pj, P, t2, q2);
          if (!(t2 > tstar(kd - 1, L, P))) break;
          thp = t2; qc = q2; kd--;
        }
        R += mc * (q[bot] - qc);
        Rc = mc * (q[bot] - qc);
        move_mass(L, P, mu, th, q, pib, bot, kd, mc, th[bot], thp, q[bot], qc);
        for (int j = kd; j < bot; j++) wxo[j] += mc;            // it crosses every interface on its way
        cin[kd] += mc;
        // what the arriving air brings beyond the receiving layer's saturation rains at once
        const real rt = mu[kd] * condense_k(kd, mu, th, q, pib, P);
        R += rt;
        Rc += rt;
      } else mc = 0;
    }
  }
  if (conv) conv[i] = mc;
  for (int j = 0; j + 1 < L; j++) wx[(size_t)j * N + i] = wxo[j];
  rain[i] = R / dt;
  if (isl) {
    pw += ice0 - m_ice;                                         // the ice's melt
    if (Ts[i] < P.Tf) {                                         // snow: its freezing warms the ground
      m_ice += R;
      Ts[i] += P.Lf * R / Cw;
    } else pw += R;
    if (Ts[i] < P.Tf && pw > 0) {                               // the pond freezes, warming the ground up to freezing
      const real room = (P.Tf - Ts[i]) * Cw / P.Lf, m = pw < room ? pw : room;
      pw -= m;
      m_ice += m;
      Ts[i] += P.Lf * m / Cw;
    }
    ice[i] = m_ice;
    if (Ts[i] >= P.Tf) {                                        // soaking in; frozen ground takes none
      const real room = P.k_inf * dt * (1 - wet / soilcap[i]);
      const real m = pw < room ? pw : room;
      if (m > 0) { pw -= m; wet += m; }
    }
    if (wet > soilcap[i]) { pw += wet - soilcap[i]; wet = soilcap[i]; }   // a spring
    soil[i] = wet;
    pond[i] = pw;
  }
  if (dgo) {
    int r = 0;
    for (int k = 0; k < L; k++) dgo[(size_t)(r++) * N + i] = Q[k];
    for (int k = 0; k < L; k++) dgo[(size_t)(r++) * N + i] = dgl[k];
    for (int j = 0; j + 1 < L; j++) { dgo[(size_t)(r++) * N + i] = gross[2 * j]; dgo[(size_t)(r++) * N + i] = gross[2 * j + 1]; }
    for (int k = 0; k < L; k++) dgo[(size_t)(r++) * N + i] = cin[k];
  }
  if (rainp) { rainp[i] = Rk[0] / dt; rainp[(size_t)N + i] = Rk[1] / dt; rainp[(size_t)2 * N + i] = Rc / dt; }

  const real E1 = column_energy(L, mu, pib, th, q, P) + Cw * Ts[i] - P.Lf * m_ice;
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; if (moist) qv[(size_t)k * N + i] = q[k]; }
  stored[i] = E1 - E0;
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
// k_a = 1/40 d, k_s = 1/4 d. The heating then crosses theta* like any other; the upper layer's crossing uses the
// heating at the interface (#40). hs[0..1] = k_a, k_s (1/s). stored = the column's enthalpy change (J/m^2).
extern "C" __global__ void hs_column(int N, int L, const real* par, const real* hs, const real* lat, real* mu_in, real* theta,
                                     real dt, real* wx, real* qnet, real* stored) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const Par P = unpack(par);
  real mu[LMAX], th[LMAX], q[LMAX], pb[LMAX], pib[LMAX], dth[LMAX];
  for (int k = 0; k < L; k++) { mu[k] = mu_in[(size_t)k * N + i]; th[k] = theta[(size_t)k * N + i]; q[k] = 0; }
  layer_exner(L, mu, P, pb, pib);
  real E0 = 0, Qs = 0;
  for (int k = 0; k < L; k++) E0 += mu[k] * pib[k] * th[k];
  const real s2 = sin(lat[i]) * sin(lat[i]), c2 = 1 - s2;
  const real ps = pb[L - 1];
  real pt = 0;
  for (int k = 0; k < L; k++) {
    const real pbar = p_of(pib[k], P);
    real Teq = (315 - 60 * s2 - 10 * log(pbar / P.p0) * c2) * pow(pbar / P.p0, P.kappa);
    if (Teq < 200) Teq = 200;
    const real theq = Teq * P.cp / pib[k];
    const real st = pt / ps, sb = pb[k] / ps;
    const real lo = st > 0.7 ? st : (real)0.7;
    const real f = sb > 0.7 ? ((sb - 0.7) * (sb - 0.7) - (lo - 0.7) * (lo - 0.7)) / (2 * 0.3 * (sb - st)) : (real)0;
    const real kt = hs[0] + (hs[1] - hs[0]) * f * c2 * c2;
    const real d = (theq - th[k]) * (1 - exp(-kt * dt));
    dth[k] = d;
    th[k] += d;
    Qs += mu[k] * pib[k] * d / dt;
    pt = pb[k];
  }
  real dth_up = dth[0];
  if (L == 2) {
    const real pi_ = pb[0], si = pi_ / ps;
    real Ti = (315 - 60 * s2 - 10 * log(pi_ / P.p0) * c2) * pow(pi_ / P.p0, P.kappa);
    if (Ti < 200) Ti = 200;
    const real thi = Ti * pow(P.p0 / pi_, P.kappa);
    const real ki = hs[0] + (hs[1] - hs[0]) * (si > 0.7 ? (si - 0.7) / 0.3 : (real)0) * c2 * c2;
    dth_up = (thi - P.theta_star) * (1 - exp(-ki * dt));
  }
  real wxo[LMAX];
  crossing(L, P, mu, th, q, dth, dth_up, pib, wxo);
  for (int j = 0; j + 1 < L; j++) wx[(size_t)j * N + i] = wxo[j];
  real E1 = 0;
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; E1 += mu[k] * pib[k] * th[k]; }
  qnet[i] = Qs;
  stored[i] = E1 - E0;
}
