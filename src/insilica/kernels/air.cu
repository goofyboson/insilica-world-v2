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
// NL, the world's layer count, given at compile time (V2-5): the column's arrays then fit in registers and its loops unroll
#ifdef NL
#define LMAX NL
#else
#define LMAX 8
#endif
#define PI_ 3.14159265358979323846
// Mixed precision (SPEC #106): these transcendentals in 32-bit, all sums, the state and the energy ledger in 64-bit.
// Kept 64-bit: layer_exner (its p x - pt xt cancels for thin layers) and the edge shape's expm1/log1p (arguments to ±700).
#if defined(MIXED)
#define FEXP(x) ((real)expf((float)(x)))
#define FPOW(x, y) ((real)powf((float)(x), (float)(y)))
#define FLOG(x) ((real)logf((float)(x)))
#define OMEXP(x) ((real)-expm1f(-(float)(x)))                  // 1 - e^-x, accurate for small x
typedef float fast;                                              // the capacity, condensation and spread arithmetic
#else
#define FEXP(x) exp(x)
#define FPOW(x, y) pow(x, y)
#define FLOG(x) log(x)
#define OMEXP(x) (1 - exp(-(x)))
typedef real fast;
#endif

struct Par {
  real S0, asw, tau0, mu_ref, eps_s, h_sens, u0, alb, c_slab, mu_bl, theta_star, floor_, sigma, grav, cp, kappa, p0;
  real tilt, year_s, day_s, tau_lin, theta_star2;
  real moist, Lv, Lf, Tf, k_evap, tau_conv, alb_ice, rho_ice, ice_cover, k_ice, Rv, sat_layer, sw_tau;
  real alb_dry, alb_wet, soil_beta, k_inf, pond_cover, alb_snow, alb_ice_melt, alb_snow_melt, alb_glacier, snow_cover, alb_ramp, land_exch, plume_deep, lw_w0, c_dry, c_wet, sat_prof, land_rs, sat_base, bl_bot, cloud, rh_crit, re_liq, re_ice, cloud_g, k_lw_liq, k_lw_ice, c_auto, q_auto, v_ice, l_det, tau_erode, k_revap, mu_cloud, theta_rise, bl_mix, mu_night, bl_ent, bl_cap, bulk_c, bulk_gust, bulk_z, bulk_w, plume_ent, l_ice, storm_area, graupel_v, ice_small, q_shape, edge_heat, freeze_dry;
  int Lm1;                                         // the column's bottom layer
  real ths[LMAX];                                  // the theta* of the interfaces above the lowest one, from the top (decision #101)
};

// The parameters arrive as an array in this order (atmosphere.PAR_ORDER).
__device__ Par unpack(const real* a) {
  Par P;
  P.S0 = a[0]; P.asw = a[1]; P.tau0 = a[2]; P.mu_ref = a[3]; P.eps_s = a[4]; P.h_sens = a[5]; P.u0 = a[6]; P.alb = a[7];
  P.c_slab = a[8]; P.mu_bl = a[9]; P.theta_star = a[10]; P.floor_ = a[11]; P.sigma = a[12]; P.grav = a[13]; P.cp = a[14];
  P.kappa = a[15]; P.p0 = a[16]; P.tilt = a[17]; P.year_s = a[18]; P.day_s = a[19]; P.tau_lin = a[20]; P.theta_star2 = a[21];
  P.moist = a[22]; P.Lv = a[23]; P.Lf = a[24]; P.Tf = a[25]; P.k_evap = a[26]; P.tau_conv = a[27]; P.alb_ice = a[28];
  P.rho_ice = a[29]; P.ice_cover = a[30]; P.k_ice = a[31]; P.Rv = a[32]; P.sat_layer = a[33]; P.sw_tau = a[34];
  P.alb_dry = a[35]; P.alb_wet = a[36]; P.soil_beta = a[37]; P.k_inf = a[38]; P.pond_cover = a[39]; P.alb_snow = a[40]; P.alb_ice_melt = a[41]; P.alb_snow_melt = a[42]; P.alb_glacier = a[43]; P.snow_cover = a[44]; P.alb_ramp = a[45]; P.land_exch = a[46]; P.plume_deep = a[47]; P.lw_w0 = a[48]; P.c_dry = a[49]; P.c_wet = a[50]; P.sat_prof = a[51]; P.land_rs = a[52]; P.sat_base = a[53]; P.bl_bot = a[54]; P.cloud = a[55]; P.rh_crit = a[56]; P.re_liq = a[57]; P.re_ice = a[58]; P.cloud_g = a[59]; P.k_lw_liq = a[60]; P.k_lw_ice = a[61]; P.c_auto = a[62]; P.q_auto = a[63]; P.v_ice = a[64]; P.l_det = a[65]; P.tau_erode = a[66]; P.k_revap = a[67]; P.mu_cloud = a[68]; P.theta_rise = a[69]; P.bl_mix = a[70]; P.mu_night = a[71]; P.bl_ent = a[72]; P.bl_cap = a[73]; P.bulk_c = a[74]; P.bulk_gust = a[75]; P.bulk_z = a[76]; P.bulk_w = a[77]; P.plume_ent = a[78]; P.l_ice = a[79]; P.storm_area = a[80]; P.graupel_v = a[81]; P.ice_small = a[82]; P.q_shape = a[83]; P.edge_heat = a[84]; P.freeze_dry = a[85]; for (int j = 0; j < LMAX; j++) P.ths[j] = a[86 + j];
  return P;
}

__device__ void layer_exner(int L, const real* mu, const Par& P, real* pb, real* pib, real* exi = 0) {
  real p = 0, xt = 0;
  for (int k = 0; k < L; k++) {
    const real pt = p;
    p = p + P.grav * mu[k];
    pb[k] = p;
    const real x = pow(p / P.p0, P.kappa);                       // one pow per interface
    pib[k] = P.cp * (p * x - pt * xt) / ((P.kappa + 1) * (p - pt));
    if (exi) exi[k] = x;                                         // (p_k / p0)^kappa at the layer's lower edge
    xt = x;
  }
}

// (p / p0)^kappa at each layer's four capacity nodes (gx[4k + j]), for layer_cap while the layers' masses stay as they are (V2-5)
__device__ void node_exner(int L, const real* mu, const Par& P, real* gx) {
  const real GX[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  for (int k = 0; k < L; k++) {
    real pt = 0;
    for (int j = 0; j < k; j++) pt += P.grav * mu[j];
    const real pb = pt + P.grav * mu[k];
    for (int j = 0; j < 4; j++) {
      const real p = (real)0.5 * (pt + pb) + (real)0.5 * (pb - pt) * GX[j];
      gx[4 * k + j] = FPOW(p / P.p0, P.kappa);
    }
  }
}

// Saturation specific humidity (v1's qsat: Tetens over water), at temperature T (K) and pressure p (Pa).
__device__ __forceinline__ fast qsat(fast T, fast p) {
  return (fast)0.622 * (fast)611.0 * (fast)FEXP((fast)17.27 * (T - (fast)273.15) / (T - (fast)35.85)) / p;
}

// The pressure whose Exner function is pib (a layer's representative pressure).
__device__ __forceinline__ real p_of(real pib, const Par& P) { return FPOW(pib / P.cp, 1 / P.kappa) * P.p0; }

// Saturation adjustment of air at Exner pib and pressure p in one Newton step: vapor beyond q_sat condenses and its
// latent heat warms the air. Updates theta and q; returns the condensate (kg/kg).
__device__ real condense(real& th, real& q, real pib, real p, const Par& P) {
  const fast T = (fast)pib * (fast)th / (fast)P.cp, qs = qsat(T, (fast)p);
  if (!(q > qs)) return 0;
  const real dq = (fast)(q - qs) / ((fast)1 + (fast)P.Lv * (fast)P.Lv * qs / ((fast)P.cp * (fast)P.Rv * T * T));
  q -= dq;
  th += P.Lv * dq / pib;
  return dq;
}

// The theta* of interface j (between layers j and j+1).
__device__ __forceinline__ real tstar(int j, int L, const Par& P) { return j == L - 2 ? P.theta_star : P.ths[j]; }

#if defined(MIXED)
// Mixed precision: the same r(s) = x solved in 32-bit on |s| <= 30, its series near 0 (no cancellation), and beyond 30 the
// exact asymptotes (e^-30 ~ 1e-13 left out): r = 1 - 1/s for s > 30, r = -1/s for s < -30.
__device__ real edge_shape(real a, real b, real m) {
  real xd = (m - a) / (b - a);
  if (xd < (real)1e-6) xd = (real)1e-6;
  if (xd > 1 - (real)1e-6) xd = 1 - (real)1e-6;
  if (xd > 1 - (real)1 / 30) return 1 / (1 - xd);
  if (xd < (real)1 / 30) return -1 / xd;
  const float x = (float)xd;
  float lo = -30.f, hi = 30.f, t = 12.f * (x - 0.5f);
  for (int it = 0; it < 30; it++) {
    float r, dr;
    if (fabsf(t) < 0.5f) {
      const float t2 = t * t;
      r = 0.5f + t * (1.f / 12 + t2 * (-1.f / 720 + t2 * (1.f / 30240 - t2 / 1209600.f)));
      dr = 1.f / 12 + t2 * (-1.f / 240 + t2 * (1.f / 6048 - t2 / 172800.f));
    } else {
      const float e1 = expm1f(t);
      r = 1.f - 1.f / t + 1.f / e1;
      dr = 1.f / (t * t) - (e1 + 1.f) / (e1 * e1);
    }
    if (r < x) lo = t; else hi = t;
    float tn = t - (r - x) / dr;
    if (r == x) break;
    if (!(tn >= lo && tn <= hi) || !(dr > 0)) tn = 0.5f * (lo + hi);
    if (fabsf(tn - t) < 1e-6f * (1.f + fabsf(t))) { t = tn; break; }
    t = tn;
  }
  return t;
}
// The mass share m's theta in a layer of shape s, as a fraction of the way between the edges: ln(1 + m (e^s - 1)) / s.
__device__ __forceinline__ fast spread_frac(fast m, fast s) {
  if (fabsf(s) < 1e-6f) return m;
  if (s > 30.f) return 1.f + logf(m) / s;
  if (s < -30.f) return log1pf(-m) / s;
  return log1pf(m * expm1f(s)) / s;
}
#define SPREAD(m, s, es) spread_frac((fast)(m), (fast)(s))
#define ES(s) ((real)0)
#else
#define SPREAD(m, s, es) (fabs(s) < (real)1e-6 ? (m) : log1p((m) * (es)) / (s))
#define ES(s) expm1(s)
__device__ real edge_shape(real a, real b, real m) {
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
  return s;
}
#endif

// Saturation adjustment of a layer of uniform theta spanning pressures [pt, pb] (Pa), against the layer's capacity: q_sat
// averaged over its pressure range (4-point Gauss-Legendre; temperatures below 160 K hold no vapor worth counting), so
// a deep layer's warm bottom counts, as it does when its relative humidity is uniform with height. One Newton step with
// d(q_sat)/d(theta) from Clausius-Clapeyron at each node. Updates theta and q; returns the condensate (kg/kg).
__device__ real condense_layer(real& th, real& q, real pib, real pt, real pb, const Par& P) {
  const real x[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  const real wq[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  fast qs = 0, dqs = 0;
  for (int j = 0; j < 4; j++) {
    const real p = (real)0.5 * (pt + pb) + (real)0.5 * (pb - pt) * x[j];
    const fast ex = FPOW(p / P.p0, P.kappa), T = (fast)th * ex;
    if (T < 160) continue;
    const fast s = qsat(T, (fast)p);
    qs += (fast)0.5 * (fast)wq[j] * s;
    dqs += (fast)0.5 * (fast)wq[j] * s * (fast)P.Lv / ((fast)P.Rv * T * T) * ex;
  }
  if (!(q > qs)) return 0;
  const real dq = (fast)(q - qs) / ((fast)1 + (fast)P.Lv / (fast)pib * dqs);
  q -= dq;
  th += P.Lv * dq / pib;
  return dq;
}

// A middle layer's capacity with its theta spread between its edges as edge_shape has it (decision #91), not uniform: its base
// is near the lower edge's theta, so far colder than the layer's mean. One Newton step, the whole spread shifting with the mean.
__device__ real condense_spread(real a, real b, real& th, real& q, real pib, real pt, real pb, const Par& P) {
  const real x[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  const real wq[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  const real s = edge_shape(a, b, th);
  fast qs = 0, dqs = 0;
  for (int j = 0; j < 4; j++) {
    const real p = (real)0.5 * (pt + pb) + (real)0.5 * (pb - pt) * x[j], m = (pb - p) / (pb - pt);
    const real tn = a + (b - a) * SPREAD(m, s, ES(s));
    const fast ex = FPOW(p / P.p0, P.kappa), T = (fast)tn * ex;
    if (T < 160) continue;
    const fast v = qsat(T, (fast)p);
    qs += (fast)0.5 * (fast)wq[j] * v;
    dqs += (fast)0.5 * (fast)wq[j] * v * (fast)P.Lv / ((fast)P.Rv * T * T) * ex;
  }
  if (!(q > qs)) return 0;
  const real dq = (fast)(q - qs) / ((fast)1 + (fast)P.Lv / (fast)pib * dqs);
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
    if (P.sat_prof > 0 && k > 0 && k < P.Lm1 && th[k] < tstar(k - 1, P.Lm1 + 1, P))
      return condense_spread(tstar(k, P.Lm1 + 1, P), tstar(k - 1, P.Lm1 + 1, P), th[k], q[k], pib[k], pt, pt + P.grav * mu[k], P);
    // the bottom layer is mixed, its vapor even through its depth: clouds at its top aren't rain; it rains by itself only when saturated at its base, and plumes take the rest (decision #93)
    if (P.sat_base > 0 && k == 2) { const real pb = pt + P.grav * mu[k]; return condense(th[k], q[k], P.cp * FPOW(pb / P.p0, P.kappa), pb, P); }
    return condense_layer(th[k], q[k], pib[k], pt, pt + P.grav * mu[k], P);
  }
  return condense(th[k], q[k], pib[k], p_of(pib[k], P), P);
}

// ---- clouds (decision #95): water that has condensed and not yet rained, cw (kg/kg) per layer ----
// A layer's vapor capacity qs and its change with theta, dqs (as condense_layer and condense_spread have them).
__device__ void layer_cap(int k, const real* mu, real th, const Par& P, real& qs, real& dqs, const real* gx = 0) {   // gx: node_exner's, if the masses are unchanged
  const real x[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  const real wq[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  real pt = 0;
  for (int j = 0; j < k; j++) pt += P.grav * mu[j];
  const real pb = pt + P.grav * mu[k];
  const bool spread = P.sat_prof > 0 && k > 0 && k < P.Lm1 && th < tstar(k - 1, P.Lm1 + 1, P);
  const real ea = spread ? tstar(k, P.Lm1 + 1, P) : (real)0, eb = spread ? tstar(k - 1, P.Lm1 + 1, P) : (real)0;
  const real s = spread ? edge_shape(ea, eb, th) : (real)0, es = spread ? ES(s) : (real)0;
  fast qf = 0, dqf = 0;
  for (int j = 0; j < 4; j++) {
    const real p = (real)0.5 * (pt + pb) + (real)0.5 * (pb - pt) * x[j], m = (pb - p) / (pb - pt);
    const real tn = !spread ? th : ea + (eb - ea) * SPREAD(m, s, es);
    const fast ex = gx ? (fast)gx[4 * k + j] : (fast)FPOW(p / P.p0, P.kappa), T = (fast)tn * ex;
    if (T < 160) continue;
    const fast v = qsat(T, (fast)p);
    qf += (fast)0.5 * (fast)wq[j] * v;
    dqf += (fast)0.5 * (fast)wq[j] * v * (fast)P.Lv / ((fast)P.Rv * T * T) * ex;
  }
  qs = qf; dqs = dqf;
}

// The cell's water spread evenly over +-b about its mean qt, b = (1 - rh_crit) qs (Sundqvist-style statistics): its
// saturated share f and the mean condensate l that share holds.
__device__ void cloud_pdf(real qt, real qs, const Par& P, real& l, real& f) {
  const real b = (1 - P.rh_crit) * qs, s = qt - qs;
  if (s <= -b) { l = 0; f = 0; }
  else if (s >= b) { l = s; f = 1; }
  else { f = (s + b) / (2 * b); l = (s + b) * f / 2; }
}

// The layer's cloud cover: the statistics', or its water spread in sheets at a cloud's typical water content, by phase (liquid q_auto, ice l_ice).
__device__ real cover_of(real cw, real mu, real liq, const Par& P) {
  return fmin((real)1, cw / (liq * P.q_auto + (1 - liq) * P.l_ice) * fmax((real)1, mu / P.mu_cloud));
}
__device__ real cloud_cover(int k, const real* mu, const real* th, const real* q, const real* cw, real liq, const Par& P, const real* gx = 0) {
  real qs, dqs, l, f;
  layer_cap(k, mu, th[k], P, qs, dqs, gx);
  cloud_pdf(q[k] + cw[k], qs, P, l, f);
  return fmax(f, cover_of(cw[k], mu[k], liq, P));
}

// Condense up to the statistics' condensate, with the latent heat's feedback on the capacity; above it, cloud erodes over tau_erode.
__device__ real cloud_adjust(int k, const real* mu, real* th, real* q, real* cw, const real* pib, real dt, const Par& P, const real* gx = 0) {
  real qs, dqs, l, f;
  layer_cap(k, mu, th[k], P, qs, dqs, gx);
  if (!(qs > 0)) return 0;
  cloud_pdf(q[k] + cw[k], qs, P, l, f);
  const real dl = l > cw[k] ? (l - cw[k]) / (1 + P.Lv / pib[k] * dqs * f) : -(cw[k] - l) * OMEXP(dt / P.tau_erode);
  cw[k] += dl; q[k] -= dl;
  th[k] += P.Lv * dl / pib[k];
  return dl;
}

// A parcel (theta, q) lifted to pressure p (Pa): its theta and vapor there after condensing what it can't hold.
__device__ real parcel(real th, real q, real p, const Par& P, real& thp, real& qp) {  // th, q by value: thp, qp may be the same air
  thp = th; qp = q;
  const real ex = FPOW(p / P.p0, P.kappa);
  condense(thp, qp, P.cp * ex, p, P);
  return ex;                                     // (p / p0)^kappa, for the caller
}

__device__ real column_energy(int L, const real* mu, const real* pib, const real* th, const real* q, const Par& P) {
  real E = 0;
  for (int k = 0; k < L; k++) E += mu[k] * pib[k] * th[k] + P.Lv * mu[k] * q[k];
  return E;
}


// Re-divide the column after moving mass m (kg/m^2, > 0) from layer `from` to layer `to`. The air leaves with theta
// th_leave and vapor q_leave per kilogram and arrives with th_arrive and q_arrive (different when it condensed on the
// way). The column's energy is the same before and after up to rounding; that residual is booked into the receiving
// layer.
__device__ void move_mass(int L, const Par& P, real* mu, real* th, real* q, real* pib, int from, int to, real m,
                          real th_leave, real th_arrive, real q_leave, real q_arrive, real* cl = 0) {
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
  if (cl) cl[to] = (mu[to] * cl[to] + m * cl[from]) / mu2[to];
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
  const real W = b - a, s = edge_shape(a, b, m);
  if (fabs(s) < (real)1e-6) { fa = (1 - s / 2) / W; fb = (1 + s / 2) / W; return; }
#if defined(MIXED)
  if (s > 30) { fa = 0; fb = s / W; return; }                    // (e^-s left out, as in edge_shape)
  if (s < -30) { fa = -s / W; fb = 0; return; }
#endif
  const real e1 = expm1(s);
  fa = s / e1 / W;
  fb = s * (e1 + 1) / e1 / W;
}

// The plume's full ascent (decision #88): bottom air (th0, q0) lifted from the bottom layer's top (pressure pi_) up through the
// layer above, condensing and raining as it goes, against that layer's air as edge_density spreads it (the theta at mass
// share x above its lower edge a is a + (b - a) ln(1 + x (e^s - 1)) / s). A: the buoyancy summed by mass up to the highest level
// it beats (K kg/m^2), negative stretches included, the bottom layer's stretch as a triangle up to its buoyancy at the
// interface. kd, thd, qd: the layer it detrains into and its theta and vapor there.
__device__ void plume_ascent(int bot, const real* mu, const real* th, real th0, real q0, real pi_, const Par& P, real& A, int& kd, real& thd, real& qd, const real* q = 0, real* qx = 0, real* lt = 0) {
  const int L = bot + 1, M = bot > 2 ? 3 : 8;
  real tp, qp;
  parcel(th0, q0, pi_, P, tp, qp);
  real sum = (real)0.5 * mu[bot] * (tp - tstar(bot - 1, L, P)), pt = pi_;
  A = 0; kd = -1; thd = tp; qd = qp;
  real cape = 0, q0c = -1, q40 = -1, p0c = 0, p40 = 0;  // lightning (#104): the parcel's buoyant energy, its vapor and pressure at 0 C and -40 C
  real qa[LMAX];                                 // what it has given each layer so far by mixing, per kg of plume
  for (int k = 0; k <= bot; k++) { qa[k] = 0; if (qx) qx[k] = 0; }
  for (int k = bot - 1; k >= 0; k--) {
    const real a = tstar(k, L, P), b = k > 0 ? tstar(k - 1, L, P) : (real)0, s = k > 0 ? edge_shape(a, b, th[k]) : (real)0;
    const real es = k > 0 ? ES(s) : (real)0;
    const real f = q && P.plume_ent > 0 ? OMEXP(P.plume_ent * P.grav * mu[k] / M) : (real)0;   // the mixing per sub-level
    bool last = false;
    for (int j = 0; j < M; j++) {
      const real x = (j + (real)0.5) / M, pl = pt - x * P.grav * mu[k];
      const real ex = parcel(tp, qp, pl, P, tp, qp);
      const real te = k == 0 ? th[0] : a + (b - a) * SPREAD(x, s, es);   // the top layer has no upper edge: its mean
      // it mixes with the air it rises through, an equal mass going each way (decision #103): dry surroundings cool and dry it, so it stops sooner
      if (q && P.plume_ent > 0) {
        qa[k] += f * (qp - q[k]);
        tp += f * (te - tp);
        qp += f * (q[k] - qp);
      }
      sum += mu[k] / M * (tp - te);
      last = tp > te;
      if (lt) {
        const real Tp = tp * ex;
        if (tp > te) cape += 287.0 * (tp - te) * ex * P.grav * mu[k] / M / pl;
        if (q0c < 0 && Tp < P.Tf) { q0c = qp; p0c = pl; }
        if (q40 < 0 && Tp < P.Tf - 40) { q40 = qp; p40 = pl; }
      }
      if (last) { A = sum; kd = k; thd = tp; qd = qp; if (qx) for (int m = 0; m <= bot; m++) qx[m] = qa[m]; }
    }
    if (!last || k == 0) break;
    pt -= P.grav * mu[k];
    parcel(tp, qp, pt, P, tp, qp);
    if (!(tp > b)) break;                       // not through the interface above
  }
  if (lt) {                                     // what freezes between 0 and -40 C (to where it stops, if sooner), that band's depth in pressure, the energy
    if (q0c >= 0 && q40 < 0) { q40 = qd; p40 = pt; }
    lt[0] = q0c >= 0 ? fmax((real)0, q0c - q40) : (real)0;
    lt[1] = q0c >= 0 ? p0c : (real)0;
    lt[2] = q0c >= 0 ? p40 : (real)0;
    lt[3] = cape;
  }
}

// Lightning from one storm's physics (decision #104): Deierling et al. (2008), a storm's flash rate is 9.0e-15 f_p f_np per minute, f_p the falling graupel's
// mass flux (kg m/s, its mass above the melting level times its fall speed) and f_np the rising small ice's (kg/s). The plume's ice flux divides among storms of
// updraft area storm_area, each rising at half its thermodynamic speed sqrt(2 CAPE), and only an updraft faster than graupel falls can hold it up to charge.
__device__ real storm_flashes(real mflux, const real* lt, const Par& P) {
  const real qi = lt[0], w = (real)0.5 * sqrt(2 * fmax((real)0, lt[3]));
  if (!(mflux > 0) || !(qi > 0) || !(w > P.graupel_v)) return 0;
  const real rho = lt[1] / (287.0 * P.Tf), phi = rho * w * P.storm_area, lift = 1 - P.graupel_v * P.graupel_v / (w * w);
  const real H = 287.0 * (P.Tf - 20) / P.grav * FLOG(lt[1] / lt[2]);
  const real up = phi * qi * lift, fnp = up * P.ice_small, fp = up * (1 - P.ice_small) * H / w * P.graupel_v;
  return mflux / phi * (real)9.0e-15 * fp * fnp / 60;       // flashes/s/m^2
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
// dtop, dbot (optional): each layer's heating (as theta) at its top and bottom edge, where the air crosses; the layer's mean
// dth otherwise (decision #111: the k-tables' sub-levels give the edges their own heating)
__device__ void crossing(int L, const Par& P, real* mu, real* th, real* q, const real* dth, real dth_up, real* pib,
                         real* wxo, real* gross = 0, real* cl = 0, const real* dtop = 0, const real* dbot = 0) {
  for (int j = 0; j + 1 < L; j++) wxo[j] = 0;
  if (P.theta_star <= 0) return;                           // theta_star <= 0: no crossing (diagnostics)
  real dmj[LMAX], thcj[LMAX];
  // a layer's edge densities from its theta before the heating, the same at both its edges: solved once per layer (V2-5)
  real efa[LMAX], efb[LMAX];
  bool got[LMAX];
  for (int k = 0; k < L; k++) got[k] = false;
  auto dens = [&](int k, real& fa, real& fb) {
    if (!got[k]) { edge_density(tstar(k, L, P), tstar(k - 1, L, P), th[k] - dth[k], efa[k], efb[k]); got[k] = true; }
    fa = efa[k]; fb = efb[k];
  };
  for (int j = 0; j + 1 < L; j++) {
    const int a = j, b = j + 1;                            // the layers above and below interface j
    const real ts_ = tstar(j, L, P);
    real dm = 0, thc = ts_;
    if (gross) { gross[2 * j] = 0; gross[2 * j + 1] = 0; }
    if (dth[b] > 0 && P.bl_mix > 0 && b == L - 1) {
      // the mixed bottom layer (decision #98): heated past the air above's lowest theta it takes that air in, deepening, and never drains up by heating
      if (th[b] >= ts_) {
        real fa, fb;
        dens(a, fa, fb);
        // (turbulence takes it in over hours at the least, however little of the air above lies near the edge; dth over a day's mass as the cap)
        dm = -fmin(mu[a] * dth[b] * fa, fmax(mu[a] - P.mu_bl, (real)0) * P.bl_cap);
        thc = ts_ - (real)0.5 * dth[b];
      }
    } else if ((dtop ? dtop[b] : dth[b]) > 0) {
      const real eh = dtop ? dtop[b] : dth[b];              // the heating at b's top edge
      if (th[b] >= ts_) { dm = mu[b] - P.floor_; thc = th[b]; }
      else if (b == L - 1) dm = mu[b] * eh / (2 * (ts_ - (th[b] - dth[b])));
      else {                                                // a middle layer's top edge
        real fa, fb;
        dens(b, fa, fb);
        dm = mu[b] * eh * fb;
      }
    }
    const real da = dbot ? dbot[a] : (a == 0 ? dth_up : dth[a]);
    if (da < 0) {
      real down;
      if (th[a] <= ts_) { down = mu[a] - P.floor_; thc = th[a]; }
      else if (a == 0) down = mu[a] * (-da) / (2 * ((th[a] - dth[a]) - ts_));
      else {                                                // a middle layer's bottom edge
        real fa, fb;
        dens(a, fa, fb);
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
    if (dm > 0 && a == 0 && L >= 3 && P.freeze_dry > 0) {
      // freeze-drying (#112): air rising into the stratosphere passes the tropopause's cold point, the interface at theta*, and
      // holds no more vapor than saturation there; the rest freezes and falls back, staying in the layer below with its cloud
      const real pi_ = P.grav * mu[0], Ti = tstar(0, L, P) * FPOW(pi_ / P.p0, P.kappa);
      const real qx = fmin(q[b], Ti < 160 ? (real)0 : (real)qsat((fast)Ti, (fast)pi_));
      const real ma = mu[a], mb = mu[b], ca = cl ? cl[a] : (real)0, cb = cl ? cl[b] : (real)0;
      move_mass(L, P, mu, th, q, pib, b, a, dm, thcj[j], thcj[j], qx, qx, cl);
      if (cl) { cl[a] = ma * ca / mu[a]; cl[b] = mb * cb / mu[b]; }
    } else if (dm > 0) move_mass(L, P, mu, th, q, pib, b, a, dm, thcj[j], thcj[j], q[b], q[b], cl);
    else if (dm < 0) move_mass(L, P, mu, th, q, pib, a, b, -dm, thcj[j], thcj[j], q[a], q[a], cl);
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
// Albedos by surface (decision #87): snow (the part of the land's ice that fell as snow, snow kg/m^2, melting first) from alb_snow
// when cold to alb_snow_melt at freezing air, over alb_ramp K, covering snow / snow_cover of the ice; the rest is bare glacier ice,
// alb_glacier. Sea ice goes from alb_ice to alb_ice_melt in the same way.
extern "C" __global__ void air_column(int N, int L_in, const real* par, int mode, const real* t, const real* insol, const real* pos,
                                      const real* uabs, real* mu_in, real* theta, real* qv, real* Ts, real* ice, real dt,
                                      real* wx, real* olr, real* sw_abs, real* stored, real* rain, real* evap, real* conv,
                                      real* tsurf, real* rainp, real* dgo, real* cap, real* qocean,
                                      real* qsw, const real* land, const real* soilcap, real* soil, real* pond, real* snow, real* cloudw, real* cloudo, const real* elev, real* flash,
                                      const real* radf, const real* swf, real* swalb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
#ifdef NL
  const int L = NL;                              // (L_in is the same)
#else
  const int L = L_in;
#endif
  Par P = unpack(par);
  // the bottom layer's upper edge rises with the ground (decision #96): air over high ground is warmer in theta, so the layer keeps its mass there, bent over the mountain
  if (elev && P.theta_rise > 0 && elev[i] > 0) P.theta_star += P.theta_rise * elev[i];
  const real ts_base = P.theta_star;
  P.Lm1 = L - 1;
  P.bl_cap = OMEXP(dt / 86400.0);
  const bool moist = P.moist > 0;
  const bool cloudy = moist && P.cloud > 0 && cloudw;
  real mu[LMAX], th[LMAX], q[LMAX], pb[LMAX], pib[LMAX], Q[LMAX], T[LMAX], eps[LMAX];
  real mt = 0;
  for (int k = 0; k < L; k++) {
    mu[k] = mu_in[(size_t)k * N + i]; th[k] = theta[(size_t)k * N + i]; q[k] = moist ? qv[(size_t)k * N + i] : (real)0;
    mt += mu[k]; Q[k] = 0;
  }
  real exi[LMAX], gx[4 * LMAX];
  layer_exner(L, mu, P, pb, pib, exi);
  if (cloudy) node_exner(L, mu, P, gx);                          // good until the first mass moves (the crossing)
  // the lowest theta of the air above the mixed bottom layer: no colder than the layer itself, and below that air's mean
  if (P.bl_mix > 0 && L > 2) P.theta_star = fmin(fmax(ts_base, th[L - 1]), th[L - 2] - 1);
  real cl[LMAX], fc[LMAX], cwp[LMAX], eps0[LMAX], liq[LMAX];
  for (int k = 0; k < L; k++) { cl[k] = cloudy ? cloudw[(size_t)k * N + i] : (real)0; fc[k] = 0; }
  real m_ice = moist ? ice[i] : (real)0;
  // the surface water's heat capacity (J/m^2/K): the ocean's mixed layer when coupled (cap), else the slab
  // Land's by its soil water when c_wet is set (decision #90): wet soil stores and conducts heat better. The ground's heat relative to T_REF stays as the water changes.
  if (cap && P.c_wet > 0 && moist && land && land[i] > (real)0.5) {
    const real r = fmin((real)1, soil[i] / soilcap[i]), Cn = P.c_dry + (P.c_wet - P.c_dry) * r;
    Ts[i] = P.Tf + (Ts[i] - P.Tf) * cap[i] / Cn;
    cap[i] = Cn;
  }
  const real Cw = cap ? cap[i] : P.c_slab;
  const real E0 = column_energy(L, mu, pib, th, q, P) + Cw * Ts[i] - P.Lf * m_ice;
  for (int k = 0; k < L; k++) T[k] = pib[k] * th[k] / P.cp;
  const int top = 0, bot = L - 1;
  // each layer's cover, in-cloud water path (kg/m^2) and liquid share (all ice below -20 C)
  if (cloudy)
    for (int k = 0; k < L; k++) {
      // a middle layer's clouds, and the plumes' outflow, sit high in it: their phase by its upper part
      const real pc = k == bot ? (real)0.5 * (pb[k] + (k ? pb[k - 1] : (real)0)) : (k ? pb[k - 1] : (real)0) + (real)0.25 * (pb[k] - (k ? pb[k - 1] : (real)0));
      liq[k] = fmax((real)0, fmin((real)1, (th[k] * FPOW(pc / P.p0, P.kappa) - P.Tf + 20) / 20));
      fc[k] = cloud_cover(k, mu, th, q, cl, liq[k], P, gx);
      cwp[k] = fc[k] > (real)1e-6 ? mu[k] * cl[k] / fc[k] : (real)0;
    }

  // ---- the air touching the ground: the bottom mu_bl of the column ----
  real wbl[LMAX], thb = 0, qb = 0, mb = 0;
  {
    // the air touching the ground is the mixed bottom layer, as deep as it is down to a share bl_bot of mu_bl; what it takes from the layer above is that layer's coldest air, at its lower edge, not its warm mean (decision #94)
    real need = P.bl_bot > 0 && mu[bot] < P.mu_bl ? fmax(mu[bot], P.bl_bot * P.mu_bl) : P.mu_bl;
    for (int k = bot; k >= 0; k--) { const real m = mu[k] < need ? mu[k] : need; wbl[k] = m; need -= m; thb += m * (P.bl_bot > 0 && k < bot && th[k] > tstar(k, L, P) ? tstar(k, L, P) : th[k]); qb += m * q[k]; mb += m; }
  }
  thb /= mb; qb /= mb;
  const real ps = pb[bot];
  const real Tsa = thb * exi[bot];
  const real rough = moist && land && land[i] > (real)0.5 ? P.land_exch : (real)1;
  // the bulk exchange (decision #102): rho C (|U| + gust) per unit difference, zero but for the gusts in calm air; bulk_c 0 keeps v1's form,
  // whose 10 W/m^2/K at calm thermally damped the bottom layer's storms in about 2 days
  // and the air's stability (Louis 1979, bulk Richardson number over a height bulk_z): warm air over a cold surface hardly mixes, cold air over a warm one mixes more
  real stab = 1;
  if (P.bulk_c > 0 && P.bulk_z > 0) {
    const real Ub = uabs[i] + P.bulk_gust, Ri = P.grav * P.bulk_z * (Tsa - Ts[i]) / (Tsa * Ub * Ub);
    stab = Ri > 0 ? 1 / (1 + 10 * Ri * (1 + 5 * Ri)) : 1 - 10 * Ri / (1 + 10 * sqrt(-Ri));
  }
  real bulk = P.bulk_c > 0 ? ps / (287.0 * Tsa) * P.bulk_c * stab * (uabs[i] + P.bulk_gust) : (real)0;
  // a thin surface layer in series, quasi-steady (a test, decision #102): what the surface gives reaches the bottom layer's air only as turbulence carries it up,
  // at bulk_w (m/s) through a neutral or stable top, three times that when the ground heats the air; so conductances add as 1/g = 1/g_surface + 1/g_top
  if (P.bulk_c > 0 && P.bulk_w > 0) {
    const real gt = ps / (287.0 * Tsa) * P.bulk_w * (Ts[i] > Tsa ? (real)3 : (real)1);
    bulk = bulk * gt / (bulk + gt);
  }
  const real hs = rough * (P.bulk_c > 0 ? P.cp * bulk : P.h_sens * (1 + uabs[i] / P.u0));

  // ---- shortwave; the surface's albedo with the ice cover ----
  real qrad[LMAX], qet[LMAX], qeb[LMAX];                         // k-tables: each layer's radiative heating, and its top and bottom sub-level's (W/m^2)
  for (int k = 0; k < L; k++) { qrad[k] = 0; qet[k] = 0; qeb[k] = 0; }
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
  const real warm = fmax((real)0, fmin((real)1, (Tsa - P.Tf + P.alb_ramp) / P.alb_ramp));
  real alb_i = P.alb_ice + (P.alb_ice_melt - P.alb_ice) * warm;
  if (isl) {
    const real fs = fmin((real)1, snow[i] / P.snow_cover);
    alb_i = fs * (P.alb_snow + (P.alb_snow_melt - P.alb_snow) * warm) + (1 - fs) * P.alb_glacier;
  }
  const real alb = fice * alb_i + (1 - fice) * alb0;
  if (swalb) swalb[i] = alb;                                     // for the next radiation step (k-tables)
  // the k-tables' shortwave (sw_column at the last radiation step): each layer's and the surface's share of the sunlight
  const real toAir = swf ? [&]() { real s = 0; for (int k = 0; k < L; k++) s += swf[(size_t)k * N + i]; return s * I; }()
                         : P.asw * (mt / P.mu_ref) * I;
  if (swf)
    for (int k = 0; k < L; k++) {
      Q[k] += I * swf[(size_t)k * N + i];
      qrad[k] += I * swf[(size_t)k * N + i];
      qet[k] += I * swf[(size_t)(L + 2 + 2 * k) * N + i];
      qeb[k] += I * swf[(size_t)(L + 3 + 2 * k) * N + i];
    }
  else if (P.sw_tau > 1) {                  // split like the vapor part (s^4) of the longwave profile (decision #59)
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
  // clouds reflect: each layer's cloud by two-stream conservative scattering of its optical depth 3 W / (2 rho r), layers at random
  // overlap, with the reflections between them and the ground
  real Tc = 1;
  if (cloudy)
    for (int k = 0; k < L; k++) {
      const real tau = (real)1.5 * cwp[k] * (liq[k] / (1000 * P.re_liq) + (1 - liq[k]) / (P.rho_ice * P.re_ice)), g1 = (1 - P.cloud_g) * tau;
      Tc *= 1 - fc[k] * g1 / (2 + g1);
    }
  const real swS = swf ? I * swf[(size_t)L * N + i] : (I - toAir) * Tc * (1 - alb) / (1 - alb * (1 - Tc));

  // ---- longwave: each layer's emissivity from its share of the optical-depth profile; the downward longwave reaching
  // the surface ----
  if (moist && P.lw_w0 > 0) {
    // the longwave's vapor part from the column's own vapor (decision #89): its optical depth grows as the square root of the vapor
    // path (strong lines), reference path lw_w0, shared among the layers by their vapor; the well-mixed part stays by pressure
    real W = 0;
    for (int k = 0; k < L; k++) W += mu[k] * (q[k] > 0 ? q[k] : (real)0);
    const real tv = P.tau0 * (1 - P.tau_lin) * sqrt(W / P.lw_w0);
    for (int k = 0; k < L; k++) {
      const real sb = pb[k] / pb[bot], st = k ? pb[k - 1] / pb[bot] : (real)0;
      eps[k] = OMEXP((P.tau0 * P.tau_lin * (sb - st) + (W > 0 ? tv * mu[k] * (q[k] > 0 ? q[k] : (real)0) / W : (real)0)));
    }
  } else
  for (int k = 0; k < L; k++) eps[k] = OMEXP((tau_p(pb[k], pb[bot], P) - tau_p(k ? pb[k - 1] : (real)0, pb[bot], P)));
  // clouds absorb and emit by their water path, on top of the gas
  if (cloudy)
    for (int k = 0; k < L; k++) {
      eps0[k] = eps[k];
      const real kl = liq[k] * P.k_lw_liq + (1 - liq[k]) * P.k_lw_ice;
      eps[k] = 1 - (1 - eps[k]) * (1 - fc[k] * OMEXP(kl * cwp[k]));
    }
  real ldown = 0;
  // the k-tables' longwave (V2-6a, lw_column at the last radiation step): rows up, dn, tr at the L + 1 layer edges from the
  // top, the clear sky's top-of-air flux and transmission; the air's emission as saved, the surface's spread by tr
  const int RW = L + 1;
  if (radf) ldown = radf[(size_t)(2 * RW - 1) * N + i];
  else
  for (int k = 0; k < L; k++) {
    real dn = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int j = k + 1; j < L; j++) dn -= eps[j] * dn;
    ldown += dn;
  }

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
  if (radf) {
    const real Es = up;
    #define RF(r, k) radf[(size_t)((r) * RW + (k)) * N + i]
    for (int k = 0; k < L; k++) {                                // the layer's net longwave: edges' net fluxes, and its share of the surface's
      const real ql = (RF(0, k + 1) - RF(1, k + 1)) - (RF(0, k) - RF(1, k)) + (RF(2, k + 1) - RF(2, k)) * Es;
      Q[k] += ql;
      // its top and bottom sub-levels' (the levels inside: rows 3 RW + 3 + 6k + {up, dn, tr} below the top sub-level, + 3 above the bottom one)
      #define RE(r) radf[(size_t)(3 * RW + 3 + 6 * k + (r)) * N + i]
      const real n1 = RE(0) - RE(1) + RE(2) * Es, n2 = RE(3) - RE(4) + RE(5) * Es;
      qrad[k] += ql;
      qet[k] += n1 - ((RF(0, k) - RF(1, k)) + RF(2, k) * Es);
      qeb[k] += ((RF(0, k + 1) - RF(1, k + 1)) + RF(2, k + 1) * Es) - n2;
      #undef RE
    }
    up = RF(0, 0) + RF(2, 0) * Es;
    if (cloudy && cloudo) {
      real clear = 1;
      for (int k = 0; k < L; k++) { cloudo[(size_t)k * N + i] = fc[k]; clear *= 1 - fc[k]; }
      cloudo[(size_t)L * N + i] = 1 - clear;
      cloudo[(size_t)(L + 1) * N + i] = swf ? I * (swf[(size_t)(L + 1) * N + i] - (1 - (toAir / fmax(I, (real)1e-30) + swf[(size_t)L * N + i])))
                                            : swS - (I - toAir) * (1 - alb);
      cloudo[(size_t)(L + 2) * N + i] = radf[(size_t)(3 * RW) * N + i] + radf[(size_t)(3 * RW + 1) * N + i] * Es - up;
      for (int k = 0; k < L; k++) {                              // in-cloud longwave and shortwave optical depths, for the next radiation step
        cloudo[(size_t)(L + 3 + k) * N + i] = fc[k] > (real)1e-6 ? (liq[k] * P.k_lw_liq + (1 - liq[k]) * P.k_lw_ice) * cwp[k] : (real)0;
        cloudo[(size_t)(2 * L + 3 + k) * N + i] = (real)1.5 * cwp[k] * (liq[k] / (1000 * P.re_liq) + (1 - liq[k]) / (P.rho_ice * P.re_ice));
      }
    }
    #undef RF
  } else {
  for (int k = bot; k >= 0; k--) {
    const real a = eps[k] * up;
    Q[k] += a;
    const real em = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    Q[k] -= 2 * em;
    up = up - a + em;
  }
  olr[i] = up;
  if (cloudy && cloudo) {                                        // cover and the cloud effects at the top (W/m^2), diagnostics
    real up0 = P.eps_s * P.sigma * tsk * tsk * tsk * tsk, clear = 1;
    for (int k = bot; k >= 0; k--) up0 = up0 - eps0[k] * up0 + eps0[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int k = 0; k < L; k++) { cloudo[(size_t)k * N + i] = fc[k]; clear *= 1 - fc[k]; }
    cloudo[(size_t)L * N + i] = 1 - clear;
    cloudo[(size_t)(L + 1) * N + i] = swS - (I - toAir) * (1 - alb);
    cloudo[(size_t)(L + 2) * N + i] = up0 - up;
  }
  for (int k = 0; k < L; k++) {
    real dn = eps[k] * P.sigma * T[k] * T[k] * T[k] * T[k];
    for (int j = k + 1; j < L; j++) { const real a = eps[j] * dn; Q[j] += a; dn -= a; }
  }
  }
  olr[i] = up;

  // ---- sensible heat and evaporation into the bottom air ----
  const real H = hs * (tsk - Tsa);
  for (int k = 0; k < L; k++) Q[k] += H * wbl[k] / mb;
  real E = 0;
  if (moist) {
    // q_shape (#112): the air at the surface holds the bottom layer's vapor at its relative humidity there, as the radiation
    // spreads it (uniform relative humidity through the layer), not the deep layer's mean
    real qa = qb;
    if (P.q_shape > 0) {
      real qsl, dqsl;
      layer_cap(bot, mu, th[bot], P, qsl, dqsl, cloudy ? gx : (real*)0);
      const real qss = qsat(Tsa, ps);
      if (qsl > 0) qa = fmin(qb * qss / qsl, qss);
    }
    const real def = qsat(tsk, ps) - qa;
    E = def > 0 ? rough * (P.bulk_c > 0 ? bulk : P.k_evap * (1 + uabs[i] / P.u0)) * def * (1 - fice) : (real)0;   // kg/m^2/s, open water only
    if (isl) {
      const real full = P.soil_beta * soilcap[i];
      real Es = (1 - pcov) * E, Ep = pcov * E;
      // soil and plants hold their water back: a surface resistance land_rs (s/m) in series with the air's (decision #92); open water has none
      if (P.land_rs > 0 && E > 0) {
        const real rho = ps / (287.0 * Tsa), ra = rho / (rough * (P.bulk_c > 0 ? bulk : P.k_evap * (1 + uabs[i] / P.u0)));
        Es *= ra / (ra + P.land_rs);
      }
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
  if (cloudy) {
    // cloud water forms and erodes; liquid turns to rain (Sundqvist 1989), ice falls out; rain evaporates in the clear air it falls through
    real Fr = 0;
    for (int k = 0; k < L; k++) {
      const real dl = cloud_adjust(k, mu, th, q, cl, pib, dt, P, gx);
      dth[k] += P.Lv * dl / pib[k];
      dgl[k] += P.Lv * mu[k] * dl / dt;
      real qs, dqs, l, f;
      layer_cap(k, mu, th[k], P, qs, dqs, gx);
      cloud_pdf(q[k] + cl[k], qs, P, l, f);
      f = fmax(f, cover_of(cl[k], mu[k], liq[k], P));
      // a cloud is thin next to a thick layer: its water sits in a depth mu_cloud of the layer's mass, so its water content is that much higher
      const real inc = f > (real)1e-6 ? cl[k] / f * fmax((real)1, mu[k] / P.mu_cloud) : (real)0, r = inc / P.q_auto;
      real pt_ = 0;
      for (int j = 0; j < k; j++) pt_ += P.grav * mu[j];
      const real pm = pt_ + (real)0.5 * P.grav * mu[k], Tk = pib[k] * th[k] / P.cp, depth = mu[k] / (pm / (287.0 * Tk));
      const real rate = liq[k] * P.c_auto * OMEXP(r * r) + (1 - liq[k]) * P.v_ice / depth;
      const real loss = cl[k] * OMEXP(rate * dt);
      cl[k] -= loss;
      Fr += mu[k] * loss / dt;
      real ev = 0;
      if (Fr > 0 && qs > q[k]) {
        ev = P.k_revap * (1 - f) * (1 - q[k] / qs) * sqrt(Fr) * mu[k];
        ev = fmin(ev, fmin(Fr, (qs - q[k]) * mu[k] / dt / (1 + P.Lv / pib[k] * dqs)));
        Fr -= ev;
        q[k] += ev * dt / mu[k];
        th[k] -= P.Lv * ev * dt / (mu[k] * pib[k]);
        dth[k] -= P.Lv * ev * dt / (mu[k] * pib[k]);
        dgl[k] -= P.Lv * ev;
      }
      Rk[k == bot ? 1 : 0] += mu[k] * loss - ev * dt;
    }
    R = Fr * dt;
  } else if (moist)
    for (int k = 0; k < L; k++) {
      const real dq = condense_k(k, mu, th, q, pib, P);
      dth[k] += P.Lv * dq / pib[k];
      dgl[k] += P.Lv * mu[k] * dq / dt;
      R += mu[k] * dq;
      Rk[k == bot ? 1 : 0] += mu[k] * dq;
    }

  // ---- dry crossing of the interfaces ----
  real wxo[LMAX];
  if (P.bl_mix > 0 && L > 2) P.theta_star = fmin(fmax(ts_base, th[bot]), th[bot - 1] - 1);
  // the k-tables' sub-levels: the edges' own heating, the layer's mean radiative heating swapped for its edge sub-level's (#111)
  real dtop[LMAX], dbot[LMAX];
  if (radf) {
    real pt_ = 0;
    for (int k = 0; k < L; k++) {
      const real pbk = pt_ + P.grav * mu[k], dp = pbk - pt_;
      const real ft = k == 0 ? (real)0.05 : (real)0.25, fb = k == 0 ? (real)0.5 : (real)0.25;
      const real pit = P.cp * FPOW((pt_ + 0.5 * ft * dp) / P.p0, P.kappa), pibt = P.cp * FPOW((pbk - 0.5 * fb * dp) / P.p0, P.kappa);
      const real mean = qrad[k] * dt / (mu[k] * pib[k]);
      dtop[k] = dth[k] - mean + qet[k] * dt / (ft * mu[k] * pit);
      dbot[k] = dth[k] - mean + qeb[k] * dt / (fb * mu[k] * pibt);
      if (k == L - 1 && P.bl_mix > 0) dtop[k] = dbot[k] = dth[k];    // the mixed bottom layer heats evenly
      pt_ = pbk;
    }
  }
  // (off, #111: with isothermal layers an edge sub-level's temperature is not the edge's, and the middle layer's base, far colder than
  // the air below it, warmed and stopped feeding the bottom layer; kept for a temperature profile continuous across the interfaces)
  // The 380 K interface alone takes the middle layer's top-edge heating (#112): the layer-mean heating there is a cooling, so the
  // stratosphere's night cooling moved air down and the day's warming at the edge never moved it back, draining the top layer
  const bool edges = P.edge_heat > 0;
  real dtop1[LMAX];
  if (radf && !edges && L > 1) {
    for (int k = 0; k < L; k++) dtop1[k] = dth[k];
    dtop1[1] = dtop[1];
  }
  crossing(L, P, mu, th, q, dth, dth[top], pib, wxo, gross, cloudy ? cl : (real*)0,
           radf ? (edges ? dtop : (L > 1 ? dtop1 : (real*)0)) : (real*)0, edges && radf ? dbot : (real*)0);
  // the surface's buoyancy flux sets the layer's depth (decision #98): rising, it entrains air from above at bl_ent of that flux across the jump to it
  // (Tennekes 1973); falling, turbulence dies and the layer gives its top back as residual air, at the whole flux, down to a thin night layer
  if (P.bl_mix > 0 && L > 2) {
    const real Bf = H + (real)0.61 * P.cp * Tsa * E, jump = fmax((real)1, P.theta_star - th[bot]);
    if (Bf > 0) {
      const real m = fmin(P.bl_ent * Bf * dt / (P.cp * jump), mu[bot - 1] - P.floor_);
      if (m > 0) { const real thc = P.theta_star; move_mass(L, P, mu, th, q, pib, bot - 1, bot, m, thc, thc, q[bot - 1], q[bot - 1], cloudy ? cl : (real*)0); wxo[bot - 1] -= m; }
    } else if (mu[bot] > P.mu_night) {
      const real m = fmin(-Bf * dt / (P.cp * jump), mu[bot] - P.mu_night);
      move_mass(L, P, mu, th, q, pib, bot, bot - 1, m, th[bot], th[bot], q[bot], q[bot], cloudy ? cl : (real*)0);
      wxo[bot - 1] += m;
    }
  }

  // ---- moist convection (decisions #46, #49): a plume of bottom-layer air lifted to the interface above it, condensing
  // on the way, compared with the air there, which is at theta*. Buoyant (theta_p > theta*), it rises; it keeps rising
  // through each further interface it is buoyant at and detrains into the layer above the last one, with its latent
  // heat; its condensate rains. Removing bottom air lowers the first interface, where the plume condenses less, so the
  // mass that would remove the buoyancy there is d / (-d(theta_p)/dp) / g; a share 1 - exp(-dt/tau_conv) of it moves per
  // step (a Betts-Miller-style relaxation; the one convective constant). ----
  real mc = 0;
  if (flash) flash[i] = 0;
  if (moist && P.theta_star > 0) {
    real pi_ = 0;
    for (int k = 0; k < bot; k++) pi_ += P.grav * mu[k];     // the bottom layer's top
    const real th0 = th[bot];
    real q0 = q[bot];
    if (cloudy) {                     // plumes root in the cell's moist air: its moistest, by the statistics
      real qs, dqs;
      layer_cap(bot, mu, th0, P, qs, dqs);
      q0 = fmax(q0, fmin(qs, q0 + (1 - P.rh_crit) * qs));
    }
    real thp, qc, thp2, qc2;
    if (P.plume_deep > 0 && bot >= 2) {
      // the air it rises through sinks to make room, warming each level by the theta it had below: over the ascent, A falls by
      // m (theta at detrainment - theta*) per mass m moved, so that mass removes the buoyancy; it relaxes over tau_conv
      real A; int kd;
      real qx[LMAX], xs = 0;
      real lt[4];
      plume_ascent(bot, mu, th, th0, q0, pi_, P, A, kd, thp, qc, q, qx, lt);
      for (int k = 0; k <= bot; k++) xs += qx[k];
      if (kd >= 0 && A > 0) {
        const real rise = thp - tstar(bot - 1, L, P);
        mc = fmin(A / (rise > 1 ? rise : (real)1), mu[bot] - P.floor_) * OMEXP(dt / P.tau_conv);
        if (mc > 0) {
          // the plume keeps l_det of its condensate and leaves it as cloud; the rest rains inside it
          const real cond = fmax((real)0, q0 - qc - xs), det = cloudy ? fmin(cond, P.l_det) : (real)0;
          if (flash) flash[i] = storm_flashes(mc / dt, lt, P);
          R += mc * (cond - det);
          Rc = mc * (cond - det);
          // (what the mixing took or gave in vapor, and any surplus were it to give more than it condensed; its heat falls to the receiving layer in move_mass)
          const real qarr = qc + (q0 - qc - xs - cond);
          for (int k = 0; k < bot; k++) q[k] = fmax((real)0, q[k] + mc * qx[k] / mu[k]);
          move_mass(L, P, mu, th, q, pib, bot, kd, mc, th0, thp, q0, qarr, cloudy ? cl : (real*)0);
          for (int j = kd; j < bot; j++) wxo[j] += mc;
          cin[kd] += mc;
          if (cloudy) {
            cl[kd] += mc * det / mu[kd];
            cloud_adjust(kd, mu, th, q, cl, pib, dt, P);
          } else {
            const real rt = mu[kd] * condense_k(kd, mu, th, q, pib, P);
            R += rt;
            Rc += rt;
          }
        } else mc = 0;
      }
    } else {
    parcel(th0, q0, pi_, P, thp, qc);
    const real d = thp - tstar(bot - 1, L, P);
    if (d > 0) {
      const real dp = 500;
      parcel(th0, q0, pi_ + dp, P, thp2, qc2);
      const real slope = (thp - thp2) / dp;                     // > 0: the buoyancy falls as the interface lowers
      const real maxup = mu[bot] - P.floor_;
      const real want = slope > 0 ? d / slope / P.grav : maxup;
      mc = fmin(want, maxup) * OMEXP(dt / P.tau_conv);
      if (mc > 0) {
        int kd = bot - 1;                                       // the layer it detrains into
        real pj = pi_;
        while (kd > 0) {                                        // buoyant at the next interface up too?
          pj -= P.grav * mu[kd];
          real t2, q2;
          parcel(th0, q0, pj, P, t2, q2);
          if (!(t2 > tstar(kd - 1, L, P))) break;
          thp = t2; qc = q2; kd--;
        }
        R += mc * (q0 - qc);
        Rc = mc * (q0 - qc);
        move_mass(L, P, mu, th, q, pib, bot, kd, mc, th0, thp, q0, qc);
        for (int j = kd; j < bot; j++) wxo[j] += mc;            // it crosses every interface on its way
        cin[kd] += mc;
        // what the arriving air brings beyond the receiving layer's saturation rains at once
        const real rt = mu[kd] * condense_k(kd, mu, th, q, pib, P);
        R += rt;
        Rc += rt;
      } else mc = 0;
    }
    }
  }
  if (conv) conv[i] = mc;
  for (int j = 0; j + 1 < L; j++) wx[(size_t)j * N + i] = wxo[j];
  rain[i] = R / dt;
  if (isl) {
    pw += ice0 - m_ice;                                         // the ice's melt, from the snow first
    real sn = snow[i] - (ice0 - m_ice);
    if (skin) {                                                 // ice covers the ground: the skin decides, and the air takes the freezing's heat (#87)
      if (tsk < P.Tf) {
        // rain and snow freeze as they land; a pond only as fast as the surface can lose the heat, to the cold ice, the cold air and the sky
        // (overnight, in the sun it does not), the rest stays liquid and runs on.
        // The heat goes into the air touching the ground, not into however thin a bottom layer (#87; decision #100)
        const real room = (P.k_ice / dice * (P.Tf - tsk) + hs * fmax((real)0, P.Tf - Tsa) + fmax((real)0, P.eps_s * P.sigma * P.Tf * P.Tf * P.Tf * P.Tf - ldown - swS)) * dt / P.Lf
                         + (sn > 0 ? (real)10.0 / 3600 * dt : (real)0);   // and into snow: meltwater soaks into firn, cold from the winter, at up to 10 mm/h (a stand-in until the firn has a temperature)
        const real mp = fmin(pw > 0 ? pw : (real)0, room), m = R + mp;
        m_ice += m;
        sn = (sn > 0 ? sn : (real)0) + R;
        pw -= mp;
        for (int k = 0; k <= bot; k++) th[k] += P.Lf * m * (wbl[k] / mb) / (mu[k] * pib[k]);
      } else pw += R;
    } else if (Ts[i] < P.Tf) {                                  // snow: its freezing warms the ground
      m_ice += R;
      sn = (sn > 0 ? sn : (real)0) + R;
      Ts[i] += P.Lf * R / Cw;
    } else pw += R;
    if (!skin && Ts[i] < P.Tf && pw > 0) {                      // the pond freezes, warming the ground up to freezing
      const real room = (P.Tf - Ts[i]) * Cw / P.Lf, m = pw < room ? pw : room;
      pw -= m;
      m_ice += m;
      Ts[i] += P.Lf * m / Cw;
    }
    ice[i] = m_ice;
    snow[i] = fmax((real)0, fmin(sn, m_ice));                   // the ice's flow may have taken what it lay on
    if (!skin && Ts[i] >= P.Tf) {                               // soaking in; frozen or ice-covered ground takes none
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
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; if (moist) qv[(size_t)k * N + i] = q[k]; if (cloudy) cloudw[(size_t)k * N + i] = cl[k]; }
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

// ---- longwave by the k-tables (V2-6a, SPEC #109, #110): one block per column, one thread per g-point ----
// Each layer is split into NSUB sub-levels (the top layer's from the top at 0.05, 0.2, 0.5 of its depth, the others even),
// each layer isothermal at its mean temperature (#111); their vapor at the
// layer's mean relative humidity (q scaled to each sub-level's saturation); ozone from a profile in pressure, CO2, N2O,
// CH4 well mixed. Clouds: each layer's cover and water path (the column's own rules), spread through it by mass, mixed
// with the clear sky in every sub-level. Two-stream, one diffusivity angle (1/1.66), source linear in optical depth.
// radf rows (each N wide): up and down from the air's emission at the L + 1 layer edges (the surface emitting nothing),
// tr (the share of the surface's emission reaching each edge), the clear sky's top up flux and transmission.
#define NSUB 4
#define NSMAX (NSUB * LMAX)
#define NCOMP 6
__device__ __forceinline__ fast lw_through(fast I, fast tau, fast Bn, fast Bf) {
  const fast x = tau * (fast)1.66, t = (fast)FEXP(-x);
  const fast lin = x > (fast)1e-3 ? 1 - (1 - t) / x : x * ((fast)0.5 - x / 6);
  return I * t + Bf * (1 - t) + (Bn - Bf) * lin;
}
// The sub-levels' properties for lw_column and sw_column (one thread per layer of a column): per sub-level s of the
// column, sub[(s * NV + v) * N + i] for v = edge temperatures (top, bottom), table positions (log p, T), amounts (H2O,
// CO2, O3, N2O, CH4; molecules / cm^2), H2O's place between its two tables, cloud longwave optical depth and cover, air
// (molecules / cm^2), the haze's share. A layer's cloud sits in one sub-level (CLOUD_SUB: the bottom layer's middle, a
// middle layer's top, where the plumes' outflow is, the top layer's bottom), so its cover mixes once in each stream.
#define NV 14
__device__ __forceinline__ int cloud_sub(int k, int L) { return k == L - 1 ? 1 : (k == 0 ? NSUB - 1 : 1); }
extern "C" __global__ void lw_setup(int N, int L_in, const real* par, int NP, int NT, real lp0, real dlp, real t0, real dtt,
                                    const float* o3, int NO3, real o3lp0, real o3dlp, real co2, real n2o, real ch4,
                                    const real* mu_in, const real* theta, const real* qv, const real* cloudo, real hs, int tmode, float* sub) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, k = blockIdx.y;
  if (i >= N) return;
#ifdef NL
  const int L = NL;
#else
  const int L = L_in;
#endif
  Par P = unpack(par);
  P.Lm1 = L - 1;
  #define W(s, v) sub[((size_t)(s) * NV + (v)) * N + i]
  {
    real mu[LMAX], th[LMAX], q[LMAX], pb[LMAX], pib[LMAX];
    for (int j = 0; j < L; j++) {
      mu[j] = mu_in[(size_t)j * N + i]; th[j] = theta[(size_t)j * N + i];
      q[j] = qv ? fmax(qv[(size_t)j * N + i], (real)0) : (real)0;    // (transport can leave a trace below zero)
    }
    layer_exner(L, mu, P, pb, pib);
    const real pt = k ? pb[k - 1] : (real)0, pbk = pb[k];
    // temperatures, each keeping the layer's mean temperature (#111; the capacity's theta spread, squeezed into a thin middle
    // layer, made an inversion whose warm top cooled the layer away): the top layer isothermal; the others, tmode 2, at the
    // standard lapse rate (T ~ p^n, n = R 6.5 K/km / g = 0.190: on AFGL columns clear-sky OLR and surface down within 4 W/m^2
    // in 3 layers), tmode 1 at uniform theta (n = kappa), tmode 0 isothermal; vapor at uniform relative humidity by uniform theta
    const real Tiso = pib[k] * th[k] / P.cp;
    auto Tad = [&](real p) -> real { return th[k] * FPOW(p / P.p0, P.kappa); };
    const real nT = tmode == 1 ? P.kappa : (real)0.190;
    const real Cn = Tiso * (nT + 1) * (pbk - pt) / (pow(pbk, nT + 1) - pow(pt, nT + 1));
    auto Tof = [&](real p) -> real { return tmode == 0 || (k == 0 && L >= 3) ? Tiso : Cn * FPOW(p, nT); };
    // the layer's cloud: cover and in-cloud longwave optical depth, as the column's last step had them (cloudo)
    const fast fcl = cloudo ? (fast)cloudo[(size_t)k * N + i] : (fast)0, tcl = cloudo ? (fast)cloudo[(size_t)(L + 3 + k) * N + i] : (fast)0;
    const real FT[5] = {0, 0.05, 0.2, 0.5, 1}, FE[5] = {0, 0.25, 0.5, 0.75, 1};
    fast qs[NSUB], qm = 0, uas[NSUB];
    real pe[NSUB + 1];
    for (int j = 0; j <= NSUB; j++) pe[j] = pt + (k == 0 ? FT[j] : FE[j]) * (pbk - pt);
    for (int j = 0; j < NSUB; j++) {
      const int s = k * NSUB + j;
      const real a = pe[j], b = pe[j + 1], pm = a > 0 ? (b - a) / log(b / a) : (real)0.5 * b;
      const real Tm = Tof((real)0.5 * (a + b));
      W(s, 0) = (float)Tof(a); W(s, 1) = (float)Tof(b);
      // (the top layer's vapor well mixed, as the stratosphere's is)
      const real Tq = Tad((real)0.5 * (a + b));
      qs[j] = k == 0 && L >= 3 ? (fast)1 : (Tq < 160 ? (fast)0 : qsat((fast)Tq, (fast)pm));
      qm += qs[j] * (fast)((b - a) / (pbk - pt));
      // table position: log pressure down from the table's top pressure, temperature up from t0
      fast x = (fast)((lp0 - log(pm)) / dlp), y = (fast)((Tm - t0) / dtt);
      x = fminf(fmaxf(x, 0.f), (fast)(NP - 1) - 1e-4f); y = fminf(fmaxf(y, 0.f), (fast)(NT - 1) - 1e-4f);
      W(s, 2) = x; W(s, 3) = y;
      const fast ua = (fast)((b - a) / (P.grav * 28.9647e-3) * 6.02214076e23 / 1e4);    // air molecules / cm^2
      fast yo = (fast)((o3lp0 - log(pm)) / o3dlp);
      yo = fminf(fmaxf(yo, 0.f), (fast)(NO3 - 1) - 1e-4f);
      const int io = (int)yo;
      const fast xo3 = o3[io] + (o3[io + 1] - o3[io]) * (yo - (fast)io);
      W(s, 5) = (float)((fast)co2 * ua); W(s, 6) = (float)(xo3 * ua); W(s, 7) = (float)((fast)n2o * ua); W(s, 8) = (float)((fast)ch4 * ua);
      W(s, 10) = 0.f; W(s, 11) = (float)fcl;
      W(s, 12) = (float)ua;
      W(s, 13) = (float)(FPOW(b / pb[L - 1], (real)8000 / hs) - FPOW(a / pb[L - 1], (real)8000 / hs));
      uas[j] = ua;
    }
    // the layer's cloud sits in the sub-level holding the median of its vapor (condensate forms where the vapor is: with
    // uniform relative humidity, in the layer's warm lower part; #111)
    {
      fast cum = 0;
      int jc = NSUB - 1;
      for (int j = 0; j < NSUB; j++) {
        cum += qm > 0 ? qs[j] * (fast)((pe[j + 1] - pe[j]) / (pbk - pt)) / qm : (fast)0.25;
        if (cum >= (fast)0.5) { jc = j; break; }
      }
      W(k * NSUB + jc, 10) = (float)tcl;
    }
    for (int j = 0; j < NSUB; j++) {
      const int s = k * NSUB + j;
      const fast qj = qm > 0 ? (fast)q[k] * qs[j] / qm : (fast)q[k];
      const fast x = qj / (1 - qj) * (fast)(28.9647 / 18.01528);                          // H2O volume mixing ratio
      W(s, 4) = (float)(x * uas[j]);
      W(s, 9) = (float)(x / (fast)0.04);
    }
  }
  #undef W
}

extern "C" __global__ void lw_column(int N, int L_in, const real* par, int NG, const float* lk, const float* pg,
                                     int NP, int NT, int NTP, real tp0,
                                     const float* sub, const real* Ts, real* radf) {
  const int i = blockIdx.x, g = threadIdx.x;
#ifdef NL
  const int L = NL;
#else
  const int L = L_in;
#endif
  __shared__ fast su[NSMAX][NCOMP - 1], sfh[NSMAX], sfp[NSMAX], sft[NSMAX], sTt[NSMAX], sTb[NSMAX], stc[NSMAX], sfc[NSMAX];
  __shared__ int sip[NSMAX], sit[NSMAX];
  __shared__ float red[3 * (LMAX + 1) + 3 + 6 * LMAX][144];   // per g-point (32-bit, NG <= 144), summed in 64-bit
  __shared__ real sums[3 * (LMAX + 1) + 3 + 6 * LMAX];
  if (g < NSUB * L) {                             // the column's sub-levels, from lw_setup
    const int s = g;
    #define R_(v) sub[((size_t)s * NV + (v)) * N + i]
    sTt[s] = R_(0); sTb[s] = R_(1);
    const fast x = R_(2), y = R_(3);
    sip[s] = (int)x; sfp[s] = x - (fast)sip[s]; sit[s] = (int)y; sft[s] = y - (fast)sit[s];
    for (int c = 0; c < NCOMP - 1; c++) su[s][c] = R_(4 + c);
    sfh[s] = R_(9); stc[s] = R_(10); sfc[s] = R_(11);
    #undef R_
  }
  __syncthreads();
  const int NS = NSUB * L;
  const int RW = L + 1, E0 = 3 * RW + 3, rows = E0 + 6 * L;
  if (g < NG) {
    // the g-point's optical depth per sub-level: bilinear in (log p, T) of each component's log k
    fast tau[NSMAX];
    const size_t cs = (size_t)NP * NT * NG;
    for (int s = 0; s < NS; s++) {
      const size_t b0 = ((size_t)sip[s] * NT + sit[s]) * NG + g;
      const fast fp = sfp[s], ft = sft[s];
      fast lkc[NCOMP];
      for (int c = 0; c < NCOMP; c++) {
        const float* K = lk + c * cs + b0;
        const fast v00 = K[0], v01 = K[NG], v10 = K[(size_t)NT * NG], v11 = K[(size_t)NT * NG + NG];
        lkc[c] = (fast)FEXP(v00 + (v10 - v00) * fp + (v01 - v00) * ft + (v00 - v01 - v10 + v11) * fp * ft);
      }
      tau[s] = fmax((fast)0, su[s][0] * (lkc[0] + (lkc[1] - lkc[0]) * sfh[s]) + su[s][1] * lkc[2] + su[s][2] * lkc[3] +
                             su[s][3] * lkc[4] + su[s][4] * lkc[5]);
    }
    auto Bof = [&](fast T) -> fast {
      fast y = T - (fast)tp0;
      y = fminf(fmaxf(y, 0.f), (fast)(NTP - 1) - 1e-4f);
      const int j = (int)y;
      return pg[(size_t)j * NG + g] + (pg[(size_t)(j + 1) * NG + g] - pg[(size_t)j * NG + g]) * (y - (fast)j);
    };
    const fast PIf = (fast)PI_;
    // down from the top; a layer's cloud fills the same share of every one of its sub-levels (overlapped within the layer,
    // at random between layers), so the clear and the cloudy part pass the layer separately and mix at its lower edge
    fast I = 0;
    red[RW][g] = 0;
    for (int k = 0; k < L; k++) {
      const fast f = sfc[k * NSUB];
      fast Ic = I, Id = I;
      for (int s = k * NSUB; s < (k + 1) * NSUB; s++) {
        const fast Bt = Bof(sTt[s]), Bb = Bof(sTb[s]);
        Ic = lw_through(Ic, tau[s], Bb, Bt);
        if (f > 0) Id = lw_through(Id, tau[s] + stc[s] / (fast)1.66, Bb, Bt);
        // the levels below the layer's top sub-level and above its bottom one (the edges' own heating, for the crossing)
        if (s == k * NSUB) red[E0 + 6 * k + 1][g] = PIf * (f > 0 ? (1 - f) * Ic + f * Id : Ic);
        if (s == (k + 1) * NSUB - 2) red[E0 + 6 * k + 4][g] = PIf * (f > 0 ? (1 - f) * Ic + f * Id : Ic);
      }
      I = f > 0 ? (1 - f) * Ic + f * Id : Ic;
      red[RW + k + 1][g] = PIf * I;
    }
    // up: the air's emission over a surface emitting nothing, the surface's transmission, and both in clear sky
    const fast Bs = Bof((fast)Ts[i]);
    fast Iu = 0, tr = 1, Iuc = 0, trc = 1;
    red[L][g] = 0; red[2 * RW + L][g] = Bs;
    for (int k = L - 1; k >= 0; k--) {
      const fast f = sfc[k * NSUB];
      fast Ic = Iu, Id = Iu, tc = tr, td = tr;
      for (int s = (k + 1) * NSUB - 1; s >= k * NSUB; s--) {
        const fast Bt = Bof(sTt[s]), Bb = Bof(sTb[s]);
        const fast t1 = (fast)FEXP(-tau[s] * (fast)1.66);
        Iuc = lw_through(Iuc, tau[s], Bt, Bb);
        trc *= t1;
        Ic = lw_through(Ic, tau[s], Bt, Bb);
        tc *= t1;
        if (f > 0) {
          Id = lw_through(Id, tau[s] + stc[s] / (fast)1.66, Bt, Bb);
          td *= t1 * (fast)FEXP(-stc[s]);
        }
        const int e = s == (k + 1) * NSUB - 1 ? 3 : (s == k * NSUB + 1 ? 0 : -1);
        if (e >= 0) {
          red[E0 + 6 * k + e][g] = PIf * (f > 0 ? (1 - f) * Ic + f * Id : Ic);
          red[E0 + 6 * k + e + 2][g] = (f > 0 ? (1 - f) * tc + f * td : tc) * Bs;
        }
      }
      Iu = f > 0 ? (1 - f) * Ic + f * Id : Ic;
      tr = f > 0 ? (1 - f) * tc + f * td : tc;
      red[k][g] = PIf * Iu; red[2 * RW + k][g] = tr * Bs;
    }
    red[3 * RW][g] = PIf * Iuc; red[3 * RW + 1][g] = trc * Bs; red[3 * RW + 2][g] = Bs;
  }
  __syncthreads();
  if (g < rows) {                                 // sums over the g-points in a fixed order
    real sum = 0;
    for (int j = 0; j < NG; j++) sum += (real)red[g][j];
    sums[g] = sum;
  }
  __syncthreads();
  if (g < rows) {
    const real Bsum = sums[3 * RW + 2];
    real v = sums[g];
    if ((g >= 2 * RW && g < 3 * RW) || g == 3 * RW + 1 || (g >= E0 && (g - E0) % 3 == 2)) v /= Bsum;   // shares of the surface's emission
    radf[(size_t)g * N + i] = v;
  }
}

// ---- shortwave by the k-tables (V2-6a): one block per column, one thread per g-point, on lw_setup's sub-levels ----
// Each sub-level: the gases' absorption from the tables, Rayleigh scattering (the g-point's cross section times the air),
// the haze (tau550 times the g-point's spectral factor times the sub-level's share; ssa, asymmetry), and in its layer's
// cloud sub-level the cloud (cover f: reflectances and transmittances mixed (1 - f) clear + f cloudy). Delta-Eddington
// in each sub-level, adding over the surface's albedo. The sun at the mean of its cosine over the coming radiation
// interval's daylight. swf rows (each N wide), per unit of sunlight on the horizontal at the top: each layer's absorbed
// share, the surface's, the clear sky's reflected share; the reflected share is what the rest leaves, so the shares
// close exactly.
__device__ void sw_layer(fast tau, fast ssa, fast g, fast mu0, fast& Rd, fast& Td, fast& Rs, fast& Ts, fast& Tdd) {
  const fast f = g * g;
  tau = (1 - ssa * f) * tau;
  ssa = (1 - f) * ssa / (1 - ssa * f);
  g = (g - f) / (1 - f);
  const fast g1 = (7 - ssa * (4 + 3 * g)) / 4, g2 = -(1 - ssa * (4 - 3 * g)) / 4, g3 = (2 - 3 * g * mu0) / 4, g4 = 1 - g3;
  const fast a1 = g1 * g4 + g2 * g3, a2 = g1 * g3 + g2 * g4;
  const fast k = sqrt(fmax((g1 - g2) * (g1 + g2), (fast)1e-12));
  const fast e = (fast)FEXP(-k * tau), e2 = e * e, k2e = 2 * k * e;
  fast rf = 1 / (k + g1 + (k - g1) * e2);
  Rd = g2 * (1 - e2) * rf;
  Td = k2e * rf;
  Tdd = (fast)FEXP(-tau / mu0);
  const fast km = k * mu0;
  fast om = 1 - km * km;
  if (fabs(om) < (fast)1e-3) om = om < 0 ? (fast)-1e-3 : (fast)1e-3;
  rf = ssa * rf / om;
  Rs = rf * ((1 - km) * (a2 + k * g3) - (1 + km) * (a2 - k * g3) * e2 - k2e * (g3 - a2 * mu0) * Tdd);
  Ts = rf * (k2e * (g4 + a1 * mu0) - Tdd * ((1 + km) * (a1 + k * g4) - (1 - km) * (a1 - k * g4) * e2));
  Rs = fmin(fmax(Rs, (fast)0), 1 - Tdd);
  Ts = fmin(fmax(Ts, (fast)0), 1 - Tdd - Rs);
}

extern "C" __global__ void sw_column(int N, int L_in, const real* par, int NG, const float* lk, int NP, int NT, real yscale,
                                     const float* sun, const float* ray, const float* hzf, const float* cssa,
                                     real tau550, real hssa, real hg, real inhom, const float* sub, int mode, const real* t, real interval,
                                     const real* pos, const real* alb, const real* cloudo, real* swf) {
  const int i = blockIdx.x, g = threadIdx.x;
#ifdef NL
  const int L = NL;
#else
  const int L = L_in;
#endif
  __shared__ fast su[NSMAX][7], sfh[NSMAX], sfp[NSMAX], sft[NSMAX], sr[NSMAX], sh[NSMAX], stc[NSMAX], sfc[NSMAX];
  __shared__ int sip[NSMAX], sit[NSMAX];
  __shared__ fast smu;
  __shared__ float red[3 * LMAX + 2][144];
  __shared__ real sums[3 * LMAX + 2];
  const Par P = unpack(par);
  if (g < NSUB * L) {
    const int s = g, k = s / NSUB;
    #define R_(v) sub[((size_t)s * NV + (v)) * N + i]
    const fast x = R_(2), y = fminf((fast)R_(3) * (fast)yscale, (fast)(NT - 1) - 1e-4f);
    sip[s] = (int)x; sfp[s] = x - (fast)sip[s]; sit[s] = (int)y; sft[s] = y - (fast)sit[s];
    for (int c = 0; c < 5; c++) su[s][c] = R_(4 + c);
    su[s][5] = (fast)0.2095 * R_(12);                         // O2
    sfh[s] = R_(9); sr[s] = R_(12); sh[s] = R_(13);
    const bool cs = R_(10) > 0 && cloudo;                    // the layer's cloud sub-level (lw_setup's)
    sfc[s] = cs ? (fast)cloudo[(size_t)k * N + i] : (fast)0;
    stc[s] = cs ? (fast)(inhom * cloudo[(size_t)(2 * L + 3 + k) * N + i]) : (fast)0;   // uneven within the cell (#111)
    #undef R_
  }
  if (g == 0) {                                   // the sun: the mean of its cosine over the interval's daylight
    fast m = 0.5;
    if (mode == 0) {
      fast sum = 0; int lit = 0;
      for (int j = 0; j < 4; j++) {
        const real tt = t[0] + (j + (real)0.5) * interval / 4;
        const real ang = 2 * PI_ * tt / P.year_s, dec = asin(sin(P.tilt) * sin(ang)), rot = 2 * PI_ * fmod(tt, P.day_s) / P.day_s;
        const real cz = pos[3 * i] * cos(dec) * cos(-rot) + pos[3 * i + 1] * cos(dec) * sin(-rot) + pos[3 * i + 2] * sin(dec);
        if (cz > 0) { sum += (fast)cz; lit++; }
      }
      m = lit ? sum / lit : (fast)0.05;
    }
    smu = fmax(m, (fast)0.02);
  }
  __syncthreads();
  const int NS = NSUB * L;
  if (g < NG) {
    const fast mu0 = smu, a = (fast)alb[i];
    const size_t cs = (size_t)NP * NT * NG;
    fast Rd[NSMAX], Td[NSMAX], Rs[NSMAX], Ts[NSMAX], Tdd[NSMAX], cRd[NSMAX], cTd[NSMAX], cRs[NSMAX], cTs[NSMAX], cTdd[NSMAX];
    for (int s = 0; s < NS; s++) {
      const size_t b0 = ((size_t)sip[s] * NT + sit[s]) * NG + g;
      const fast fp = sfp[s], ft = sft[s];
      fast lkc[7];
      for (int c = 0; c < 7; c++) {
        const float* K = lk + c * cs + b0;
        const fast v00 = K[0], v01 = K[NG], v10 = K[(size_t)NT * NG], v11 = K[(size_t)NT * NG + NG];
        lkc[c] = (fast)FEXP(v00 + (v10 - v00) * fp + (v01 - v00) * ft + (v00 - v01 - v10 + v11) * fp * ft);
      }
      const fast tg = fmax((fast)0, su[s][0] * (lkc[0] + (lkc[1] - lkc[0]) * sfh[s]) + su[s][1] * lkc[2] + su[s][2] * lkc[3] +
                                    su[s][3] * lkc[4] + su[s][4] * lkc[5] + su[s][5] * lkc[6]);
      const fast tr = ray[g] * sr[s], th = (fast)tau550 * hzf[g] * sh[s];
      const fast tt = tg + tr + th, sc = tr + (fast)hssa * th;
      const fast ssa = tt > 0 ? fmin(sc / tt, (fast)(1 - 1e-6)) : (fast)0, ga = sc > 0 ? (fast)hg * (fast)hssa * th / sc : (fast)0;
      sw_layer(tt, ssa, ga, mu0, cRd[s], cTd[s], cRs[s], cTs[s], cTdd[s]);
      const fast f = sfc[s];
      if (f > 0 && stc[s] > 0) {                  // the cloudy part: the cloud's scattering added
        const fast tc = stc[s], cw = cssa[g], sc2 = sc + cw * tc, t2 = tt + tc;
        const fast g2 = (ga * sc + (fast)P.cloud_g * cw * tc) / sc2;
        fast r1, t1, r2, t3, d1;
        sw_layer(t2, fmin(sc2 / t2, (fast)(1 - 1e-6)), g2, mu0, r1, t1, r2, t3, d1);
        Rd[s] = (1 - f) * cRd[s] + f * r1; Td[s] = (1 - f) * cTd[s] + f * t1;
        Rs[s] = (1 - f) * cRs[s] + f * r2; Ts[s] = (1 - f) * cTs[s] + f * t3; Tdd[s] = (1 - f) * cTdd[s] + f * d1;
      } else { Rd[s] = cRd[s]; Td[s] = cTd[s]; Rs[s] = cRs[s]; Ts[s] = cTs[s]; Tdd[s] = cTdd[s]; }
    }
    // adding: the albedos of what lies below each level, then the fluxes down from the top
    fast Ad[NSMAX + 1], As[NSMAX + 1], etop[LMAX], ebot[LMAX];
    auto add = [&](const fast* Rd, const fast* Td, const fast* Rs, const fast* Ts, const fast* Tdd, fast* absk, fast& surf) -> fast {
      Ad[NS] = a; As[NS] = a;
      for (int s = NS - 1; s >= 0; s--) {
        const fast den = 1 / (1 - Ad[s + 1] * Rd[s]);
        Ad[s] = Rd[s] + Td[s] * Td[s] * Ad[s + 1] * den;
        As[s] = Rs[s] + (Tdd[s] * As[s + 1] + Ts[s] * Ad[s + 1]) * Td[s] * den;
      }
      fast Fdir = 1, Fdif = 0, net = 1 - As[0];
      for (int k = 0; k < L; k++) absk[k] = 0;
      for (int s = 0; s < NS; s++) {
        const fast den = 1 / (1 - Ad[s + 1] * Rd[s]);
        const fast Fdn = Fdir * Tdd[s];
        Fdif = (Td[s] * Fdif + Rd[s] * As[s + 1] * Fdn + Ts[s] * Fdir) * den;
        Fdir = Fdn;
        const fast n2 = Fdir + Fdif - (Ad[s + 1] * Fdif + As[s + 1] * Fdir);
        absk[s / NSUB] += net - n2;
        if (s % NSUB == 0) etop[s / NSUB] = net - n2;              // the layers' top and bottom sub-levels (the crossing's edges)
        if (s % NSUB == NSUB - 1) ebot[s / NSUB] = net - n2;
        net = n2;
      }
      surf = net;
      return As[0];
    };
    fast absk[LMAX], surf, cabs[LMAX], csurf, et[LMAX], eb[LMAX];
    add(Rd, Td, Rs, Ts, Tdd, absk, surf);
    for (int k = 0; k < L; k++) { et[k] = etop[k]; eb[k] = ebot[k]; }
    const fast crefl = add(cRd, cTd, cRs, cTs, cTdd, cabs, csurf);
    const fast w = sun[g];
    for (int k = 0; k < L; k++) { red[k][g] = w * absk[k]; red[L + 2 + 2 * k][g] = w * et[k]; red[L + 3 + 2 * k][g] = w * eb[k]; }
    red[L][g] = w * surf;
    red[L + 1][g] = w * crefl;
  }
  __syncthreads();
  if (g < 3 * L + 2) {
    real sum = 0;
    for (int j = 0; j < NG; j++) sum += (real)red[g][j];
    sums[g] = sum;
    swf[(size_t)g * N + i] = sum;
  }
}
