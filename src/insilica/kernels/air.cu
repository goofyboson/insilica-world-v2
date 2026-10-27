// Column physics of the two air layers over a slab surface (stages V2-2a and V2-2b, decisions #32-#35, #40, #45-#48):
// v1's radiation and surface exchange, heating as potential temperature, air crossing the isentropic interface
// theta* only when heated or cooled across it; with moisture (V2-2b) vapor in both layers, evaporation from open
// water, condensation and rain, moist convection across theta*, and thermodynamic sea ice on the slab.
// One thread per cell; layers k = 0 (top) .. L-1 (bottom), mass mu_k (kg/m^2), specific humidity q_k (kg/kg).
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
  real tilt, year_s, day_s, tw_top, tw_bottom;
  real moist, Lv, Lf, Tf, k_evap, tau_conv, alb_ice, rho_ice, ice_cover, k_ice, Rv;
};

// The parameters arrive as an array in this order (atmosphere.PAR_ORDER).
__device__ Par unpack(const real* a) {
  Par P;
  P.S0 = a[0]; P.asw = a[1]; P.tau0 = a[2]; P.mu_ref = a[3]; P.eps_s = a[4]; P.h_sens = a[5]; P.u0 = a[6]; P.alb = a[7];
  P.c_slab = a[8]; P.mu_bl = a[9]; P.theta_star = a[10]; P.floor_ = a[11]; P.sigma = a[12]; P.grav = a[13]; P.cp = a[14];
  P.kappa = a[15]; P.p0 = a[16]; P.tilt = a[17]; P.year_s = a[18]; P.day_s = a[19]; P.tw_top = a[20]; P.tw_bottom = a[21];
  P.moist = a[22]; P.Lv = a[23]; P.Lf = a[24]; P.Tf = a[25]; P.k_evap = a[26]; P.tau_conv = a[27]; P.alb_ice = a[28];
  P.rho_ice = a[29]; P.ice_cover = a[30]; P.k_ice = a[31]; P.Rv = a[32];
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

// Re-divide the column after moving mass dm (kg/m^2, > 0 up) between the two layers. The air leaves its layer with
// theta th_leave and vapor q_leave per kilogram and arrives with th_arrive and q_arrive (different when it condensed on
// the way). The column's energy is the same before and after up to rounding; that residual is booked into the
// receiving layer.
__device__ void move_mass(int L, const Par& P, real* mu, real* th, real* q, real* pib, real dm,
                          real th_leave, real th_arrive, real q_leave, real q_arrive) {
  if (dm == 0) return;
  const int top = 0, bot = 1;
  const int from = dm > 0 ? bot : top, to = dm > 0 ? top : bot;
  const real m = dm > 0 ? dm : -dm;
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

// Crossing theta* (two layers, decisions #34, #40): a layer's theta spans [lo, hi] evenly about its mean, with theta* at
// the shared edge; the convectively mixed lower layer crosses by its own mean heating, the stratified upper layer by
// the heating of its air at theta* (dth_up). Mass crossing mu |d theta| / (2 |theta* - theta|); a layer wholly beyond
// theta* crosses down to its floor; crossing air carries theta* (or its whole layer's theta) and its layer's vapor.
// Floor air stays on its own side of theta*. Returns the mass moved up (kg/m^2).
__device__ real crossing(int L, const Par& P, real* mu, real* th, real* q, const real* dth, real dth_up, real* pib) {
  if (L != 2 || P.theta_star <= 0) return 0;               // theta_star <= 0: no crossing (diagnostics)
  const int top = 0, bot = 1;
  const real ts_ = P.theta_star;
  real dm = 0, thc = ts_;
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
  const real qd = dm > 0 ? q[bot] : q[top];
  move_mass(L, P, mu, th, q, pib, dm, thc, thc, qd, qd);
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

// mode 0: the sun at time t (diurnal and seasonal, v1's sunVector); mode 1: insol[i], a given mean insolation.
// Outputs per cell: wx (mass moved up across theta*, dry crossing + moist convection), olr, sw_abs (W/m^2), stored (the
// column's energy change, J/m^2), rain and evap (kg/m^2/s), conv (kg/m^2 moved up by moist convection this step).
// qv and ice are read and written only when moist.
extern "C" __global__ void air_column(int N, int L, const real* par, int mode, const real* t, const real* insol, const real* pos,
                                      const real* uabs, real* mu_in, real* theta, real* qv, real* Ts, real* ice, real dt,
                                      real* wx, real* olr, real* sw_abs, real* stored, real* rain, real* evap, real* conv) {
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
  const real E0 = column_energy(L, mu, pib, th, q, P) + P.c_slab * Ts[i] - P.Lf * m_ice;
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
  const real alb = fice * P.alb_ice + (1 - fice) * P.alb;
  const real toAir = P.asw * (mt / P.mu_ref) * I;
  for (int k = 0; k < L; k++) Q[k] += toAir * mu[k] / mt;
  const real swS = (I - toAir) * (1 - alb);

  // ---- longwave optical depths; the downward longwave reaching the surface ----
  for (int k = 0; k < L; k++) {
    const real w = (k == bot) ? P.tw_bottom : P.tw_top;
    eps[k] = 1 - exp(-P.tau0 * w * mu[k] / P.mu_ref);
  }
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
    for (int k = 0; k < L; k++) q[k] += E * dt * wbl[k] / mb / mu[k];
  }
  evap[i] = E;

  // ---- heating as potential temperature ----
  real dth[LMAX];
  for (int k = 0; k < L; k++) { dth[k] = Q[k] * dt / (mu[k] * pib[k]); th[k] += dth[k]; }

  // ---- the slab: open water takes the surface fluxes (evaporation costs it Lv per kg); under ice, the conduction up
  // into the skin. Freezing below Tf turns slab heat into ice; ice melts with slab heat above Tf and from the top
  // with the skin's surplus ----
  real qS;
  if (skin) qS = -cond;
  else qS = swS + ldown - P.eps_s * P.sigma * ts * ts * ts * ts - H - P.Lv * E;
  real Tn = ts + qS * dt / P.c_slab;
  if (moist) {
    m_ice -= melt_top * dt / P.Lf;
    if (m_ice < 0) { Tn += -m_ice * P.Lf / P.c_slab; m_ice = 0; }   // more top melt than ice: the rest warms the slab
    if (Tn < P.Tf) { m_ice += (P.Tf - Tn) * P.c_slab / P.Lf; Tn = P.Tf; }
    else if (m_ice > 0) {
      const real m = (Tn - P.Tf) * P.c_slab / P.Lf;
      if (m <= m_ice) { m_ice -= m; Tn = P.Tf; } else { Tn = P.Tf + (m - m_ice) * P.Lf / P.c_slab; m_ice = 0; }
    }
    ice[i] = m_ice;
  }
  Ts[i] = Tn;
  sw_abs[i] = toAir + swS;

  // ---- condensation: one-step saturation adjustment per layer (v1 SPEC 18 #25); the latent heat warms the layer and
  // counts as heating for the crossing of theta*; the water rains out ----
  real R = 0;
  if (moist)
    for (int k = 0; k < L; k++) {
      const real dq = condense(th[k], q[k], pib[k], p_of(pib[k], P), P);
      dth[k] += P.Lv * dq / pib[k];
      R += mu[k] * dq;
    }

  // ---- dry crossing of theta* ----
  const real dm = crossing(L, P, mu, th, q, dth, dth[top], pib);

  // ---- moist convection (decision #46): a parcel of lower-layer air lifted to the interface, condensing on the way,
  // compared with the air there, which is at theta*. Buoyant (theta_p > theta*), it rises into the upper layer with its
  // latent heat; its condensate rains. Removing lower air lowers the interface, where the parcel condenses less, so
  // the mass that would remove the buoyancy is d / (-d(theta_p)/dp) / g; a share 1 - exp(-dt/tau_conv) of it moves
  // per step (a Betts-Miller-style relaxation; the one convective constant). ----
  real mc = 0;
  if (moist && L == 2) {
    const real pi_ = P.grav * mu[top];
    real thp, qc, thp2, qc2;
    parcel(th[bot], q[bot], pi_, P, thp, qc);
    const real d = thp - P.theta_star;
    if (d > 0 && P.theta_star > 0) {
      const real dp = 500;
      parcel(th[bot], q[bot], pi_ + dp, P, thp2, qc2);
      const real slope = (thp - thp2) / dp;                     // > 0: the buoyancy falls as the interface lowers
      const real maxup = mu[bot] - P.floor_;
      const real want = slope > 0 ? d / slope / P.grav : maxup;
      mc = fmin(want, maxup) * (1 - exp(-dt / P.tau_conv));
      if (mc > 0) {
        R += mc * (q[bot] - qc);
        move_mass(L, P, mu, th, q, pib, mc, th[bot], thp, q[bot], qc);
        // what the arriving air brings beyond the upper layer's saturation rains at once
        R += mu[top] * condense(th[top], q[top], pib[top], p_of(pib[top], P), P);
      } else mc = 0;
    }
  }
  if (conv) conv[i] = mc;
  wx[i] = dm + mc;
  rain[i] = R / dt;

  const real E1 = column_energy(L, mu, pib, th, q, P) + P.c_slab * Ts[i] - P.Lf * m_ice;
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
  wx[i] = crossing(L, P, mu, th, q, dth, dth_up, pib);
  real E1 = 0;
  for (int k = 0; k < L; k++) { theta[(size_t)k * N + i] = th[k]; E1 += mu[k] * pib[k] * th[k]; }
  qnet[i] = Qs;
  stored[i] = E1 - E0;
}
