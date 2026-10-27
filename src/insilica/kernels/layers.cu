// Stacked-layer shallow water with a split-explicit fast mode (SPEC section 3, decisions #24-#27).
//
// Layers k = 0 (top) .. L-1 (bottom), Boussinesq, thickness h_k, normal wind u_k. The Montgomery potential of
// layer k is M_k = g eta + M'_k, with eta = b + sum_j h_j the free surface and
//   M'_0 = 0,  M'_k = M'_{k-1} + g'_{k-1} z_k,   z_k = b + sum_{j>=k} h_j (the top of layer k),
// g'_{k-1} the reduced gravity across the interface between layers k-1 and k. The g eta part drives the fast
// external mode, sub-stepped at its true speed; everything else is slow (time-split RK3, Wicker & Skamarock
// 2002, as MPAS-Atmosphere). Thicknesses stay the prognostic truth: eta is diagnosed from them.
//
// Layer-major arrays x[k * n + i]. Kernels that need the whole column loop over layers in one thread
// (LMAX layers at most); the others take the layer from blockIdx.y.

typedef REAL real;
#define LMAX 8

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;
#define LOOP(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return; const int l = blockIdx.y;

// The free surface eta = b + sum_k h_k (summed bottom up), and the column's barotropic wind
// ubar = sum_k h_e,k u_k / H_e on edges (h_e,k the mean of the edge's two cells).
extern "C" __global__ void lay_eta(int N, int L, const real* h, const real* b, real* eta) {
  LOOP1(i, N)
  real z = b[i];
  for (int k = L - 1; k >= 0; k--) z += h[(size_t)k * N + i];
  eta[i] = z;
}

extern "C" __global__ void lay_ubar(int N, int E, int L, const real* h, const real* u, const int* eI, const int* eJ, real* ubar) {
  LOOP1(e, E)
  real H = 0, U = 0;
  for (int k = 0; k < L; k++) {
    const real he = (real)0.5 * (h[(size_t)k * N + eI[e]] + h[(size_t)k * N + eJ[e]]);
    H += he;
    U += he * u[(size_t)k * E + e];
  }
  ubar[e] = U / H;
}

// Slow Bernoulli function B_k = M'_k + K_k at cells (the g eta part is the fast mode's).
extern "C" __global__ void lay_bern(int N, int E, int L, const real* h, const real* u, const real* b, const real* gp,
                                    const int* cellEdge, const real* le, const real* de, const real* areaCell, real* B) {
  LOOP1(i, N)
  real z[LMAX];
  real s = b[i];
  for (int k = L - 1; k >= 0; k--) { s += h[(size_t)k * N + i]; z[k] = s; }
  real Mp = 0;
  for (int k = 0; k < L; k++) {
    if (k > 0) Mp += gp[k - 1] * z[k];
    const real* ul = u + (size_t)k * E;
    real ke = 0;
    for (int m = 0; m < 6; m++) {
      const int e = cellEdge[i * 6 + m];
      if (e < 0) break;
      ke += le[e] * de[e] * (real)0.25 * ul[e] * ul[e];
    }
    B[(size_t)k * N + i] = Mp + ke / areaCell[i];
  }
}

// Air layers (decision #32): h_k is the layer's mass per area (kg/m^2), theta_k its mean potential temperature.
// Hydrostatics from p = 0 at the top: interface pressures p_k (bottom of layer k) = g sum_{j<=k} h_j; Exner
// pi(p) = cp (p/p0)^kappa. Montgomery potential M = theta pi + Phi: M_{L-1} = g b + theta_{L-1} pi(p_s) and
// M_k = M_{k+1} + (theta_k - theta_{k+1}) pi(p_k) (Hsu & Arakawa). The pressure force in a layer is
// -grad M_k + pibar_k grad theta_k (pibar_k the layer's mass-mean Exner function; the second term is in
// lay_slow). The fast substeps take the linear part Phi_k = sum_{m>=k} S_m P_m of every layer's M (air_stiff; P_m the mass
// above interface m), so B_k = M_k - Phi_k + K_k (decision #55).
extern "C" __global__ void air_bern(int N, int E, int L, const real* h, const real* u, const real* theta, const real* b,
                                    real grav, real cp, real kappa, real p0, const real* S, const int* cellEdge, const real* le,
                                    const real* de, const real* areaCell, real* B, real* pibar) {
  LOOP1(i, N)
  // one pow per interface: x_k = (p_k / p0)^kappa; then pi_k = cp x_k and p^(kappa+1) / p0^kappa = p x
  real xb[LMAX];
  real p = 0, mu = 0, xt = 0;
  for (int k = 0; k < L; k++) {
    const real hk = h[(size_t)k * N + i];
    const real pt = p;
    p = p + grav * hk;
    mu += hk;
    const real x = pow(p / p0, kappa);
    xb[k] = x;
    // mass-mean Exner over [pt, p]: cp (p x_p - pt x_pt) / ((kappa+1)(p - pt))
    pibar[(size_t)k * N + i] = cp * (p * x - pt * xt) / ((kappa + 1) * (p - pt));
    xt = x;
  }
  real M = grav * b[i] + theta[(size_t)(L - 1) * N + i] * cp * xb[L - 1];
  real Mk[LMAX];
  Mk[L - 1] = M;
  for (int k = L - 2; k >= 0; k--) {
    M = M + (theta[(size_t)k * N + i] - theta[(size_t)(k + 1) * N + i]) * cp * xb[k];
    Mk[k] = M;
  }
  real P[LMAX], Phi[LMAX];
  real a = 0;
  for (int k = 0; k < L; k++) { a += h[(size_t)k * N + i]; P[k] = a; }
  real ph = 0;
  for (int k = L - 1; k >= 0; k--) { ph += S[(size_t)k * N + i] * P[k]; Phi[k] = ph; }
  for (int k = 0; k < L; k++) {
    const real* ul = u + (size_t)k * E;
    real ke = 0;
    for (int m = 0; m < 6; m++) {
      const int e = cellEdge[i * 6 + m];
      if (e < 0) break;
      ke += le[e] * de[e] * (real)0.25 * ul[e] * ul[e];
    }
    B[(size_t)k * N + i] = Mk[k] - Phi[k] + ke / areaCell[i];
  }
}

// The air column's linear stiffness for the fast substeps (decision #55): d M_k / d mu_j = sum_{m >= max(k, j)} S_m with
// S_m = (theta_m - theta_m+1) g d pi / dp at interface m (the bottom's term theta_L-1 g d pi / dp at the surface), so the
// linear Montgomery potential is Phi_k = sum_{m >= k} S_m P_m, P_m the mass above interface m. A statically unstable
// interface (theta_m < theta_m+1) contributes nothing (its waves would grow; the slow step keeps that part). Computed from
// the step's starting state and held through its stages.
extern "C" __global__ void air_stiff(int N, int L, const real* h, const real* theta, real grav, real cp, real kappa, real p0,
                                     real* S) {
  LOOP1(i, N)
  real p = 0;
  for (int k = 0; k < L; k++) {
    p += grav * h[(size_t)k * N + i];
    const real dth = theta[(size_t)k * N + i] - (k + 1 < L ? theta[(size_t)(k + 1) * N + i] : (real)0);
    const real dpi = kappa * cp * pow(p / p0, kappa) / p;
    S[(size_t)k * N + i] = dth > 0 ? dth * grav * dpi : (real)0;
  }
}

// ---- Air fast substeps (decision #55): every layer's mass and wind, forward-backward, from t^n over the stage. ----
// mu_k <- mu_k - tau div(h_e,k u_k), h_e,k the stage's edge thickness; then u_k <- u_k + tau (-grad Phi*_k + T_k), with
// Phi* from the forward-weighted masses mu + gamma (mu - mu_prev) (divergence damping of every layer, #28) and T_k the
// stage's full slow tendency T'_k + G. acc sums the winds the mass steps used.
extern "C" __global__ void af_mu(int N, int E, const real* hedge, const real* uf, const int* cellEdge, const signed char* cellSign,
                                 const real* le, const real* areaCell, real tau, real* mu, real* mu_prev) {
  LOOP(i, N)
  const real* hl = hedge + (size_t)l * E;
  const real* ul = uf + (size_t)l * E;
  const size_t o = (size_t)l * N + i;
  mu_prev[o] = mu[o];
  real div = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    div += (real)cellSign[i * 6 + m] * le[e] * hl[e] * ul[e];
  }
  mu[o] = mu[o] - tau * div / areaCell[i];
}

extern "C" __global__ void af_phi(int N, int L, const real* mu, const real* mu_prev, real gamma, const real* S, real* phi) {
  LOOP1(i, N)
  real P[LMAX];
  real a = 0;
  for (int k = 0; k < L; k++) {
    const size_t o = (size_t)k * N + i;
    a += mu[o] + gamma * (mu[o] - mu_prev[o]);
    P[k] = a;
  }
  real ph = 0;
  for (int k = L - 1; k >= 0; k--) { ph += S[(size_t)k * N + i] * P[k]; phi[(size_t)k * N + i] = ph; }
}

extern "C" __global__ void af_u(int N, int E, const real* phi, const real* Tp, const real* G, const int* eI, const int* eJ,
                                const real* de, real tau, real* uf, real* acc) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  const real v = uf[o];
  acc[o] += v;
  const real* pl = phi + (size_t)l * N;
  uf[o] = v + tau * (-(pl[eJ[e]] - pl[eI[e]]) / de[e] + Tp[o] + G[e]);
}

// The stage's layer fluxes: the stage's edge thickness times the mean wind of the substeps.
extern "C" __global__ void af_flux(int E, const real* hedge, const real* acc, real inv_n, real* F) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  F[o] = hedge[o] * acc[o] * inv_n;
}

// The fast mode's coefficient per cell for air (decisions #37, #43): the column's external stiffness, the mass-weighted
// change of the layers' Montgomery potential per unit column mass when every layer's mass changes in proportion (the
// external mode), at fixed theta. The bottom gives R T_s / mu (T_s from the lowest layer holding real air, above twice
// the floor); each interface j below layer j adds (theta_j - theta_j+1) R (p_j / p0)^kappa P_j / mu^2, P_j the mass
// above it. With the bottom term alone (#37) the interfaces' share of the external restoring force stayed in the slow
// step, where RK3 cannot follow a wave turning ~3 radians per step: a 2-dt column-mass mode grew at the pentagons once
// the air was warm. With the full stiffness the barotropic slow forcing holds no external stiffness, and what is left
// in the slow step is second order. Computed from the step's starting state and held through its stages.
extern "C" __global__ void air_gfast(int N, int L, const real* h, const real* theta, real grav, real kappa, real p0, real rgas,
                                     real floor_, real* Gc) {
  LOOP1(i, N)
  real mu = 0;
  for (int k = 0; k < L; k++) mu += h[(size_t)k * N + i];
  int kb = L - 1;
  while (kb > 0 && h[(size_t)kb * N + i] < 2 * floor_) kb--;
  const real Ts = theta[(size_t)kb * N + i] * pow(grav * mu / p0, kappa);
  real gc = rgas * Ts / mu;
  real above = 0;
  for (int j = 0; j < L - 1; j++) {
    above += h[(size_t)j * N + i];
    const real dth = theta[(size_t)j * N + i] - theta[(size_t)(j + 1) * N + i];
    if (dth > 0) gc += dth * rgas * pow(grav * above / p0, kappa) * above / (mu * mu);
  }
  Gc[i] = gc;
}

// Per stage: each edge's layer thickness h_e and potential vorticity q_e (the mean of its two vertices), read by
// lay_slow for every neighbour instead of being recomputed from cells and vertices there.
extern "C" __global__ void lay_edge_hq(int N, int E, int NV, const real* h, const real* q, const int* eI, const int* eJ,
                                       const int* ev1, const int* ev2, real* he, real* qe) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  he[o] = (real)0.5 * (h[(size_t)l * N + eI[e]] + h[(size_t)l * N + eJ[e]]);
  qe[o] = (real)0.5 * (q[(size_t)l * NV + ev1[e]] + q[(size_t)l * NV + ev2[e]]);
}

// Slow momentum tendencies T_k = PV flux - grad(B_k) - (r_k + c_k / h_e,k) u_k - nu4 del4 u_k, their thickness-
// weighted column mean G (the barotropic forcing), T'_k = T_k - G, and the stage input's edge thickness H_e and
// barotropic wind. r_k is a drag rate (1/s); c_k a linear stress coefficient (m/s): a stress acting on the layer
// decelerates it in proportion to 1/h, so a layer thinning toward the floor is braked hard (decision #29).
extern "C" __global__ void lay_slow(int N, int E, int NV, int L, const real* hedge, const real* u, const real* qedge, const real* B,
                                    const int* eoeStart, const int* eoe, const real* w, const int* eI, const int* eJ,
                                    const int* ev1, const int* ev2, const real* de, const real* drag, const real* stress,
                                    real floor_, real nu4, const real* lap4, const real* theta, const real* pibar,
                                    real dts, real thin, real* Tp, real* G, real* He, real* ubs) {
  LOOP1(e, E)
  real T[LMAX], hev[LMAX];
  real H = 0, U = 0, GT = 0;
  for (int k = 0; k < L; k++) {
    const real* hl = hedge + (size_t)k * E;
    const real* ul = u + (size_t)k * E;
    const real* ql = qedge + (size_t)k * E;
    const real* Bl = B + (size_t)k * N;
    const real qa = ql[e];
    const real he = hl[e];
    // PV flux. Where a layer thins out (an outcrop front), a neighbour's thickness counts for at most `thin` times
    // this edge's own: otherwise a floor-thin vertex's q = (zeta + f) / h_v times a thick neighbour's flux amplifies
    // the Coriolis force on the thin layer by their thickness ratio (decision #38). Smooth layers never reach the cap.
    const real cap = thin * he;
    real Q = 0;
    for (int m = eoeStart[e]; m < eoeStart[e + 1]; m++) {
      const int e2 = eoe[m];
      real h2 = hl[e2];
      if (h2 > cap) h2 = cap;
      const real F2 = h2 * ul[e2];
      Q += w[m] * F2 * (real)0.5 * (qa + ql[e2]);
    }
    real d = Q - (Bl[eJ[e]] - Bl[eI[e]]) / de[e] - drag[k] * ul[e];
    if (stress[k] != 0) {
      // exact exponential decay over the stage (rate r = c / h): the effective rate (1 - e^(-r dts)) / dts is r for
      // weak friction and never exceeds 1/dts, so a nearly massless layer is stopped, not overshot (decision #36)
      const real r = stress[k] / (he > floor_ ? he : floor_);
      d = d - ((r * dts > (real)1e-8) ? (1 - exp(-r * dts)) / dts : r) * ul[e];
    }
    if (theta) {                       // air: + pibar grad(theta), the pressure force of theta varying in a layer
      const real* tl = theta + (size_t)k * N;
      const real* pl = pibar + (size_t)k * N;
      d = d + (real)0.5 * (pl[eI[e]] + pl[eJ[e]]) * (tl[eJ[e]] - tl[eI[e]]) / de[e];
    }
    if (nu4 != 0) d = d - nu4 * lap4[(size_t)k * E + e];
    T[k] = d;
    hev[k] = he;
    H += he;
    U += he * ul[e];
    GT += he * d;
  }
  const real g = GT / H;
  for (int k = 0; k < L; k++) Tp[(size_t)k * E + e] = T[k] - g;
  G[e] = g;
  He[e] = H;
  ubs[e] = U / H;
}

// Fast mode, forward-backward: eta <- eta - tau div(H_e ubar); then ubar <- ubar + tau (-g grad eta* + G), with
// eta* = eta + gamma (eta - eta_prev) the forward-weighted surface: divergence damping (Skamarock & Klemp 1992;
// MPAS-Atmosphere, WRF; decision #28), which damps the fast divergent waves and leaves rotational flow untouched.
// acc sums the ubar values the eta steps used: their mean is the stage's barotropic transport velocity, so layer
// thicknesses sum to the sub-stepped surface.
extern "C" __global__ void bt_eta(int N, const real* He, const real* ub, const int* cellEdge, const signed char* cellSign,
                                  const real* le, const real* areaCell, real tau, real* eta, real* eta_prev) {
  LOOP1(i, N)
  eta_prev[i] = eta[i];
  real div = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    div += (real)cellSign[i * 6 + m] * le[e] * He[e] * ub[e];
  }
  eta[i] = eta[i] - tau * div / areaCell[i];
}

extern "C" __global__ void bt_u(int E, const real* eta, const real* eta_prev, real gamma, const real* G, const int* eI,
                                const int* eJ, const real* de, const real* Gc, real tau, real* ub, real* acc) {
  LOOP1(e, E)
  const real v = ub[e];
  acc[e] += v;
  const int i = eI[e], j = eJ[e];
  const real ej = Gc[j] * (eta[j] + gamma * (eta[j] - eta_prev[j]));
  const real ei = Gc[i] * (eta[i] + gamma * (eta[i] - eta_prev[i]));
  ub[e] = v + tau * (-(ej - ei) / de[e] + G[e]);
}

// Layer thickness fluxes for the stage: each layer's baroclinic flux plus its share of the barotropic transport,
// F_k = h_e,k (u_k - ubar*) + h_e,k ubar_avg, so sum_k F_k = H_e ubar_avg, the transport the fast mode moved.
extern "C" __global__ void lay_flux(int N, int E, const real* h, const real* u, const real* ubs, const real* acc, real inv_n,
                                    const int* eI, const int* eJ, real* F) {
  LOOP(e, E)
  const real he = (real)0.5 * (h[(size_t)l * N + eI[e]] + h[(size_t)l * N + eJ[e]]);
  F[(size_t)l * E + e] = he * (u[(size_t)l * E + e] - ubs[e]) + he * (acc[e] * inv_n);
}

// Thickness floor (#26): a layer may send out at most what it holds above the floor at the step's start, so no
// layer drops below the floor in any stage. Limiting fluxes adds no mass.
extern "C" __global__ void lay_scale(int N, int E, const real* h0, const real* F, const int* cellEdgeAsc, const int* eI,
                                     const real* le, const real* areaCell, real dts, real floor_, real* scale) {
  LOOP(i, N)
  const real* Fl = F + (size_t)l * E;
  real out = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdgeAsc[i * 6 + m];
    if (e < 0) break;
    const real f = Fl[e] * le[e] * dts;
    if (eI[e] == i) { if (f > 0) out += f; }
    else if (!(f > 0)) out -= f;
  }
  const real avail = h0[(size_t)l * N + i] - floor_;
  const real cap = (avail > 0 ? avail : (real)0) * areaCell[i];
  scale[(size_t)l * N + i] = out > cap ? cap / out : (real)1;
}

extern "C" __global__ void lay_limit(int N, int E, const int* eI, const int* eJ, const real* scale, const real* le, real dt,
                                     real* F, real* Fm) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  real f = F[o];
  f *= f > 0 ? scale[(size_t)l * N + eI[e]] : scale[(size_t)l * N + eJ[e]];
  F[o] = f;
  if (Fm) Fm[o] = f * le[e] * dt;
}

// The stage's new layers: h_k = h0_k - c dt div F_k; u_k = u0_k + c dt T'_k + (ubar_end - ubar0).
extern "C" __global__ void lay_h(int N, int E, const real* h0, const real* F, const int* cellEdge, const signed char* cellSign,
                                 const real* le, const real* areaCell, real cdt, real* hn) {
  LOOP(i, N)
  const real* Fl = F + (size_t)l * E;
  real div = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    div += (real)cellSign[i * 6 + m] * le[e] * Fl[e];
  }
  const size_t o = (size_t)l * N + i;
  hn[o] = h0[o] - cdt * div / areaCell[i];
}

extern "C" __global__ void lay_u(int E, const real* u0, const real* Tp, const real* ub_end, const real* ub0, real cdt, real* un) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  un[o] = u0[o] + cdt * Tp[o] + (ub_end[e] - ub0[e]);
}

// ---- Exchange between layers (#27): the hook convection, entrainment and overturning will use. ----
// wx[k * N + i] (m) is the thickness moved up across the interface below layer k (from layer k+1 into layer k);
// negative moves down. Mass, tracer (upwind: the donor's value) and momentum (upwind, on edges) move together,
// so every total is conserved. The caller keeps each donor above the floor.

extern "C" __global__ void xch_cell(int N, int L, const real* wx, const real* areaCell, real* h, real* M, real* psi) {
  LOOP1(i, N)
  real tr[LMAX], dm[LMAX];
  for (int k = 0; k < L; k++) { tr[k] = 0; dm[k] = 0; }
  for (int k = 0; k + 1 < L; k++) {
    const real v = wx[(size_t)k * N + i];
    const real p = v > 0 ? psi[(size_t)(k + 1) * N + i] : psi[(size_t)k * N + i];
    const real vol = v * areaCell[i];
    dm[k] += vol; dm[k + 1] -= vol;
    tr[k] += vol * p; tr[k + 1] -= vol * p;
  }
  for (int k = 0; k < L; k++) {
    const size_t o = (size_t)k * N + i;
    const real m1 = M[o] + dm[k];
    psi[o] = (M[o] * psi[o] + tr[k]) / m1;
    M[o] = m1;
    h[o] = h[o] + dm[k] / areaCell[i];
  }
}

// Momentum, given the edge thicknesses before (h_e from hold) and the exchange: h_e,k u_k is conserved column-wise.
// Mixing momentum between layers removes kinetic energy; diss, if given, receives it per unit area at the edge
// (0.5 sum_k h_e,k u_k^2 before minus after, the energy measure's own form; J/m^2 for air), for the caller to book as heat.
extern "C" __global__ void xch_edge(int N, int E, int L, const real* wx, const real* hold, const int* eI, const int* eJ, real* u,
                                    real* diss) {
  LOOP1(e, E)
  real mom[LMAX], hn[LMAX];
  real k0 = 0;
  for (int k = 0; k < L; k++) {
    const real he = (real)0.5 * (hold[(size_t)k * N + eI[e]] + hold[(size_t)k * N + eJ[e]]);
    const real uk = u[(size_t)k * E + e];
    mom[k] = he * uk;
    hn[k] = he;
    k0 += he * uk * uk;
  }
  for (int k = 0; k + 1 < L; k++) {
    const real v = (real)0.5 * (wx[(size_t)k * N + eI[e]] + wx[(size_t)k * N + eJ[e]]);
    const real ud = v > 0 ? u[(size_t)(k + 1) * E + e] : u[(size_t)k * E + e];
    mom[k] += v * ud; mom[k + 1] -= v * ud;
    hn[k] += v; hn[k + 1] -= v;
  }
  real k1 = 0;
  for (int k = 0; k < L; k++) {
    const real un = mom[k] / hn[k];
    u[(size_t)k * E + e] = un;
    k1 += hn[k] * un * un;
  }
  if (diss) diss[e] = (real)0.5 * (k0 - k1);
}

// ---- Interfacial stress (decision #39): turbulent momentum exchange between neighbouring layers. ----
// The stress across the interface below layer k is tau = c (u_k - u_{k+1}); it decelerates each layer as tau / h,
// so a thick layer barely feels it while a layer thinning toward its floor is locked to the layer next to it. Over a
// step the pair's difference decays exactly as exp(-c (1/h_k + 1/h_{k+1}) dt) and their thickness-weighted mean
// (the column's momentum) is unchanged. Interfaces are taken from the top down. The kinetic energy removed,
// 0.5 h_k h_{k+1} / (h_k + h_{k+1}) (du0^2 - du1^2) per unit area at the edge, is returned in diss (J/m^2).
extern "C" __global__ void lay_vmix(int N, int E, int L, const real* h, const int* eI, const int* eJ, real c, real dt,
                                    real* u, real* diss) {
  LOOP1(e, E)
  real d = 0;
  for (int k = 0; k + 1 < L; k++) {
    const real ha = (real)0.5 * (h[(size_t)k * N + eI[e]] + h[(size_t)k * N + eJ[e]]);
    const real hb = (real)0.5 * (h[(size_t)(k + 1) * N + eI[e]] + h[(size_t)(k + 1) * N + eJ[e]]);
    real* ua = u + (size_t)k * E + e;
    real* ub = u + (size_t)(k + 1) * E + e;
    const real mean = (ha * *ua + hb * *ub) / (ha + hb);
    const real du0 = *ua - *ub;
    const real du1 = du0 * exp(-c * (1 / ha + 1 / hb) * dt);
    *ua = mean + hb / (ha + hb) * du1;
    *ub = mean - ha / (ha + hb) * du1;
    d += (real)0.5 * ha * hb / (ha + hb) * (du0 * du0 - du1 * du1);
  }
  diss[e] = d;
}

// The kinetic energy the interfacial stress, the wind stress or the crossings' momentum mixing removed, back as heat
// (air; decisions #39, #65, #73): each edge's loss
// l_e d_e diss_e (J) goes half to each of its cells, spread over the column's layers in proportion to their mass, so
// every layer warms by the same amount per kilogram: d theta_k = Q / (mu_col pibar_k).
extern "C" __global__ void vmix_heat(int N, int L, const real* h, const real* diss, const int* cellEdge, const real* le,
                                     const real* de, const real* areaCell, real grav, real cp, real kappa, real p0, real* theta) {
  LOOP1(i, N)
  real q = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    q += (real)0.5 * le[e] * de[e] * diss[e];
  }
  if (q == 0) return;
  q /= areaCell[i];
  real mu = 0;
  for (int k = 0; k < L; k++) mu += h[(size_t)k * N + i];
  real p = 0, xt = 0;
  for (int k = 0; k < L; k++) {
    const real pt = p;
    p = p + grav * h[(size_t)k * N + i];
    const real x = pow(p / p0, kappa);
    const real pib = cp * (p * x - pt * xt) / ((kappa + 1) * (p - pt));
    theta[(size_t)k * N + i] += q / (mu * pib);
    xt = x;
  }
}
