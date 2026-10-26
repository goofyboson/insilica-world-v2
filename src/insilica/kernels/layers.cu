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

// Slow momentum tendencies T_k = PV flux - grad(B_k) - (r_k + c_k / h_e,k) u_k - nu4 del4 u_k, their thickness-
// weighted column mean G (the barotropic forcing), T'_k = T_k - G, and the stage input's edge thickness H_e and
// barotropic wind. r_k is a drag rate (1/s); c_k a linear stress coefficient (m/s): a stress acting on the layer
// decelerates it in proportion to 1/h, so a layer thinning toward the floor is braked hard (decision #29).
extern "C" __global__ void lay_slow(int N, int E, int NV, int L, const real* h, const real* u, const real* q, const real* B,
                                    const int* eoeStart, const int* eoe, const real* w, const int* eI, const int* eJ,
                                    const int* ev1, const int* ev2, const real* de, const real* drag, const real* stress,
                                    real floor_, real nu4, const real* lap4, real* Tp, real* G, real* He, real* ubs) {
  LOOP1(e, E)
  real T[LMAX], hev[LMAX];
  real H = 0, U = 0, GT = 0;
  for (int k = 0; k < L; k++) {
    const real* hl = h + (size_t)k * N;
    const real* ul = u + (size_t)k * E;
    const real* ql = q + (size_t)k * NV;
    const real* Bl = B + (size_t)k * N;
    const real qa = (real)0.5 * (ql[ev1[e]] + ql[ev2[e]]);
    real Q = 0;
    for (int m = eoeStart[e]; m < eoeStart[e + 1]; m++) {
      const int e2 = eoe[m];
      const real F2 = (real)0.5 * (hl[eI[e2]] + hl[eJ[e2]]) * ul[e2];
      Q += w[m] * F2 * (real)0.5 * (qa + (real)0.5 * (ql[ev1[e2]] + ql[ev2[e2]]));
    }
    const real he = (real)0.5 * (hl[eI[e]] + hl[eJ[e]]);
    real d = Q - (Bl[eJ[e]] - Bl[eI[e]]) / de[e] - drag[k] * ul[e];
    if (stress[k] != 0) d = d - stress[k] / (he > floor_ ? he : floor_) * ul[e];
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
                                const int* eJ, const real* de, real grav, real tau, real* ub, real* acc) {
  LOOP1(e, E)
  const real v = ub[e];
  acc[e] += v;
  const real ej = eta[eJ[e]] + gamma * (eta[eJ[e]] - eta_prev[eJ[e]]);
  const real ei = eta[eI[e]] + gamma * (eta[eI[e]] - eta_prev[eI[e]]);
  ub[e] = v + tau * (-grav * (ej - ei) / de[e] + G[e]);
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
extern "C" __global__ void xch_edge(int N, int E, int L, const real* wx, const real* hold, const int* eI, const int* eJ, real* u) {
  LOOP1(e, E)
  real mom[LMAX], hn[LMAX];
  for (int k = 0; k < L; k++) {
    const real he = (real)0.5 * (hold[(size_t)k * N + eI[e]] + hold[(size_t)k * N + eJ[e]]);
    mom[k] = he * u[(size_t)k * E + e];
    hn[k] = he;
  }
  for (int k = 0; k + 1 < L; k++) {
    const real v = (real)0.5 * (wx[(size_t)k * N + eI[e]] + wx[(size_t)k * N + eJ[e]]);
    const real ud = v > 0 ? u[(size_t)(k + 1) * E + e] : u[(size_t)k * E + e];
    mom[k] += v * ud; mom[k + 1] -= v * ud;
    hn[k] += v; hn[k + 1] -= v;
  }
  for (int k = 0; k < L; k++) u[(size_t)k * E + e] = mom[k] / hn[k];
}
