// TRiSK shallow-water operators and FCT tracer transport: a port of v1's makeSolver (one layer) and
// transportStep (src/sim/56-swe.js) as CUDA kernels. REAL is double for the engine (float only for the
// speed probe). Compiled with --fmad=false, and every sum runs in v1's order (gathers walk a cell's or
// vertex's edges in the order v1's scatter loops visit them), so results follow v1's arithmetic.
//
// Layered arrays are stored layer-major: x[l * n + i]. blockIdx.y is the layer; layers are independent
// shallow-water systems here (the vertical coupling arrives in V2-1). Index tables are int32 with a fixed
// width of 6 per cell (pad -1) and 3 per vertex.

typedef REAL real;

#define LOOP(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return; const int l = blockIdx.y;

// F = h_e * u, h_e the mean of the two cells (v1: fr * 0.5 * (h_i + h_j) * u with fr = 1).
extern "C" __global__ void edge_flux(int N, int E, const real* h, const real* u, const int* eI, const int* eJ, real* F) {
  LOOP(e, E)
  const real* hl = h + (size_t)l * N;
  F[(size_t)l * E + e] = (real)0.5 * (hl[eI[e]] + hl[eJ[e]]) * u[(size_t)l * E + e];
}

// Positivity limiter, part 1 (v1 `limit`): each cell's outgoing thickness volume in the stage, in v1's
// ascending edge order, against half the layer; scale[i] <= 1.
extern "C" __global__ void cell_outflow_scale(int N, int E, const real* h, const real* F, const int* cellEdgeAsc,
                                              const int* eI, const real* le, const real* areaCell, real dts, real* scale) {
  LOOP(i, N)
  const real* Fl = F + (size_t)l * E;
  real out = 0;
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real f = Fl[e] * le[e] * dts;
    if (eI[e] == i) { if (f > 0) out += f; }
    else if (!(f > 0)) out -= f;
  }
  const real hv = h[(size_t)l * N + i];
  const real cap = (real)0.5 * (hv > 0 ? hv : (real)0) * areaCell[i];
  scale[(size_t)l * N + i] = out > cap ? cap / out : (real)1;
}

// Part 2: outgoing fluxes scaled by the sender's factor (conservative: the receiver gets the same).
extern "C" __global__ void edge_limit(int N, int E, real* F, const int* eI, const int* eJ, const real* scale) {
  LOOP(e, E)
  real* f = F + (size_t)l * E + e;
  *f *= *f > 0 ? scale[(size_t)l * N + eI[e]] : scale[(size_t)l * N + eJ[e]];
}

// Cell tendencies: dh = -div F, and the Bernoulli function B = g(h + b) + K.
extern "C" __global__ void cell_tend(int N, int E, const real* h, const real* b, const real* F, const real* u,
                                     const int* cellEdge, const signed char* cellSign, const real* le, const real* de,
                                     const real* areaCell, real grav, real* dh, real* B) {
  LOOP(i, N)
  const real* Fl = F + (size_t)l * E;
  const real* ul = u + (size_t)l * E;
  real div = 0, ke = 0;
  for (int k = 0; k < 6; k++) {
    const int e = cellEdge[i * 6 + k];
    if (e < 0) break;
    div += (real)cellSign[i * 6 + k] * le[e] * Fl[e];
    ke += le[e] * de[e] * (real)0.25 * ul[e] * ul[e];
  }
  const size_t o = (size_t)l * N + i;
  dh[o] = (real)0 - div / areaCell[i];
  B[o] = grav * (h[o] + b[i]) + (real)0 + ke / areaCell[i];
}

// Potential vorticity at vertices: q = (zeta + f) / h_v, with h_v the kite-weighted thickness.
extern "C" __global__ void vertex_pv(int N, int E, int NV, const real* h, const real* u, const int* vCell, const signed char* vSlot,
                                     const real* kite, const int* vEdge, const signed char* vEdgeSign, const real* de,
                                     const real* areaV, const real* fV, real* q) {
  LOOP(v, NV)
  const real* hl = h + (size_t)l * N;
  const real* ul = u + (size_t)l * E;
  real hv = 0, z = 0;
  for (int k = 0; k < 3; k++) {
    const int c = vCell[v * 3 + k];
    hv += kite[c * 6 + vSlot[v * 3 + k]] * hl[c] * (real)1;
  }
  for (int k = 0; k < 3; k++) {
    const int e = vEdge[v * 3 + k];
    z += (real)vEdgeSign[v * 3 + k] * de[e] * ul[e];
  }
  q[(size_t)l * NV + v] = (z / areaV[v] + fV[v]) / (hv / areaV[v]);
}

extern "C" __global__ void edge_qe(int NV, int E, const real* q, const int* ev1, const int* ev2, real* qe) {
  LOOP(e, E)
  const real* ql = q + (size_t)l * NV;
  qe[(size_t)l * E + e] = (real)0.5 * (ql[ev1[e]] + ql[ev2[e]]);
}

// Edge tendency: du = sum w F' (q + q')/2 - dB/dn - r u - nu4 del4u (TRiSK, energy-conserving PV flux).
extern "C" __global__ void edge_tend(int N, int E, const real* F, const real* qe, const real* B, const real* u,
                                     const int* eoeStart, const int* eoe, const real* w, const int* eI, const int* eJ,
                                     const real* de, const real* drag, real nu4, const real* lap4, real* du) {
  LOOP(e, E)
  const real* Fl = F + (size_t)l * E;
  const real* ql = qe + (size_t)l * E;
  const real* Bl = B + (size_t)l * N;
  real Q = 0;
  const real qa = ql[e];
  for (int k = eoeStart[e]; k < eoeStart[e + 1]; k++) {
    const int e2 = eoe[k];
    Q += w[k] * Fl[e2] * (real)0.5 * (qa + ql[e2]);
  }
  const size_t o = (size_t)l * E + e;
  real d = Q - (Bl[eJ[e]] - Bl[eI[e]]) / de[e] - drag[l] * u[o];
  if (nu4 != 0) d = d - nu4 * lap4[o];
  du[o] = d;
}

// Vector Laplacian on edges (v1 `lap`): dD/dn - dzeta/dt, from cell divergence and vertex vorticity.
extern "C" __global__ void cell_div(int N, int E, const real* u, const int* cellEdge, const signed char* cellSign,
                                    const real* le, const real* areaCell, real* D) {
  LOOP(i, N)
  const real* ul = u + (size_t)l * E;
  real s = 0;
  for (int k = 0; k < 6; k++) {
    const int e = cellEdge[i * 6 + k];
    if (e < 0) break;
    s += (real)cellSign[i * 6 + k] * le[e] * ul[e];
  }
  D[(size_t)l * N + i] = s / areaCell[i];
}

extern "C" __global__ void vertex_curl(int E, int NV, const real* u, const int* vEdge, const signed char* vEdgeSign,
                                       const real* de, const real* areaV, real* Z) {
  LOOP(v, NV)
  const real* ul = u + (size_t)l * E;
  real z = 0;
  for (int k = 0; k < 3; k++) {
    const int e = vEdge[v * 3 + k];
    z += (real)vEdgeSign[v * 3 + k] * de[e] * ul[e];
  }
  Z[(size_t)l * NV + v] = z / areaV[v];
}

extern "C" __global__ void edge_lap(int N, int E, int NV, const real* D, const real* Z, const int* eI, const int* eJ,
                                    const int* ev1, const int* ev2, const real* de, const real* le, real* out) {
  LOOP(e, E)
  const real* Dl = D + (size_t)l * N;
  const real* Zl = Z + (size_t)l * NV;
  out[(size_t)l * E + e] = (Dl[eJ[e]] - Dl[eI[e]]) / de[e] - (Zl[ev2[e]] - Zl[ev1[e]]) / le[e];
}

// out = x + c*d over n values (RK stages); out may alias x.
extern "C" __global__ void axpy(long long n, const real* x, real c, const real* d, real* out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + c * d[i];
}

// Mass flux volume of the final stage: Fm = F * l_e * dt.
extern "C" __global__ void edge_volume(int E, const real* F, const real* le, real dt, real* Fm) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  Fm[o] = F[o] * le[e] * dt;
}

// ---- Tracer transport (v1 transportStep): one tracer per layer; Fm (mass per step, eI -> eJ), M0, M1. ----

extern "C" __global__ void tr_upwind_flux(int N, int E, const real* Fm, const real* psi, const int* eI, const int* eJ, real* FL) {
  LOOP(e, E)
  const real f = Fm[(size_t)l * E + e];
  const real* p = psi + (size_t)l * N;
  FL[(size_t)l * E + e] = f * (f >= 0 ? p[eI[e]] : p[eJ[e]]);
}

// out = (base*M0 - sum out-fluxes + sum in-fluxes) / div, summed in v1's ascending edge order. With frac and
// Fm given, div is M0 moved by frac*Fm (v1's RK stage); otherwise div = M1.
extern "C" __global__ void tr_cell_update(int N, int E, const real* M0, const real* base, const real* flux, real frac,
                                          const real* Fm, const real* M1, const int* cellEdgeAsc, const int* eI, real* out) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* fl = flux + (size_t)l * E;
  real t = M0[o] * base[o];
  real mx = M0[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real f = Fm ? frac * fl[e] : fl[e];
    if (eI[e] == i) t -= f; else t += f;
    if (Fm) { const real m = frac * Fm[(size_t)l * E + e]; if (eI[e] == i) mx -= m; else mx += m; }
  }
  out[o] = t / (Fm ? mx : M1[o]);
}

// 3rd-order upwind-biased edge flux (Skamarock & Gassmann 2011), stencil weights from d2_stencil (host):
// a and b are the scaled second derivatives at eI and eJ: f (mean - a - b) + |f| (b - a) beta, which with
// beta = 1 uses the upwind cell's curvature (-2a for f >= 0, -2b for f < 0) and with beta = 0 is 4th-order
// centered. v1bias = 1 reproduces v1's form, which takes -2a for both signs (the downwind cell when f < 0;
// SPEC section 9 #20); only test K1 uses it.
extern "C" __global__ void tr_ho_flux(int N, int E, const real* Fm, const real* p, const int* idx, const real* wt,
                                      const int* eI, const int* eJ, int v1bias, real beta, real* out) {
  LOOP(e, E)
  const real* ps = p + (size_t)l * N;
  const real f = Fm[(size_t)l * E + e];
  const int o = e * 8;
  const real a = wt[o] * ps[idx[o]] + wt[o + 1] * ps[idx[o + 1]] + wt[o + 2] * ps[idx[o + 2]] + wt[o + 3] * ps[idx[o + 3]];
  const real b = wt[o + 4] * ps[idx[o + 4]] + wt[o + 5] * ps[idx[o + 5]] + wt[o + 6] * ps[idx[o + 6]] + wt[o + 7] * ps[idx[o + 7]];
  const real up = v1bias ? (f >= 0 ? (b - a) : (a - b)) : (b - a) * beta;
  out[(size_t)l * E + e] = f * ((real)0.5 * (ps[eI[e]] + ps[eJ[e]]) - (a + b)) + fabs(f) * up;
}

extern "C" __global__ void tr_antidiff(int E, const real* H, const real* FL, real* A) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  A[o] = H[o] - FL[o];
}

// Zalesak limiter, cell part: allowed rise and fall against the local bounds, and the ratios R+ and R-.
extern "C" __global__ void tr_fct_ratios(int N, int E, const real* psi, const real* td, const real* M1, const real* A,
                                         const int* nbr, const int* cellEdgeAsc, const int* eI, real* Rp, real* Rm) {
  LOOP(i, N)
  const real* p = psi + (size_t)l * N;
  const real* t = td + (size_t)l * N;
  const real* Al = A + (size_t)l * E;
  real hi = fmax(p[i], t[i]), lo = fmin(p[i], t[i]);
  for (int k = 0; k < 6; k++) {
    const int j = nbr[i * 6 + k];
    if (j < 0) break;
    hi = fmax(hi, fmax(p[j], t[j]));
    lo = fmin(lo, fmin(p[j], t[j]));
  }
  const real m1 = M1[(size_t)l * N + i];
  const real qmax = (hi - t[i]) * m1, qmin = (t[i] - lo) * m1;
  real Pp = 0, Pm = 0;
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real a = Al[e];
    if (eI[e] == i) { if (a > 0) Pm += a; else Pp -= a; }
    else { if (a > 0) Pp += a; else Pm -= a; }
  }
  Rp[(size_t)l * N + i] = Pp > 0 ? fmin((real)1, qmax / Pp) : (real)0;
  Rm[(size_t)l * N + i] = Pm > 0 ? fmin((real)1, qmin / Pm) : (real)0;
}

extern "C" __global__ void tr_fct_edge(int N, int E, const real* A, const real* Rp, const real* Rm, const int* eI, const int* eJ, real* CA) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  const real a = A[o];
  const real* rp = Rp + (size_t)l * N;
  const real* rm = Rm + (size_t)l * N;
  const real c = a > 0 ? fmin(rp[eJ[e]], rm[eI[e]]) : fmin(rp[eI[e]], rm[eJ[e]]);
  CA[o] = c * a;
}

// Final FCT update: psi = (td*M1 - sum c*A out + sum c*A in) / M1.
extern "C" __global__ void tr_fct_apply(int N, int E, const real* td, const real* M1, const real* CA, const int* cellEdgeAsc,
                                        const int* eI, real* psi) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* ca = CA + (size_t)l * E;
  real t = td[o] * M1[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    if (eI[e] == i) t -= ca[e]; else t += ca[e];
  }
  psi[o] = t / M1[o];
}

// M1 = M0 - sum Fm out + sum Fm in (the receiving mass of a transport step), in v1's edge order.
extern "C" __global__ void mass_update(int N, int E, const real* M0, const real* Fm, const int* cellEdgeAsc, const int* eI, real* M1) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* f = Fm + (size_t)l * E;
  real m = M0[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    if (eI[e] == i) m -= f[e]; else m += f[e];
  }
  M1[o] = m;
}

// ---- Deterministic sums: fixed-order tree reduction (no atomics), so totals are bit-identical run to run. ----
// Each block reduces SUM_CHUNK*256 consecutive values: thread t adds x[base + t + 256 k] for k = 0..CHUNK-1,
// then a shared-memory tree halves the 256 partial sums. Optional y: sums x*y.
#define SUM_THREADS 256
#define SUM_CHUNK 8
extern "C" __global__ void block_sum(long long n, const real* x, const real* y, real* out) {
  __shared__ real s[SUM_THREADS];
  const long long base = (long long)blockIdx.x * SUM_THREADS * SUM_CHUNK;
  real acc = 0;
  for (int k = 0; k < SUM_CHUNK; k++) {
    const long long i = base + threadIdx.x + (long long)k * SUM_THREADS;
    if (i < n) acc += y ? x[i] * y[i] : x[i];
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int h = SUM_THREADS / 2; h > 0; h >>= 1) {
    if (threadIdx.x < h) s[threadIdx.x] += s[threadIdx.x + h];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
