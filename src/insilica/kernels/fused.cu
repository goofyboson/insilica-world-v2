// Fused TRiSK shallow-water and FCT transport kernels (SPEC section 9 #18): v1's makeSolver and transportStep
// (src/sim/56-swe.js) doing v1's arithmetic in v1's order (gathers walk a cell's or vertex's edges in the order
// v1's scatter loops visit them; compiled with --fmad=false), with fewer launches and fewer arrays than one
// kernel per v1 loop: the thickness flux is recomputed inside the limiter's cell pass, q at edges inside
// the edge tendency, the RK update is folded into the tendency kernels, and transport recomputes its edge
// fluxes inside the cell passes that sum them.
//
// Layout as in trisk.cu: layer-major arrays x[l * n + i], blockIdx.y = layer, int32 index tables.

typedef REAL real;

#define LOOP(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return; const int l = blockIdx.y;

__device__ __forceinline__ real hflux(const real* hl, const real* ul, const int* eI, const int* eJ, int e) {
  return (real)0.5 * (hl[eI[e]] + hl[eJ[e]]) * ul[e];
}

// Stage pass 1 (cells): the positivity limiter's scale factor, from fluxes computed on the fly.
extern "C" __global__ void swe_scale(int N, int E, const real* h, const real* u, const int* cellEdgeAsc, const int* eI,
                                     const int* eJ, const real* le, const real* areaCell, real dts, real* scale) {
  LOOP(i, N)
  const real* hl = h + (size_t)l * N;
  const real* ul = u + (size_t)l * E;
  real out = 0;
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real f = hflux(hl, ul, eI, eJ, e) * le[e] * dts;
    if (eI[e] == i) { if (f > 0) out += f; }
    else if (!(f > 0)) out -= f;
  }
  const real hv = hl[i];
  const real cap = (real)0.5 * (hv > 0 ? hv : (real)0) * areaCell[i];
  scale[(size_t)l * N + i] = out > cap ? cap / out : (real)1;
}

// Stage pass 2 (edges): the limited thickness flux F; on the last stage also the volume Fm = F l dt.
extern "C" __global__ void swe_flux(int N, int E, const real* h, const real* u, const int* eI, const int* eJ,
                                    const real* scale, const real* le, real dt, real* F, real* Fm) {
  LOOP(e, E)
  const size_t o = (size_t)l * E + e;
  real f = hflux(h + (size_t)l * N, u + (size_t)l * E, eI, eJ, e);
  f *= f > 0 ? scale[(size_t)l * N + eI[e]] : scale[(size_t)l * N + eJ[e]];
  F[o] = f;
  if (Fm) Fm[o] = f * le[e] * dt;
}

// Stage pass 3 (cells): B = g(h + b) + K, and the RK update hn = h0 + c (-div F).
extern "C" __global__ void swe_cell(int N, int E, const real* h, const real* b, const real* F, const real* u,
                                    const int* cellEdge, const signed char* cellSign, const real* le, const real* de,
                                    const real* areaCell, real grav, const real* h0, real c, real* hn, real* B) {
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
  const real dh = (real)0 - div / areaCell[i];
  B[o] = grav * (h[o] + b[i]) + (real)0 + ke / areaCell[i];
  hn[o] = h0[o] + c * dh;
}

// Stage pass 4 (vertices): potential vorticity q = (zeta + f) / h_v.
extern "C" __global__ void swe_vertex(int N, int E, int NV, const real* h, const real* u, const int* vCell, const signed char* vSlot,
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

// Stage pass 5 (edges): the momentum tendency and the RK update un = u0 + c du; q at edges from the vertices.
extern "C" __global__ void swe_edge(int N, int E, int NV, const real* F, const real* q, const real* B, const real* u,
                                    const int* eoeStart, const int* eoe, const real* w, const int* eI, const int* eJ,
                                    const int* ev1, const int* ev2, const real* de, const real* drag, real nu4,
                                    const real* lap4, const real* u0, real c, real* un) {
  LOOP(e, E)
  const real* Fl = F + (size_t)l * E;
  const real* ql = q + (size_t)l * NV;
  const real* Bl = B + (size_t)l * N;
  const real qa = (real)0.5 * (ql[ev1[e]] + ql[ev2[e]]);
  real Q = 0;
  for (int k = eoeStart[e]; k < eoeStart[e + 1]; k++) {
    const int e2 = eoe[k];
    const real qb = (real)0.5 * (ql[ev1[e2]] + ql[ev2[e2]]);
    Q += w[k] * Fl[e2] * (real)0.5 * (qa + qb);
  }
  const size_t o = (size_t)l * E + e;
  real d = Q - (Bl[eJ[e]] - Bl[eI[e]]) / de[e] - drag[l] * u[o];
  if (nu4 != 0) d = d - nu4 * lap4[o];
  un[o] = u0[o] + c * d;
}

// ---- Transport ----

__device__ __forceinline__ real upwind(const real* Fml, const real* p, const int* eI, const int* eJ, int e) {
  const real f = Fml[e];
  return f * (f >= 0 ? p[eI[e]] : p[eJ[e]]);
}

// 3rd-order flux with upwind weight beta (SPEC section 9 #20): f (mean - a - b) + |f| (b - a) beta.
__device__ __forceinline__ real hoflux(const real* Fml, const real* ps, const int* idx, const real* wt, const int* eI,
                                       const int* eJ, int e, real beta) {
  const real f = Fml[e];
  const int o = e * 8;
  const real a = wt[o] * ps[idx[o]] + wt[o + 1] * ps[idx[o + 1]] + wt[o + 2] * ps[idx[o + 2]] + wt[o + 3] * ps[idx[o + 3]];
  const real b = wt[o + 4] * ps[idx[o + 4]] + wt[o + 5] * ps[idx[o + 5]] + wt[o + 6] * ps[idx[o + 6]] + wt[o + 7] * ps[idx[o + 7]];
  return f * ((real)0.5 * (ps[eI[e]] + ps[eJ[e]]) - (a + b)) + fabs(f) * ((b - a) * beta);
}

// The upwind solution td = (M0 psi - upwind fluxes) / M1.
extern "C" __global__ void tr_td(int N, int E, const real* M0, const real* psi, const real* Fm, const real* M1,
                                 const int* cellEdgeAsc, const int* eI, const int* eJ, real* td) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* p = psi + (size_t)l * N;
  const real* Fml = Fm + (size_t)l * E;
  real t = M0[o] * psi[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real f = upwind(Fml, p, eI, eJ, e);
    if (eI[e] == i) t -= f; else t += f;
  }
  td[o] = t / M1[o];
}

// An RK stage of the high-order solution: out = (M0 psi - frac sum  H(src)) / (M0 - frac sum  Fm).
extern "C" __global__ void tr_stage(int N, int E, const real* M0, const real* psi, const real* src, real frac, const real* Fm,
                                    const int* idx, const real* wt, const int* cellEdgeAsc, const int* eI, const int* eJ, real beta,
                                    real* out) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* s = src + (size_t)l * N;
  const real* Fml = Fm + (size_t)l * E;
  real t = M0[o] * psi[o];
  real mx = M0[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real f = frac * hoflux(Fml, s, idx, wt, eI, eJ, e, beta);
    const real m = frac * Fml[e];
    if (eI[e] == i) { t -= f; mx -= m; } else { t += f; mx += m; }
  }
  out[o] = t / mx;
}

// Zalesak ratios, with the antidiffusive fluxes A = H(p2) - upwind(psi) computed on the fly.
extern "C" __global__ void tr_ratios(int N, int E, const real* psi, const real* td, const real* M1, const real* p2, const real* Fm,
                                     const int* idx, const real* wt, const int* nbr, const int* cellEdgeAsc, const int* eI,
                                     const int* eJ, real beta, real* Rp, real* Rm) {
  LOOP(i, N)
  const real* p = psi + (size_t)l * N;
  const real* t = td + (size_t)l * N;
  const real* s = p2 + (size_t)l * N;
  const real* Fml = Fm + (size_t)l * E;
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
    const real a = hoflux(Fml, s, idx, wt, eI, eJ, e, beta) - upwind(Fml, p, eI, eJ, e);
    if (eI[e] == i) { if (a > 0) Pm += a; else Pp -= a; }
    else { if (a > 0) Pp += a; else Pm -= a; }
  }
  Rp[(size_t)l * N + i] = Pp > 0 ? fmin((real)1, qmax / Pp) : (real)0;
  Rm[(size_t)l * N + i] = Pm > 0 ? fmin((real)1, qmin / Pm) : (real)0;
}

// The limited update: psi_out = (td M1 - sum  c A out + sum  c A in) / M1. Writes a separate array (neighbors still
// read psi).
extern "C" __global__ void tr_apply(int N, int E, const real* psi, const real* td, const real* M1, const real* p2, const real* Fm,
                                    const real* Rp, const real* Rm, const int* idx, const real* wt, const int* cellEdgeAsc,
                                    const int* eI, const int* eJ, real beta, real* psi_out) {
  LOOP(i, N)
  const size_t o = (size_t)l * N + i;
  const real* p = psi + (size_t)l * N;
  const real* s = p2 + (size_t)l * N;
  const real* Fml = Fm + (size_t)l * E;
  const real* rp = Rp + (size_t)l * N;
  const real* rm = Rm + (size_t)l * N;
  real t = td[o] * M1[o];
  for (int k = 0; k < 6; k++) {
    const int e = cellEdgeAsc[i * 6 + k];
    if (e < 0) break;
    const real a = hoflux(Fml, s, idx, wt, eI, eJ, e, beta) - upwind(Fml, p, eI, eJ, e);
    const real c = a > 0 ? fmin(rp[eJ[e]], rm[eI[e]]) : fmin(rp[eI[e]], rm[eJ[e]]);
    if (eI[e] == i) t -= c * a; else t += c * a;
  }
  psi_out[o] = t / M1[o];
}
