// The mountains' drag on the air (V2-6b, decision #127): Lott & Miller (1997), the scheme of ECMWF's model and LMDZ's
// (orografi.f90), on the isentropic layers. A cell's sub-grid orography is its heights' standard deviation mu about the cell's
// mean, the slope sigma along its principal axis, the anisotropy gamma and the axis's angle theta from east (Baines & Palmer
// 1990). Over the mountains' height H = 2 mu:
//  - the low-level flow, density and stability are the mass-weighted means between mu and 2 mu;
//  - the flow is blocked below Z_blk, where the integral of N / U_p down from H reaches H_nc (U_p, the wind along the drag's
//    direction): the air below goes around the hills, and each layer there feels the form drag per unit mass
//      Cd max(2 - 1/r, 0) (sigma / 4 mu) sqrt((Z_blk - z) / (z + mu)) (B cos^2 psi + C sin^2 psi) |U| U,
//    r the hills' aspect ratio seen by the wind (cos^2 psi + gamma sin^2 psi) / (gamma cos^2 psi + sin^2 psi), psi the angle
//    between the wind and the axis, B = 1 - 0.18 gamma - 0.04 gamma^2, C = 0.48 gamma + 0.3 gamma^2; taken implicitly, its
//    height profile averaged over each layer's part below Z_blk;
//  - the air above Z_blk flows over and launches gravity waves at H with stress
//      tau = G rho N U_p (sigma |D| / 4 mu) (H - Z_blk)^2,  D = (B - (B - C) sin^2 psi, (B - C) sin psi cos psi);
//    going up, a layer where the wave's Richardson number Ri (1 - a) / (1 + a sqrt(Ri))^2 (a = N dh / U_p, dh its
//    amplitude) falls below Ri_c keeps only the stress that leaves it at Ri_c (Lindzen's saturation); a layer whose U_p is not
//    positive (a critical level) takes all of it; the top layer takes what is left (no momentum leaves through the top). The
//    force on a layer is its stress divergence along the drag's direction, times |U_low| / U_p,low (as LMDZ).
// The reaction is on the ground: the air loses that momentum to the solid Earth; the kinetic energy removed is the air's heat
// (added to diss, booked as the wind stress's).

typedef REAL real;
#define LMAX 8
#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

struct Oro { real grav, p0, kappa, rd, G, cd, hnc, ric, dt, mu_min; };

__device__ __forceinline__ real blk_int(real t, real A) {       // F(t) with F' = sqrt((A - t) / t), 0 <= t <= A
  t = fmin(fmax(t, (real)0), A);
  return sqrt(t * (A - t)) + A * asin(sqrt(t / A));
}

// Per cell: each layer's implicit blocked-drag factor beta (dt x rate) and the waves' force per area (N/m^2; east, north).
// op: the cell's wind (east, north) from its edges' normal winds, 2 coefficients per edge slot (Perot's reconstruction).
extern "C" __global__ void oro_cell(int N, int E, int L, const real* mu, const real* theta, const real* u, const int* cellEdge,
                                   const real* op, const real* sso, const real* par, real* beta, real* F) {
  LOOP1(i, N)
  const Oro P = {par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8], par[9]};
  for (int k = 0; k < L; k++) { beta[(size_t)k * N + i] = 0; F[(size_t)(2 * k) * N + i] = 0; F[(size_t)(2 * k + 1) * N + i] = 0; }
  const real ms = sso[i], sig = sso[N + i], gam = fmax(sso[2 * N + i], (real)1e-7), tho = sso[3 * N + i];
  if (!(ms > P.mu_min) || !(sig > 0)) return;
  real U[LMAX], V[LMAX], zb[LMAX], zt[LMAX], zm[LMAX], rho[LMAX], nn[LMAX], ri[LMAX];
  for (int k = 0; k < L; k++) {
    real a = 0, b = 0;
    for (int m = 0; m < 6; m++) {
      const int e = cellEdge[i * 6 + m];
      if (e < 0) break;
      const real ue = u[(size_t)k * E + e];
      a += op[(size_t)(i * 6 + m) * 2] * ue; b += op[(size_t)(i * 6 + m) * 2 + 1] * ue;
    }
    U[k] = a; V[k] = b;
  }
  // the layers' heights above the ground (bottom up), densities, stability and Richardson numbers
  real pt[LMAX], pb[LMAX], T[LMAX], p = 0;
  for (int k = 0; k < L; k++) { pt[k] = p; p += P.grav * mu[(size_t)k * N + i]; pb[k] = p; }
  real z = 0;
  for (int k = L - 1; k >= 0; k--) {
    const real pm = (real)0.5 * (pt[k] + pb[k]);
    T[k] = theta[(size_t)k * N + i] * pow(pm / P.p0, P.kappa);
    rho[k] = pm / (P.rd * T[k]);
    const real H = P.rd * T[k] / P.grav;
    zb[k] = z; zm[k] = z + H * log(pb[k] / pm);
    zt[k] = k > 0 ? z + H * log(pb[k] / fmax(pt[k], (real)1e-3 * pb[k])) : (real)1e30;
    z = zt[k];
  }
  for (int k = 0; k < L; k++) {
    real s2 = 0, nsum = 0, ssum = 0;
    int c = 0;
    for (int j = k; j <= k + 1; j++) {                      // the interfaces above (j = k) and below (j = k + 1) layer k
      if (j < 1 || j > L - 1) continue;
      const real dz = fmax(zm[j - 1] - zm[j], (real)1);
      const real th0 = theta[(size_t)(j - 1) * N + i], th1 = theta[(size_t)j * N + i];
      nsum += P.grav * (th0 - th1) / ((real)0.5 * (th0 + th1) * dz);
      const real du = U[j - 1] - U[j], dv = V[j - 1] - V[j];
      ssum += (du * du + dv * dv) / (dz * dz);
      c++;
    }
    s2 = ssum / c;
    const real n2 = fmax(nsum / c, (real)1e-6);
    nn[k] = sqrt(n2);
    ri[k] = fmax(n2 / fmax(s2, (real)1e-12), P.ric);
  }
  // the low-level flow between mu and 2 mu
  const real H = 2 * ms;
  real wsum = 0, ul = 0, vl = 0, nl = 0, rl = 0;
  for (int k = L - 1; k >= 0; k--) {
    const real ov = fmin(zt[k], H) - fmax(zb[k], ms);
    if (ov <= 0) continue;
    const real w = rho[k] * ov;
    wsum += w; ul += w * U[k]; vl += w * V[k]; nl += w * nn[k]; rl += w * rho[k];
  }
  if (wsum <= 0) return;
  ul /= wsum; vl /= wsum; nl /= wsum; rl /= wsum;
  const real sl = sqrt(ul * ul + vl * vl);
  if (sl < (real)0.1) return;
  const real B = 1 - (real)0.18 * gam - (real)0.04 * gam * gam, C = (real)0.48 * gam + (real)0.3 * gam * gam;
  const real psi = tho - atan2(vl, ul), sp = sin(psi), cs = cos(psi);
  const real d1 = B - (B - C) * sp * sp, d2 = (B - C) * sp * cs, dm = sqrt(d1 * d1 + d2 * d2);
  const real ex = (ul * d1 - vl * d2) / (sl * dm), ey = (vl * d1 + ul * d2) / (sl * dm);   // the drag's direction (unit)
  real up[LMAX];
  for (int k = 0; k < L; k++) up[k] = U[k] * ex + V[k] * ey;
  const real upl = ul * ex + vl * ey;
  // the blocked layer's top: N / U_p integrated down from H reaches H_nc
  real acc = 0, zblk = 0;
  for (int k = 0; k < L; k++) {
    const real top = fmin(zt[k], H), bot = zb[k];
    if (top <= bot) continue;
    if (up[k] <= (real)1e-3) { zblk = top; break; }
    const real rate = nn[k] / up[k];
    if (acc + rate * (top - bot) >= P.hnc) { zblk = top - (P.hnc - acc) / rate; break; }
    acc += rate * (top - bot);
  }
  zblk = fmin(fmax(zblk, (real)0), H);
  // the blocked flow's form drag, implicit (beta = dt x rate)
  if (zblk > 0) {
    const real A = zblk + ms;
    for (int k = L - 1; k >= 0; k--) {
      if (zb[k] >= zblk) break;
      const real s = sqrt(U[k] * U[k] + V[k] * V[k]);
      if (s <= 0) continue;
      const real pk = tho - atan2(V[k], U[k]), c2 = cos(pk) * cos(pk), s2 = 1 - c2;
      const real r = (c2 + gam * s2) / (gam * c2 + s2);
      const real fac = fmax(2 - 1 / r, (real)0) * (B * c2 + C * s2);
      const real I = blk_int(fmin(zt[k], zblk) + ms, A) - blk_int(zb[k] + ms, A);
      const real dz = (k > 0 ? zt[k] : zblk) - zb[k];                // the layer's own depth
      beta[(size_t)k * N + i] = P.dt * P.cd * fac * sig / (4 * ms) * s * I / fmax(dz, (real)1);
    }
  }
  // the gravity waves launched at H, deposited going up
  if (upl <= 0) return;
  const real heff = H - zblk, q = sig * dm / (4 * ms);
  real tau = P.G * rl * nl * upl * q * heff * heff;
  int kh = 0;
  for (int k = L - 1; k >= 0; k--) if (zt[k] > H) { kh = k; break; }
  const real scale = sl / upl;
  for (int k = kh; k >= 0 && tau > 0; k--) {
    real out;
    if (k == 0 || up[k] <= 0) out = 0;
    else {
      const real norm = P.G * rho[k] * nn[k] * up[k] * q;
      const real dh2 = tau / norm, a = nn[k] * sqrt(dh2) / up[k], sq = sqrt(ri[k]);
      const real riw = ri[k] * (1 - a) / ((1 + a * sq) * (1 + a * sq));
      out = tau;
      if (riw < P.ric) {
        const real del = 4 / sq / P.ric + 1 / (P.ric * P.ric) + 4 / P.ric, bb = 1 / P.ric + 2 / sq;
        const real ac = (real)0.5 * (-bb + sqrt(del));
        out = fmin(tau, norm * (up[k] * ac / nn[k]) * (up[k] * ac / nn[k]));
      }
    }
    real f = (tau - out) * scale;                                      // N/m^2 against the drag's direction
    const real mk = mu[(size_t)k * N + i], sk = sqrt(U[k] * U[k] + V[k] * V[k]);
    f = fmin(f, mk * sk / P.dt);                                       // never more than stops the layer in a step
    F[(size_t)(2 * k) * N + i] = -f * ex; F[(size_t)(2 * k + 1) * N + i] = -f * ey;
    tau = out;
  }
}

// Per edge and layer: the blocked drag (implicit) and the waves' force (along the edge's normal), the kinetic energy removed
// added to diss (J/m^2 per edge, as the wind stress's). nrm: the edge's normal (east, north) at its two cells.
extern "C" __global__ void oro_edge(int N, int E, int L, const real* mu, const int* eI, const int* eJ, const real* nrm,
                                   const real* beta, const real* F, real dt, real* u, real* diss) {
  LOOP1(e, E)
  const int a = eI[e], b = eJ[e];
  real d = 0;
  for (int k = 0; k < L; k++) {
    const real m = (real)0.5 * (mu[(size_t)k * N + a] + mu[(size_t)k * N + b]);
    const real fn = (real)0.5 * (F[(size_t)(2 * k) * N + a] * nrm[4 * e] + F[(size_t)(2 * k + 1) * N + a] * nrm[4 * e + 1]
                               + F[(size_t)(2 * k) * N + b] * nrm[4 * e + 2] + F[(size_t)(2 * k + 1) * N + b] * nrm[4 * e + 3]);
    const real bt = (real)0.5 * (beta[(size_t)k * N + a] + beta[(size_t)k * N + b]);
    if (fn == 0 && bt == 0) continue;
    real* pu = u + (size_t)k * E + e;
    const real u0 = *pu, u1 = (u0 + dt * fn / m) / (1 + bt);
    *pu = u1;
    d += (real)0.5 * m * (u0 * u0 - u1 * u1);
  }
  diss[e] += d;
}
