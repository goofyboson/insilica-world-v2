// The ocean's coupling to the air and its mixed layer (stage V2-3, decisions #64-#71).
//
// Ocean layers k = 0 (the mixed layer) to L - 1 (the abyss), thickness h (m) and temperature T (K; heat only, the layers'
// density steps g' are fixed, decision #64), on the same grid as the air (layer-major arrays, one thread per cell or edge).

typedef REAL real;
#define LMAX 8

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

// Wind stress (decision #65): the air's bottom layer (mass per area ma) and the ocean's mixed layer (mo = rho_w h) are
// pulled toward each other by the stress tau = c (u_air - u_ocean), integrated exactly over the step: their difference
// decays as exp(-c (1/ma + 1/mo) dt) and their mass-weighted mean (the column's momentum) is unchanged. The kinetic
// energy removed, 0.5 ma mo / (ma + mo) (du0^2 - du1^2) per unit area at the edge, goes to diss (J/m^2; booked as the
// air's heat); the stress the ocean received (N/m^2) is added to tau_acc as tau^2 dt (for the mixed layer's u*).
extern "C" __global__ void couple_stress(int N, int E, int La, const real* ha, real* ua, const real* ho, real* uo, const int* eI,
                                         const int* eJ, real c, real rho_w, real dt, real* diss, real* tau_acc) {
  LOOP1(e, E)
  const int b = La - 1;
  const real ma = (real)0.5 * (ha[(size_t)b * N + eI[e]] + ha[(size_t)b * N + eJ[e]]);
  const real mo = rho_w * (real)0.5 * (ho[eI[e]] + ho[eJ[e]]);
  real* pa = ua + (size_t)b * E + e;
  real* po = uo + e;
  const real mean = (ma * *pa + mo * *po) / (ma + mo);
  const real du0 = *pa - *po;
  const real du1 = du0 * exp(-c * (1 / ma + 1 / mo) * dt);
  const real a0 = *pa;
  *pa = mean + mo / (ma + mo) * du1;
  *po = mean - ma / (ma + mo) * du1;
  diss[e] = (real)0.5 * ma * mo / (ma + mo) * (du0 * du0 - du1 * du1);
  const real tau = ma * (a0 - *pa) / dt;
  tau_acc[e] += tau * tau * dt;
}

// Convective adjustment (decisions #66, #71): wherever a layer is colder than the one below, the two mix their heat
// completely (heat conserved, interfaces unmoved: with fixed g' the layers' density never inverts, so convection is heat); repeated until the column is stable (L - 1 passes suffice, as a
// mixed pair is merged into a block that is mixed again with the next layer).
__device__ void adjust(int N, int L, int i, const real* h, real* T) {
  real hb[LMAX], Tb[LMAX];
  int first[LMAX], nb = 0;
  for (int k = 0; k < L; k++) {                                 // blocks of mixed layers, top down
    hb[nb] = h[(size_t)k * N + i]; Tb[nb] = T[(size_t)k * N + i]; first[nb] = k; nb++;
    while (nb > 1 && Tb[nb - 2] < Tb[nb - 1]) {                 // the block above is denser: merge
      Tb[nb - 2] = (hb[nb - 2] * Tb[nb - 2] + hb[nb - 1] * Tb[nb - 1]) / (hb[nb - 2] + hb[nb - 1]);
      hb[nb - 2] += hb[nb - 1];
      nb--;
    }
  }
  for (int j = 0; j < nb; j++) {
    const int k1 = j + 1 < nb ? first[j + 1] : L;
    for (int k = first[j]; k < k1; k++) T[(size_t)k * N + i] = Tb[j];
  }
}

// Sunlight passing depth z (Paulson & Simpson 1977, clear water; decision #70) and its integral from the surface.
__device__ __forceinline__ real sun_f(real z, real R, real z1, real z2) { return R * exp(-z / z1) + (1 - R) * exp(-z / z2); }
__device__ __forceinline__ real sun_int(real z, real R, real z1, real z2) {
  return -(R * z1 * expm1(-z / z1) + (1 - R) * z2 * expm1(-z / z2));
}

// The surface buoyancy flux that a mixed layer of depth h must work against (decision #74): heat absorbed at depth z
// takes (h/2 - z) per unit buoyancy to mix through the layer against h/2 at the surface, so the sunlight counts with
// 1 + f(h) - (2/h) int_0^h f dz (the light passing the base not at all), the other fluxes in full. Heating > 0.
__device__ __forceinline__ real kt_buoy(real h, real Bns, real Bsw, real R, real z1, real z2) {
  return Bns + Bsw * (1 + sun_f(h, R, z1, z2) - 2 * sun_int(h, R, z1, z2) / h);
}

// The turbulent energy left at the base of a layer of depth h (Kraus-Turner): wind stirring 2 m u*^3 less the work
// against heating (all of it, h B) or plus a share n of the convection's (n h |B|, cooling) (decision #74).
__device__ __forceinline__ real kt_energy(real h, real us3, real B, real m_kt, real n_kt) {
  return 2 * m_kt * us3 - (real)0.5 * h * ((1 + n_kt) * B + (1 - n_kt) * fabs(B));
}

// The mixed layer over one mixing interval (decisions #66, #67, #71, #74), per column:
// 1. Convective adjustment (above).
// 2. Entrainment (Kraus-Turner, as simplified by Gaspar 1988): the turbulent energy available at the layer's base,
//    P(h) = kt_energy (above) with the interval's mean buoyancy flux B(h) (kt_buoy, from qns, the non-solar flux, and
//    qsun, the sunlight entering the water), deepens the layer at w_e = P / (h db) (db = g alpha (T0 - T1)) when P > 0;
//    when P <= 0 the layer shoals to the depth where P = 0 (the Monin-Obukhov depth, by bisection since B depends on h),
//    detraining its own water. u*^3 = (tau / rho_w)^(3/2) from the interval's rms stress.
// The mixed layer trades water with the thermocline layer below it only; the deeper interfaces move with the dynamics
// (and the adjustment mixes their temperatures). wx (L - 1 rows, m, > 0 up into the layer above) is then applied with
// Layers.exchange (mass, temperature and momentum together); each layer keeps at least hmin.
// Entrainment needs a stratified base: below a jump of dTmin the layers are effectively mixed already (the adjustment did
// it, by mixing T without moving the interface) and entraining would only move thickness; the mixed layer stays at most
// hmax deep (Earth's deepest winter mixed layers) (decision #68).
extern "C" __global__ void ocn_column(int N, int L, const real* h, real* T, const real* taum, const real* qns, const real* qsun,
                                      real galpha, real rho_w, real cw, real m_kt, real n_kt, real hmin, real hmax, real dTmin,
                                      real R, real z1, real z2, real dt, real* wx) {
  LOOP1(i, N)
  adjust(N, L, i, h, T);
  const real h0 = h[i], h1 = h[(size_t)N + i];
  const real T0 = T[i], T1 = T[(size_t)N + i];
  const real us = sqrt(taum[i] / rho_w);                        // u* from the interval's rms stress (N/m^2)
  const real us3 = us * us * us;
  const real Bns = galpha * qns[i] / (rho_w * cw), Bsw = galpha * qsun[i] / (rho_w * cw);
  const real P = kt_energy(h0, us3, kt_buoy(h0, Bns, Bsw, R, z1, z2), m_kt, n_kt);
  const real db = galpha * (T0 - T1);
  real dm = 0;
  if (P > 0) {
    if (db > galpha * dTmin) dm = P / (h0 * db) * dt;
  } else {
    // shoal to the depth in [hmin, h0] where the energy balances (P(h0) <= 0), or to hmin if it is short there too
    real lo = hmin, hi = h0;
    if (kt_energy(lo, us3, kt_buoy(lo, Bns, Bsw, R, z1, z2), m_kt, n_kt) > 0)
      for (int it = 0; it < 50; it++) {
        const real mid = (real)0.5 * (lo + hi);
        if (kt_energy(mid, us3, kt_buoy(mid, Bns, Bsw, R, z1, z2), m_kt, n_kt) > 0) lo = mid; else hi = mid;
      }
    dm = lo - h0;
  }
  real up_max = h1 - hmin;
  if (hmax - h0 < up_max) up_max = hmax - h0;
  const real dn_max = h0 - hmin;
  if (dm > up_max) dm = up_max > 0 ? up_max : (real)0;
  if (-dm > dn_max) dm = dn_max > 0 ? -dn_max : (real)0;
  wx[i] = dm;
  for (int k = 1; k < L - 1; k++) wx[(size_t)k * N + i] = 0;
}

// Convective adjustment before every ocean step (decision #66): heat mixed downward wherever surface cooling made a layer
// colder than the one below (with T-dependent density, left for the daily mixing, g' went negative between mixings).
extern "C" __global__ void ocn_overturn(int N, int L, const real* h, real* T) {
  LOOP1(i, N)
  adjust(N, L, i, h, T);
}

// Sunlight below the mixed layer (decision #70), every air step: of the sunlight the air column put into the mixed layer,
// the part passing depth z, f(z) (sun_f), heats the layers below instead, each the part absorbed within it. Heat is
// moved, not made. qns_acc and sun_acc sum the mixed layer's non-solar flux and the sunlight entering the water for the
// mixing (decision #74).
extern "C" __global__ void ocn_pen(int N, int L, const real* h, real* T, const real* qocean, const real* qsw, real rho_cw,
                                   real R, real z1, real z2, real dt, real* qns_acc, real* sun_acc) {
  LOOP1(i, N)
  real z = h[i];
  real f = sun_f(z, R, z1, z2);
  const real pen = qsw[i] * f;
  T[i] -= pen * dt / (rho_cw * h[i]);
  for (int k = 1; k < L; k++) {
    const real hk = h[(size_t)k * N + i];
    z += hk;
    const real fb = k < L - 1 ? sun_f(z, R, z1, z2) : (real)0;
    T[(size_t)k * N + i] += qsw[i] * (f - fb) * dt / (rho_cw * hk);
    f = fb;
  }
  qns_acc[i] += (qocean[i] - qsw[i]) * dt;
  sun_acc[i] += qsw[i] * dt;
}

// The mixed layer's heat capacity per cell for the air's surface physics: rho_w c_w h0 (J/m^2/K).
extern "C" __global__ void ocn_cap(int N, const real* h, real rho_cw, real* cap) {
  LOOP1(i, N)
  cap[i] = rho_cw * h[i];
}
