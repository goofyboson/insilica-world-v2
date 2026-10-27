// The ocean's coupling to the air and its mixed layer (stage V2-3, decisions #64-#69).
//
// Ocean layers k = 0 (the mixed layer) and 1 (the deep layer), thickness h (m), temperature T (K, its density
// rho0 (1 - alpha (T - T0))), on the same grid as the air (layer-major arrays, one thread per cell or edge).

typedef REAL real;

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

// The mixed layer over one ocean step (decisions #66, #67), per column:
// 1. Overturning: a mixed layer colder (denser) than the deep layer mixes with it completely (heat conserved).
// 2. Entrainment (Kraus-Turner, as simplified by Gaspar 1988): the turbulent energy available at the layer's base,
//    P = 2 m u*^3 - (h / 2) ((1 + n) B0 - (1 - n) |B0|), with B0 the surface buoyancy flux (g alpha Q / (rho c); heating
//    > 0), deepens the layer at w_e = P / (h db) (db = g alpha (T0 - T1)) when P > 0; when P < 0 the layer shoals to the
//    Monin-Obukhov depth 2 m u*^3 / B0 (detraining its own water). u*^3 = (tau / rho_w)^(3/2) from the step's mean stress.
// wx (m, > 0 up into the mixed layer) is then applied with Layers.exchange (mass, temperature and momentum together);
// each layer keeps at least hmin.
// Entrainment needs a stratified base: below a jump of dTmin the layers are effectively mixed already (the overturning did
// it, by mixing T without moving the interface) and entraining would only move thickness; the mixed layer stays at most
// hmax deep (Earth's deepest winter mixed layers), so the deep layer stays thick (decision #68).
extern "C" __global__ void ocn_column(int N, real* h, real* T, const real* taum, const real* qmean, real galpha, real rho_w,
                                      real cw, real m_kt, real n_kt, real hmin, real hmax, real dTmin, real dt, real* wx) {
  LOOP1(i, N)
  const real h0 = h[i], h1 = h[(size_t)N + i];
  real T0 = T[i], T1 = T[(size_t)N + i];
  if (T0 < T1) {                                                // overturning
    const real Tm = (h0 * T0 + h1 * T1) / (h0 + h1);
    T0 = Tm; T1 = Tm;
    T[i] = Tm; T[(size_t)N + i] = Tm;
  }
  const real us = sqrt(taum[i] / rho_w);                        // u* from the step's rms stress (N/m^2)
  const real us3 = us * us * us;
  const real B0 = galpha * qmean[i] / (rho_w * cw);
  const real P = 2 * m_kt * us3 - (real)0.5 * h0 * ((1 + n_kt) * B0 - (1 - n_kt) * fabs(B0));
  const real db = galpha * (T0 - T1);
  real dm = 0;
  if (P > 0) {
    if (db > galpha * dTmin) dm = P / (h0 * db) * dt;
  } else if (B0 > 0) {
    const real hmo = 2 * m_kt * us3 / B0;
    const real target = hmo > hmin ? hmo : hmin;
    if (target < h0) dm = target - h0;                        // detrain down to the Monin-Obukhov depth
  }
  real up_max = h1 - hmin;
  if (hmax - h0 < up_max) up_max = hmax - h0;
  const real dn_max = h0 - hmin;
  if (dm > up_max) dm = up_max > 0 ? up_max : (real)0;
  if (-dm > dn_max) dm = dn_max > 0 ? -dn_max : (real)0;
  wx[i] = dm;
}

// Overturning before every ocean step (decision #66): a mixed layer colder (denser) than the deep layer mixes with it
// completely (heat conserved). Left for the daily mixing, the stratification's g' went negative between mixings and the
// interface grew without bound (the layered model's Rayleigh-Taylor instability).
extern "C" __global__ void ocn_overturn(int N, const real* h, real* T) {
  LOOP1(i, N)
  const real h0 = h[i], h1 = h[(size_t)N + i], T0 = T[i], T1 = T[(size_t)N + i];
  if (T0 < T1) {
    const real Tm = (h0 * T0 + h1 * T1) / (h0 + h1);
    T[i] = Tm; T[(size_t)N + i] = Tm;
  }
}

// The mixed layer's heat capacity per cell for the air's surface physics: rho_w c_w h0 (J/m^2/K).
extern "C" __global__ void ocn_cap(int N, const real* h, real rho_cw, real* cap) {
  LOOP1(i, N)
  cap[i] = rho_cw * h[i];
}
