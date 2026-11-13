// Asynchronous coupling for the deep ocean's spin-up (V2-5, lever 5): a coupled year's forcing of the ocean recorded in
// nslot slots of the year (one a day by default: the mixing is nonlinear in the day's wind and heat, so their mean over
// longer slots overstirs it), then replayed under the ocean alone, interpolated between the slots' centres.
// Slot s covers [s, s + 1) year / nslot; t (s) is the world's clock on the device, year its length. Replayed rates are 32-bit.

typedef REAL real;

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

__device__ __forceinline__ int slot_of(real t, real year, int nslot) {
  real x = fmod(t, year);
  if (x < 0) x += year;
  int s = (int)(x / year * nslot);
  return s < 0 ? 0 : (s >= nslot ? nslot - 1 : s);
}

// Recording, after the wind stress: the momentum it gave the mixed layer (u0 now minus before).
extern "C" __global__ void rec_du(int E, const real* t, real year, int nslot, const real* u0, const real* before, real* acc) {
  LOOP1(e, E)
  acc[(size_t)slot_of(t[0], year, nslot) * E + e] += u0[e] - before[e];
}

// Recording, after the air's column: its heating of the mixed layer (J/m^2), and the seconds each slot has seen.
extern "C" __global__ void rec_heat(int N, const real* t, real year, int nslot, const real* T0, const real* before, const real* cap,
                                    real dt, real* acc, real* secs) {
  LOOP1(i, N)
  const int s = slot_of(t[0], year, nslot);
  acc[(size_t)s * N + i] += cap[i] * (T0[i] - before[i]);
  if (i == 0) secs[s] += dt;
}

// Recording, before the daily mixing: the day's sums the mixing uses, and the mixed layer's temperature.
extern "C" __global__ void rec_day(int N, int E, const real* t, real back, real year, int nslot, const real* q_acc, const real* sun_acc,
                                   const real* fw_acc, const real* tau_acc, const real* T0, real* qa, real* sa, real* fa, real* ta,
                                   real* Ta, real* days) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int s = slot_of(t[0] - back, year, nslot);              // (the day that just ended: the clock has reached its end)
  if (i < E) ta[(size_t)s * E + i] += tau_acc[i];
  if (i >= N) return;
  const size_t o = (size_t)s * N + i;
  qa[o] += q_acc[i];
  sa[o] += sun_acc[i];
  fa[o] += fw_acc[i];
  Ta[o] += T0[i];
  if (i == 0) days[s] += 1;
}

// The two slots around time t (their centres) and the weight of the second.
__device__ __forceinline__ void slots_at(real t, real year, int nslot, int& s0, int& s1, real& w) {
  real x = fmod(t, year);
  if (x < 0) x += year;
  real y = x / year * nslot - (real)0.5;
  if (y < 0) y += nslot;
  s0 = (int)y;
  if (s0 >= nslot) s0 = nslot - 1;
  w = y - s0;
  s1 = s0 + 1 == nslot ? 0 : s0 + 1;
}

// Replaying, every ocean step (dt): the wind's momentum on the mixed layer and its stress for the mixing.
extern "C" __global__ void play_edge(int E, const real* t, real year, int nslot, const float* du, const float* tau, real dt, real* u0,
                                     real* tau_acc) {
  LOOP1(e, E)
  int s0, s1; real w;
  slots_at(t[0], year, nslot, s0, s1, w);
  u0[e] += ((1 - w) * du[(size_t)s0 * E + e] + w * du[(size_t)s1 * E + e]) * dt;
  tau_acc[e] += ((1 - w) * tau[(size_t)s0 * E + e] + w * tau[(size_t)s1 * E + e]) * dt;
}

// Replaying, every ocean step, at sea cells: the air's heating with the surface's feedback lam (W/m^2/K) toward the recorded
// mixed-layer temperature (Haney 1971), the same feedback in the mixing's heat; its sunlight (for ocn_pen) and fresh water.
// Rates are per second (the recording's sums over its seconds or days).
extern "C" __global__ void play_cell(int N, const real* t, real year, int nslot, const float* Q, const float* Tr, const float* qns,
                                     const float* sun, const float* fw, real lam, real S_ref, real rho_w, real dt, const real* h,
                                     const real* cap, real* T, real* S, real* qocean, real* qsw, real* fw_acc, const real* land, real fw_local,
                                     real* hw, real* M, const real* area, real Tf) {
  LOOP1(i, N)
  if (land && land[i] > (real)0.5) return;
  int s0, s1; real w;
  slots_at(t[0], year, nslot, s0, s1, w);
  const size_t a = (size_t)s0 * N + i, b = (size_t)s1 * N + i;
  const real fb = lam * (((1 - w) * Tr[a] + w * Tr[b]) - T[i]);
  const real sw = (1 - w) * sun[a] + w * sun[b];
  T[i] += ((1 - w) * Q[a] + w * Q[b] + fb) * dt / cap[i];
  qsw[i] = sw;
  qocean[i] = (1 - w) * qns[a] + w * qns[b] + fb + sw;
  const real f = ((1 - w) * fw[a] + w * fw[b]) * dt;
  if (fw_local > 0) {                                           // (#165, as ocn_fresh)
    const real h1 = fmax(h[i] + f / rho_w, fmin(h[i], (real)10));   // (a sea with no way in evaporating dry: kept at 10 m, its salt still kept)
    S[i] *= h[i] / h1;
    hw[i] = h1;
    M[i] = h1 * area[i];
  } else S[i] -= S_ref * f / (rho_w * h[i]);
  // (#165) no sea ice is replayed: the surface water stops at freezing, as under the coupled model's ice (Tf > 0)
  if (Tf > 0 && T[i] < Tf) T[i] = Tf;
  fw_acc[i] += f;
}

extern "C" __global__ void play_time(real* t, real dt) { t[0] += dt; }
