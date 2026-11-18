// Geology's accumulators (V2-6e, decision #220): what the climate hands a later geology model (docs/v2-plan.md, V2-9),
// summed per cell over a run, one thread per cell, every row of geo (G x N) a running sum or state; geology.py lists them.
// Off unless a world has them: then they run inside its step graph, on fixed buffers, reading a tick clock on the device.

typedef REAL real;

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

// rows of geo (the same in geology.py)
#define G_RAIN 0       // rain, kg/m^2
#define G_EVAP 1       // evaporation, kg/m^2
#define G_TSUM 2       // the ground's (or sea's) surface temperature x time, K s
#define G_FROZEN 3     // time below freezing, s (land)
#define G_CRACK 4      // time in the frost-cracking window, s (land)
#define G_FTC 5        // freeze-thaw cycles (land)
#define G_FSTATE 6     // 1 while frozen (the cycles' hysteresis)
#define G_RX1 7        // the largest rain in one day, kg/m^2
#define G_BUCKET 8     // rain in the current slot of the day, kg/m^2
#define G_ICE 9        // ice sheet mass x time, kg s/m^2
#define G_RIVER 10     // surface water flowing out of the cell, kg/m^2 of its area (land.cu, land_apply)
#define G_RUNOFF 11    // the cell's own runoff: its pond and soil water's net loss to the routing, kg/m^2 (land)
#define G_WEATHER 12   // runoff weighted by the silicate weathering's temperature factor, kg/m^2 (land)
#define G_ICEFLOW 13   // ice flowing out of the cell, kg/m^2 of its area (land.cu, ice_apply)

// Every tick, after the column: rain, evaporation, temperature, frost, the ice's mass, and the day's rain in slots of
// slot_ticks ticks; when a slot fills, the last nslot slots (one day) are summed and the largest such day kept.
extern "C" __global__ void geo_column(int N, const real* land, const real* rain, const real* evap, const real* Ts, const real* ice,
                                      real dt, real tf, real band, real crack_lo, real crack_hi, int slot_ticks, int nslot,
                                      const real* clock, real* geo, real* ring) {
  LOOP1(i, N)
  const real T = Ts[i];
  geo[G_RAIN * N + i] += rain[i] * dt;
  geo[G_EVAP * N + i] += evap[i] * dt;
  geo[G_TSUM * N + i] += T * dt;
  if (!land || land[i] > (real)0.5) {
    if (T < tf) geo[G_FROZEN * N + i] += dt;
    if (T >= crack_lo && T <= crack_hi) geo[G_CRACK * N + i] += dt;
    real& s = geo[G_FSTATE * N + i];
    if (s > (real)0.5) { if (T > tf + band) { s = 0; geo[G_FTC * N + i] += 1; } }
    else if (T < tf - band) s = 1;
    if (ice) geo[G_ICE * N + i] += ice[i] * dt;
  }
  real& b = geo[G_BUCKET * N + i];
  b += rain[i] * dt;
  const long n = (long)clock[0] + 1;                              // ticks done after this one
  if (n % slot_ticks == 0) {
    ring[((n / slot_ticks - 1) % nslot) * N + i] = b;
    b = 0;
    real day = 0;
    for (int k = 0; k < nslot; k++) day += ring[k * N + i];
    if (day > geo[G_RX1 * N + i]) geo[G_RX1 * N + i] = day;
  }
}

extern "C" __global__ void geo_tick(real* clock) {
  if (blockIdx.x == 0 && threadIdx.x == 0) clock[0] += 1;
}

// On each routing step, after land_apply: the cell's runoff, the pond and soil water it lost to the routing (its outflow less
// its inflow), and that runoff weighted by exp(Ea/R (1/t0 - 1/Ts)), the temperature factor of silicate dissolution.
extern "C" __global__ void geo_runoff(int N, const real* land, const real* pond, const real* pond1, const real* soil, const real* soil1,
                                      const real* Ts, real ear, real t0, real* geo) {
  LOOP1(i, N)
  if (!(land[i] > (real)0.5)) return;
  const real r = (pond[i] + soil[i]) - (pond1[i] + soil1[i]);
  geo[G_RUNOFF * N + i] += r;
  geo[G_WEATHER * N + i] += r * exp(ear * ((real)1 / t0 - (real)1 / Ts[i]));
}
