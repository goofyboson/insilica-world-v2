// Water moving over and through the land (stage V2-4b, decision #85), one thread per cell; a sea cell only receives.
//
// Surface water (pond, kg/m^2) runs toward lower water surfaces z = b + pond / rho_w + ice / rho_ice: through an edge of
// length le, a depth min(head, the donor's depth) at the speed 3/4 U, so a flow crosses cells at about U on any grid (a
// hexagon's downhill edges carry 2 le U per unit depth at speed 1: A = 1.5 le d, hence the 3/4). At most 1/12 of the head
// per edge, as v1, so that a level surface stays level (decision #85). Soil water seeps between water tables
// zt = (soil / soilcap) b by Darcy's law, K_seep (le / de) (zt_i - zt_j), and out to a sea cell whose table is 0; at most
// 1/12 of what levels the two tables. What reaches a sea cell waits in mouth (kg/m^2 of that cell) for the ocean's fresh water.

typedef REAL real;

#define LOOP1(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return;

// The surface flow and the seepage from cell a to cell b across edge e (kg per step), >= 0 when a gives. Only land gives.
__device__ __forceinline__ void edge_flows(int a, int b, int e, const real* land, const real* z, const real* pond, const real* zt,
                                           const real* soil, const real* soilcap, const real* le, const real* de, const real* area,
                                           real rho_w, real uflow, real kseep, real& fp, real& fs) {
  fp = 0; fs = 0;
  if (!(land[a] > (real)0.5)) return;
  const bool lb = land[b] > (real)0.5;
  const real ma = area[a] < area[b] ? area[a] : area[b];
  const real head = z[a] - (lb ? z[b] : (real)0);
  if (head > 0 && pond[a] > 0) {
    const real d = pond[a] / rho_w, depth = head < d ? head : d;
    const real q = (real)0.75 * uflow * depth * le[e], cap = head / 12 * ma;
    fp = rho_w * (q < cap ? q : cap);
  }
  const real dz = zt[a] - (lb ? zt[b] : (real)0);
  if (dz > 0 && soil[a] > 0) {
    const real ra = zt[a] / soil[a];                             // table height per kg/m^2 (m^3/kg)
    const real rb = lb && soilcap[b] > 0 ? zt[b] > 0 && soil[b] > 0 ? zt[b] / soil[b] : ra : (real)0;
    const real q = rho_w * kseep * le[e] / de[e] * dz, cap = dz / (ra + rb) / 12 * ma;
    fs = q < cap ? q : cap;
  }
}

// The heights the flows read: water surface z and water table zt.
extern "C" __global__ void land_heights(int N, const real* b, const real* land, const real* pond, const real* ice, const real* soil,
                                        const real* soilcap, real rho_w, real rho_ice, real* z, real* zt) {
  LOOP1(i, N)
  if (!(land[i] > (real)0.5)) { z[i] = 0; zt[i] = 0; return; }
  z[i] = b[i] + pond[i] / rho_w + ice[i] / rho_ice;
  zt[i] = soilcap[i] > 0 && b[i] > 0 ? soil[i] / soilcap[i] * b[i] : (real)0;
}

// Each land cell's scale: what it would give in total, limited to half its pond and all its soil (as v1's limiter).
extern "C" __global__ void land_scale(int N, const real* land, const real* z, const real* pond, const real* zt, const real* soil,
                                      const real* soilcap, const int* cellEdge, const int* eI, const int* eJ, const real* le,
                                      const real* de, const real* area, real rho_w, real uflow, real kseep, real* sp, real* ss) {
  LOOP1(i, N)
  sp[i] = 1; ss[i] = 1;
  if (!(land[i] > (real)0.5)) return;
  real op = 0, os = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    const int o = eI[e] == i ? eJ[e] : eI[e];
    real fp, fs;
    edge_flows(i, o, e, land, z, pond, zt, soil, soilcap, le, de, area, rho_w, uflow, kseep, fp, fs);
    op += fp; os += fs;
  }
  const real ap = pond[i] > 0 ? (real)0.5 * pond[i] * area[i] : (real)0, as = soil[i] > 0 ? soil[i] * area[i] : (real)0;   // rounding can leave -1e-15
  if (op > 0 && op > ap) sp[i] = ap / op;
  if (os > 0 && os > as) ss[i] = as / os;
}

// New pond and soil from the flows out of this cell and into it; a sea cell adds what it receives to mouth.
extern "C" __global__ void land_apply(int N, const real* land, const real* z, const real* pond, const real* zt, const real* soil,
                                      const real* soilcap, const int* cellEdge, const int* eI, const int* eJ, const real* le,
                                      const real* de, const real* area, real rho_w, real uflow, real kseep, const real* sp,
                                      const real* ss, real* pond1, real* soil1, real* mouth) {
  LOOP1(i, N)
  real gp = 0, gs = 0;
  const bool li = land[i] > (real)0.5;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    const int o = eI[e] == i ? eJ[e] : eI[e];
    real fp, fs;
    if (li) {
      edge_flows(i, o, e, land, z, pond, zt, soil, soilcap, le, de, area, rho_w, uflow, kseep, fp, fs);
      gp -= fp * sp[i]; gs -= fs * ss[i];
    }
    edge_flows(o, i, e, land, z, pond, zt, soil, soilcap, le, de, area, rho_w, uflow, kseep, fp, fs);
    gp += fp * sp[o]; gs += fs * ss[o];
  }
  if (li) {
    pond1[i] = pond[i] + gp / area[i];
    soil1[i] = soil[i] + gs / area[i];
  } else {
    pond1[i] = 0; soil1[i] = 0;
    mouth[i] += (gp + gs) / area[i];
  }
}

// Ice sheets (stage V2-4b, decision #86): Glen's shallow-ice flow, n = 3, without sliding. The surface s = b + ice / rho_ice falls
// toward a neighbour across an edge; through it, a volume gam H^5 |grad s|^2 (head / de) le per step, H the donor's thickness, with
// |grad s|^2 from the TRiSK kinetic energy's rule at each end, so that the flow does not depend on the edges' directions. At most
// 1/12 of the head per edge. A sea cell's surface is 0: ice reaching it becomes that cell's sea ice, which melts as any other,
// taking its latent heat from the sea, and the sea's fresh water counts its change from ice_prev, so the salt changes on melting.

extern "C" __global__ void ice_surface(int N, const real* b, const real* land, const real* ice, real rho_ice, real* s) {
  LOOP1(i, N)
  s[i] = land[i] > (real)0.5 ? b[i] + ice[i] / rho_ice : (real)0;
}

extern "C" __global__ void ice_slope(int N, const real* land, const real* s, const int* cellEdge, const int* eI, const int* eJ,
                                     const real* le, const real* de, const real* area, real* g2) {
  LOOP1(i, N)
  real sum = 0;
  if (land[i] > (real)0.5)
    for (int m = 0; m < 6; m++) {
      const int e = cellEdge[i * 6 + m];
      if (e < 0) break;
      const int o = eI[e] == i ? eJ[e] : eI[e];
      const real ds = (s[o] - s[i]) / de[e];
      sum += (real)0.5 * le[e] * de[e] * ds * ds;
    }
  g2[i] = sum / area[i];
}

// The ice from cell a to cell b across edge e (kg per step), >= 0 when a gives. Only land gives.
__device__ __forceinline__ real ice_flow(int a, int b, int e, const real* land, const real* s, const real* g2, const real* ice,
                                         const real* le, const real* de, const real* area, real rho_ice, real gam) {
  if (!(land[a] > (real)0.5) || !(ice[a] > 0)) return 0;
  const real head = s[a] - s[b];
  if (!(head > 0)) return 0;
  const real H = ice[a] / rho_ice, H2 = H * H;
  const real g = land[b] > (real)0.5 ? (real)0.5 * (g2[a] + g2[b]) : g2[a];
  const real q = gam * H2 * H2 * H * g * head / de[e] * le[e];
  const real ma = area[a] < area[b] ? area[a] : area[b], cap = head / 12 * ma;
  return rho_ice * (q < cap ? q : cap);
}

extern "C" __global__ void ice_scale(int N, const real* land, const real* s, const real* g2, const real* ice, const int* cellEdge,
                                     const int* eI, const int* eJ, const real* le, const real* de, const real* area, real rho_ice,
                                     real gam, real* si) {
  LOOP1(i, N)
  si[i] = 1;
  if (!(land[i] > (real)0.5)) return;
  real out = 0;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    out += ice_flow(i, eI[e] == i ? eJ[e] : eI[e], e, land, s, g2, ice, le, de, area, rho_ice, gam);
  }
  const real have = ice[i] > 0 ? ice[i] * area[i] : (real)0;
  if (out > have) si[i] = have / out;
}

extern "C" __global__ void ice_apply(int N, const real* land, const real* s, const real* g2, const real* ice, const int* cellEdge,
                                     const int* eI, const int* eJ, const real* le, const real* de, const real* area, real rho_ice,
                                     real gam, const real* si, real* ice1, real* ice_prev, real* calved) {
  LOOP1(i, N)
  real d = 0;
  const bool li = land[i] > (real)0.5;
  for (int m = 0; m < 6; m++) {
    const int e = cellEdge[i * 6 + m];
    if (e < 0) break;
    const int o = eI[e] == i ? eJ[e] : eI[e];
    if (li) d -= ice_flow(i, o, e, land, s, g2, ice, le, de, area, rho_ice, gam) * si[i];
    d += ice_flow(o, i, e, land, s, g2, ice, le, de, area, rho_ice, gam) * si[o];
  }
  d /= area[i];
  ice1[i] = ice[i] + d;
  if (!li) {
    calved[i] += d;
    if (ice_prev) ice_prev[i] += d;
  }
}
