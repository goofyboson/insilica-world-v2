// Support kernels next to the fused engine (fused.cu): the vector Laplacian for del4 hyperviscosity, the
// receiving masses of a transport step, and the deterministic tree sum. REAL is double for the engine (float
// only for the speed probe); compiled with --fmad=false; sums run in v1's order (a cell's edges by ascending
// edge index, a vertex's edges ascending), as fused.cu's do.
//
// Layered arrays are stored layer-major: x[l * n + i]; blockIdx.y is the layer. Index tables are int32 with a
// fixed width of 6 per cell (pad -1) and 3 per vertex.

typedef REAL real;

#define LOOP(i, n) const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (n)) return; const int l = blockIdx.y;

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
