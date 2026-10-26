# InSilica World V2 — specification

The plan, physics and stage series are in [docs/v2-plan.md](docs/v2-plan.md). This file records what is built: the
code layout, the numerical conventions, the interfaces, every stage's tests with their thresholds, and (§9) the
decision log of agreed changes. v1's SPEC (`../insilica-world/SPEC.md`, decision log §18 #1–#52) stays the reference
for the physics still to be ported.

## 1. Layout

| Path | What |
|---|---|
| `src/insilica/constants.py` | physical constants (from v1's `00-core.js`) |
| `src/insilica/grid/icosa.py` | icosahedral Voronoi mesh with Lloyd/SCVT (port of v1 `buildGrid`, `lloydSCVT`); Hilbert and Morton renumbering |
| `src/insilica/grid/trisk.py` | TRiSK C-grid geometry (port of v1 `getTrisk`), the GPU gather tables, the Perot wind rebuild `cell_winds` |
| `src/insilica/grid/__init__.py` | `Grid`, `get(n, order)` with the disk cache `cache/grids/n{n}-{order}-{generator hash}.npz` |
| `src/insilica/kernels/*.cu` | CUDA C kernels (ASCII only), compiled at run time by NVRTC through CuPy: `fused.cu` (the engine), `trisk.cu` (the ∇⁴ operator, `mass_update`, `block_sum`) |
| `src/insilica/gpu.py` | kernel loading, `DeviceGrid`, deterministic sums `dsum` |
| `src/insilica/swe.py`, `transport.py` | shallow water on L stacked layers; FCT tracer transport (ports of v1 `makeSolver`, `transportStep`), fused kernels |
| `src/insilica/world.py` | the V2-0 world: presets, stepping, fields, save/load |
| `src/insilica/io/save.py` | save-file format (§6) |
| `src/insilica/runcache.py` | spin-up cache `cache/runs/` |
| `src/insilica/server/`, `viewer/` | the engine server (`python -m insilica.server`) and the browser page (§5) |
| `src/insilica/probe.py` | performance probe (`python -m insilica.probe`) |
| `tests/` | pytest suite; `tests/v1ref.py` builds v1 reference data with `tools/v1_dump.mjs` |
| `tools/run_tests.py` | the runner (§7) |
| `cache/` | everything rebuilt on demand (gitignored); `INSILICA_CACHE` moves it |

Environment: Python 3.14 in `.venv`, `pip install -e ".[test]"`. CuPy is `cupy-cuda12x[ctk]`, which brings CUDA 12.9
runtime, NVRTC and headers as pip packages (§9 #6).

## 2. Grid

v1's mesh: an icosahedron with vertices at both poles (cells 0 and 11), midpoint
subdivision n times, Jacobi Lloyd iterations toward a spherical centroidal Voronoi tessellation (at most 400, stop
when no generator moves 1e-11 rad, poles pinned), corners at circumcenters, neighbors sorted counterclockwise from
east, a fixed 6-slot neighbor table (pad −1), edges listed by lower cell then slot (`eI < eJ`, normal from eI to eJ),
TRiSK vertices numbered by first appearance. Each corner is computed once and shared by its three cells (#19).
Built in v1's numbering, then by default renumbered along a 3-D Hilbert curve of the cell positions for memory
locality (`order='hilbert'`; `'v1'` and `'morton'` are available); the grid keeps `perm` (cell k is v1-order cell
`perm[k]`), and edges and vertices are renumbered by the same rules. From n = 6 the Lloyd iteration runs on the GPU (same formulas; results
differ only by rounding). Arrays keep v1's names (`pos, area, nbr, deg, dist, edgeLen, edgeDir, corners, eI, eJ,
cellEdge, cellSign, ev1, ev2, de, le, xe, ne, kite, areaCell, areaV, circ1, circ2, fV, eoeStart, eoe, w`), plus
`xm`, the Voronoi edge's midpoint (#17): `xe` (where the line between the cell centers crosses the edge) stays the
kites' corner and TRiSK's edge point; `xm` is used by the Perot rebuild `cell_winds` and for sampling analytic
winds onto edges.

GPU gather tables (`trisk.gather_tables`) let each thread sum what v1's loops scatter, in v1's order:
`cellEdgeAsc` (a cell's edges by ascending edge index), `vEdge/vEdgeSign` (a vertex's three edges, ascending),
`vCell/vSlot` (a vertex's three kites by ascending cell).

| n | cells | edges | build | Lloyd last move (rad) | area max/min |
|---|---|---|---|---|---|
| 4 | 2,562 | 7,680 | 1.5 s | 1.2e-11 | 1.33 |
| 5 | 10,242 | 30,720 | 5.5 s | 8.1e-7 | 1.42 |
| 6 | 40,962 | 122,880 | 3.7 s (GPU) | 9.4e-6 | 1.54 |
| 7 | 163,842 | 491,520 | 6.1 s (GPU) | 9.4e-6 | 1.69 |
| 8 | 655,362 | 1,966,080 | 25 s (GPU) | 5.1e-6 | 1.79 |

From n = 5 the Lloyd iteration stops at its 400-iteration cap before converging (as v1's does at n = 5).

## 3. Numerics on the GPU

- 64-bit throughout; kernels are templated on `REAL` only so the speed probe can time 32-bit.
- NVRTC options `--std=c++17 --fmad=false`: no fused multiply-adds, so products round as in JavaScript and NumPy.
- Every kernel is a gather (one thread per cell, edge or vertex), with no atomics, summing in v1's order. Results are
  bit-identical run to run (D1).
- Layered arrays are layer-major: `x[l * n + i]`, `blockIdx.y` = layer.
- `dsum`: fixed-order tree reduction (256 threads × 8 values per block, shared-memory halving, repeated), the only
  way totals are formed.
- Shallow water (`swe.SWE`): v1's one-layer TRiSK solver per layer: Wicker–Skamarock RK3; v1's positivity limiter
  in every stage (a cell sends at most half its layer per stage); ∇⁴ hyperviscosity from the step's starting wind,
  held through the stages; linear drag. Fused kernels (#18): five per stage (limiter scale, limited flux, cell
  tendency + RK update, vertex PV, edge tendency + RK update), 15 per step (21 with ∇⁴), against V2-0's 27.
- Transport (`transport.Transport`): 3rd-order fluxes f (mean − a − b) + |f| (b − a) β in RK3 stages (a, b the
  scaled curvatures at the edge's two cells; β = 1 is 3rd-order upwind, β = 0 4th-order centered; default 0.25, #20, #21) with Zalesak
  FCT against the upwind solution, on M·psi with the step's mass fluxes; `mass_after` gives the receiving masses in
  flux form, so the tracer total is conserved to rounding. Fused: five cell passes with the edge fluxes recomputed
  where they are summed.
- The fused kernels do v1's arithmetic in v1's order (V1-ENGINE).
- A world tick (SWE step + transport) allocates nothing and is captured once as a CUDA graph, then replayed with one
  launch per tick (#18).

## 4. The V2-0 world

`World(n, preset, layers)`: state `h (L,N)`, `u (L,E)`, `M (L,N)` (layer volume per cell, moved by the same fluxes
as h), `psi (L,N)`. Presets: `w2` (Williamson 2), `w5` (Williamson 5, flow over a mountain), `galewsky` (Galewsky
jet), each carrying v1 N6's cosine bell as tracer. dt = 450 s · 2^(4−n). A tick is one SWE step plus one FCT step.
This is a stand-in for the tools; V2-1 replaces it with the layered fluid core.

## 5. Viewer protocol

`python -m insilica.server [--n 4] [--preset w5] [--port 8765]` serves the page at `http://127.0.0.1:8765/` and a
WebSocket at `/ws` (localhost only). Text messages are JSON; binary frames carry arrays.

| Client → server | Reply |
|---|---|
| `{"type":"hello"}` | `info` (n, N, E, preset, dt, tick, fields, presets), then the grid frame |
| `{"type":"get","field":f,"layer":l,"dtype":"f4"\|"f8"}` | one field frame |
| `{"type":"subscribe","field":f,"layer":l}` | that field now, then after every step or frame |
| `{"type":"play","tpf":k}`, `{"type":"pause"}` | runs k ticks per frame at up to 30 frames/s; `status` |
| `{"type":"step","k":k}` | k ticks, then `status` and the subscribed field |
| `{"type":"new","n":n,"preset":p}` | a new world; `info` and grid to every client |

Errors come back as `{"type":"error","message":…}` without closing the connection. Binary frame: uint32 LE header
length H, H bytes of JSON header, zero padding to a multiple of 8, then the parts, each padded to 8. Field header
`{"kind":"field","name","layer","tick","dtype","count","min","max"}`; grid header `{"kind":"grid","n","N","parts":
[pos f4 3N, corners f4 18N, deg u1 N]}`. Display fields are 32-bit; `"dtype":"f8"` gives the engine's exact bytes.
The page uses three.js r128, vendored in `viewer/vendor/` (MIT licence) so it works offline.

## 6. Save files

An uncompressed zip: `meta.json` (format version 1, kind, n, preset, layers, dt, tick, params, grid generator hash,
engine source hash, array list) and one `.npy` per state array (raw little-endian float64). Entries are written in
sorted order with a fixed timestamp, so the same state always gives the same bytes. Loading rebuilds the grid from n
and refuses a file made with a different grid generator.

## 7. Tests and runner

**Method (agreed 2026-10-26, #22).** Least redundancy and run time, most integrity: one test per link of the
integrity chain (mesh → agreement with v1 → accuracy → conservation → reproducibility → viewer), each as strong as
possible, and nothing a stronger test already implies. A new stage adds tests only for what no existing test can
see, extends an existing test where it can, and retires checks its new tests supersede. Thresholds change only with
the owner's agreement, recorded here.

`python tools/run_tests.py [--perf] [--jobs J] [--cold]` prints the GPU environment, runs the suite in up to 5
processes (pytest-xdist, `--dist loadgroup`; tests sharing a grid share an xdist group) and prints every measured
value against its threshold. `--perf` then runs PERF alone (at every stage close); `--cold` uses an empty cache.

| Test | What it checks | Threshold |
|---|---|---|
| GRID (n = 0–8) | topology (N = 10·4ⁿ+2, E = 30·4ⁿ, V = 20·4ⁿ, 12 pentagons, Euler, every edge in 2 cells and vertex in 3); Σ areas = 4πR² three ways; each cell's kites = its polygon; dual ⟂ primal edges; TRiSK weight antisymmetry; discrete curl(grad) = 0; area max/min; at n = 4 the disk cache returns the freshly built grid | topology exact; ≤ 1e-12; < 1.5 at n = 4; cache bit-identical |
| V1-GRID (n = 2–5) | the mesh built in v1's numbering is v1's mesh: same edges, vertices, neighbor cycles, weight lists, circulation signs; all geometry against v1's dump | same mesh; ≤ 1e-12 · 4^(n−4) (#16) |
| V1-ENGINE | the engine as it runs (fused kernels, CUDA graph, Hilbert-ordered grid) against v1's own results: Williamson 2 at n = 4 (1 and 960 steps of 450 s), a steep mountain with ∇⁴, drag and the positivity limiter active (10 steps), upwind transport of v1's cosine bell (1 and 576 steps) | 1 step ≤ 1e-12; 576–960 steps ≤ 1e-9; limiter exercised |
| ACCURACY | N1 Williamson 2, 5 days at n = 4 (height L2); N6 cosine bell once around at n = 4 (FCT3 against upwind); Perot rebuild of solid-body rotation at n = 4, 5, 6 | N1 < 1e-3; N6 ≥ 2× better, min > −1e-9, tracer drift < 1e-12; Perot max ≥ ×1.8 and rms ≥ ×3.5 per level |
| CONSERVE (n = 4, 6) | Williamson 5 + bell with ∇⁴ and drag, 5 days at n = 4 and 1 day at n = 6: total volume Σh·A, layer mass ΣM, tracer ΣM·psi | relative change ≤ 1e-13 |
| REPRO | repeat runs, graph against direct ticks and repeated tree sums; save → load → save bytes and save-at-T + continue against a straight run; spin-up cache miss then hit; two identical layers against each other and a one-layer run; the Hilbert-ordered grid against v1's numbering after 960 steps | bit-identical; numbering ≤ 1e-12 |
| VIEWER | the protocol with a Python client: grid, exact f8 bytes, f4 bytes, step +1/+5, play pushes, pause holds, error reply, page and three.js served | all correct |
| PERF (`--perf`) | ns per layer-edge per RK3 step at n = 4–8 for L = 1, 2, 4 in 64- and 32-bit; launch overhead; what the graph and the cell numbering each give | report; gate: n = 8 ≤ 16 ns |

**Results (2026-10-26, 22/22):** GRID topology exact at every level, worst geometry 1.6e-13 (orthogonality at
n = 8), area ratio 1.33 at n = 4, cache identical. V1-GRID 1.2e-14 / 5.3e-14 / 2.3e-13 / 1.29e-12 against
6.3e-14 / 2.5e-13 / 1e-12 / 4e-12. V1-ENGINE 2.3e-15 (1 step), 4.3e-14 (960 steps), mountain 4.4e-14, upwind
4.2e-15 / 2.1e-14. ACCURACY N1 2.01e-4; N6 FCT3 0.305, ×3.01; Perot ×2.16 / ×3.85 per level. CONSERVE ≤ 1.8e-16.
REPRO all bit-identical, numbering 2.3e-14. VIEWER correct. PERF n = 8 2.9 ns per layer-edge-step, two layers
11.4 ms per step; the graph ×2.0 at n = 4; Hilbert ×1.26 at n = 8. Suite 9 s warm, 71 s from an empty cache; PERF 52 s.

**Stage history.** V2-0 closed with 41/41 and V2-0b with 45/45 under the earlier, finer-grained suite (IDs F0,
G1–G4, K1, K2, C1, D1, S1, P1, W1, R1, E1–E4); its results are in #15–#21 and the git history (`880afb0`,
`224b4dd`). #22 maps those checks onto the suite above.

## 8. Git

One branch per stage (`stage-v2-0`, …), checkpoints pushed there, merged to `main` when the stage closes.

## 9. Decision log

| # | Date | Area | Decision |
|---|---|---|---|
| 1 | 2026-10-26 | Engine | CUDA kernels in 64-bit driven from Python through CuPy; the browser is the viewer (plan §6, §9). |
| 2 | 2026-10-26 | Repository | A new repository next to v1; v1 is paused and read-only (plan §9). |
| 3 | 2026-10-26 | Layer thickness | Set by physics with a minimum-thickness floor; free ocean interface with entrainment; isentropic air interface, hybrid with a terrain-following lower layer; the remap nudge is a fallback, off by default (plan §3). |
| 4 | 2026-10-26 | Fast modes | Split-explicit at true wave speeds (plan §4), not semi-implicit. |
| 5 | 2026-10-26 | Resolution | Showcase air n = 6 + ocean n = 8; tests at n = 4–5; a variable-resolution region stays a later option. |
| 6 | 2026-10-26 | Python and CuPy | Python 3.14 in a project `.venv` (CuPy 14.2 has Python 3.14 wheels). `cupy-cuda12x`, with the `[ctk]` extra: CuPy 14's bundled CCCL headers need CUDA ≥ 12.8, and with the installed 12.6 toolkit CuPy's own reductions (e.g. `cupy.sum`) failed to compile; the pip CUDA 12.9 runtime, NVRTC and headers fixed it. |
| 7 | 2026-10-26 | Tests | pytest with pytest-xdist (≤ 5 processes), a runner that prints measured values against thresholds, and a spin-up cache keyed by a hash of every engine source. |
| 8 | 2026-10-26 | Viewer | JSON control messages and binary frames; display fields 32-bit, exact 64-bit on request; three.js r128 vendored; localhost port 8765. |
| 9 | 2026-10-26 | Porting | v1's grid, TRiSK, solver and transport ported step for step in v1's order, checked against dumps of v1's own results (G4, K1). Kernels compiled with `--fmad=false`. |
| 10 | 2026-10-26 | Lloyd | v1's rule at every n (≤ 400 iterations, tolerance 1e-11); n ≥ 6 on the GPU. |
| 11 | 2026-10-26 | Cell order | v1's numbering in V2-0; a Morton (Z-order) renumbering is measured by P1. |
| 12 | 2026-10-26 | Saves | The zip/npy format of §6; v1 save files are not imported. |
| 13 | 2026-10-26 | Git | One branch per stage, merged to `main` at the stage's close (as v1 §18 #48). |
| 14 | 2026-10-26 | G4 procedure | Edges and vertices are compared through maps matched by their cells (an edge by its cell pair, a vertex by its cell triple), and each cell's slots through the rotation that aligns its neighbor cycle. Where a neighbor lies exactly due west of a cell (angle ±π, on the icosahedron's symmetry lines), rounding picks which neighbor comes first, in v1 and in V2 alike: 3–6 cells per level start their slot cycle at a different neighbor, which renumbers some edges. The mesh itself (cells, edges, vertices, cycles, weight lists, signs) is identical. |
| 15 | 2026-10-26 | Mesh precision | Asked for by the owner before closing V2-0, so that G2 and G4 hold at their 1e-12 thresholds. v1's formulas (circumcenter from (b − a) × (c − a), area from a·(b × c), arc from atan2(|a × b|, a·b)) lose precision on small triangles: against a 40-digit reference with the points on the unit sphere (`tools/precision_check.py`), at n = 8 corners are off by 5.6e-14 rad, areas by 3.0e-12 and arcs by 1.1e-14 (relative). V2 now forms them from chords in a tangent frame at the first point, with radial parts from a·b − 1 = −|b − a|²/2 and arcs as 2 asin(|b − a|/2): 2.5e-16, 6.8e-16 and 2.7e-16. G2's cell-vs-kites check went from 1.4e-12 / 7.3e-12 / 2.7e-11 (n = 6, 7, 8, failing) to 1.7e-14 / 3.2e-14 / 7.2e-14. Against v1, G4 moved from 1.4e-14 / 7.0e-14 / 3.5e-13 / 1.41e-12 to 1.2e-14 / 5.2e-14 / 2.3e-13 / 1.29e-12 (n = 2–5). From v1's own positions at n = 5, v1's cell areas are off by 5.1e-13 and V2's by 4.6e-15, so the remaining G4 gap at n = 5 is v1's own rounding (≈ 1e-12 in the TRiSK weights), plus 8.9e-15 rad of position difference from the unconverged Lloyd iteration. K1 is unchanged (2.3e-15 after 1 step, 4.2e-14 after 960). G4 at n = 5 stays failing until the owner decides (decided in #16). |
| 16 | 2026-10-26 | G4 threshold | Agreed with the owner: G4's geometry bound scales with the grid level as 1e-12 · 4^(n−4), because v1's own small-triangle rounding (#15) grows about 4× per level. The bound stays 1e-12 at n = 4, where v1's tests ran; it is stricter below (6.3e-14 at n = 2, 2.5e-13 at n = 3) and 4e-12 at n = 5. Measured 1.2e-14 / 5.2e-14 / 2.3e-13 / 1.29e-12: all pass. G2 keeps its flat 1e-12, which it now meets at every level. |
| 17 | 2026-10-26 | Edge midpoint | Agreed with the owner: the Perot wind rebuild and the sampling of analytic winds onto edges use the Voronoi edge's midpoint `xm`. With v1's edge point `xe` the Perot error's maximum stayed near 2% at every level (rms halved); with `xm` the maximum falls ×2.1 and the rms ×3.9 per level (E1). `xe` remains the kites' corner and TRiSK's edge point, since the kites are the exact cell–triangle overlaps only with it. N1 moved from 1.99e-4 to 2.01e-4. |
| 18 | 2026-10-26 | Fused kernels, CUDA graph | Agreed with the owner. Kernels fused to 15 launches per RK3 step (from 27) and 5 transport passes (from 11), with bit-identical results; each world tick captured once as a CUDA graph and replayed with one launch. The unfused kernels stay as the test reference (E2). |
| 19 | 2026-10-26 | Cell order | Agreed with the owner: cells renumbered along a 3-D Hilbert curve by default (measured against Morton: equal at n = 8, faster at n = 6–7). Tests that compare with v1 map through `perm`. Each corner is now computed once and shared by its three cells: per-cell copies differed in the last bits, so an edge's length depended on which cell it was read from (E3 read 6e-15 at n = 4 before the change, 2.9e-16 after). |
| 20 | 2026-10-26 | Transport bug from v1 | Found by the renumbering: v1's 3rd-order flux, f (mean − a − b) + \|f\| (f ≥ 0 ? b − a : a − b), takes the curvature of the edge's eI cell for both flux signs, so for f < 0 it is downwind-biased and the result depends on how edges are oriented (FCT3 differed by 6e-3 after one step on the renumbered grid). Corrected to f (mean − a − b) + \|f\| (b − a) β, which uses the upwind cell's curvature and is orientation-independent (E4). β (the upwind weight) is a parameter, default 1 pending the owner's choice; measured FCT3 L2 on the one-revolution bell: β = 1 0.445 / 0.126 (n = 4 / 5), 0.5 0.357 / 0.076, 0.25 (MPAS's default) 0.305 / 0.054, 0 0.263 / 0.063. v1's mixed form gave 0.363 at n = 4; its partly downwind (anti-diffusive) half sharpened the bell. N6 with β = 1: ×2.06 (threshold ≥ 2×). K1 checks v1's form through the reference kernel on v1's numbering. |
| 21 | 2026-10-26 | Transport β | Agreed with the owner: the engine's default upwind weight is β = 0.25, MPAS's default (`transport.BETA`; a world can override it with `params['beta']`). N6 FCT3 L2 0.305, ×3.01 better than upwind (threshold ≥ 2×); at n = 5 the bell's L2 is 0.054 (β = 1: 0.126). E2 compares the fused and reference kernels at this β. |
| 22 | 2026-10-26 | Test method | Agreed with the owner, and to be kept for every later stage: least redundancy and run time with the most integrity (§7). The 45 results of V2-0/V2-0b became seven tests with every threshold kept: GRID = G1 + G2 + G3 + grid cache; V1-GRID = G4; V1-ENGINE = K1, now run on the engine as it runs (fused kernels, CUDA graph, Hilbert order); ACCURACY = N1 + N6 + E1; CONSERVE = C1 (world; the bell's budget is in N6); REPRO = D1 + S1 + spin-up cache + E2's graph check + E4, plus a new check that two identical layers stay bit-identical to each other and to one layer (replacing E2's multi-layer coverage); VIEWER = W1. Retired: F0 (every test needs the GPU; the runner prints the environment), E3 (implied by V1-ENGINE and REPRO's numbering check), E2's fused-against-reference comparison together with the reference kernels (`reference.py` and the one-loop-per-kernel kernels in `trisk.cu`; V1-ENGINE checks the engine against v1 directly), K1's check of v1's old 3rd-order form (no longer run, #20), R1 (the runner's summary line). PERF runs only with `--perf`, at every stage close. Suite 206 s → 9 s warm. |
