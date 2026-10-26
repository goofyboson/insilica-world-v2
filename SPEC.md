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
| `src/insilica/kernels/*.cu` | CUDA C kernels (ASCII only), compiled at run time by NVRTC through CuPy: `layers.cu` (the layered core), `fused.cu` (one-layer TRiSK and FCT transport), `trisk.cu` (the ∇⁴ operator, `mass_update`, `block_sum`) |
| `src/insilica/gpu.py` | kernel loading, `DeviceGrid`, deterministic sums `dsum` |
| `src/insilica/layers.py` | the layered core: stacked layers, time-split RK3 with the fast mode sub-stepped, floor, exchange hook (§3) |
| `src/insilica/swe.py`, `transport.py` | v1's one-layer shallow water (kept for V1-ENGINE) and FCT tracer transport (ports of v1 `makeSolver`, `transportStep`), fused kernels |
| `src/insilica/cases.py` | initial states: single-layer benchmarks and the layered cases, with their settings (`PRESETS`) |
| `src/insilica/world.py` | the world: a case on the layered core with tracers; stepping, fields, save/load (§4) |
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
- **Layered core** (`layers.Layers`, V2-1): L Boussinesq layers (k = 0 on top), thickness h_k and wind u_k, over
  topography b. Layer k feels the Montgomery potential M_k = g·η + M'_k, with η = b + Σh the free surface and
  M'_k = Σ_{m≤k} g'_{m−1} z_m (z_m the top of layer m, g'_{m−1} the reduced gravity across the interface above it),
  so each layer is v1's TRiSK shallow water with that potential (#24). Momentum also has drag r_k u and a stress
  term (c_k/h_k) u (#29) and ∇⁴ hyperviscosity ν4 = U·d³/π⁴, refused at start-up if ν4·λ²·dt > 1 (λ the vector
  Laplacian's largest eigenvalue, ≈ 27/d²; #25).
- **Time stepping**: time-split RK3 (Wicker & Skamarock 2002, as MPAS-Atmosphere; #28). In each RK3 stage the slow
  tendencies (everything but g·∇η) come from the stage's input state; their thickness-weighted column mean G drives
  the fast mode, which is sub-stepped from tⁿ by forward-backward steps at the true external speed √(gH)
  (τ·√(gH)/d_min ≤ 0.5), with divergence damping γ = 0.1 (the pressure gradient takes η + γ(η − η_prev)). Each layer
  then moves with its own baroclinic flux plus its share of the stage's mean barotropic transport, so the layers
  sum to the sub-stepped surface; η is never a separate variable.
- **Floor** (#26): a layer may send out, in any stage, only what it held above the floor (1 m) at the step's start;
  limiting fluxes adds no mass.
- **Exchange** (#27): `Layers.exchange` moves thickness across interfaces with its tracer and momentum (upwind),
  conserving every total; the hook convection, entrainment and overturning will use.
- A world tick (layer step + transport) allocates nothing and is captured once as a CUDA graph, then replayed with one
  launch per tick (#18).

## 4. The world

`World(n, preset, params, layers, state)`: a case from `cases.PRESETS` on the layered core, with state `h (L,N)`,
`u (L,E)`, `M (L,N)` (layer volume per cell, moved by the same fluxes as h) and `psi (L,N)` (v1 N6's cosine bell
in every layer). A tick is one layer step plus one FCT step with its fluxes. Presets:

| Preset | Layers | Step | What |
|---|---|---|---|
| `w2`, `w5`, `rh4`, `galewsky` | 1 | 450 s · 2^(4−n) (rh4 300 s) | Williamson 2, 5, 6 and the Galewsky jet |
| `phillips` | 2 | 1200 s · 2^(5−n) | a balanced jet at 45°N over a lower layer at rest, 3× the critical shear: baroclinic instability |
| `outcrop` | 2 | 1 h | two ocean layers flowing over a ridge where the lower layer thins to the floor; bottom stress 2.5e-4 m/s |

`params` overrides a case's settings: `dt`, `gprime`, `floor`, `drag`, `stress`, `nu4` or `visc4_U`, `rotation`,
`beta`, `div_damp`, `nfast` (fast sub-steps per step; by default 1.2 × what √(gH) of the deepest column needs, then
saved with the world). `layers=K` splits a one-layer case into K equal layers with g' = 0 (they move as one: REPRO).
Fields for the viewer: `h`, `z` (the layer's top), `eta`, `psi`, `speed`, `vort`, per layer.

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
| ACCURACY | through the layered core: N1 Williamson 2, 5 days at n = 4, 5, 6; W5, 15 days at n = 4; W6 Rossby–Haurwitz, 7 days at n = 4 (v1 N4's phase fit); Galewsky at n = 5 and 6 against an unperturbed twin (v1 N3's metric); Williamson 2 for 30 days at n = 5 (v1 N5); N6 cosine bell at n = 4; Perot rebuild of solid-body rotation at n = 4, 5, 6 | N1 < 1e-3 at n = 4, falls ≥ ×3 per level; W5 energy drift < 1e-4; W6 phase speed within ±10%; Galewsky day 6 > 0.3, growth > ×20, max |u| < 200; 30-day wind error < 1 m/s; N6 ≥ 2× better, min > −1e-9, drift < 1e-12; Perot max ≥ ×1.8, rms ≥ ×3.5 per level |
| WAVES | at n = 5 with long steps (fast mode sub-stepped): an external wave in air (300 m/s, 30-min steps) and ocean (200 m/s, 1-h steps), and a two-layer internal wave against the exact free-surface speed | external within 2%, internal within 3% |
| BAROCLINIC | Phillips two-layer instability at n = 6: a balanced jet at 45°N over a lower layer at rest, at 3× and 0.3× the critical shear 2βL_d², disturbance energy against an unperturbed twin from day 1 to day 20 | ≥ ×100 at 3×; ≤ ×2 at 0.3× |
| CONSERVE (n = 4, 6) | the outcrop case (15 days at n = 4, 1 day at n = 6): each layer's volume, mass and tracer, the thinnest layer; then an exchange of up to 40% of each donor: volume, tracer, momentum and M = h·A; and energy on the subcritical two-layer jet with no dissipation at all | ≤ 1e-13; never below the floor; exchange ≤ 1e-13; energy drift < 1e-4 |
| REPRO | repeat runs, graph against direct ticks and repeated tree sums; save → load → save bytes and save-at-T + continue against a straight run; spin-up cache miss then hit; a one-layer case split into two equal layers with g' = 0 against itself and the one-layer run; the Hilbert-ordered grid against v1's numbering after 960 steps | bit-identical; numbering ≤ 1e-12 |
| VIEWER | the protocol with a Python client: grid, exact f8 bytes, f4 bytes, step +1/+5, play pushes, pause holds, error reply, page and three.js served | all correct |
| PERF (`--perf`) | ns per layer-edge per slow step at n = 4–8 for L = 1, 2, 4; launch overhead; cost per model hour of whole world ticks for air (n = 6, `phillips`) and ocean (n = 8, `outcrop`) | n = 8 ≤ 16 ns; air ≤ 15 ms and ocean ≤ 180 ms per model hour (the plan's §5 estimates) |

**Results (2026-10-26, 22/22):** GRID topology exact at every level, worst geometry 1.6e-13 (orthogonality at
n = 8), area ratio 1.33 at n = 4, cache identical. V1-GRID 1.2e-14 / 5.3e-14 / 2.3e-13 / 1.29e-12 against
6.3e-14 / 2.5e-13 / 1e-12 / 4e-12. V1-ENGINE 2.3e-15 (1 step), 4.3e-14 (960 steps), mountain 4.4e-14, upwind
4.2e-15 / 2.1e-14. ACCURACY N1 2.01e-4; N6 FCT3 0.305, ×3.01; Perot ×2.16 / ×3.85 per level. CONSERVE ≤ 1.8e-16.
REPRO all bit-identical, numbering 2.3e-14. VIEWER correct. PERF n = 8 2.9 ns per layer-edge-step, two layers
11.4 ms per step; the graph ×2.0 at n = 4; Hilbert ×1.26 at n = 8. Suite 9 s warm, 71 s from an empty cache; PERF 52 s.

**V2-1 results (2026-10-26, 27/28 + PERF; in review):** ACCURACY N1 1.96e-4 at n = 4, falling ×3.79 then **×2.94**
per level (threshold ×3; see #30); W5 energy drift 9.4e-7; W6 phase speed −7.8%; Galewsky day 6 0.815 (×136, n = 5)
and 1.048 (×169, n = 6); 30-day wind error 0.074 m/s; N6 ×3.01; Perot ×2.16 / ×3.85. WAVES air 300.0 vs 300 m/s
(+0.01%, 8 fast sub-steps per 30-min step), ocean 200.9 vs 200 (+0.43%, 10 per hour), internal 35.14 vs 35.59
(−1.26%). BAROCLINIC ×1,710 at 3× critical shear, ×0.85 at 0.3×. CONSERVE per-layer budgets ≤ 2.4e-16, thinnest layer
exactly 1 m, exchange exact, energy drift with no dissipation 3.8e-10 (15 d). REPRO all bit-identical. PERF slow step
5.4 ns per layer-edge at n = 8; air n = 6 11.3 ms and ocean n = 8 78 ms per model hour. Suite 91 s warm (ACCURACY's
long runs are most of it).

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
| 23 | 2026-10-26 | Layer thickness | Agreed with the owner: physics alone controls the layers (heating and cooling across isentropes, convection, entrainment, Ekman pumping, overturning). The remap toward a target thickness (ALE), #3's fallback, is not built; it is to be recommended again only if a phenomenon cannot be made to emerge physically. With two layers per fluid a remap acts as artificial vertical mixing and would hide physics errors. |
| 24 | 2026-10-26 | Layered core | Boussinesq density layers with the Montgomery potential M_k = g·η + M'_k (§3); the potential is one function, so V2-2 can add isentropic air layers. |
| 25 | 2026-10-26 | Hyperviscosity | ν4 = U·d³/π⁴ (`visc4_U`, d the mean cell spacing): one setting serves every level. Measured: the vector Laplacian's largest eigenvalue is ≈ 27/d², not (π/d)², so the grid's shortest modes damp at ≈ 7.5·U/d; a world refuses ν4·λ²·dt > 1 at start-up (the Phillips case first had visc4_U = 50, unstable at 2.13; it uses 20, 0.85–0.98). |
| 26 | 2026-10-26 | Floor | 1 m, enforced by limiting each layer's outflow per stage to what it held above the floor at the step's start (no mass added). |
| 27 | 2026-10-26 | Exchange | A conservative inter-layer exchange of thickness with its tracer and momentum (upwind), for the vertical processes of V2-2 and V2-3. |
| 28 | 2026-10-26 | Fast mode | Agreed with the owner: time-split RK3 (MPAS-Atmosphere style), with MPAS-Ocean's barotropic averaging filter to be added only if the ocean case showed fast-mode noise. It did: a balanced two-layer ocean (n = 6, 1-h steps, 15 fast sub-steps) grew its free-surface error from 2.8e-6 m to 1.2e-2 m in 165 days, against 3.0e-5 m for 60-s steps without sub-stepping; at 15-min steps it blew up by day ~100. The averaging filter inside RK3 stages (sub-stepping over twice the stage interval and averaging, with the averaged transport for the layers too) made every two-layer run blow up within days. Divergence damping γ = 0.1 (Skamarock & Klemp 1992, the standard companion of time-split RK3 in MPAS-Atmosphere and WRF) fixed it: 3.0e-6 → 1.5e-5 m over 180 days at both 1-h and 15-min steps, matching the short-step reference. **Pending the owner's agreement** (the agreed remedy was the filter). |
| 29 | 2026-10-26 | Bottom stress | A layer thinning to the floor over a ridge ran away (its wind 16 → 111 m/s in three days, then NaN), with or without ∇⁴ and at shorter steps: a fixed-rate drag cannot hold a nearly massless layer. Bottom friction now acts as a stress, decelerating a layer as c/h (the physical form; c = 2.5e-4 m/s, ocean-typical, on the outcrop case's lower layer): 30 days stable, the thin layer's fastest wind under 4.3 m/s. CONSERVE's energy check moved to a case with no dissipation (the subcritical jet), since the outcrop case now dissipates by design. **Pending the owner's agreement.** |
| 30 | 2026-10-26 | Mesh smoothing (proposed) | ACCURACY N1's convergence (×2.94 from n = 5 to 6, threshold ×3) is limited by the mesh: with v1's rule the Lloyd iteration stops at 400 iterations, unconverged from n = 5 up (area max/min 1.42 / 1.54 / 1.69 / 1.79 at n = 5–8). Iterated to its 1e-11 tolerance, N1 falls ×3.80 then ×3.08 (the n = 6 mesh after 4,000 iterations). Measured cost to the tolerance, once per level (grids are cached): n = 6 5,412 iterations, 106 s; n = 7 19,497, 388 s; n = 8 not reached after 18,000 iterations (979 s, last move 5e-8), about 55,000 (~50 min) projected. Proposal: iterate to the tolerance at every level for the engine's grids, keeping v1's 400-iteration rule only for the grids V1-GRID and V1-ENGINE compare with v1. **Pending the owner's agreement.** |
