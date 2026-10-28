# InSilica World V2: layered ocean and atmosphere on one GPU fluid core (plan)

Status: plan, 2026-10-26. Nothing here is built. Decided so far (§9): this repository sits next to v1
(`../insilica-world`), v1 is paused as is, the engine runs as CUDA kernels in 64-bit driven from Python through
CuPy with the browser as viewer, and layer thickness is set by physics rather than nudged (§3). Open decisions are
listed at the end. v1 is the reference: its SPEC, code and tests are read-only background for this project.

## 1. Goal

One fluid engine runs every layer of both fluids:

- **Two ocean layers and two air layers**, all fully moving.
- **Variable layer thickness** in each, so the ocean thermocline can rise and sink (upwelling, Kelvin waves, cold tongues) and the air layers can tilt (fronts, the sloping interface that drives storms).
- **The same equations in air and water**, with different substrate properties: density and stratification, equation of state, friction, heat capacity.
- **Fast waves at their real speeds**: no reduced external gravity (v1's `G_EXT` / `G_RED` trick). They are resolved by shorter time steps on the fast part only.
- **A mesh fine enough for ocean eddies** where we want them, while tests stay quick at low resolution.
- **64-bit throughout**, so conservation stays exact the way it is in v1.

## 2. Why this is a new major version

| v1 | V2 |
|---|---|
| JavaScript, CPU, single thread, single HTML file | Python engine with CUDA kernels on the GPU (64-bit); the browser page is a viewer connected to the local engine |
| Air: two layers, each always half the column (σ-style) | Layers with their own thickness, moved by the flow, then gently remapped toward a target per fluid (ALE) |
| Fast mode slowed to ~100 m/s by reduced gravity | Real speeds (air ~300 m/s, ocean ~200 m/s), sub-cycled |
| Ocean: a 100 m slab plus a resting deep reservoir | Two moving ocean layers, thermocline = interface |
| n = 3–5 (960–240 km cells) | Air n = 6 (120 km); ocean n = 8 (30 km, eddy-permitting); tests at n = 4–5 |
| Exact 64-bit ledgers (1e-15) | Same: 64-bit on the GPU |

That changes the language, grid sizes, state layout, time stepping, every dynamics kernel and the test harness, so a clean start with its own spec and stage series is simpler than migrating v1 stage by stage. v1's physics (radiation, water cycle, ice, terrain, the tides plan) is ported, not reinvented, and v1 stays as the working, portable reference.

## 3. One fluid core, two substrates

Each fluid is a stack of layers k = 1..L (L = 2 for both). Per layer: mass per area μ_k (thickness), normal velocity u_k on edges (TRiSK C-grid, as v1's Stage 3c-1), temperature/heat and tracers (vapor in air; salinity optional later) in cells.

**Equations (same for both fluids):**
- Mass: ∂μ_k/∂t = −∇·(μ_k u_k) + vertical exchange (convection, entrainment, remapping).
- Momentum (vector-invariant, energy-conserving TRiSK form): ∂u_k/∂t = q_k F_k^⊥ − ∇(M_k + K_k) − drag_k − ∇⁴ hyperviscosity + vertical momentum exchange.
- M_k is the layer's **Montgomery potential** (pressure plus height energy), from hydrostatics with the fluid's equation of state. This is the only place air and water differ dynamically.
- Tracers: 3rd-order FCT transport (v1 Stage 3c-1).

**Substrate properties:**

| | Ocean | Air |
|---|---|---|
| Equation of state | linear ρ(T) (Boussinesq), optional salinity later | ideal gas; layers carry potential temperature |
| Stratification | interface density jump Δρ/ρ ~ 0.002 → internal gravity ~0.02 m/s² | from layer θ difference, internal gravity ~0.2–0.5 m/s² |
| Wave speeds | external ~200 m/s (4 km deep), internal 1–3 m/s | external (Lamb) ~300 m/s, internal 30–60 m/s |
| Deformation radius | ~30 km mid-latitude, ~200 km equator | ~1,000 km |
| Friction | bottom drag on the lower layer; wind stress on the upper | surface drag on the lower layer (wind stress, equal and opposite, goes into the ocean) |
| Heat capacity | water (4.2 MJ/m³K) | air |
| Vertical exchange | entrainment at the thermocline, convective overturning when the upper layer is denser | moist convection (v1 3c), radiative cooling aloft |

**Variable thickness and the remap (ALE).** Each step moves layers with the flow (Lagrangian); a remap then nudges the interface toward a target coordinate with a timescale τ_remap:
- Air: target = a mass fraction (σ-like) with a long τ (days), so storms and fronts can tilt the interface but the upper layer can't drain (the failure of v1's first 3c attempt). τ → 0 recovers v1's σ behavior.
- Ocean: isopycnal (no remap) with a minimum thickness, so the thermocline moves freely; where the upper layer outcrops (thins to the minimum, e.g. strong upwelling or polar winter), the remap and entrainment take over.
Remapping moves mass, heat, tracers and momentum between layers conservatively.

**Decided (2026-10-26): thickness set by physics, with a floor, instead of a nudge.** The nudge above stays only
as a fallback setting (off by default, τ = ∞), used if a stage's tests show it is needed.
- **Ocean:** the interface is free. Ekman pumping moves the thermocline; wind stirring and winter convective
  cooling deepen the upper layer by entraining deep water; where upwelling thins the upper layer toward a minimum
  thickness, entrainment strengthens and brings deep water up (cold tongues, coastal upwelling).
- **Air:** the interface is an isentrope (a fixed potential temperature θ*). Air crosses it only when heated or
  cooled: convection lifts lower air past θ*, radiative cooling aloft (~1–2 K/day) brings upper air below it; the
  cross-interface mass flux is about mass × heating rate ÷ the layer's θ range, with no tuning constant. The layer
  thicknesses (and the Hadley circulation) then come out of radiative–convective balance. v1's first 3c attempt
  drained its upper layer because its return flow was an invented 5–10-day relaxation.
- **Near the ground (air):** where surface air is warmer than θ* (tropical summer, hot deserts) a purely isentropic
  lower layer would vanish ("intersect the ground"). The standard fix is a hybrid: the lower layer follows the
  terrain (σ-like, as v1) and the interface above it is isentropic (as NOAA's icosahedral FIM model's hybrid σ–θ
  coordinate). A minimum thickness is the safety floor in both fluids.
- **Test early:** V2-2 starts with a two-layer radiative–convective equilibrium run, alone, before any weather.

## 4. Time stepping: real fast waves, sub-cycled

Physics (radiation, condensation, surface fluxes, ice) stays on the 1-hour tick. Inside it, each fluid is **split-explicit** (as MPAS and ROMS do): the slow (baroclinic) part takes the largest stable step; the fast external mode, a 2-D problem, is sub-cycled with short steps at its true speed. Nothing is slowed artificially.

Approximate steps (RK3, Courant ≈ 0.8):

| Grid | Spacing | Air slow step (jets 80 + internal 50 m/s) | Air fast sub-steps | Ocean slow step | Ocean fast sub-steps |
|---|---|---|---|---|---|
| n = 4 | 480 km | 1 h | ~2 per step | 1 h | ~4 / h |
| n = 5 | 240 km | 30 min | ~2 | 1 h | ~8 / h |
| n = 6 | 120 km | 12 min | ~2–3 | 1 h | ~15 / h |
| n = 7 | 60 km | 6 min | ~2–3 | 1 h | ~30 / h |
| n = 8 | 30 km | 3 min | ~2–3 | 1 h | ~60 / h |

The ocean is cheap per hour (slow currents and internal waves); the air is the expensive fluid at high resolution.

## 5. Resolution and cost

| Grid | Cells | Spacing | Resolves |
|---|---|---|---|
| n = 4 | 2,562 | 480 km | coarse storms (tests) |
| n = 5 | 10,242 | 240 km | atmospheric storms (B4/B5 pass in v1) |
| n = 6 | 40,962 | 120 km | storms well; equatorial ocean waves |
| n = 7 | 163,842 | 60 km | ocean eddies at low latitudes |
| n = 8 | 655,362 | 30 km | **eddy-permitting ocean** at mid-latitudes |
| n = 9 | 2,621,442 | 15 km | eddy-resolving (likely beyond 12 GB with two fluids) |

**The 4070 in 64-bit.** Consumer NVIDIA cards run 64-bit arithmetic at 1/64 of 32-bit (~0.45 TFLOPS on the 4070), but the fluid kernels are limited by memory traffic (504 GB/s; well under one 64-bit operation per byte moved), so 64-bit mainly doubles the bytes moved. Expected: 2–4× slower than 32-bit would be, still ~10× faster than a 6-core CPU. V2-0 measures this.

Rough cost per model hour, from v1's measured 0.22 µs per layer-edge per RK3 step (JavaScript, one thread) and an assumed 4–8 ns per layer-edge-step on the 4070 in 64-bit at large sizes:

| Configuration | CPU, one thread | 4070, 64-bit (estimate) | Model year (estimate) |
|---|---|---|---|
| Air n = 6 | ~0.4 s | 8–15 ms | ~2 min of dynamics |
| Air n = 8 | ~25 s | 0.4–0.9 s | 1–2 h |
| Ocean n = 8 | ~5 s | 90–180 ms | 15–25 min |
| **Air n = 6 + ocean n = 8** (recommended showcase) | impractical | ~100–200 ms plus physics | ~15–30 min |
| Both n = 8 | impractical | 0.5–1 s | 1.5–2.5 h |

So: an eddy-permitting ocean needs the GPU, and pairing a coarser atmosphere (storms only need ~120 km) with a fine ocean is about 5× cheaper than running both fine. The two meshes are coupled by precomputed conservative remap weights (icosahedral levels nest closely, so each n = 6 cell overlaps ~16 n = 8 cells). Variable-resolution SCVT (fine where we watch, coarse elsewhere, as MPAS meshes do) is a later option on the same machinery.

## 6. Architecture (decided: CUDA via CuPy, 64-bit)

**Engine:** a Python package. The world's state lives in GPU memory as 64-bit CuPy arrays; every per-cell and per-edge loop is a CUDA C kernel compiled at run time by NVRTC (`cupy.RawKernel` / `RawModule`), so no Visual Studio or CMake is needed. Python orchestrates the tick: kernel launches, sub-cycling, physics order, save/load. CUDA's 64-bit math library covers what the physics needs (`exp`, `log`, `pow`, `sin`, …).
- Why CUDA rather than Vulkan/wgpu: complete 64-bit math (Vulkan's standard math set has no 64-bit exp/log/pow/sin), the best profiler (Nsight), and the most mature 64-bit path on NVIDIA hardware.
- Why CuPy rather than a C++ host: same kernel speed, no MSVC toolchain, and scans, analysis and tests become Python. The CUDA C kernels move unchanged to a C++ host later if ever needed.
- Kernel launch overhead (~10–20 µs from Python) matters at n = 4 (tests) more than at n = 6–8; fused kernels and CUDA graphs are available if it shows up in profiles.

**Precision and conservation:** 64-bit everywhere, so v1's exact ledgers carry over unchanged (energy, water and air totals to ~1e-15). No integer-quantum bookkeeping and no separate CPU reference are needed. Global sums use fixed-order tree reductions (no 64-bit atomic adds), so runs are bit-identical on the same machine and driver.

**Viewer:** the browser page (three.js, as in v1) connects to the engine over a local WebSocket and draws snapshots it requests (a chosen set of fields, at a chosen rate; binary frames, ~2–15 MB per full snapshot at n = 6–8). Controls (play, speed, sliders, new world) go back over the same connection. The engine starts with one command; then the page is opened. v1 remains the portable single-file version; a 32-bit browser build of V2 could come later.

**Tests and tooling:** pytest on the real GPU kernels. v1's parallel runner and spin-up cache carry over (save/load is part of V2-0). Numerical-method tests (Williamson suite etc.) run at n = 4–6 on the GPU directly.

**Environment:** RTX 4070 (12 GB, compute capability 8.9), driver 610.88, CUDA toolkits 12.6 and 13.2 installed. Python 3.14 is installed; if CuPy wheels don't support it yet, a Python 3.12/3.13 is installed alongside via the `py` launcher (V2-0 checks).

## 7. Stage series (V2)

Each stage keeps v1's rules: restate tests and plan, pause for go, commit as we go, record changes in the spec.

| Stage | Build | Key tests |
|---|---|---|
| **V2-0 Foundation** | Repo, Python package, CuPy + NVRTC kernels, grid (port v1's SCVT and TRiSK geometry), save/load, WebSocket viewer skeleton, parallel test runner, 64-bit performance probe on the 4070 | Measured ns per layer-edge (64-bit); a ported v1 kernel matches v1's JavaScript result; exact conservation of a transported tracer; save/load bit-identical continuation |
| **V2-1 Fluid core** | Multi-layer TRiSK: variable thickness, ALE remap, split-explicit fast mode at true speeds, FCT transport | Williamson 1, 2, 5, 6 + Galewsky (v1's N1–N6) at n = 5–7; Phillips two-layer baroclinic instability; layer vanishing/outcrop handling; real-speed external waves (Lamb ~300 m/s) |
| **V2-2 Atmosphere** | Air layers with θ (two in V2-2a; three in V2-2b, isentropic interfaces at 305 and 380 K, SPEC #45), port v1 radiation, convection, condensation, surface exchange | Two-layer radiative–convective equilibrium alone first; Held–Suarez dry benchmark (the standard test for atmosphere engines); v1 B1–B10 and climate tests T1.11/T1.12 equivalents |
| **V2-3 Ocean** ✅ | Three fixed-density ocean layers (decided in stage, SPEC #64, #71), wind stress, heat and fresh water (salinity, #76), sea ice, entrainment; aquaplanet first, then one walled basin (#79) | Ocean Kelvin wave at c₁ (#78); ocean heat transport, thermocline tilt and east–west SST contrast in the basin (#80, replacing the equatorial cold tongue, which does not form with these trades); coupled budgets (#81); ocean baroclinic eddies at n = 8 moved to V2-5 |
| **V2-4 Land and coasts** | Land masks, no-flow coastlines, v1 surface water, soil, ice sheets, terrain | Western intensification in a closed basin (gyres, boundary currents); rain shadow (v1 T3.6) |
| **V2-5 Two resolutions** | Air n = 6 / ocean n = 8 with conservative remap coupling | Coupled conservation exact; speed target ~5–10 ticks/s |
| **V2-6 Ports and polish** | Moon and tides, geology (v1 4b/4c plans), UI | v1's tidal tests T1–T6; viewer controls |
| **V2-7 Optimization and speedup** | Once the full air–ocean–land model exists, with the 64-bit model as the reference truth (§7.1): ensembles in one launch, 32-bit where safe, mixed time steps, faster spin-up, step engineering; then approximations (a reduced-complexity companion, a learned emulator) | Every climate and weather test passes within agreed tolerances of the 64-bit reference; speed per model year and per ensemble member against targets set at planning (~100× from speedups, more from approximations) |

### 7.1 Speed: the method as we go, and the V2-7 stage

**As we go (owner's rule, 2026-10-27):** take easy speedups now that don't sacrifice any of the phenomena the model is
meant to show — kernel fusion, CUDA graphs, fewer host round-trips, cached and restartable spin-ups, cheaper
equivalent arithmetic (one pow per interface, #42), measured with PERF broken out by component. Defer anything that
changes results (32-bit, mixed time steps, approximations) to V2-7, so the accuracy work is done once on the finished
model rather than repeated every stage. Keep the code ready for V2-7: column-local physics with no hidden host
round-trips (batchable), every kernel behind the `REAL` precision switch, and equilibrated states kept for restarts.

**Where the time goes (2026-10-27).** n = 4 moist three-layer air: ~2 min per model year per process, the GPU mostly
idle (2,562 cells; kernel launch overhead dominates). n = 6: 43 ms per model hour (~6 min per model year). The largest
wait in practice is spin-up: a 100 m slab takes about a decade of model time to settle.

**V2-7 levers, in order of payoff (estimates):**

1. **Ensembles in one launch** (~10–100× throughput at n = 4): many worlds (calibration members, sweeps, spin-ups) in
   the same kernels, since the GPU is idle at small grids. Same per-world speed, far more worlds.
2. **32-bit where safe** (~5–20× on compute-bound parts at n ≥ 6): the 4070 runs 64-bit at 1/64 rate, and the physics'
   pow/exp/saturation integrals are bound by it. Ledgers that need 1e-15 closure stay 64-bit (or compensated sums);
   an accuracy study against the 64-bit model decides each component.
3. **Mixed time steps** (~2–5×): radiation every 1–3 h, slab and ocean on longer steps than the air, only the dynamics
   at the short step.
4. **Faster spin-up** (~5–20× on time to equilibrium): spin up at n = 3–4 and interpolate up; accelerate the slow
   components during spin-up only (e.g. reduced slab or ocean heat capacity); or solve for the equilibrium directly.
   None of these touch the measured years.
5. **Step engineering** (~1.5–3×): fuse the column physics with neighbouring kernels, many ticks per graph launch, no
   host synchronization inside loops.
6. **Two resolutions** (V2-5): the fine grid only where it is needed.
7. **CPU and GPU together** (owner's question, 2026-10-27): not by splitting one tick (the physics needs the dynamics'
   output and vice versa, so overlap needs the physics lagged a step, and hides at most its ~25% share: ≤ ~1.3×; six
   Ryzen cores and the 4070 have similar 64-bit peaks, ~0.4 TFLOPS, and the state crosses PCIe in ~0.1 ms at n = 6).
   Better: the CPU runs independent work beside the GPU (small-grid ensemble members, spin-ups, sweeps), the slow
   components on long steps (land, ice sheets, geology, perhaps the deep ocean) coupled every few hours or days, and
   everything around the model (viewer, diagnostics, saves) off the GPU's critical path with asynchronous copies.

Together ~100× for climate-length runs at n = 4–5. **1000× and beyond needs approximation:** a reduced-complexity
companion (energy balance with parameterized transports) calibrated against the full model for quick what-ifs, and a
learned emulator trained on the full model's output (NeuralGCM / ACE style; 100–1000×+ within the climates it was
trained on). The full model stays the reference truth.

## 8. Risks

- **Python 3.14 and CuPy wheels**: V2-0 checks first; a side-by-side Python 3.12/3.13 is the fallback.
- **64-bit speed on the 4070** is an estimate until V2-0 measures it; if it is closer to 1/64 than to 1/2–1/4 of 32-bit, the n = 8 ocean slows and the showcase may drop to n = 7.
- **Variable-thickness air** drained the upper layer in v1's first 3c attempt; the remap with a target and minimum thickness is designed against that, and V2-1 tests it in isolation.
- **Launch overhead at small grids** could make the n = 4 test suite slower than expected; mitigations are kernel fusion and CUDA graphs.
- **Scale of work:** V2-0 through V2-3 is roughly the size of v1's Stages 1–3c. The showcase (eddying ocean) arrives at V2-3/V2-5.
- **Portability:** V2 runs where an NVIDIA GPU and CUDA are present; v1 stays the portable version.

## 9. Decisions

Decided (2026-10-26):
- **Engine:** CUDA kernels in 64-bit, driven from Python via CuPy; the browser is the viewer.
- **Where:** a new repository `insilica-world-v2`, next to v1 (`../insilica-world`), not inside it. v1 is attached
  to V2 sessions as a read-only reference.
- **v1:** paused as is. Stages 1–3c-1 complete on v1's `main` (v0.3c1); Stage 3c built but unclosed on branch
  `stage-3c` (52/56; T3.6 rain shadow and B10 moving tropical rain open). v1 later stages (4a ocean, 4b tides,
  4c geology, 5 save/load and polish) move to V2.
- **Layer thickness:** set by physics with a minimum-thickness floor (§3): free ocean interface with entrainment;
  isentropic air interface crossed only by heating and cooling, hybrid with a terrain-following lower layer near the
  ground. The remap-toward-target nudge is a fallback setting, off by default.

Open (to settle when V2-0/V2-1 are planned):
1. **Fast modes:** split-explicit at true speeds (recommended), or semi-implicit (one global elliptic solve per step; larger steps but more complex).
2. **Resolution targets:** air n = 6 + ocean n = 8 for the showcase, tests at n = 4–5 (recommended); or a fine region (variable resolution) instead of a whole-planet eddying ocean.
