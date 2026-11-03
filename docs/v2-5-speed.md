# V2-5 speed ledger

Every speedup, measured with `tools/profile.py` on the RTX 4070, against the baseline below. "Identical" means the state
hash after a fixed run matches the baseline's bit for bit; anything that changes results says by how much.

## Baseline (2026-11-02, commit 19ee5b0, `earth` preset with its full physics)

| | n = 4 | n = 5 | n = 6 |
|---|---|---|---|
| Cells | 2,562 | 10,242 | 40,962 |
| Step | 1800 s | 900 s | 450 s |
| Per step, captured graph | 1.76 ms | 2.50 ms | 8.89 ms |
| Per model hour | 3.5 ms | 10.0 ms | 71.1 ms |
| Model years per day | 2,805 | 988 | 139 |
| Kernel launches per step | 135 | 119 | 111 |

Where n = 6's step goes (timed kernels 9.40 ms):

| Part | ms | Share | Largest kernels |
|---|---|---|---|
| Air column physics | 3.29 | 35% | `air_column` 3.27 ms, one launch |
| Air dynamics | 2.76 | 29% | `lay_slow` 0.79, `af_mu` 0.35, `air_bern` 0.32 |
| Transport (θ, vapor, cloud water) | 2.29 | 24% | `tr_stage` 0.94, `tr_ratios` 0.65, `tr_apply` 0.59 |
| Ocean dynamics | 0.69 | 7% | `bt_u`, `bt_eta` |
| Ocean transport, physics, land | 0.37 | 4% | |

At n = 4 the GPU is mostly idle: 135 small launches cost 1.76 ms with the graph and 5.31 ms without, so launches, not
arithmetic, set the pace. `air_column` costs 3.3 ms on `earth` but 1.0 ms on the plain aquaplanet: clouds, the full plume
and the layer capacities add about 2.3 ms, much of it 64-bit `pow`, `exp` and `log` (the 4070 runs 64-bit at 1/64 of
32-bit) recomputed for the same layer several times a step.

## Where the plan needs to be

| Configuration | Today (measured or estimated) | Needed |
|---|---|---|
| V2-8: air n = 6 with ocean n = 8, 3 layers each | about 190 ms per model hour, about 50 model years per day (air 71 measured; ocean n = 8 about 120 estimated from PERF's 2-layer 82 ms) | about 250 model years per day, so a 1,000-year deep-ocean spin-up takes days, not weeks |
| The same after V2-6's physics (bands, gases, more tracers; estimate ×1.5–2) | about 25–35 model years per day | the same 250 |
| Searches like V2-4's, at n = 4 | 2,805 model years per day, one world at a time | many worlds at once |

So V2-8 after V2-6 needs roughly ×7–10 overall. Not all of that can come without changing results; V2-5 measures how
much does.

## Targets

| # | Lever | Target | Changes results? |
|---|---|---|---|
| 1 | Column physics: compute each layer's pressures, exponents and capacity once per step, not three or four times | `air_column` ×2–3 (3.3 → ≤ 1.5 ms at n = 6) | No: identical |
| 2 | Transport: carry θ, vapor and cloud water through one set of launches | transport ×1.5–2 (2.3 → ≤ 1.3 ms) | No: identical |
| 3 | Dynamics: fuse small launches, cut `lay_slow`'s memory traffic | air dynamics ×1.3 | No: identical |
| 4 | Ensembles: many worlds per launch | n = 4: ≥ 10× worlds per second at 16 worlds | No: each member identical to its single run |
| 5 | Spin-up: deep ocean accelerated, spin-ups cached and reused | 1,000-year deep ocean in ≤ 1 day at V2-8 | Spin-up path only, measured against a plain spin-up |
| 6 | First 32-bit trial, column physics first (most transcendental), then the rest | measure: speed and the climate's change | Yes: measured against 64-bit, decision for V2-7 |

Overall V2-5 target: `earth` n = 6 at ≥ 2× (≥ 280 model years per day) with identical results from levers 1–3; ≥ 10× search
throughput at n = 4 from lever 4; levers 5–6 measured and reported.

## First 32-bit trial (lever 6, 2026-11-03)

The whole model in 32-bit by a trial shim (every CuPy float array float32, kernels compiled with `REAL=float`), measured
on an idle GPU against the same commit in 64-bit. The shim is not part of the engine; V2-7 decides what goes to 32-bit.

| `earth`, per step (captured graph) | n = 4 | n = 5 | n = 6 |
|---|---|---|---|
| 64-bit | 1.95 ms | 2.45 ms | 8.90 ms (139 model years per day) |
| 32-bit | 0.54 ms | 0.86 ms | 3.01 ms (409 model years per day) |
| Speedup | ×3.6 | ×2.8 | ×3.0 |

n = 6 in 32-bit by part: air dynamics 1.56 ms, transport 1.13, air column 0.73 (from 3.29: the transcendentals), ocean
dynamics 0.55. Ocean `lay_mask` (12 launches of almost no work) costs 0.20 ms either way: a fusion target, not a 32-bit
effect.

Climate, 16 years from the same start (64-bit → 32-bit): Ts 289.15 → 289.02 K, TOA +4.76 → +5.00 W/m², rain 2.88 → 2.88
mm/day (convective share 49 → 48%), vapor 34.4 → 34.2 kg/m², cloud cover 58 → 58%, cloud effect on sunlight −43.1 → −43.3
and longwave +28.8 → +28.5 W/m², albedo 0.230 → 0.230, rain by band 0–10° 4.88 → 4.78, 15–30° 1.91 → 1.92, 40–60° 3.02 →
3.01, layer masses 156/588/232 → 155/587/233 hPa, fastest wind 77 → 85 m/s. Regions agree within run-to-run noise except
the near-dry Amazon (0.36 → 1.28 mm/day).

Budgets, `earth` n = 4, 30 days after a 5-day settle, every step, sums in 64-bit (64-bit → 32-bit):

| Budget | 64-bit | 32-bit |
|---|---|---|
| Air water against evaporation − rain, worst step | 3.8e-16 | 6.4e-9 |
| The same over the run | −3.1e-16 | −5.0e-7 (−4e-7 mm/day) |
| Land water against rain − evaporation − rivers, over the run | −2.0e-15 | −2.7e-6 (−2.8e-4 mm/day global, ~0.1 m per 1,000 years) |
| Energy against TOA, worst step | 1.0e-10 | 2.1e-7 |
| Energy change − TOA, over the run | +0.10 W/m² (the air core's known #81) | −0.77 W/m² |

So 32-bit gives ×3 everywhere and leaves the 16-year climate nearly unchanged, but it loses energy at −0.77 W/m² (a
fifth of a CO₂ doubling's forcing) and land water at about 0.1 m per 1,000 years: too much for 1,000-year spin-ups and ice
sheets. V2-7's choice is therefore mixed precision, not plain 32-bit: keep the stored state and its ledgers in 64-bit
and do the arithmetic in 32-bit, then measure again against these numbers.

## Gains

| Date | Change | Commit | n = 6 per step | ×baseline | Identical? |
|---|---|---|---|---|---|
| 2026-11-02 | Baseline | 19ee5b0 | 8.89 ms | 1.00 | — |
| 2026-11-03 | Lever 1: column computed once where it repeated, compiled for its layer count (`air_column` 3.28 → 2.50 ms) | 24afdb8 | 8.19 ms | 1.09 | yes: 8 presets |
| 2026-11-03 | Mixed precision (SPEC #106): column transcendentals and capacity in 32-bit (2.50 → 1.08 ms), transport stencil in 32-bit (2.30 → 1.69 ms) | f141497 | 6.09 ms | 1.46 | no: measured below; 64-bit reference still identical |
| 2026-11-03 | Lever 2: every tracer in the same launches (rows tracer × layer), each 3rd-order edge flux once per edge (was 8 evaluations per edge, now 3) (transport 1.69 → 0.95 ms) | 7b5ca7d | 5.37 ms | 1.66 | yes: 8 presets, both precisions |
| 2026-11-03 | Lever 3: the fast sub-step's mass and linear-potential passes in one kernel (`af_mu` + `af_phi` → `af_muphi`, 8 launches fewer per step) | 65f5a32 | 5.28 ms | 1.68 | yes: 8 presets, both precisions |

### Mixed precision (2026-11-03, SPEC #106)

`INSILICA_PRECISION` = `mixed` (the default) or `double` (the 64-bit reference, bit-identical to before: 8 presets).
The state, every sum and the energy and water ledgers stay 64-bit. 32-bit: the column's `exp`, `pow`, `log` (`1 − e^−x` as
`−expm1f(−x)`, accurate for small x), its capacity and condensation arithmetic, the edge-shape solve (32-bit Newton on
|s| ≤ 30, its series near 0, the exact asymptotes beyond), and the transport stencil's weights and curvature sums. Kept 64-bit
after measuring or on reasoning: `layer_exner` (its p x − pt xt cancels for thin layers), pressure gradients, and the PV flux
sum (tried: 0.79 → 0.77 ms, memory-bound, reverted).

| `earth`, per step | n = 4 | n = 5 | n = 6 |
|---|---|---|---|
| 64-bit, after lever 1 | 1.95 ms | 2.45 ms | 8.19 ms |
| Mixed | 1.32 ms (3,750 model years per day) | 1.87 ms (1,319) | 6.09 ms (203) |

Budgets (`earth` n = 4, 30 days, as for the 32-bit trial; 64-bit → mixed): air water worst step 3.8e-16 → 5.1e-16; land water
over the run −2.0e-15 → −2.1e-15; energy worst step 1.0e-10 → 1.0e-10; energy change − TOA +0.103 → +0.104 W/m² (the air
core's #81 in both). Plain 32-bit gave 6.4e-9, −2.7e-6 and −0.77 W/m². CONSERVE (n = 4 and 5, its limits unchanged): passed.

Climate, 16 years (64-bit → mixed): Ts 289.15 → 289.14 K, TOA +4.76 → +4.68 W/m², rain 2.88 → 2.89 mm/day (convective 49 →
49%), vapor 34.4 → 34.4 kg/m², cover 58 → 58%, cloud effect on sunlight −43.1 → −42.9 and longwave +28.8 → +28.5 W/m², albedo
0.230 → 0.229, rain by band 0–10° 4.88 → 4.90, 15–30° 1.91 → 1.87, 40–60° 3.02 → 3.03, layer masses 156/588/232 → 156/587/233
hPa, fastest wind 77 → 73 m/s; Amazon 0.36 → 0.25, E USA 4.63 → 4.78 mm/day (run-to-run noise in small regions).

### Lever 1 notes (2026-11-03)

Identical by state hash after 3 days at n = 4 of `earth`, `earth` with `bl_mix`, `aqua`, `aqua5`, `continent`,
`held_suarez` (2 layers), `rce` and `aqua_dry`. Changes, each reusing a value computed again from the same inputs by the
same expression (`--fmad=false` keeps rewritten code's rounding the same):

| Change | `air_column` at n = 6 |
|---|---|
| Before | 3.28 ms |
| Capacity nodes' (p/p0)^κ once per step (were 3× per layer); plume's Exner from its parcel, its entrainment and expm1(s) per layer, not per sub-level; crossing's edge shape once per layer (was up to 3×); Tsa's power from `layer_exner` | 2.61 ms |
| Compiled for the world's layer count (`-DNL=L`): local memory 3,088 → 1,344 bytes per thread | 2.50 ms |

Short of the ≤ 1.5 ms target. What is left is 64-bit arithmetic, not memory: 254 registers, and the 4070 has 2 FP64 units
per SM. Switching pieces off (timing only) shows where it goes: clouds about 0.95 ms (about ten layer-capacity evaluations
per column, four `exp` each), the deep plume 0.43 ms, the middle layer's spread capacity 0.3 ms. Skipping a repeat in only
some columns gains nothing (a warp waits for its slowest column), so the next gains change results: fewer or cheaper
capacity evaluations, or 32-bit transcendentals (the whole column was 0.68 ms in the 32-bit trial). Those belong with V2-7's
mixed-precision decision.

### Levers 2–4 notes (2026-11-03)

**Suite at lever 2 (7b5ca7d, mixed precision):** 41 of 42 with PERF, ICE the known failure (a started polar sheet now
−148 kg/m² a year against −183 in 64-bit; the limit is 50). PERF: air n = 6 29.3 ms per model hour (337 model years per
day) against 39.6 at the stage's start.

**Lever 3** stopped at one fusion: what is left in air dynamics is memory traffic, not launches or 64-bit arithmetic.
`lay_slow` (0.77 ms, the PV flux's ten-neighbour gathers) was tried in 32-bit (0.79 → 0.77 ms, reverted); the other kernels
are under 0.32 ms each.

**Lever 4, ensembles** (`insilica.ensemble.Ensemble`): each member keeps its own stream and captured graph, and the ensemble
queues every member's tick before waiting. Members are identical to single runs (REPRO now checks three).

| `earth` n = 4 | Per member step | Model years per day, all members |
|---|---|---|
| 1 world | 1.10 ms | 4,480 |
| 4 | 0.47 ms | 10,563 |
| 8 | 0.39 ms | 12,504 |
| 16 | 0.38 ms | 12,871 |
| 32 | 0.38 ms | 12,936 |

×2.9 at 8 or more members, short of the ×10 target: the target assumed one n = 4 world left the GPU about 90% idle, but
it leaves it about two-thirds idle. At 8 members the GPU is full: 0.38 ms per member is the same work per cell as n = 6
(5.28 ms for 16× the cells). Capturing a day of ticks per graph (fewer host launches) changed nothing and was removed.

### Lever 5: deep-ocean spin-up by asynchronous coupling (2026-11-03, SPEC #107)

`insilica.spinup.spin_up(world, cycles, ocean_years)`: each cycle a coupled year with a `Recorder`, then `ocean_years` of the
ocean alone (`OceanOnly`) on that year's forcing, one slot a day, with the surface heat's feedback λ = 10 W/m²/K and the
year's net fresh water balanced. Normal runs are unchanged (state hashes identical): the recording is off unless attached.

A replayed year against the coupled year it was recorded from (`earth` n = 4, after 1 year):

| | Coupled | Ocean alone, 24 slots | Ocean alone, daily slots |
|---|---|---|---|
| Mixed layer depth | 75.6 m | 108.1 m | 80.8 m |
| Heat gained over the sea | +8.59 W/m² | +4.55 W/m² | +7.97 W/m² |
| SST against coupled | — | +0.24 K mean, 0.65 K rms | +0.02 K mean, 0.24 K rms |
| Thermocline change | +0.179 K | +0.093 K | +0.169 K |

Spin-up from the `earth` start, n = 4 (layer means: mixed layer / thermocline):

| Run | Year 50 | Cost per model year (alone on the GPU) |
|---|---|---|
| Plain, coupled | 19.83 / 17.06 °C, S 34.75 / 34.90 | 20 s |
| 1 + 9, λ = 40 (fresh water balanced) | 18.61 / 16.24 °C | 5.4 s (×3.7) |
| 1 + 4, λ = 40 | 19.19 / 16.65 °C | 7.0 s (×2.8) |
| 1 + 9, λ = 10 | 19.29 / 16.81 °C, S 34.86 / 34.98 | 5.4 s (×3.7) |

With λ = 10 the thermocline is 0.25 K behind plain's +2.9 K at year 50: about ×3.4 toward equilibrium. Without the
fresh-water balance (λ = 40, 100 years) the thermocline freshened to 34.57 psu against plain's 34.89.

**For V2-8:** the ocean alone, `earth` sea floor and land, 3 layers: n = 6 4.6 ms per 1-h step (2,167 model years per day),
n = 8 147.5 ms (67 model years per day; 102 fast sub-steps an hour against 24). The 1,000-year target at V2-8 is met by
spinning the deep ocean up at n = 6 (asynchronously, about half a day) and moving it to n = 8 for the last decades; the
remapping is V2-8's work. The n = 8 barotropic sub-steps are the next ocean speed target.

## Close (2026-11-03): 42/43 with PERF, ICE the known failure

| Target | Result |
|---|---|
| `earth` n = 6 ≥ 2× (≥ 280 model years per day) | ×1.68: 8.89 → 5.28 ms per step, 139 → 234 model years per day (levers 1–3 identical, mixed precision measured). **Not met**: what is left is the dynamics' memory traffic. |
| `air_column` 3.3 → ≤ 1.5 ms | 1.08 ms (2.50 identical, the rest mixed precision) |
| Transport 2.3 → ≤ 1.3 ms | 0.95 ms (identical) |
| Air dynamics ×1.3 | ×1.05 (2.77 → 2.63 ms). **Not met** |
| Ensembles ≥ 10× at n = 4 | ×2.9 at 8+ members, the GPU full. **Not met**: the target overestimated the idle GPU |
| 1,000-year deep ocean ≤ 1 day at V2-8 | Met on the n = 6 route: asynchronous coupling (×3.4 at n = 4) and the ocean alone at 2,167 model years per day at n = 6; n = 8 alone is 67 |
| 32-bit measured | Plain 32-bit ×3 but leaks energy (−0.77 W/m²); mixed precision closes the books as 64-bit |

PERF: air n = 6 29.4 ms per model hour (335 model years per day) against 39.6 at the stage's start; coupled n = 6 33.5 ms,
land n = 6 34.0 ms. SPINUP (new): a replayed year's SST rms 0.29 K against the coupled year, heat uptake +8.35 against +8.59
W/m², salt 3.7e-15. REPRO checks three ensemble members.
