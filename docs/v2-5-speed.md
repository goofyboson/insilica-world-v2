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

## Gains

| Date | Change | Commit | n = 6 per step | ×baseline | Identical? |
|---|---|---|---|---|---|
| 2026-11-02 | Baseline | 19ee5b0 | 8.89 ms | 1.00 | — |
