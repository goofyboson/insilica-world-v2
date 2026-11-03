# InSilica World V2

A planet simulator with two moving ocean layers and three moving air layers, run by one 64-bit GPU fluid engine
(CUDA kernels driven from Python through CuPy). A browser page connects to the local engine as the viewer.

- **Plan:** [docs/v2-plan.md](docs/v2-plan.md) — goals, physics, time stepping, resolution and cost, architecture,
  stages, risks and decisions.
- **Spec:** [SPEC.md](SPEC.md) — code layout, numerics, viewer protocol, save format, every stage's tests and
  thresholds, and the decision log.
- **v1:** [goofyboson/insilica-world](https://github.com/goofyboson/insilica-world) — the portable single-file
  JavaScript version (Stages 1–3c), kept as the reference.

## Current status

**Now: V2-5 Speed (in progress, docs/v2-5-speed.md).** `earth` n = 6 per step 8.89 → 5.28 ms (139 → 234 model years per
day), n = 4 1.76 → 1.10 ms; ensembles of 8+ worlds ×2.9 at n = 4. Mixed precision by default (SPEC #106: 64-bit state and
ledgers, 32-bit column math and transport stencil; the 64-bit reference stays bit-identical), its 16-year climate within
noise of 64-bit and its budgets closing as 64-bit's. Suite 41/42 with PERF (ICE open). Next: deep-ocean spin-up.

**V2-4b Rivers, lakes and ice closed 2026-11-01 (42/42 with PERF).** Rivers flow at the same speed on any grid (0.998–1.022 m/s against 1 m/s,
n = 4–8; SPEC #85); ice sheets follow Glen's law, within 3.8% of the exact dome height at n = 6 (#86). Snow, glacier ice and sea ice have their own albedos, and a started polar sheet sustains itself (#87). Not yet by themselves: no lake forms on `relief`, and no ice sheet grows there from nothing. Earlier:
**V2-4a Land closed 2026-10-31 (40/40 with PERF).**
Land of any shape over the coupled ocean: ground heat, soil bucket, snow, runoff to the coast and terrain under the air. An
ocean gyre intensifies in the west (×4.2, interior within 14% of Sverdrup), land's 45–65°N seasons reach 36 K against 8 K at
sea, and every land budget closes. Land at n = 6 costs 44.7 ms per model hour. **Two known shortfalls stay open** (SPEC #84):
the rain shadow is backwards (×0.57 against a target above 2×), and equatorial land is nearly dry (about 0.4 mm/day against
2.8 at sea), not fixed by albedo, height or two changes to convection. Earlier:
**V2-3 Ocean closed 2026-10-28 (36/36 with PERF).** Three ocean layers
(mixed layer, thermocline, abyss) with fixed density steps, temperature and salinity under the three air layers:
Kraus–Turner mixing with sunlight counted where it is absorbed, convection, sea ice with a polar halocline, exact wind
stress. A `basin` preset (a thin pole-to-pole wall) gives gyres that carry ~0.5 PW poleward and an east–west tilt of the
equatorial thermocline; the equatorial Kelvin wave runs within 0.7% of c₁. The coupled world at n = 6 costs 45 ms per
model hour (~220 model years a day). Along the way the air's energy now closes better (the crossings' and ∇⁴ kinetic
energy are heat) and nearly empty air layers take their neighbours' wind, so the air runs at n = 4–7 with one setting
(SPEC §9 #64–#83). Earlier:
**V2-2b Moist atmosphere closed 2026-10-27 (34/34).** Three moist air layers
(isentropic interfaces at 305 and 380 K) over a slab aquaplanet with sea ice: an ITCZ (2.2× the subtropics), a Hadley
cell, trades and surface westerlies, mid-latitude storms, 2.2 mm/day of rain at 290 K; every gravity wave runs in the
fast substeps. Air at n = 6 costs 38 ms per model hour (~260 model years a day). Known near miss: mid-latitude rain is
slightly too even (B9, to revisit at n = 6 and with the ocean). Ice-dependent targets wait for the ocean (SPEC #43–#63).
Earlier:
**V2-2a Dry atmosphere closed 2026-10-26 (32/32).** Two air layers
with an isentropic interface that only heating and cooling move air across: radiative–convective equilibrium settles
(TOA +0.03 W/m², dry surface 294.8 K) and the Held–Suarez benchmark runs stably with jets and surface westerlies. Air at
n = 6 with its physics costs 19 ms per model hour (~510 model years a day) (SPEC §9 #32–#42). Earlier: **V2-1 Fluid core closed 2026-10-26 (29/29).** Stacked layers that each move freely, with the fast
external wave sub-stepped at its true speed (time-split RK3): external waves run within 0.4% of √(gH), internal waves
within 1.3%, a supercritical two-layer jet grows baroclinic eddies (×1,700 in 20 days) while a subcritical one stays
stable, and every layer's mass and tracer are conserved to 1e-15 even where a layer thins to its floor. Air at n = 6
costs 11 ms and ocean at n = 8 79 ms per model hour (SPEC §9 #23–#31). Earlier: V2-0b fused the kernels (×4.5 at n = 4) and fixed a transport
bug carried over from v1; V2-0 ported v1's grid and solver to 64-bit CUDA.

| Stage | Scope | Status |
|---|---|---|
| V2-0 | Foundation: Python package, CuPy kernels, grid port, save/load, WebSocket viewer skeleton, 64-bit speed probe on the RTX 4070 | ✅ Done (41/41) |
| V2-0b | Engine refinements: Voronoi-edge midpoint for wind rebuild, fused kernels + CUDA graph, Hilbert cell order, v1 transport fix | ✅ Done (45/45) |
| V2-1 | Fluid core: stacked layers, real-speed fast waves (time-split RK3), thickness floor, bottom stress, exchange hook | ✅ Done (29/29) |
| V2-2a | Dry atmosphere: two air layers with an isentropic interface, radiation, slab surface, RCE and Held–Suarez | ✅ Done (32/32) |
| V2-2b | Moist atmosphere: three air layers, vapor, evaporation, condensation, moist convection, rain, sea ice; climate and weather tests | ✅ Done (34/34) |
| V2-3 | Ocean: three layers, wind stress, thermocline, entrainment, salinity, sea ice; aquaplanet and one basin | ✅ Done (36/36) |
| V2-4a | Land: masks, coasts, ground heat, soil bucket, snow, runoff to the coast, terrain; gyres; two known shortfalls (SPEC #84) | ✅ Done (40/40) |
| V2-4b | Rivers, lakes, infiltration, seepage, snow, ice-sheet flow | ✅ Done (42/42) |
| V2-4c | Surface coupling, full plume, clouds, `earth` preset with sea floor, mixed-layer trial, all behind switches | ✅ Closed (41/42; ICE open, SPEC #100) |
| V2-4d | General layers: any number of air and ocean layers from lists; defaults stay at 3 and 3 (SPEC #101) | ✅ Closed (41/42; ICE open) |
| V2-4e | Clouds, convection and rain on `earth`, then lightning from storm physics; weak surface winds paused to V2-6 (SPEC #102) | ✅ Closed (41/42; ICE open; earth, clouds and lightning reported, untested) |
| V2-5 | Speed for development: ensembles in one launch, faster spin-up (the deep ocean's centuries), kernel profiling and removing wasted work, a first 32-bit trial; results unchanged or measured against 64-bit; mixed precision now too (SPEC #106) | 🔨 In progress (levers 1–4 and 32-bit trial done) |
| V2-6 | Real physics for the stand-ins (radiation (air and haze scattering, vapor-dependent longwave, sunlight absorbed by vapor and ozone, stratosphere), ice in the air with freezing heat, mountain and gravity-wave drag, stability-dependent surface exchange, land cover map, plume entrainment and downdrafts, rain evaporation, Earth's orbit, sea-ice motion, deep-water formation, firn with stored cold (closes the ICE failure, SPEC #100), gas composition (H₂O, CO₂, CH₄, O₃, N₂O by amount, N₂ and O₂ setting heat capacity and molecular weight), transported composition tracers with exact budgets, per-world planet settings), Moon and tides, UI polish | ⏳ Planned |
| V2-7 | Final optimization: mixed time steps, 32-bit sign-off, then approximations (reduced model, learned emulator) against the 64-bit reference ([plan §7.1](docs/v2-plan.md)) | ⏳ Planned |
| V2-8 | Two resolutions: air n = 6 coupled to ocean n = 8 | ⏳ Planned |
| V2-9 | Geology: separate long-time-scale model in the same codebase and viewer (Climate, Geology and Coupled modes), starting from real Earth; controllable events (core heat, volcanoes, earthquakes and tsunamis, storm model, meteor impacts up to a supported size); rock and soil elements (C, N, P, S, Fe, Ca, K, Si and others), weathering, ocean chemistry | 💬 To design |
| V2-10 | Other Worlds: stress-test and generalize the climate model and find its limits, on Mars, Venus, Jupiter, Saturn and invented worlds; hot-regime model for sterilizing impacts | 💬 To design |
| V2-11 | Coming to Life: microbes, plants and ecosystems that change the air, ocean and land they live on, with every element's ledger closed. Target: **O₂ emerges from photosynthetic organisms** | 💬 To design |
| V2-12 | Elemental Guardians: story mode where Air, Water, Fire and Earth, each with a personality, narrate from the model's data through a local LLM, read-only, on triggers of high balance and high imbalance and a regular heartbeat, with interactions from the ledgers' measured exchanges; early prototype after V2-5 | 💬 To design |

## Running it

The first run builds the mesh levels once (n = 8 takes about an hour; later runs load them from `cache/grids/`).
Needs an NVIDIA GPU with a current driver, Python 3.12+ and (for the v1 comparison tests) Node.js with v1 checked
out next to this repository.

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
```

Engine and viewer (then open http://127.0.0.1:8765/):

```bash
.venv/Scripts/python -m insilica.server --n 5 --preset aqua
```

A saved world opens with `--load` instead: for example the 20-year `basin` spin-up the tests use (save it once with
`coupled_spun(4, 'basin').save('cache/basin-n4-spun.iw2')` from `tests/test_climate.py`; the launch configuration
`engine-basin` serves it on port 8766). Every preset, `coupled` and `basin` included, can also be started from rest in
the viewer's New world panel; the ocean's fields (SST, salinity, mixed-layer depth, thermocline and abyss temperatures,
current, sea ice) are in the Field menu.

```bash
.venv/Scripts/python -m insilica.server --load cache/basin-n4-spun.iw2
```

Tests (27 tests in up to 5 processes, about 7 minutes warm, about 15 after an engine change while the spin-ups are rebuilt; prints every measured value against its threshold; add
`--perf` for the speed probe):

```bash
.venv/Scripts/python tools/run_tests.py
```

Speed probe on its own:

```bash
.venv/Scripts/python -m insilica.probe
```
