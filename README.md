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

**V2-4a Land closed 2026-10-31 (40/40 with PERF). Next: V2-4b (rivers, lakes, infiltration, seepage, snow, ice-sheet flow).**
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
| V2-4b | Rivers, lakes, infiltration, seepage, snow, ice-sheet flow | ⏳ Next |
| V2-5 | Two resolutions: air n = 6 coupled to ocean n = 8 | ⏳ Planned |
| V2-6 | Moon and tides, geology, UI polish | ⏳ Planned |
| V2-7 | Optimization and speedup: ensembles in one launch, 32-bit where safe, mixed time steps, faster spin-up; then approximations (reduced model, learned emulator) against the 64-bit reference ([plan §7.1](docs/v2-plan.md)) | ⏳ Planned |

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
