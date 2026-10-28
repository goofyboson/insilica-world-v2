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

**V2-3 Ocean in progress (branch `stage-v2-3`).** Three fixed-density ocean layers (mixed layer, thermocline, abyss)
with temperature and salinity under the three air layers: Kraus–Turner mixing with sunlight counted where it is absorbed,
convection, sea ice with a polar halocline, wind-stress coupling; the equatorial Kelvin wave runs within 0.7% of c₁
(SPEC #64–#81). Done so far: stability, energy closure (−0.009 W/m²), WAVES, a cached 20-year coupled spin-up at n = 4, and a `basin` preset (a thin pole-to-pole wall, one ocean basin: gyres carry ~0.5 PW poleward, the equatorial thermocline tilts; #79).
CLIMATE coupled on `basin` passes (#80); CONSERVE coupled passes, with the air's ∇⁴ loss now booked as heat (#81). PERF times the coupled world (45 ms per model hour at n = 6, gate 50, #82). Still to do: viewer ocean fields, the stage's records; then, once
the stage's changes are in, **an n = 6 coupled spin-up and climate check** (agreed 2026-10-27).

**V2-2b Moist atmosphere closed 2026-10-27 (34/34), merged to `main`.** Three moist air layers
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
| V2-3 | Ocean: three layers, wind stress, thermocline, entrainment, salinity, sea ice; aquaplanet first | 🔧 In progress |
| V2-4 | Land and coasts: gyres and boundary currents, surface water, soil, ice sheets, terrain | ⏳ Planned |
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

Tests (thirteen tests in up to 5 processes, about 2 minutes warm, longer the first time while spin-ups are cached; prints every measured value against its threshold; add
`--perf` for the speed probe):

```bash
.venv/Scripts/python tools/run_tests.py
```

Speed probe on its own:

```bash
.venv/Scripts/python -m insilica.probe
```
