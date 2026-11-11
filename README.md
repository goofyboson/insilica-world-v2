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

**V2-6c in progress (2026-11-09):** ice in the air with its freezing heat aloft, snow, melting and sublimation (SPEC #134); RAIN against GPCP (new); the surface's wind from the reconstructed cell wind (#135); rain and snow evaporation by Kessler's form as the IFS's (#136); plumes rooted in the mixed layer's vapor plus the thermals' excess (#137); the cloud layer's entrainment with as much detrainment (#138), then as the IFS's (#139); a shallow plume (#140); the plumes' ascent through the cloud layer resolved in ~25 hPa steps (#141); `earth6`, a cumulus layer above the mixed layer (#142), the tests now on it; the plumes through the cumulus and cloud layers (#143); the cumulus layer's top from the mixed layer's air lifted saturated 200 hPa (#144); the plumes' buoyancy in virtual θ (#145) with the air they take out of the mixed layer replaced by the cumulus layer's, up to cloud base (#146); the mixed layer's cloud not raining, tried in three forms and reverted (#147); bare soil evaporating from its drying top (#148); the cumulus top freed above 305 K (#149) and the cumulus layer on its own moist adiabat (#150) the plumes' rain evaporating above cloud base (#151) and shallow plumes closed by the boundary layer's w* (#152), all tried and reverted; the air in seven layers, the 305–380 K layer split at 330 K (earth7, #153: tropical precipitable water 25.4 → 32.2 kg/m²); ice clouds with their own sunlight optics (#154); the deep plume entraining and detraining through every layer it crosses (#155). EARTH passes (surface sunlight +1.2%); CONSERVE passes; RAIN fails (3.30 against 2.69 mm/day, +23%); WINDS fails on the jet and the equatorial stratosphere.

**V2-6b Boundary layer and winds closed 2026-11-08 (44/46 with PERF; ICE and WINDS' jet position known failures to V2-6d). Next: V2-6c moist physics.** `earth5`: five air layers (stratosphere split at 500 K, upper and lower troposphere, a mixed boundary layer), surface exchange by stability, mountain and gravity-wave drag, land cover. EARTH passes, every term within 3% (net +0.6 W/m²); WINDS within on trades, surface westerlies, the northern jet and the stratosphere; the southern jet sits at 35°S (55°S observed), held there by a warm Southern Ocean and weak eddies (SPEC #131). PERF air n = 6 23.5 ms per model hour. docs/v2-6b-boundary-layer.md.

**V2-6b record (2026-11-07 to 11-08):** the four-layer earth (`earth4`, SPEC #115–#116); EARTH measures at equilibrium (#117); `earth4`'s surface given a daily ground skin, stomata and salt (#118); air leaving a nearly emptied layer now carries its own heat, which ends the runaway that crashed `earth4` (#119); the mixed layer takes in colder air through an emptied layer, so tropical land is no longer hot and dry (Ts 313 → 301 K, rain 1.9 → 5.2 mm/day; #120); the 380 K edge heated by radiation alone, so the top layer no longer fills (#121); three faults found on the way fixed: a heat pump in #120's reach (#122), backward θ spans (#123), and a mixed layer that grew into the whole troposphere, now ending at its cloud base (#124); EARTH measures `earth4` and passes, each budget term within 3% (#125). WINDS (#126) measures the same year against the NCEP 1991–2020 reanalysis: trades, surface westerlies and the 305–380 K jet within limits (jets 10° equatorward); above 380 K 3–5× too strong (peaks 97.5/49.2 against 26.5/9.9 m/s, tropics −22 against −5), a known failure for the drag. The mountains' drag (Lott & Miller 1997, #127): the northern stratospheric jet 49 → 9 m/s (10 observed), the southern 98 → 51 (27), the tropics −22 → −33 (−5); the southern jet now 20° equatorward, so WINDS fails on the jet and the stratosphere; EARTH passes (net −0.1). The southern jet's shift traced to the drag together with the Southern Ocean it builds (not the stratospheric vortex); the gravity waves not tied to the ground (Alexander & Dunkerton 1999, #128): southern stratospheric jet 51 → 43 (27), northern 10 (10), tropics −33 (−5, out of a single stratospheric layer's reach); EARTH passes (net +0.7). The stratosphere in two layers, split at 500 K (`earth5`, #129), and two faults in the drags fixed (a wave never pushes a layer past its phase speed; the waves' spectrum centred on the source wind, #130): WINDS' stratosphere now within (peaks 34/21 against 27/10 m/s, tropics −9 against −5), WINDS failing only on the southern jet's position (35°S against 55°S); EARTH on `earth5` passes (net +1.5). `earth4` at equilibrium: Ts 289.6 K, net +1.0 W/m², tropical top layer 804 kg/m², stable through a single-stepped spin-up. Open: the extratropical top layer is light; subtropical land is dry; save/load is not bit-exact. Suite 43/45 (ICE, WINDS on the jet's position, both known failures to V2-6d; GRID n = 8 passes alone). The southern jet's position is a known failure to V2-6d (#131): it stays at 35°S with the drags off, held there by a Southern Ocean 7–8 K too warm and weak eddies. Land cover from NCAR's MODIS-derived plant types and soil colours with Noah's values (#132): each land cell's roughness, albedo, snow under trees and stomata; Sahara albedo 0.28 → 0.36, Amazon 0.13 → 0.10, Siberian boreal winter 0.65 → 0.39; EARTH passes (net +0.5). Left in V2-6b: closing the stage (full suite with `--perf`).

**V2-6a Radiation closed 2026-11-04 (43/45 with PERF; EARTH and ICE open). Next: V2-6b boundary layer and winds.** Radiation by correlated-k from HITRAN line lists: longwave within 0.04% of line-by-line OLR on seven AFGL columns (LONGWAVE), a k-table shortwave, the stratosphere as the top layer above 380 K, freeze-dried. `earth` now runs them by default, with moist physics fixed on the way (SPEC #112): the tropical troposphere within 1–3 K of observed, clear-sky OLR 268 (266 measured). **EARTH** (the global budget within 5% of measured) is a known failure to V2-6b: too little low cloud lets 179 W/m² of sunlight reach the surface (161 measured), and rain is 4.2 mm/day (about 2.7). ICE's drift (−146 → −201) is weather, not a change. PERF air n = 6 23.2 ms per model hour. docs/v2-6a-radiation.md.

**V2-5 Speed closed 2026-11-03 (42/43 with PERF; ICE open). Next: V2-6 real physics.** `earth` n = 6 per step 8.89 → 4.22 ms
(×2.11: 139 → 281 model years per day; PERF air n = 6 39.6 → 22.9 ms per model hour), n = 4 1.76 → 1.10 ms; ensembles ×2.9 at
n = 4; mixed precision by default (SPEC #106: 64-bit state and ledgers, 32-bit where the arithmetic is bound, its 16-year climate
within noise of 64-bit, `INSILICA_PRECISION=double` the bit-identical reference); the deep ocean's asynchronous spin-up (SPEC
#107, test SPINUP), ×3.4 at n = 4, the ocean alone 2,167 model years per day at n = 6. CLIMATE coupled now averages 3 years
(#108). Short of target: ensembles ×2.9 (×10 aimed). docs/v2-5-speed.md.

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
| V2-5 | Speed for development: ensembles in one launch, faster spin-up (the deep ocean's centuries), kernel profiling and removing wasted work, a first 32-bit trial; results unchanged or measured against 64-bit; mixed precision now too (SPEC #106) | ✅ Closed (42/43; ICE open; n = 6 ×2.11, ensembles ×2.9) |
| V2-6a | Radiation: correlated-k longwave and shortwave from line lists, stratosphere, `earth` on them (SPEC #109–#112) | ✅ Closed (43/45; EARTH known failure to V2-6b, ICE open) |
| V2-6b | Boundary layer and winds: mixed boundary layer as its own air layer, surface exchange by stability, mountain and gravity-wave drag, stratosphere in two layers (`earth5`), land cover; EARTH and WINDS against measured (SPEC #113–#133) | ✅ Closed (44/46; ICE and WINDS' jet position known failures to V2-6d) |
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
