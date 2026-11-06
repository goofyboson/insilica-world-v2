# V2-6b Boundary layer and winds (design)

V2-6a left `earth` with too little low cloud (surface sunlight 179 against 161 W/m², EARTH a known failure) and winds
that #102 found 3–5× weak. The missing physics is the boundary layer: a mixed layer whose top is an inversion set by
entrainment against subsidence, saturated at its top where stratocumulus and trade cumulus sit, with cloud-top cooling
driving entrainment that thins those clouds. SPEC #109 scope: stability, mountain and gravity-wave drag, land cover. SPEC #113.

## Method

1. **Baseline.** A year of `earth` (n = 4, the k-tables, year 9): zonal winds by layer, the eddy momentum flux and the
   surface stress it asks for (#102's measure), against observed.
2. **Surface exchange by stability.** The bulk form with a stability function of the bulk Richardson number (Louis 1979)
   and gusts, from the switches #102 left (`bulk_c`, `bulk_z`, `bulk_w`), made physical and the default.
3. **The boundary layer as a fourth air layer (owner).** Air layers from the top: the stratosphere (above θ* 380 K), the
   upper troposphere (305–380 K), the lower free troposphere (from the boundary layer's top to 305 K), and the mixed
   boundary layer. The boundary layer's top is not an isentrope: its θ* follows the layer's own θ (as `bl_mix`, #98,
   already does: θ* = max(base, θ_bl), below the air above), so its depth comes from physics:
   - its own horizontal outflow and inflow (in a layered model, that is subsidence and convergence);
   - entrainment of the air above, driven by the surface buoyancy flux (Tennekes 1973, `bl_ent` 0.2) and by cloud-top
     radiative cooling (Lilly 1968; Nicholls & Turton 1986), the entrained air bringing the vapor of the free
     troposphere's base (edge vapor, #112) — the drying that thins stratocumulus;
   - plumes leaving it, and its collapse to a thin night layer over cooling ground (#98).
   Its vapor and θ are even through its depth, so it is nearest saturation at its top, and its cloud forms there
   (`bl_q`, #112, now on the layer it was made for).
4. **Drag.** Gravity-wave drag in the stratosphere from waves not tied to the ground (Alexander & Dunkerton 1999); drag
   on air flowing at and over mountains, and the gravity waves they launch (Lott & Miller 1997).
5. **Land cover.** Roughness, albedo and plants' resistance to evaporation by surface type, from a land-cover map
   (MODIS MCD12C1 proposed; download asked first).
6. **Weak eddies (#102)** re-measured after 2–4; investigated further only if still short.

## Tests

| Test | Against | Limit (proposed) |
|---|---|---|
| EARTH (extended from V2-6a) | Wild et al. 2013, CERES EBAF | each within 5%, now passing; low-cloud cover in the stratocumulus regions and globally reported |
| WINDS (new): `earth` n = 4, annual means | observed climatology (to be checked against ERA5 before the limits are fixed) | trades at 5–25° in the bottom layers, westerlies at 35–55°, the jet, the stratosphere's annual-mean jet, eddy momentum flux at 30–50°: limits agreed after step 1 |
| CONSERVE (extended) | its ledgers | unchanged (≤ 1e-13) with the boundary layer's exchanges and the drags |

## Baseline (step 1)

`earth` n = 4, year 9 (k-tables, V2-6a's defaults), annual zonal means, east wind (m/s):

| Latitude | 65S | 55S | 45S | 35S | 25S | 15S | 5S | 5N | 15N | 25N | 35N | 45N | 55N | 65N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stratosphere (105 hPa of mass) | 108.7 | 130.5 | 97.2 | 50.2 | 15.4 | −2.0 | −7.2 | −8.1 | −5.5 | 0.7 | 13.6 | 33.1 | 53.0 | 66.2 |
| Troposphere 305–380 K (653 hPa) | 17.3 | 28.8 | 24.8 | 16.9 | 15.0 | 6.9 | −1.1 | −1.9 | 1.8 | 9.5 | 13.5 | 14.0 | 12.8 | 12.0 |
| Bottom layer (219 hPa) | 4.4 | 13.2 | 9.7 | 2.4 | −1.8 | −5.3 | −2.9 | −0.7 | −3.0 | −2.2 | 0.5 | 1.2 | 1.5 | 0.8 |

Column eddy momentum flux (kg/s²): −285,000 at 35°S, +155,000 at 35°N: 5–8× #102's aquaplanet 3.6e4, so the eddies are
no longer the main shortfall. Short now: the northern trades (−2 to −3 m/s) and northern westerlies (+1 to +1.5 m/s at
the bottom, where the eddies ask for 0.05–0.08 N/m² of surface stress); the stratosphere's jets, 130 m/s at 55°S as an
annual mean (observed about 30–40), with nothing to drag them.

## Step 2: the surface layer (built, off: SPEC #114)

`sfc` (kernel `surface_layer`): rho C_D U and rho C_H U per cell from the log law at a tenth of the bottom layer's depth, Charnock's sea
roughness, `z0_land` 0.1 m, Louis (1979) stability from the virtual temperature difference, free-convection gusts. The drag uses it at
every edge, land and sea.

`earth` n = 4 with it on, 9 years: zonal-mean bottom-layer wind (m/s), against the baseline:

| Latitude | 55S | 45S | 15S | 15N | 25N | 45N | 55N | 65N |
|---|---|---|---|---|---|---|---|---|
| baseline | 13.2 | 9.7 | −5.3 | −3.0 | −2.2 | 1.2 | 1.5 | 0.8 |
| `sfc` | 15.3 | 14.3 | −5.5 | −3.3 | −2.8 | 3.7 | 6.7 | 7.3 |

Over the sea the surface air read from the deep bottom layer's mean theta is 2–5 K warmer than the sea (stable 88–97% of the time), so
the sea's heat exchange falls to 5–8 W/m²/K and the planet warms (Ts +0.5 K a year at year 9). The stability needs the boundary layer
as its own mixed layer (step 3); `sfc` is re-measured on the four-layer earth.

## Step 3: the four-layer earth (`earth4`, SPEC #115)

Layers from the top: stratosphere (above 380 K), upper troposphere (305–380 K), lower free troposphere (the mixed layer's top to 305 K),
the mixed boundary layer (`bl_mix`, `bl_q`, `sfc`). Plumes leave the mixed layer only where its top is saturated (Neggers, Stevens &
Neelin 2006). Six numerical faults that thin and emptied layers exposed are fixed (SPEC #115).

| `earth4` n = 4, year 9 | value | observed |
|---|---|---|
| absorbed sunlight | 242.2 | 240 |
| OLR | 239.7 | 239 |
| clear-sky OLR | 259.3 | 266 |
| surface longwave down | 351.9 | 342 |
| surface sunlight | 170.4 | 161 |
| net (still falling) | +2.4 | ~0.6 |
| mixed layer, tropical / subtropical / mid-latitude sea, land (hPa) | 20 / 74 / 101 / 74 | |
| eddy momentum flux at 35°S / 35°N (kg/s²) | −21,000 / +62,000 | about −200,000 / +150,000 |

The radiation budget is now near observed; the eddies have collapsed, and with them the surface winds: the next thing to find.

Moving the middle interface does not bring the eddies back. `earth4` was started fresh with it at each of 295, 300, 310 and 315 K, run 9 years, and measured for a year (eddy flux in kg/s²):

| Middle interface (K) | 295 | 300 | 305 | 310 | 315 | 3-layer |
|---|---|---|---|---|---|---|
| eddy flux 35°S | −36,000 | −46,000 | −51,000 | −62,000 | −59,000 | −284,000 |
| eddy flux 35°N | +54,000 | +51,000 | +40,000 | +46,000 | +57,000 | +160,000 |
| the interface, tropics → 55°S (hPa) | 963 → 551 | 969 → 476 | 934 → 485 | 841 → 411 | 732 → 376 | 879 → 640 |
| Ts (K) | 289.1 | 289.9 | 291.9 | 292.5 | 292.7 | |

Four plain isentropic layers, without the mixed-layer physics (`bl_mix`, `bl_q` off), recover only part of the eddy flux (kg/s²):

| Run (θ* in K) | 35°S | 35°N | layer 1 θ, tropics / 55°S (K) |
|---|---|---|---|
| 380 / 305 / 290, `sfc` | −86,000 | +64,000 | 321.8 / 315.6 |
| 380 / 305 / 290, no `sfc` | −37,000 | +47,000 | 319.2 / 315.8 |
| 380 / 305 / 295, `sfc` | −85,000 | +89,000 | 323.3 / 318.5 |
| `earth4` (mixed layer) | −51,000 | +40,000 | 324.0 / 317.1 |
| 3-layer `earth` | −284,000 | +160,000 | 331.1 / 314.3 |

The fourth layer itself loses the eddies. In every four-layer run, the 305–380 K layer is 7–9 K colder in the tropics than in the 3-layer earth, so its equator-to-pole contrast, and with it its baroclinicity, is a half to a third as large.

Where the tropics' heat lands: 30 days from year 9 over the tropical sea (|lat| < 15°), from `air_column`'s diagnostics:

| | 3-layer `earth` | `earth4` | four plain layers (290 K) |
|---|---|---|---|
| bottom layer (hPa) | 116 | 15 (mixed layer); lower free troposphere 44 | 1 (emptied) |
| plume mass into 305–380 K (kg/m²/day) | 1,007 | 243 (+99 into the layer below) | 0 |
| rain (mm/day): total / convective | 8.9 / 8.9 | 5.3 / 4.1 | 6.2 / 0 (large-scale in layer 1) |
| 305–380 K layer's radiative cooling (W/m²) | 212 | 115 | 109 |

The four-layer tropics convect a quarter as much into the upper troposphere, which heats it less: the colder tropical upper troposphere, and so the weaker eddies.

What holds the plumes back, over the tropical sea for 30 days, from a scratch copy of the kernel with the plume closure's limits instrumented:

| | 3-layer `earth` | `earth4` |
|---|---|---|
| plumes buoyant (share of steps) | 98% | 49% |
| closure asks, when buoyant (kg/m²/day) | 1,030 | 12,600 |
| plume mass delivered (kg/m²/day) | 1,007 | 342 |
| limit that binds: closure / layer mass / rho a w* / moist root | 100 / 0 / 0 / 0% | 27 / 42 / 30 / 0% |
| plumes stopping below 305 K | 0% | 29% |

The four-layer earth's plumes are starved. They take air only from the mixed layer, and nothing refills it as fast as the closure asks to empty it: the model has no convective downdrafts and no entrainment of the cloud layer's air into the updrafts.

## Downdrafts, cloud-layer entrainment and the energy leak (SPEC #116)

Both switches are on in `earth4` and off in `earth`, each with Tiedtke's (1989) value:

- **`downdraft` 0.3:** air from the layer above the mixed layer is saturated by evaporating the plumes' rain and sinks into the mixed layer. Its mass is 0.3 of the plume's base mass, as far as the rain allows.
- **`ent_cloud` 1e-4 per m:** as the updraft crosses that layer, it takes in the surrounding air.

Extending CONSERVE to the four-layer column found an energy leak. The plumes' mixing counted the vapor it gives the layers it passes twice. That created **+28.8 W/m² from nothing in `earth`** and +14.2 in `earth4`, all of it in the layer where the plumes detrain. It is now fixed.

Tropical sea and eddy flux on the fixed code (eddy flux in kg/s²):

| | 3-layer before the fix | 3-layer fixed | `earth4` before | `earth4` + downdrafts and entrainment |
|---|---|---|---|---|
| mixed or bottom layer (hPa) | 116 | 157 | 15 | 53 |
| plume mass into 305–380 K (kg/m²/day) | 1,007 | 1,116 | 243 | 459 |
| rain (mm/day) | 8.9 | 7.4 | 5.3 | 6.5 |
| 305–380 K layer θ, tropics (K) | 331.1 | 321.8 | 324.0 | 320.4 |
| eddy flux 35°S / 35°N | −284,000 / +160,000 | −198,000 / +88,000 | −51,000 / +40,000 | −38,000 / +64,000 |

EARTH on the fixed code, year 9: absorbed 238.5, OLR 221.8, net +16.7. The moist physics of V2-6a (#112) was set while the leak was in place.
