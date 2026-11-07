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

### `earth` re-checked on the fixed code

At year 9 the planet is not in balance, and the leak had hidden that. The whole planet's energy closes, with the change in stored energy matching the TOA flux to within 0.04 W/m² over 2 days. Over year 9 it stores 16.8 W/m², almost all of it in the ocean's thermocline layer:

- The thermocline starts at 286.1 K and is still warming at year 8.
- The deep layer stays at 273.15 K.
- Before the fix, the thermocline took up about 32 W/m². That was the TOA's +3.2 plus the leak's +28.8. So #112's near-balance at year 9 came from the leak, not from equilibrium.

Taken to equilibrium with the asynchronous spin-up (#107): from the cached year-8 state, 4 × (1 coupled year + 10 ocean-only years), then a settling year and a measured year:

| | year 9 (EARTH now) | equilibrium | observed |
|---|---|---|---|
| absorbed sunlight | 238.5 | 237.9 (−0.9%) | 240 |
| OLR | 221.8 | 235.6 (−1.4%) | 239 |
| clear-sky OLR | 258.9 | 270.0 (+1.5%) | 266 |
| surface longwave down | 337.6 | 360.5 (+5.4%) | 342 |
| surface sunlight | 174.6 | 172.1 (+6.9%) | 161 |
| net | +16.7 | +2.3 | ~0.6 |
| Ts (K) | 289.3 | 292.7 | ~288 |
| thermocline (K) | 288.3 | 291.1 | |

A 90-day sample of the equilibrium state:

| | equilibrium | observed |
|---|---|---|
| rain (mm/day) | 4.9 | about 2.7 |
| tropical rain (mm/day) | 7.3 | |
| cloud cover | 0.82 | about 0.67 |
| cloud effect on sunlight (W/m²) | −48 | −47 |
| cloud effect on longwave (W/m²) | +36 | +28 |

The fixed `earth` balances, but about 4.7 K too warm. The causes are too much high cloud (longwave effect +8 W/m²), too little low cloud (surface sunlight +11 W/m²) and rain nearly twice the observed.

### Both earths at equilibrium: the surface budget (SPEC #117)

EARTH's procedure (#117) applied to both earths: 3 coupled years, then 5 × (1 coupled + 20 ocean-only), a settling year, then a year measured. Fluxes in W/m².

| | `earth` | `earth4` | observed |
|---|---|---|---|
| absorbed sunlight / OLR | 237.9 / 235.5 | 245.3 / 246.9 | 240 / 239 |
| clear-sky OLR | 270.2 | 266.5 | 266 |
| surface sunlight / longwave down | 172.1 / 360.6 | 172.8 / 365.1 | 161 / 342 |
| Ts (K) | 292.8 | 292.1 | ~288 |
| rain (mm/day) | 4.9 | 4.3 | ~2.7 |
| cloud cover; cloud effect on sunlight / longwave | 0.82; −49 / +35 | 0.64; −44 / +20 | ~0.67; −47 / +28 |
| surface net radiation | 115 | 119 | ~106 |
| latent heat | 141 | 123 | ~84 |
| sensible heat (net radiation − latent − storage) | about −27 | about −3 | ~20 |

Over the sea, a year in each latitude band:

| | `earth` | `earth4` |
|---|---|---|
| bottom-layer air at the surface − sea surface temperature (K): 0–15° / 15–35° / 35–60° | +1.1 / +1.9 / +1.9 | −0.4 / −0.7 / −0.5 |
| bottom layer's vapor / saturation at the sea's temperature | 0.30 / 0.30 / 0.24 | 0.80 / 0.66 / 0.55 |
| evaporation (mm/day) | 7.4 / 7.0 / 5.0 | 6.3 / 5.3 / 3.7 |

In the 3-layer earth, the surface air is read from the deep bottom layer's mean. That air is warmer than the sea and dry, so the air heats the sea and the sea evaporates the surplus: sensible heat is negative and latent heat is 1.7 times observed. The four-layer earth's mixed layer gets the sign right over the sea. Even so, both earths shed nearly all of the surface's energy by evaporation: global sensible heat is about 0, against an observed +20. Both run 4–5 K warm.

**Surface emissivity, tried and left at 0.95.** The surface absorbs all of the longwave reaching it but emits ε σT⁴, with ε = 0.95. That is not Kirchhoff-consistent, though energy is conserved. Trial with `eps_s` 1, at equilibrium:

| | `earth` | `earth4` |
|---|---|---|
| Ts (K) | 292.4 (−0.4) | 293.8 (+1.7) |
| rain (mm/day) | 4.6 | 3.7 |
| cloud cover | 0.80 | 0.60 |

The emissivity is not the warm bias.

## earth4's sensible heat, measured directly

The "about −3" above was a residual (net radiation − latent heat) sampled four times a day. Measured directly instead: H = c_p ρC_H U (Ts − T_air), 24 samples a day for 20 days from the equilibrium state. Sun up means sunlight absorbed in the column. Fluxes are in W/m².

| | H day / night / all | LH day / night / all | Ts − T_air (K), all | U (m/s) | soil / capacity |
|---|---|---|---|---|---|
| sea | 6.4 / 7.1 / 6.7 | 130 / 140 / 135 | +0.27 | 9.4 | – |
| sea 0–15° | 3.1 / 3.7 / 3.3 | 164 / 192 / 178 | +0.40 | 10.3 | – |
| land | 15.0 / 4.4 / 10.4 | 73 / 84 / 81 | −1.66 | 6.1 | 0.39 |
| land < 35° | 25.4 / 20.3 / 23.2 | 104 / **137** / 124 | +1.00 | 5.3 | 0.34 |
| land 35–60° | 15.3 / −0.1 / 7.4 | 68 / 53 / 60 | −1.26 | 7.0 | 0.49 |

Global H is about +8 (sea 6.7 × 0.715 + land 10.4 × 0.285), against an observed ~20. The two surfaces fall short for different reasons:

**Land evaporates like a sea, and as much by night as by day.** Land latent heat is 81 W/m² (2.8 mm/day), against an observed ~40 (~1.4 mm/day). Its Bowen ratio is 0.13, against an observed ~1. Tropical land evaporates more at night (137) than by day (104), and stays 0.8 K warmer than its air at night. Two departures from real land:

- **The ground's heat capacity.** It is one slab of 2e6 J/m²/K (SPEC #84), about 1 m of soil. The daily cycle reaches only the top ~0.1 m, about 2e5 J/m²/K (ρc √(κP/π) with κ ≈ 5e-7 m²/s). So the ground carries the day's heat into the night and keeps evaporating.
- **No surface resistance.** `land_rs` is 0 (it was tried and switched off in SPEC #92), so wet soil evaporates through the air's resistance alone. Real leaves close their stomata at night and resist ~50–100 s/m by day.

**Trial: ground capacity 2e5.** A diagnostic only: `sea.cap` was set on land from the equilibrium state, then 30 days of settling and 10 days measured. The baseline was run over the same days.

| land | H day / night / all | LH day / night / all | Bowen |
|---|---|---|---|
| baseline, 2e6 | 18.3 / 7.1 / 13.5 | 94 / 91 / 95 | 0.14 |
| 2e5 | 40.8 / −7.2 / 17.5 | 119 / 36 / 80 | 0.22 |
| baseline, < 35° | 24.6 / 20.3 / 23.2 | 106 / 134 / 123 | 0.19 |
| 2e5, < 35° | 51.5 / 1.9 / 27.5 | 123 / 57 / 95 | 0.29 |

The thin ground gives land a real daily cycle: stable, nearly still nights and doubled daytime sensible heat. The Bowen ratio is still 0.22, because the wet soil has no resistance by day. Tropical land's daytime mixed layer stays thin (25–35 hPa, against 100–200 observed). That ties to the open "desert mixed layers" item.

**The sea evaporates about a third too much (135 against ~100), and its sensible heat is half observed (6.7 against 10–15).**

- The air in the mixed layer is only 0.27 K colder than the sea (observed ~1 K).
- The saturation over the sea is that of fresh water. Salt lowers it by 2%, which is about 0.4 g/kg of the 5.4 g/kg deficit, or about 7% of the latent heat.
- The wind of 9.4 m/s (with gusts) is at the high end. The exchange coefficient itself, C_H ≈ 1.0e-3, is right.

## The surface: ground skin, stomata, salt (SPEC #118)

The three fixes, in `earth4` only:

- **Ground skin** (`c_deep`): a daily-cycle skin of 1.2e5 J/m²/K over 2e6 of deep ground, coupled at the force-restore rate and stepped implicitly.
- **Stomata:** 70 s/m in full sun, closing to 5000 s/m in the dark (Jarvis).
- **Salt:** the sea's saturation × 0.98.

At equilibrium, measured over a year (budget) and over 20 days at 24 samples a day (fluxes):

| | `earth4` before | after | observed |
|---|---|---|---|
| land: sensible / latent heat (W/m²) | 10 / 81 | 34 / 31 | ~30–40 / ~40 |
| land: Bowen ratio | 0.13 | 1.1 | ~1 |
| tropical land by night: latent heat (W/m²) | 137 | 3 | small |
| sea: sensible / latent heat (W/m²) | 6.7 / 135 | 7.8 / 133 | 10–15 / ~100 |
| rain (mm/day) | 4.26 | 3.67 | ~2.7 |
| Ts (K) | 292.1 | 294.4 | ~288 |
| cloud cover | 0.64 | 0.57 | ~0.67 |
| absorbed sunlight / OLR (W/m²) | 245.3 / 246.9 | 246.8 / 249.9 | 240 / 239 |

Left out one at a time:

| left out | Ts (K) | rain (mm/day) | cover |
|---|---|---|---|
| the skin | 294.1 | 4.11 | 0.60 |
| the stomata | 293.75 | 3.66 | 0.57 |

The run without salt went NaN.

The land now behaves like land: the Bowen ratio is about 1, and nights are still and stable. The sea is barely changed. Salt's −2% shows up as only −1.5% in the sea's latent heat. The sea's excess evaporation stays: its air is only 0.5 K colder than the sea, and the winds are strong.

**The planet warms, because cloud thins as the land dries (0.64 → 0.57).** The warm bias is in the clouds, not the surface.

**The NaN, an existing fragility.** At day 990 of the run without salt, off Japan (38°N, 144°E), the layer above the mixed layer was emptied to its floor (0.98 hPa) and already stood at θ 862 K. One column step heated it to 1912 K. Two steps later the column and the winds failed (θ −320,000 K, winds 776 m/s). Hot near-empty layers exist in the earlier `earth4` states too. For example, the old equilibrium state has a mixed layer at 1.5 hPa with θ 424 K, and about 500 cells with a layer under 2 hPa. A layer at its floor should not take heating it cannot hold.

## Air leaving a nearly emptied layer (SPEC #119)

The NaN in #118's run without salt, traced step by step at the cell where it began (38°N 144°E, day 990). The column step heated the layer above the mixed layer by roughly twice as much each step:

| step | layer above the mixed layer: mass (kg/m²) after the column | θ (K) | taken by entrainment (kg/m²) | layer 1's air sinking into it, at θ (K) |
|---|---|---|---|---|
| 47533 | 10.0 | 305.0 | 27.7 | 305 |
| 47534 | 24.1 | 348.9 | 31.4 | 307 |
| 47535 | 26.7 | 421.5 | 29.3 | 349 |
| 47536 | 21.1 | 570.3 | 30.1 | 407 |
| 47537 | 21.9 | 862.2 | 33.2 | 541 |
| 47538 | 21.6 | 1912.0 | 51.0 | 841 |

Two errors fed each other:

1. **Entrainment took the layer's coldest air only.** The growing mixed layer took up to three quarters of the layer above, all at that layer's bottom-edge θ. The remnant kept the excess heat. Now the slice taken holds the edge's θ plus f of the way to the layer's mean, as the vapor already did (#112).
2. **The crossing's net flow took the wrong θ.** A remnant warmer than its top edge rises at its own θ. Layer 1's sinking air outweighed it, and the net, now sinking, still carried the remnant's θ. Layer 1 lost heat it never held: on the last step 1838 kg/m² at 1892 K, and its θ went to −640 K. Now the net carries its own direction's θ.

`earth4` at equilibrium (3 coupled years, 5 × (1 + 20), a settling year, the next year measured):

| | #118 | #119 | without salt |
|---|---|---|---|
| Ts (K) | 294.41 | 293.93 | 293.51 |
| absorbed sunlight / OLR (W/m²) | 246.8 / 249.9 | 247.7 / 249.2 | 246.8 / 248.2 |
| net (W/m²) | −3.1 | −1.5 | −1.4 |
| rain (mm/day) | 3.67 | 3.68 | 3.72 |
| cloud cover | 0.57 | 0.57 | 0.57 |
| near-empty (< 2 hPa) cells above the mixed layer, hottest θ (K) | 555, 318 (layer max 387) | 434, 313 | 423, 307 |

**Still open.**

- **Near-empty mixed layers trap heat.** 73 cells hold under 2 hPa, up to θ 428 K over Tibet (362 K air over 272 K ground). They sit just above the floor (11 kg/m²), where the floor clamp does not reach them.
- **Tropical land is hot and dry since #118:**

| land 0–15° | before #118 | #118 / #119 |
|---|---|---|
| Ts (K) | 304 | 320 |
| deep ground (K) | (not used) | 314 |
| soil (share of capacity) | 0.57 | 0.12–0.17 |

## The mixed layer reaches through an emptied layer (SPEC #120)

After #118 the land's sensible heat rose (Bowen 0.13 → 1.1), the tropical mixed layer warmed past 305 K, and the cloud layer above it went to its floor. The mixed
layer then took air only from that empty layer, so it stayed thin and hot under colder air: absolutely unstable, with nothing to mix it.

| Tropical land 0–15°, 3 h into a day | mixed layer (kg/m²) | its θ (K) | θ above (K) | cloud layer (kg/m²) |
|---|---|---|---|---|
| before (#119) | 168 | 327.9 | 305.3 | 11 |
| after (#120)  | 1824 | 308.6 | 305.8 | 41 |

The rule (`bl_reach`): air above colder than the mixed layer belongs in it. From each layer up, it takes a slice from the bottom edge until the mixed layer's θ equals
the slice's top θ (θ linear in mass), at most the crossing's daily intake cap; a layer taken whole lets the next one in. Tennekes entrainment takes from the lowest layer
above that holds air.

| `earth4` at equilibrium, measured year | #119 | #120 |
|---|---|---|
| land 0–15°: rain / evaporation (mm/day) | 1.91 / 1.65 | 5.18 / 4.61 |
| land 0–15°: Ts (K), soil (of capacity) | 313.0, 0.16 | 301.3, 0.76 |
| land 15–35°: rain / evaporation, Ts | 0.92 / 0.84, 304.2 | 3.11 / 2.87, 297.9 |
| global Ts (K), net (W/m²) | 293.93, −1.46 | 294.95, −3.75 |
| global rain (mm/day), cover | 3.68, 0.57 | 4.44, 0.64 |
| top layer (above 380 K), tropics / south of 30°S (kg/m²) | 1021 / 598 | 1448 / 1178 |

Open: the top layer fills. From the #119 state, 10 days with and without the rule: plumes deliver nothing to the top layer either way; the dry crossing at 380 K carries
29.2 kg/m²/day up in the tropics with it (7.5 without) against 6.0 down (5.5), and the extratropics give none back (south: 3.1 up, 4.1 down).

## The top layer's filling, and three faults behind it (SPEC #121–#124)

**Why the top layer filled after #120 (#121).** Over 5 days at the #120 equilibrium, per kg/m²/day across 380 K:

| | Tropics | South of 30°S | North of 30°N |
|---|---|---|---|
| Up / down | 13.1 / 10.7 | 2.0 / 9.9 | 3.3 / 6.7 |
| Layer 1's mass per kelvin at 380 K | 24 kg/K | 2 | 3 |
| Up driven by layer 1's latent heating alone | 11.5 of 13.4 | 1.6 of 2.6 | 1.6 of 2.9 |

The cloud layer's condensation, spread through it, was counted at its 380 K edge. Leaving it out (`dry_edge`) brought the tropical top layer from 1407 to 1033 kg/m².

**Faults found while verifying it:**

| # | Where, when | What happened | Fix |
|---|---|---|---|
| 122 | Sumatra, day 47 after #121's equilibrium | Layer 1's mean 372 → 380 K in 8 steps under radiative cooling; then all 5612 kg/m² crossed 380 K | The reached layer's lower edge is the mixed layer's θ: #120's slices were taken at 305 K from air no colder than 323 K, a heat pump |
| 123 | Tibet, spin-up cycle 2 | The mixed layer's top θ* (326 K) above the cloud layer's top (305 K): negative edge density, the mixed layer's intake became an outflow of all of it | A backward span holds no air at its edges |
| 124 | Amazon, day 16 of a fresh run | The mixed layer held 8000–8500 kg/m² at θ 355 K, condensing 1300–2000 W/m² inside itself | The mixed layer ends at its condensation level; its air above leaves into the cloud layer |

The faults were found by stepping one step at a time with a rolling history and stopping at the first collapsed column (a top layer over 3000 kg/m², a θ below 150 K, or winds over 400 m/s). Restarting from a save does not repeat a run exactly, because radiation counts its steps from the load, so every hunt reran from scratch (scratchpad `eq4w*.py`).

**`earth4` at equilibrium with #121–#124** (measured year; #120 in brackets):

| Measure | Value |
|---|---|
| Ts, net at the top | 289.59 K (294.95), +1.06 W/m² (−3.75) |
| Rain, cover, surface longwave down | 3.55 mm/day (4.44), 0.65 (0.64), 345 W/m² (382) |
| Tropical top layer, up / down across 380 K | 804 kg/m² (1448), 5.9 / 4.1 kg/m²/day (13.1 / 10.7) |
| Layer 1's hottest θ, deepest mixed layer | 340 K (367), 5466 kg/m² (6775) |
| Land 0–15°: rain, evaporation, Ts, soil | 4.36, 3.36 mm/day, 298.9 K, 0.49 (5.18, 4.61, 301.3, 0.76) |
| Land 15–35°: rain, soil | 1.53 mm/day, 0.19 (3.11, 0.41) |
| Tropical sea Ts | 298.9 K (304.5) |

Open: the extratropical top layer stays light (574–684 kg/m²); subtropical land is dry; save/load is not bit-exact.
