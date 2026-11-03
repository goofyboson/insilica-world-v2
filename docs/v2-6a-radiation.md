# V2-6a Radiation (design)

Replaces the gray longwave (`tau0`, `tau_lin`, `lw_w0`) and the shortwave split (`asw`, `sw_tau`) with radiation from the
gases by amount, so the energy budget, its response to CO₂ and vapor, and other worlds' air (V2-10, V2-11) come from
physics, not tuned constants. SPEC #109.

## Method

1. **A radiative column from the layers.** Each air layer's pressure range is split into sub-levels (about 4 per layer);
   temperature follows the layer's θ (or its spread, for a middle layer, as the capacity uses) through T = θ (p/p₀)^κ; vapor
   is distributed in each layer by a fixed relative humidity profile scaled to the layer's mean q; ozone from a profile; CO₂,
   CH₄, N₂O well mixed by amount. The sub-levels' heating is summed back into each layer, so the energy books stay the
   layers' (every flux leaving one sub-level enters another or space or the ground).
2. **Longwave: correlated-k.** About 8 bands from 10 to 3,000 cm⁻¹ (the H₂O rotation band, the CO₂ 15 µm band and its
   wings, the window with the H₂O continuum and O₃ 9.6 µm, CH₄ and N₂O near 7.7 µm, the H₂O 6.3 µm band), a few g-points
   each (about 16–32 in all). Absorption tables in (band, g, p, T, vapor) are made offline by `tools/make_ktables.py` from
   line-by-line spectra: HITRAN line lists (Voigt lines with a 25 cm⁻¹ cut-off) and the MT_CKD water-vapor continuum.
   Gases overlap by random overlap within a band. Two-stream emission with linear-in-τ source, no scattering in the
   longwave (clouds: absorption by water path, as now).
3. **Shortwave: a few bands** (UV-visible, near-IR) with Rayleigh scattering from the column mass, H₂O and O₃ absorption
   from the same line data (O₃'s Chappuis and Hartley–Huggins cross sections), a background haze optical depth, and the
   clouds' two-stream reflection as now, all combined by adding layers.
4. **A stratosphere:** a fourth air layer above the tropopause (θ* about 400 K), where ozone's sunlight heats the air.
5. **Validation offline, then in the model.** The tool's line-by-line spectra are the reference for the k-tables (fluxes
   and heating per sub-level on standard profiles); the model's radiation against the tool's on the same columns; then
   `earth`'s global energy budget against measured values.

## Tests

| Test | Against | Limit (proposed) |
|---|---|---|
| k-tables against line-by-line, US standard, tropical and subarctic-winter columns | the tool's line-by-line | OLR and surface downward longwave within 1%, heating within 0.2 K/day per sub-level |
| CO₂ doubling, instantaneous TOA and tropopause forcing, US standard column | published line-by-line results (Collins et al. 2006, RTMIP) | within 10% |
| Clear-sky OLR, US standard column | published line-by-line (about 265 W/m²) | within 2% |
| `earth` global energy budget, all-sky and clear-sky | measured (Wild et al. 2013 and CERES EBAF): absorbed sunlight 240, OLR 239, clear-sky OLR 266, surface downward longwave 342, surface absorbed sunlight 161 W/m² | within 5% each |
| Budgets | CONSERVE's | unchanged (≤ 1e-13) |

Retired: `tau0`, `tau_lin`, `lw_w0`, `asw`, `sw_tau` (kept only for the gray test cases RCE and Held–Suarez, which are
defined gray).
