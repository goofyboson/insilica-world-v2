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

## Built so far (longwave tables)

- `tools/lbl.py`: line-by-line on the GPU; absorption within 0.1% (mean) of HAPI's. Checked whole against Wijngaarden
  & Happer (2020) in their line shape (no continuum, sech² wings), US standard: OLR 274.0 against 277, 2×CO₂ forcing
  2.94 against 3.0 at the top and 5.44 against 5.5 at the tropopause (W/m²).
- Clear-sky references with MT_CKD (400 ppm): US standard OLR 258.9, surface down 287.4; tropical 286.6 / 390.5;
  subarctic winter 198.5 / 173.9. US standard 2×CO₂: 2.67 at the top, 5.20 at the tropopause, 2.98 at the surface.
- `tools/make_ktables.py`: 10 bands, 136 g-points. Each band's wavenumbers are grouped by k-means on the logarithm of
  the optical depth from the top down and from the surface up in three AFGL columns; a g-point's k per gas is the mean
  over its wavenumbers in (p, T), H₂O at two vapor fractions. The plain mean overstates absorption where k spreads
  inside a g-point, so each g-point's k (all gases alike, so τ stays linear in amounts) carries a factor smooth in log p
  (12 nodes), fitted by Levenberg–Marquardt to line-by-line net fluxes and heating on four columns (US standard 400
  and 800 ppm, tropical, subarctic winter), with the model's single diffusivity angle against three-angle truth.
- Result, including three columns not fitted (tropical 800 ppm; subarctic winter with twice the vapor; US standard
  10 K warmer and twice as wet below 100 hPa): OLR within 0.04%, surface down within 0.16%, heating within 0.10 K/day
  in every sub-level below 1 hPa (0.11–0.44 in the 0.01–1 hPa one); 2×CO₂ forcing within 1.4% at the top and
  tropopause, 4.7% at the surface (tropical).

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
