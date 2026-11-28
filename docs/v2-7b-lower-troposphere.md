# V2-7b: the lower troposphere and the upper ocean

Agreed 2026-11-27 (SPEC #305). This is the design for the stage's first part, the mixed layer, and the plan for the rest. It is
reviewed before any code is written.

## 1. Why

The desert-rain trace (SPEC #281–#304) ends in two pieces of structure that no setting on the present layers can supply:

- **The lowest troposphere.** The mixed layer (the bottom air layer, `bl_mix`) mixes dry θ and vapor evenly and stops at its
  condensation level by rule (#124). So it holds no cloud (`ml_clear`, #157). The cloud that should crown it, stratocumulus
  under an inversion and trade cumulus at cloud base, can only sit in the cumulus layer above. Over the stratocumulus seas
  that layer is boundary-layer air blended with inversion air: 4–6 K warmer than the mixed layer and at 0.56 of saturation.
  It cuts the sunlight by 28 W/m², against ~60–90 observed (#302). Over the tropical sea the mixed layer is dry (relative
  humidity 0.69 against ~0.8) and 2.5 K under the sea, so it evaporates ~20% too much and holds the coupled tropics 5–6 K
  cold (#289–#293).
- **The upper ocean.** The 700 m layer under the ocean's mixed layer evens out the upper ocean. The tropics end up cold, the
  high latitudes warm, and the currents carry 0.4–0.6 PW poleward against ~1.5–2 (#289).

Each change tried on the present layers that moistened the low air also thinned the low cloud. The planet then overheated
(#157, #214, #298).

## 2. Part 1: a mixed layer with a cloudy top (`ml_moist`)

### 2.1 What changes

Real boundary layers are well mixed in the quantities that condensation conserves: the liquid-water potential temperature
θ_l = θ − (L_v / c_p Π) l and the total water q_t = q + l. Both are even through the layer. Below the condensation level the
air is clear. Above it the air is cloud, saturated at each level with l(p) = q_t − q_s(T(p)), where T(p) comes from θ_l and
q_t by saturation adjustment. This is how a stratocumulus-topped layer holds its cloud (Lilly 1968).

With `ml_moist` on, the mixed layer:

1. **Its cloud from θ_l and q_t.** `mixed_cloud`'s four levels take each level's temperature from (θ_l, q_t) by saturation
   adjustment, instead of from the layer's mean θ. Today the mean θ, warmed by its own cloud's latent heat, is applied at
   every level, so the cloud undercounts itself. Each level's cloud is its saturation excess (as `ml_clear`'s), its cover 1
   where saturated. `cloud_adjust` brings the layer's prognostic cloud water to the levels' mean, as it does now.
   - θ_l is the layer's θ less L_v·l/(c_p Π̄), from the cloud water it holds. Condensation and evaporation conserve it exactly.
   - The prognostic variables stay μ, θ, q and cloud water. Nothing new is carried.
2. **May reach past its condensation level where an inversion holds it.** #124's release ("air above cloud base is cloud:
   it leaves through the top") becomes a buoyancy test.
   - The cloudy top's virtual θ (θ_l, q_t and l at the top level, saturated) is compared with the virtual θ of the air
     above, at the next layer's lower edge, by `edge_q` and the edge rules.
   - If it is colder, the inversion holds it: the air stays and the mixed layer is stratocumulus-topped.
   - If it is buoyant, it is cumulus and leaves as now. That is the Amazon case #124 was made for, which the test still
     catches.
   - The condensation level's other uses (the room for its growth, the compensating subsidence's fill, the night's floor)
     follow the same test (#303 showed they must move together).
   - A margin (K) on the buoyancy test is the only new constant. It is set from physics: the cloud-top entrainment
     instability criterion (Randall 1980, Deardorff 1980) bounds when such a top breaks up. Stated with its source when
     built.
3. **Its entrainment driven by its cloud top too.** The buoyancy flux in #98's entrainment gains the cloudy top's longwave
   cooling (#304's term: the layer's cover × its radiative cooling) and the evaporative-cooling term of the jump. The jump
   to the air above is taken in virtual θ.
4. **The plumes' closure unchanged.** `buoy_v` (#145) already scales the cumulus base mass flux by the mixed layer's top
   saturated share. Under an inversion the plumes find no buoyancy above the top and stop, so a stratocumulus layer gets no
   cumulus. In the trades the top is buoyant and cumulus leave as now.

### 2.2 What must stay exact

- **Energy:** every move keeps the column's energy (`column_energy`); the latent heat stays in θ through the cloud water.
- **Water:** vapor + cloud water + rain + evaporation balance per column (CONSERVE air ≤ 1e-13).
- **Off is bit-identical:** with `ml_moist` 0, earth9 gives the same hashes (5984c6cb6a071387 a day from the test's
  equilibrium), and mars and every other preset are unchanged.
- **Mixed precision:** the CONSERVE ledgers hold as now.

### 2.3 Measured, before any equilibrium (the observed-sea trials, as #289–#304)

A year (or 60 + 30 days for first looks) with WOA23's annual surface relaxed daily, from the test's equilibrium:

- **Tropical sea (10°S–10°N):** evaporation (now 5.02 mm/day, observed ~3.5–4), the mixed layer's relative humidity (0.69,
  ~0.8), the air–sea temperature gap (2.5 K, ~1), sensible heat (17 W/m², ~8–10).
- **Stratocumulus seas (California, Peru, Namibia):** the mixed layer's depth and cloud water path, total cover, cloud
  effect on sunlight (−28 W/m², ~−60 to −90; against CERES once downloaded).
- **The trades (10–30°) and 30–50°:** cloud effect on sunlight; the rain bands against GPCP.
- **The coupled year from the same state:** the sea's net heat (now +4.95 W/m² globally), to see whether the planet warms
  or cools before committing to an equilibrium (#298's lesson).

### 2.4 Then, only if needed: a cloud-base transition layer (`earth10`)

If the plumes still stall ~25 hPa above cloud base (#180, #213: the cumulus layer's lower edge pinned to the mixed layer's θ),
a tenth air layer is added between the mixed layer and the cumulus layer.

- The code allows 10 layers (`LMAX` 10 in layers.cu and drag.cu, air.cu compiled per L). The cost is about 11% more run time.
- About 40 places in air.cu treat "the layer above the mixed layer" (bot − 1) as the cumulus layer. Each is audited and
  assigned to the transition layer or the cumulus layer (bot − 2) before the preset is built.

## 3. Part 2: an upper-ocean thermocline layer

After Part 1's air, the 700 m layer under the ocean's mixed layer is split into an upper thermocline layer (~100–200 m) and
the rest.

- The mixed layer then sits on warm water it takes back in season, so the equator-to-pole contrast can survive.
- The earlier splits overheated the tropics (#189–#190: 30–33 °C) or drained their warm water (#193–#199). They ran with
  the present air, whose excess evaporation (#290) and missing low cloud (#302) they could not offset.
- The design (layer depths, density steps, start from WOA23's profile, #193's method) is written after Part 1's numbers,
  as an addendum here.
- **Measured against:** coupled SST against WOA23 (#288's bias map: global −1.57 K, rms 4.68), the ocean's heat transport
  (now 0.4–0.6 PW), the sea's heat uptake.

## 4. Part 3: the new Earth

A preset with the stage's switches on, its own cached equilibrium (the test's procedure: 3 coupled years, 5 × (1 + 20),
the year), then the full suite with `--perf`.

## 5. Tests (lean suite: extended, not added; limits agreed after the first numbers)

- **RAIN's run gains an observed-sea year:** after the coupled year, the same state for a year under WOA23's annual
  surface (relaxed daily). Reported and limited:
  - tropical-sea evaporation;
  - the mixed layer's relative humidity there;
  - the cloud effect on sunlight over the stratocumulus seas and in 10° bands, against CERES EBAF-TOA (download to ask
    for: file, source and size);
  - the rain bands under the observed sea (which #287 showed separate the air's rain errors from the ocean's).

  It adds ~7 minutes to the EARTH/RAIN/WINDS run.
- **EARTH's coupled year adds SST against WOA23:** zonal-mean bias in 10° bands 60°S–60°N and the global rms, from the
  data already cached (cache/woa23).
- **CONSERVE air** covers `ml_moist`'s moves (energy and water exact).
- **Everything else unchanged:** MARS has no water, so the switch is inert there.

## 6. Steps and pauses

1. Build `ml_moist` (2.1) behind its switch, off bit-identical; CONSERVE with it on. **Pause:** report.
2. Observed-sea trials (2.3) and the coupled year's sign. **Pause:** report; decide on `earth10` (2.4).
3. The observed-sea test and the SST check built; CERES asked for. **Pause:** first numbers, limits agreed.
4. Part 2's design addendum, then the ocean trials. **Pause.**
5. The new Earth's equilibrium and the full suite with `--perf`. **Pause:** V2-7b closes or carries named failures.

## 7. Risks

- **Runaway cloud or clearing:** a stratocumulus layer's cloud and its entrainment feed each other. The buoyancy test's
  margin and the room it is given decide whether the deck holds or breaks up. Measured by the cover's day-to-day spread in
  the trials.
- **The planet's balance:** more low cloud cools. If EARTH's absorbed sunlight (now −5.3%) falls further, the warming
  needed elsewhere (the tropical sea) must come from the evaporation's fix, not from thinning cloud again. This is checked
  in the coupled year before any equilibrium.
- **Run time:** none for Part 1 (the same layers); ~11% for `earth10`; small for a fourth ocean layer.
