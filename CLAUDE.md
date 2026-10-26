# InSilica World V2

A planet simulator: two moving ocean layers and two moving air layers on one 64-bit GPU fluid engine (CUDA kernels
driven from Python through CuPy), with a browser page as the viewer. The plan, its decisions and the stage series
are in [docs/v2-plan.md](docs/v2-plan.md). Read it first.

## Working rules (agreed with the user)

- **Build in stages and pause after every stage.** Before building a stage, restate its tests and plan and flag
  anything that needs a decision; wait for the user's go. After a stage, report test results and wait again.
- **Commit as we go** (checkpoints the user can step back to) and push to GitHub
  (`https://github.com/goofyboson/insilica-world-v2.git`, once the user has created it).
- **Keep README.md current:** the full stage table and the current progress.
- **Testing method: least redundancy and run time, most integrity** (SPEC §7, decision #22). One test per link of the
  integrity chain, each as strong as possible; nothing a stronger test already implies. A new stage extends an
  existing test where it can, adds one only for what no test can see, and retires checks it supersedes. The default
  suite stays fast (seconds warm); the speed probe runs with `--perf` at every stage close.
- **Never change test thresholds or spec rules without asking.** Record every agreed change, with date and reason,
  in the decision log (SPEC §18 in v1; this project's equivalent once its SPEC exists).
- **Report with numbers:** measured values against thresholds, not just pass/fail.
- **Grid cache:** grids are cached under `GRID_VERSION` (`src/insilica/grid/__init__.py`); bump it whenever a change
  alters grid numbers, or stale grids are reused (the n = 8 grid takes about an hour to rebuild; SPEC #31).
- **Long runs in the background,** at most 5 processes at once (the CPU has 6 physical cores; oversubscribing
  doubled every run's time in v1).

## v1: read-only reference

v1 lives next to this repository at `../insilica-world` (GitHub `goofyboson/insilica-world`). Read it freely,
never modify it. Useful starting points:

- `SPEC.md` — the original build spec; **§18 is the decision log (#1–#52)** with every agreed deviation and why.
- `README.md` — stage table; v1 is paused at Stage 3c.
- `src/sim/*.js` — physics in plain JavaScript, built into `insilica-world.html` by `build.mjs`:
  `10-grid.js` (icosahedral grid, Lloyd/SCVT), `12-trisk.js` (TRiSK C-grid geometry), `56-swe.js` (shallow-water
  solver, 3rd-order FCT transport), `55-dynamics.js` (two-layer air dynamics), `20-state.js` (state, radiation
  split, thermal-wind term), `30-energy.js` (radiation, conduction), `40-water.js`, `50-air.js`, `70-phase-c.js`
  (convection, condensation, freeze/melt), `80-ledger-step.js` (tick order, ledgers), `95-save.js` (save/load).
- `test/run-tests.mjs` — every test with its procedure; `test/run-parallel.mjs` — parallel runner;
  `test/hysteresis.mjs` — snowball sweeps.
- `docs/atmosphere-dynamics-plan.md` — earlier dynamics, ocean (Step C) and tides plans.
- Branch `stage-3c` holds the latest v1 work (two-layer atmosphere, 52/56 tests); `main` is v0.3c1.

## Lessons from v1 to carry over

- **Grid:** A-grid (cell-centered) winds were unstable at pentagons and at n = 5; the TRiSK C-grid on a Lloyd/SCVT
  Voronoi mesh fixed it (v1 §18 #45–#46). Positivity needs a per-stage outflow limiter over high terrain.
- **Energy bookkeeping:** energies are stored relative to T_REF = 273.15 K (§18 #12); moving air carries dry static
  energy (heat + g·z) and the air's height energy is in the ledger (§18 #36); every kinetic-energy change is booked
  to air heat. Ledgers close to ~1e-15 in 64-bit.
- **Pressure from temperature:** compare column temperatures reduced to sea level with the lapse rate *from the
  column's mid-height*, or a standard atmosphere drives wind over mountains; start new terrain on the lapse rate.
- **Layer draining:** an invented slow subsidence drained v1's upper air layer; V2 sets layer thickness by physics
  (docs/v2-plan.md §3).
- **Climate facts found:** an all-ocean planet's 60°N seasons are ~6–8 K (land gives 14–19 K); a dry tropical ridge
  becomes a sinking desert high in weak trades; two-layer trades came out weak (~−2 m/s); global rain is
  calibrated by evaporation (K_EVAP 2e-3 gave ~2.3 mm/day).
- **Tests:** cache spin-ups, never measured phases; run groups in parallel; keep thresholds fixed unless agreed.

## Environment

- Windows 11, AMD Ryzen 5 5600X (6 cores / 12 threads), NVIDIA RTX 4070 (12 GB, compute capability 8.9,
  driver 610.88), CUDA toolkits 12.6 and 13.2 installed, no Visual Studio C++ compiler (CuPy compiles kernels at
  run time with NVRTC, so none is needed).
- Python 3.14 is installed; CuPy wheels may need 3.12 or 3.13 alongside it (`py` launcher). V2-0 checks this.
- Shells: Git Bash and PowerShell. Node.js 24 is available (v1's tests run with it).
