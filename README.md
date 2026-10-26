# InSilica World V2

A planet simulator with two moving ocean layers and two moving air layers, run by one 64-bit GPU fluid engine
(CUDA kernels driven from Python through CuPy). A browser page connects to the local engine as the viewer.

- **Plan:** [docs/v2-plan.md](docs/v2-plan.md) — goals, physics, time stepping, resolution and cost, architecture,
  stages, risks and decisions.
- **Spec:** [SPEC.md](SPEC.md) — code layout, numerics, viewer protocol, save format, every stage's tests and
  thresholds, and the decision log.
- **v1:** [goofyboson/insilica-world](https://github.com/goofyboson/insilica-world) — the portable single-file
  JavaScript version (Stages 1–3c), kept as the reference.

## Current status

**V2-0 Foundation built on branch `stage-v2-0`: 41/41 tests pass.** v1's grid, TRiSK geometry, shallow-water
solver and FCT transport run as 64-bit CUDA kernels and match v1's own results; save/load, the viewer, the test
runner and the speed probe work (n = 8 in 64-bit: 3.7 ns per layer-edge-step). Results: the V2-0 section of SPEC.md.

| Stage | Scope | Status |
|---|---|---|
| V2-0 | Foundation: Python package, CuPy kernels, grid port, save/load, WebSocket viewer skeleton, 64-bit speed probe on the RTX 4070 | ✅ 41/41, awaiting merge |
| V2-1 | Fluid core: multi-layer TRiSK, variable thickness, real-speed fast waves (split-explicit), FCT transport | ⏳ Next |
| V2-2 | Atmosphere: two air layers (isentropic interface), radiation, convection, condensation | ⏳ Planned |
| V2-3 | Ocean: two layers, wind stress, thermocline, entrainment, sea ice; aquaplanet first | ⏳ Planned |
| V2-4 | Land and coasts: gyres and boundary currents, surface water, soil, ice sheets, terrain | ⏳ Planned |
| V2-5 | Two resolutions: air n = 6 coupled to ocean n = 8 | ⏳ Planned |
| V2-6 | Moon and tides, geology, UI polish | ⏳ Planned |

## Running it

Needs an NVIDIA GPU with a current driver, Python 3.12+ and (for the v1 comparison tests) Node.js with v1 checked
out next to this repository.

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install -e ".[test]"
```

Engine and viewer (then open http://127.0.0.1:8765/):

```bash
.venv/Scripts/python -m insilica.server --n 5 --preset galewsky
```

Tests (up to 5 processes, then the speed probe alone; prints every measured value against its threshold):

```bash
.venv/Scripts/python tools/run_tests.py
```

Speed probe on its own:

```bash
.venv/Scripts/python -m insilica.probe
```
