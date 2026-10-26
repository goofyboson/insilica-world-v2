# InSilica World V2

A planet simulator with two moving ocean layers and two moving air layers, run by one 64-bit GPU fluid engine
(CUDA kernels driven from Python through CuPy). A browser page connects to the local engine as the viewer.

- **Plan:** [docs/v2-plan.md](docs/v2-plan.md) — goals, physics, time stepping, resolution and cost, architecture,
  stages, risks and decisions.
- **v1:** [goofyboson/insilica-world](https://github.com/goofyboson/insilica-world) — the portable single-file
  JavaScript version (Stages 1–3c), kept as the reference.

## Current status

**Planning complete; nothing built yet.** Next: V2-0 Foundation.

| Stage | Scope | Status |
|---|---|---|
| V2-0 | Foundation: Python package, CuPy kernels, grid port, save/load, WebSocket viewer skeleton, 64-bit speed probe on the RTX 4070 | ⏳ Next |
| V2-1 | Fluid core: multi-layer TRiSK, variable thickness, real-speed fast waves (split-explicit), FCT transport | ⏳ Planned |
| V2-2 | Atmosphere: two air layers (isentropic interface), radiation, convection, condensation | ⏳ Planned |
| V2-3 | Ocean: two layers, wind stress, thermocline, entrainment, sea ice; aquaplanet first | ⏳ Planned |
| V2-4 | Land and coasts: gyres and boundary currents, surface water, soil, ice sheets, terrain | ⏳ Planned |
| V2-5 | Two resolutions: air n = 6 coupled to ocean n = 8 | ⏳ Planned |
| V2-6 | Moon and tides, geology, UI polish | ⏳ Planned |
