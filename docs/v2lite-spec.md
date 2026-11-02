# V2Lite: small spec

A distillation of V2's climate model into one HTML file that runs in a browser. Its own project and repository, outside
V2's stages. Not started; it waits until V2's climate model is put together and generalized.

## Scope

- **In:** the climate model only: air, ocean, land, clouds and rain, on a chosen world, with a globe viewer.
- **Out, for now:** the geology model, Elemental Guardians, and anything that needs a server or an install.

## Approach

- One self-contained `.html` file: physics, viewer, grid and starting state inside it, no network needed.
- Physics as WebGPU compute shaders (WGSL) driven from JavaScript, in 32-bit, since WebGPU has no general 64-bit floats.
- About n = 4 to start, with n = 5 if speed allows. A clear message where WebGPU is missing.
- Embedded data, compressed: the grid (or code that builds it), the Earth table, and a spun-up starting climate.
- Keep it simple first: the fewest air and ocean layers and the least physics that still pass the checks below. V2's
  optimization stage decides what is safe to drop or simplify.

## Checks

V2 at 64 bits is the reference, so a port is judged against it, not by eye.

- **Column physics:** single kernels match saved V2 test cases, within what 32-bit allows.
- **Climate:** global temperature, energy balance, rain and its pattern, winds and clouds match V2 at the same n, within
  limits to set when the project starts.
- **Budgets:** water and energy close as well as 32-bit allows, with drift measured over long runs.
- **Practical:** file size and speed measured on real machines; targets to set when the project starts.

## What V2 should keep doing meanwhile

- Keep column physics simple and self-contained, so it ports cleanly.
- Keep the viewer's fields and controls behind a simple interface that doesn't depend on Python.
- Record which parts survive 32-bit in V2's first 32-bit trial and at its final sign-off.
- Save small reference cases when the climate model is finished.

## Open questions

- Which layer counts and physics are the smallest set that still behaves like V2.
- How long a spun-up start lasts before 32-bit drift shows.
- Whether a learned emulator is worth its file size for any part.
