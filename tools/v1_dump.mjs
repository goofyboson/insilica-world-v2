// Reference dump from v1 (read-only): loads v1's sim core from ../insilica-world/insilica-world.html and
// writes its grid, TRiSK geometry and kernel results as .npy files, which tests/test_v1_match.py
// compares with V2's ports (tests G4, K1). Nothing in v1 is written.
//   node tools/v1_dump.mjs <out-dir> [v1-dir]
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';

const out = process.argv[2];
const v1 = process.argv[3] || path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(\w:)/, '$1')), '..', '..', 'insilica-world');
if (!out) { console.error('usage: node tools/v1_dump.mjs <out-dir> [v1-dir]'); process.exit(2); }
const html = fs.readFileSync(path.join(v1, 'insilica-world.html'), 'utf8');
const m = html.match(/<script id="sim-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('sim-core script not found in v1'); process.exit(2); }
vm.runInThisContext(m[1], { filename: 'sim-core.js' });
const S = globalThis.SimCore, K = S.K, TR = S.trisk, GRAV = 9.80616;

// Minimal .npy writer (format 1.0, little-endian, C order).
function npy(file, arr, shape) {
  const descr = arr instanceof Float64Array ? '<f8' : arr instanceof Int32Array ? '<i4' : arr instanceof Int8Array ? '|i1' : arr instanceof Uint8Array ? '|u1' : null;
  if (!descr) throw new Error(`npy: unsupported array for ${file}`);
  const shp = shape || [arr.length];
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shp.join(', ')}${shp.length === 1 ? ',' : ''}), }`;
  const pad = 64 - ((10 + header.length + 1) % 64);
  header += ' '.repeat(pad % 64) + '\n';
  const pre = Buffer.alloc(10);
  pre.write('\x93NUMPY', 0, 'latin1'); pre[6] = 1; pre[7] = 0; pre.writeUInt16LE(header.length, 8);
  fs.writeFileSync(file, Buffer.concat([pre, Buffer.from(header, 'latin1'), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)]));
}

let commit = 'unknown';
try { commit = execFileSync('git', ['-C', v1, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(); } catch { /* not a git checkout */ }
const meta = { v1Dir: v1, v1Commit: commit, version: S.VERSION || null, grids: {}, runs: {} };

for (let n = 2; n <= 5; n++) {
  const t0 = Date.now();
  const g = S.buildGrid(n), T = TR.getTrisk(g), d = path.join(out, `n${n}`);
  fs.mkdirSync(d, { recursive: true });
  const N = g.N, E = g.E;
  for (const [k, a, s] of [
    ['pos', g.pos, [N, 3]], ['lat', g.lat], ['lon', g.lon], ['area', g.area], ['east', g.east, [N, 3]], ['north', g.north, [N, 3]],
    ['nbr', g.nbr, [N, 6]], ['deg', g.deg], ['dist', g.dist, [N, 6]], ['edgeLen', g.edgeLen, [N, 6]], ['edgeDir', g.edgeDir, [N, 6, 3]],
    ['corners', g.corners, [N, 6, 3]], ['eI', g.eI], ['eJ', g.eJ], ['eKi', g.eKi], ['eKj', g.eKj],
    ['cornerV', T.cornerV, [N, 6]], ['cellEdge', T.cellEdge, [N, 6]], ['cellSign', T.cellSign, [N, 6]], ['ev1', T.ev1], ['ev2', T.ev2],
    ['de', T.de], ['le', T.le], ['xe', T.xe, [E, 3]], ['ne', T.ne, [E, 3]], ['kite', T.kite, [N, 6]], ['areaCell', T.areaCell],
    ['areaV', T.areaV], ['circ1', T.circ1], ['circ2', T.circ2], ['fV', T.fV], ['eoeStart', T.eoeStart], ['eoe', T.eoe], ['w', T.w],
  ]) npy(path.join(d, `${k}.npy`), a, s);
  meta.grids[n] = { N, E, NV: T.NV, totalArea: g.totalArea, seconds: (Date.now() - t0) / 1000 };
}

// K1: Williamson 2 at n = 4 (v1's w2Setup and N1), states after 0, 1 and 960 steps of 450 s.
{
  const g = S.buildGrid(4), T = TR.getTrisk(g), a = K.R_PLANET, Om = K.OMEGA, u0 = 2 * Math.PI * a / (12 * 86400), gh0 = 2.94e4;
  const h = new Float64Array(g.N), b = new Float64Array(g.N), u = new Float64Array(g.E);
  for (let i = 0; i < g.N; i++) { const s = Math.sin(g.lat[i]); h[i] = (gh0 - (a * Om * u0 + u0 * u0 / 2) * s * s) / GRAV; }
  for (let e = 0; e < g.E; e++) {
    const x = [T.xe[3 * e], T.xe[3 * e + 1], T.xe[3 * e + 2]];
    const lat = Math.asin(x[2]), lon = Math.atan2(x[1], x[0]);
    u[e] = u0 * Math.cos(lat) * (-Math.sin(lon) * T.ne[3 * e] + Math.cos(lon) * T.ne[3 * e + 1]);
  }
  const d = path.join(out, 'w2');
  fs.mkdirSync(d, { recursive: true });
  npy(path.join(d, 'h0.npy'), h); npy(path.join(d, 'u0.npy'), u);
  const sol = TR.makeSolver(g, { grav: GRAV, b, nu4: 0 });
  const Fm = new Float64Array(g.E);
  sol.step(h, u, 450, Fm);
  npy(path.join(d, 'h1.npy'), h); npy(path.join(d, 'u1.npy'), u); npy(path.join(d, 'Fm1.npy'), Fm);
  for (let s = 1; s < 960; s++) sol.step(h, u, 450);
  npy(path.join(d, 'h960.npy'), h); npy(path.join(d, 'u960.npy'), u);
  // The same with ∇⁴ hyperviscosity and the positivity limiter both active (a steep mountain and a high
  // ν4), so the ported kernels' optional branches are compared too.
  const h2 = new Float64Array(g.N), b2 = new Float64Array(g.N), u2 = new Float64Array(g.E);
  for (let i = 0; i < g.N; i++) { b2[i] = Math.max(0, 5000 * (1 - Math.acos(Math.max(-1, Math.min(1, g.pos[3 * i + 2] * 0.5 + g.pos[3 * i] * Math.sqrt(0.75)))) / 0.5)); h2[i] = Math.max(1, 5100 - b2[i] - 600 * g.pos[3 * i + 2] ** 2); }
  for (let e = 0; e < g.E; e++) u2[e] = 3 * u[e];
  npy(path.join(d, 'b2.npy'), b2); npy(path.join(d, 'h2_0.npy'), h2); npy(path.join(d, 'u2_0.npy'), u2);
  const sol2 = TR.makeSolver(g, { grav: GRAV, b: b2, nu4: 1e15, drag: 1e-5 });
  for (let s = 0; s < 10; s++) sol2.step(h2, u2, 450);
  npy(path.join(d, 'h2_10.npy'), h2); npy(path.join(d, 'u2_10.npy'), u2);
}

// K1/K2: cosine bell once around (v1's N6), FCT3 and upwind; states after 1 and 576 steps.
for (const order of [1, 3]) {
  const g = S.buildGrid(4), T = TR.getTrisk(g), a = K.R_PLANET, u0 = 2 * Math.PI * a / (12 * 86400), rB = a / 3, lc = 3 * Math.PI / 2, alpha = Math.PI / 2 - 0.05;
  const psi = new Float64Array(g.N), M = Float64Array.from(T.areaCell), M1 = new Float64Array(g.N), Fm = new Float64Array(g.E);
  for (let i = 0; i < g.N; i++) {
    const dist = a * Math.acos(Math.max(-1, Math.min(1, Math.cos(g.lat[i]) * Math.cos(g.lon[i] - lc))));
    psi[i] = dist < rB ? 500 * (1 + Math.cos(Math.PI * dist / rB)) : 0;
  }
  const om = [-Math.sin(alpha) * u0 / a, 0, Math.cos(alpha) * u0 / a], steps = 576, dt = 12 * 86400 / steps;
  for (let e = 0; e < g.E; e++) {
    const x = [T.xe[3 * e] * a, T.xe[3 * e + 1] * a, T.xe[3 * e + 2] * a];
    const V = [om[1] * x[2] - om[2] * x[1], om[2] * x[0] - om[0] * x[2], om[0] * x[1] - om[1] * x[0]];
    Fm[e] = (V[0] * T.ne[3 * e] + V[1] * T.ne[3 * e + 1] + V[2] * T.ne[3 * e + 2]) * T.le[e] * dt;
  }
  const d = path.join(out, `bell${order}`);
  fs.mkdirSync(d, { recursive: true });
  npy(path.join(d, 'psi0.npy'), psi); npy(path.join(d, 'Fm.npy'), Fm);
  const sc = TR.transportScratch(g);
  for (let s = 0; s < steps; s++) {
    M1.set(M); for (let e = 0; e < g.E; e++) { M1[g.eI[e]] -= Fm[e]; M1[g.eJ[e]] += Fm[e]; }
    TR.transportStep(g, M, M1, Fm, psi, order, sc);
    M.set(M1);
    if (s === 0) npy(path.join(d, 'psi1.npy'), psi);
  }
  npy(path.join(d, 'psi576.npy'), psi);
}

fs.writeFileSync(path.join(out, 'meta.json'), JSON.stringify(meta, null, 2));
console.log(`v1 dump written to ${out} (v1 ${commit.slice(0, 7)})`);
