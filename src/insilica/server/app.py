"""The engine as a local server: one port serves the viewer page (HTTP) and a WebSocket at /ws.

Protocol (SPEC §5). Text messages are JSON; binary frames carry arrays.
Client → server:
  {"type": "hello"}                                    → info (JSON), then the grid (binary)
  {"type": "get", "field": "h", "layer": 0, "dtype": "f4"|"f8"}   → one field frame (binary)
  {"type": "subscribe", "field": "psi", "layer": 0}    → that field is pushed (f4) after every step or frame
  {"type": "play", "tpf": 4} / {"type": "pause"}       → run tpf ticks per frame (about 30 frames/s) / stop
  {"type": "step", "k": 1}                             → k ticks now
  {"type": "new", "n": 4, "preset": "w5", "params": {...}}  → a new world (then info and grid again); params (V2-6g): the
        planet's settings (PARAMS: planet radius, omega, grav, day_s, year_s, S0, tilt, ecc, lam_peri, gases; tides, gas_tracers, geology)
  {"type": "save", "name": "x"} / {"type": "load", "name": "x"}  → the world saved to / loaded from cache/saves/x.iw2 (V2-6g)
Server → client: {"type": "info"|"status"|"error", ...} and binary frames. A status carries `sky` (World.sky(): the Sun's and the
Moon's directions in the planet's frame, as the model places them) for air worlds; info the planet's settings and the saves.
Binary frame: uint32 little-endian header length H, H bytes of JSON header, zero padding to a multiple of 8, then
the payload. Header: {"kind": "field", "name", "layer", "tick", "dtype", "count", "min", "max"} or
{"kind": "grid", "n", "N", "parts": [{"name", "dtype", "count"}, ...]} with each part padded to 8 bytes.

  python -m insilica.server [--n 4] [--preset w5] [--load world.iw2] [--port 8765]   (--load: start from a saved world)
"""
import argparse
import asyncio
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from ..world import FIELDS, PRESETS, World

VIEWER = Path(__file__).resolve().parents[1] / 'viewer'
# (V2-6g) what a new world's params may set from the browser: the planet's settings, and the tides, gas tracers and geology switches
PLANET_KEYS = {'radius', 'omega', 'grav', 'day_s', 'year_s', 'S0', 'tilt', 'ecc', 'lam_peri', 'gases'}
GAS_KEYS = {'n2', 'o2', 'ar', 'co2', 'ch4', 'h2', 'he', 'ne'}
MOON_KEYS = {'gm', 'a', 'e', 'i'}


def clean_params(p):
    """A browser's new-world params, kept to PLANET_KEYS (numbers; gases a dict of known gases) and the switches tides (True, False
    or a dict with moon: None or MOON_KEYS), gas_tracers and geology; anything else is refused."""
    num = lambda v: float(v) if isinstance(v, (int, float)) and math.isfinite(v) else (_ for _ in ()).throw(ValueError(f'not a number: {v!r}'))
    out = {}
    for k, v in (p or {}).items():
        if k == 'planet':
            pl = {}
            for a, b in v.items():
                if a not in PLANET_KEYS:
                    raise ValueError(f'planet setting {a!r} not allowed')
                pl[a] = {g: num(x) for g, x in b.items() if g in GAS_KEYS} if a == 'gases' else num(b)
            out['planet'] = pl
        elif k == 'tides':
            if isinstance(v, bool):
                out['tides'] = v
            else:
                m = v.get('moon', {})
                out['tides'] = {'moon': None if m is None else {a: num(b) for a, b in m.items() if a in MOON_KEYS}}
        elif k in ('gas_tracers', 'geology'):
            out[k] = 1 if v else 0
        else:
            raise ValueError(f'param {k!r} not allowed')
    if not out.get('tides', True):
        out.pop('tides')
    return out
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'), '/index.html': ('index.html', 'text/html; charset=utf-8'),
          '/vendor/three.min.js': ('vendor/three.min.js', 'application/javascript')}
FRAME_S = 1 / 30


def frame(header, parts):
    """A binary frame: header length, JSON header, padding, then each part padded to 8 bytes."""
    h = json.dumps(header, separators=(',', ':')).encode()
    out = [struct.pack('<I', len(h)), h]
    n = 4 + len(h)
    out.append(b'\0' * (-n % 8))
    for p in parts:
        b = np.ascontiguousarray(p).tobytes()
        out += [b, b'\0' * (-len(b) % 8)]
    return b''.join(out)


def parse_frame(data):
    """Inverse of frame() for tests and tools: returns (header, [arrays])."""
    (hl,) = struct.unpack_from('<I', data, 0)
    header = json.loads(data[4:4 + hl])
    off = 4 + hl
    off += -off % 8
    parts = header['parts'] if header['kind'] == 'grid' else [dict(name=header['name'], dtype=header['dtype'], count=header['count'])]
    arrays = []
    for p in parts:
        dt = np.dtype({'f4': '<f4', 'f8': '<f8', 'u1': 'u1'}[p['dtype']])
        a = np.frombuffer(data, dtype=dt, count=p['count'], offset=off)
        arrays.append(a)
        off += a.nbytes
        off += -off % 8
    return header, arrays


class Engine:
    def __init__(self, n=4, preset='w5', load=None, saves=None):
        self.world = World.load(load) if load else World(n=n, preset=preset)
        from ..grid import CACHE_DIR
        self.saves = Path(saves) if saves else CACHE_DIR / 'saves'      # (V2-6g) where the browser saves and loads
        self.playing, self.tpf = False, 1
        self.clients = {}            # websocket → subscription (field, layer) or None
        self.lock = asyncio.Lock()

    def info(self):
        w = self.world
        planet = dict(w.planet)
        if w.air:
            p = w.physics.p
            planet.update({k: float(p[k]) for k in ('day_s', 'year_s', 'S0', 'tilt', 'ecc', 'lam_peri') if k in p})
            planet['gases'] = (w.params.get('planet', {}) or {}).get('gases') or (w.params.get('physics', {}) or {}).get('gases')
        return dict(type='info', n=w.n, N=w.grid.N, E=w.grid.E, preset=w.preset, layers=w.L, dt=w.dt, tick=w.tick,
                    playing=self.playing, tpf=self.tpf, fields=w.fields(), presets=list(PRESETS), planet=planet,
                    tides=getattr(w, 'tides', None) is not None, sky=w.sky(), soil_layers=int(w.sea.Tg.shape[0]) if w.sea is not None and w.sea.soil is not None else 0,
                    saves=sorted(f.stem for f in self.saves.glob('*.iw2')) if self.saves.exists() else [])

    def status(self):
        w = self.world
        return dict(type='status', tick=w.tick, playing=self.playing, tpf=self.tpf, day=w.tick * w.dt / 86400, sky=w.sky())

    def save_path(self, name):
        """cache/saves/<name>.iw2 for a plain name (letters, digits, - and _)."""
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', str(name)):
            raise ValueError('a save name is 1-64 letters, digits, - or _')
        return self.saves / f'{name}.iw2'

    async def broadcast_world(self):
        for c in list(self.clients):
            await c.send(json.dumps(self.info()))
            await c.send(self.grid_frame())
            if self.clients.get(c):
                await c.send(self.field_frame(*self.clients[c]))

    def grid_frame(self):
        g = self.world.grid
        return frame(dict(kind='grid', n=g.n, N=g.N, parts=[dict(name='pos', dtype='f4', count=3 * g.N),
                                                            dict(name='corners', dtype='f4', count=18 * g.N),
                                                            dict(name='deg', dtype='u1', count=g.N)]),
                     [g.pos.astype('<f4'), g.corners.astype('<f4'), g.deg.astype('u1')])

    def field_frame(self, name, layer=0, dtype='f4'):
        a = self.world.field(name, layer).get()
        a = a.astype('<f8' if dtype == 'f8' else '<f4')
        ok = np.isfinite(a)                     # a blown-up run must not put NaN into the JSON header
        lo, hi = (float(a[ok].min()), float(a[ok].max())) if ok.any() else (0.0, 0.0)
        return frame(dict(kind='field', name=name, layer=layer, tick=self.world.tick, dtype=dtype, count=a.size,
                          min=lo, max=hi, finite=bool(ok.all())), [a])

    async def push(self):
        for ws, sub in list(self.clients.items()):
            try:
                await ws.send(json.dumps(self.status()))
                if sub:
                    await ws.send(self.field_frame(*sub))
            except Exception:
                self.clients.pop(ws, None)

    async def advance(self, k):
        async with self.lock:
            await asyncio.to_thread(self._step, k)

    def _step(self, k):
        self.world.step(k)
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

    async def loop(self):
        while True:
            if self.playing:
                await self.advance(self.tpf)
                await self.push()
            await asyncio.sleep(FRAME_S)

    async def handle(self, ws):
        self.clients[ws] = None
        try:
            async for msg in ws:
                try:
                    m = json.loads(msg)
                    t = m.get('type')
                    if t == 'hello':
                        await ws.send(json.dumps(self.info()))
                        await ws.send(self.grid_frame())
                    elif t == 'get':
                        await ws.send(self.field_frame(m['field'], int(m.get('layer', 0)), m.get('dtype', 'f4')))
                    elif t == 'subscribe':
                        self.clients[ws] = (m['field'], int(m.get('layer', 0)))
                        await ws.send(self.field_frame(*self.clients[ws]))
                    elif t == 'play':
                        self.playing, self.tpf = True, max(1, int(m.get('tpf', self.tpf)))
                        await ws.send(json.dumps(self.status()))
                    elif t == 'pause':
                        self.playing = False
                        await ws.send(json.dumps(self.status()))
                    elif t == 'step':
                        await self.advance(max(1, int(m.get('k', 1))))
                        await self.push()
                    elif t == 'new':
                        self.playing = False
                        params = clean_params(m.get('params'))
                        async with self.lock:
                            self.world = await asyncio.to_thread(World, int(m.get('n', 4)), m.get('preset', 'w5'), params)
                        await self.broadcast_world()
                    elif t == 'save':
                        f = self.save_path(m.get('name'))
                        f.parent.mkdir(parents=True, exist_ok=True)
                        async with self.lock:
                            await asyncio.to_thread(self.world.save, f)
                        await self.broadcast_world()
                    elif t == 'load':
                        f = self.save_path(m.get('name'))
                        self.playing = False
                        async with self.lock:
                            self.world = await asyncio.to_thread(World.load, f)
                        await self.broadcast_world()
                    else:
                        raise ValueError(f'unknown message type {t!r}')
                except Exception as e:      # report and keep the connection
                    await ws.send(json.dumps(dict(type='error', message=f'{type(e).__name__}: {e}')))
        finally:
            self.clients.pop(ws, None)


def http(connection, request):
    path = request.path.split('?')[0]
    if path == '/ws':
        return None
    if path in STATIC:
        f, ctype = STATIC[path]
        body = (VIEWER / f).read_bytes()
        return Response(200, 'OK', Headers([('Content-Type', ctype), ('Content-Length', str(len(body))), ('Cache-Control', 'no-cache')]), body)
    return connection.respond(404, 'Not found\n')


async def main(n=4, preset='w5', host='127.0.0.1', port=8765, ready=None, load=None, saves=None):
    engine = Engine(n, preset, load, saves)
    n, preset = engine.world.n, engine.world.preset
    async with serve(engine.handle, host, port, process_request=http, max_size=2 ** 26, compression=None) as server:
        loop_task = asyncio.create_task(engine.loop())
        actual = server.sockets[0].getsockname()[1]
        if ready is not None:
            ready(actual, engine)
        else:
            print(f'InSilica World V2 engine: n = {n}, preset {preset}. Open http://{host}:{actual}/', flush=True)
        try:
            await asyncio.Future()
        finally:
            loop_task.cancel()


def cli():
    ap = argparse.ArgumentParser(description='InSilica World V2 engine with the browser viewer')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--preset', default='w5', choices=PRESETS)
    ap.add_argument('--load', help='start from this saved world (.iw2) instead of a new one')
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    try:
        asyncio.run(main(a.n, a.preset, port=a.port, load=a.load))
    except KeyboardInterrupt:
        pass
