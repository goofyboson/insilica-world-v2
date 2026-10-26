"""The engine as a local server: one port serves the viewer page (HTTP) and a WebSocket at /ws.

Protocol (SPEC §5). Text messages are JSON; binary frames carry arrays.
Client → server:
  {"type": "hello"}                                    → info (JSON), then the grid (binary)
  {"type": "get", "field": "h", "layer": 0, "dtype": "f4"|"f8"}   → one field frame (binary)
  {"type": "subscribe", "field": "psi", "layer": 0}    → that field is pushed (f4) after every step or frame
  {"type": "play", "tpf": 4} / {"type": "pause"}       → run tpf ticks per frame (about 30 frames/s) / stop
  {"type": "step", "k": 1}                             → k ticks now
  {"type": "new", "n": 4, "preset": "w5"}              → a new world (then info and grid again)
Server → client: {"type": "info"|"status"|"error", ...} and binary frames.
Binary frame: uint32 little-endian header length H, H bytes of JSON header, zero padding to a multiple of 8, then
the payload. Header: {"kind": "field", "name", "layer", "tick", "dtype", "count", "min", "max"} or
{"kind": "grid", "n", "N", "parts": [{"name", "dtype", "count"}, ...]} with each part padded to 8 bytes.

  python -m insilica.server [--n 4] [--preset w5] [--port 8765]
"""
import argparse
import asyncio
import json
import struct
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from ..world import FIELDS, PRESETS, World

VIEWER = Path(__file__).resolve().parents[1] / 'viewer'
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
    def __init__(self, n=4, preset='w5'):
        self.world = World(n=n, preset=preset)
        self.playing, self.tpf = False, 1
        self.clients = {}            # websocket → subscription (field, layer) or None
        self.lock = asyncio.Lock()

    def info(self):
        w = self.world
        return dict(type='info', n=w.n, N=w.grid.N, E=w.grid.E, preset=w.preset, layers=w.L, dt=w.dt, tick=w.tick,
                    playing=self.playing, tpf=self.tpf, fields=list(FIELDS), presets=list(PRESETS))

    def status(self):
        return dict(type='status', tick=self.world.tick, playing=self.playing, tpf=self.tpf, day=self.world.tick * self.world.dt / 86400)

    def grid_frame(self):
        g = self.world.grid
        return frame(dict(kind='grid', n=g.n, N=g.N, parts=[dict(name='pos', dtype='f4', count=3 * g.N),
                                                            dict(name='corners', dtype='f4', count=18 * g.N),
                                                            dict(name='deg', dtype='u1', count=g.N)]),
                     [g.pos.astype('<f4'), g.corners.astype('<f4'), g.deg.astype('u1')])

    def field_frame(self, name, layer=0, dtype='f4'):
        a = self.world.field(name, layer).get()
        a = a.astype('<f8' if dtype == 'f8' else '<f4')
        return frame(dict(kind='field', name=name, layer=layer, tick=self.world.tick, dtype=dtype, count=a.size,
                          min=float(a.min()), max=float(a.max())), [a])

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
                        async with self.lock:
                            self.world = await asyncio.to_thread(World, int(m.get('n', 4)), m.get('preset', 'w5'))
                        for c in list(self.clients):
                            await c.send(json.dumps(self.info()))
                            await c.send(self.grid_frame())
                            if self.clients.get(c):
                                await c.send(self.field_frame(*self.clients[c]))
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


async def main(n=4, preset='w5', host='127.0.0.1', port=8765, ready=None):
    engine = Engine(n, preset)
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
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args()
    try:
        asyncio.run(main(a.n, a.preset, port=a.port))
    except KeyboardInterrupt:
        pass
