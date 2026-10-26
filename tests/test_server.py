"""W1: the viewer protocol, driven by a Python WebSocket client against a real engine on a free port."""
import asyncio
import json
import threading
import urllib.request

import numpy as np
import pytest
from websockets.asyncio.client import connect

from insilica.server.app import main, parse_frame

pytestmark = pytest.mark.xdist_group('server')


@pytest.fixture(scope='module')
def server():
    box, started = {}, threading.Event()

    def run():
        loop = asyncio.new_event_loop()
        box['loop'] = loop

        def ready(port, engine):
            box['port'], box['engine'] = port, engine
            started.set()

        box['task'] = loop.create_task(main(n=3, preset='w5', port=0, ready=ready))
        try:
            loop.run_until_complete(box['task'])
        except asyncio.CancelledError:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert started.wait(120)
    yield box
    box['loop'].call_soon_threadsafe(box['task'].cancel)
    t.join(10)


async def recv_json(ws, kind):
    while True:
        m = await ws.recv()
        if isinstance(m, str) and json.loads(m)['type'] == kind:
            return json.loads(m)


async def recv_bin(ws):
    while True:
        m = await ws.recv()
        if isinstance(m, bytes):
            return parse_frame(m)


def test_W1_protocol(record, server):
    """W1 Viewer protocol: connect, grid, a field's bytes equal the engine's array, pause/step/play round trip"""
    port, engine = server['port'], server['engine']
    problems = []

    async def session():
        async with connect(f'ws://127.0.0.1:{port}/ws', max_size=2 ** 26) as ws:
            await ws.send(json.dumps({'type': 'hello'}))
            info = await recv_json(ws, 'info')
            gh, (pos, corners, deg) = await recv_bin(ws)
            if not (gh['N'] == info['N'] == engine.world.grid.N and np.array_equal(deg, engine.world.grid.deg)):
                problems.append('grid')
            if not np.array_equal(pos, engine.world.grid.pos.astype('<f4').ravel()):
                problems.append('grid positions')
            # exact 64-bit field
            await ws.send(json.dumps({'type': 'get', 'field': 'h', 'dtype': 'f8'}))
            fh, (h,) = await recv_bin(ws)
            if h.tobytes() != engine.world.h[0].get().astype('<f8').tobytes():
                problems.append('f8 field bytes')
            await ws.send(json.dumps({'type': 'get', 'field': 'psi'}))
            fh, (p,) = await recv_bin(ws)
            if p.tobytes() != engine.world.psi[0].get().astype('<f4').tobytes():
                problems.append('f4 field bytes')
            t0 = engine.world.tick
            await ws.send(json.dumps({'type': 'step', 'k': 1}))
            st = await recv_json(ws, 'status')
            if st['tick'] != t0 + 1:
                problems.append(f'step: tick {st["tick"]} after {t0}')
            await ws.send(json.dumps({'type': 'step', 'k': 5}))
            st = await recv_json(ws, 'status')
            if st['tick'] != t0 + 6:
                problems.append(f'step 5: tick {st["tick"]}')
            await ws.send(json.dumps({'type': 'subscribe', 'field': 'speed'}))
            fh, _ = await recv_bin(ws)
            await ws.send(json.dumps({'type': 'play', 'tpf': 2}))
            seen = []
            while len(seen) < 3:
                m = await ws.recv()
                if isinstance(m, bytes):
                    seen.append(parse_frame(m)[0])
            if any(s['name'] != 'speed' for s in seen) or not seen[-1]['tick'] > seen[0]['tick']:
                problems.append('play push')
            await ws.send(json.dumps({'type': 'pause'}))
            st = await recv_json(ws, 'status')
            await asyncio.sleep(0.2)
            t1 = engine.world.tick
            await asyncio.sleep(0.3)
            if engine.world.tick != t1 or st['playing']:
                problems.append('pause')
            await ws.send(json.dumps({'type': 'bogus'}))
            err = await recv_json(ws, 'error')
            return fh, err

    fh, err = asyncio.run(session())
    page = urllib.request.urlopen(f'http://127.0.0.1:{port}/').read()
    three = urllib.request.urlopen(f'http://127.0.0.1:{port}/vendor/three.min.js').read()
    if b'InSilica World V2' not in page or len(three) < 500_000:
        problems.append('static files')
    ok = not problems
    record('W1', 'all round trips correct (grid, f8 bytes exact, f4 bytes, step +1/+5, play pushes, pause holds, error reply, page + three.js served)'
           if ok else '; '.join(problems), 'all correct', ok)
    assert ok
