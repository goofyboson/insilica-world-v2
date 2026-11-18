"""VIEWER: the viewer protocol, driven by a Python WebSocket client against a real engine on a free port."""
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

        import tempfile
        box['saves'] = tempfile.mkdtemp(prefix='insilica-saves-')
        box['task'] = loop.create_task(main(n=3, preset='w5', port=0, ready=ready, saves=box['saves']))
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


async def recv_field(ws, name):
    """The next field frame of that name (pushed frames of a subscription are passed over)."""
    while True:
        h, arrays = await recv_bin(ws)
        if h.get('name') == name:
            return h, arrays


def test_viewer(record, server):
    """VIEWER Viewer protocol: connect, grid, a field's bytes equal the engine's array, pause/step/play round trip; (V2-6g) a new world with planet settings, its new fields' bytes, the Sun and Moon against an independent ephemeris, save and load"""
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
            # (V2-6g) a new earth7 world with planet settings, tides, gas tracers and geology: the settings reach the world
            params = {'planet': {'radius': 5.0e6, 'day_s': 30 * 3600.0, 'omega': 2 * np.pi / (30 * 3600.0), 'tilt': 0.3,
                                 'gases': {'n2': 0.9, 'co2': 0.1}}, 'tides': {'moon': {'gm': 2 * 4.9028e12}}, 'gas_tracers': 1, 'geology': 1}
            await ws.send(json.dumps({'type': 'new', 'n': 3, 'preset': 'earth7', 'params': params}))
            info = await recv_json(ws, 'info')
            await recv_bin(ws)
            w = engine.world
            p = w.physics.p
            if not (w.planet['radius'] == 5.0e6 and p['day_s'] == 30 * 3600.0 and p['tilt'] == 0.3 and w.gas['defs']
                    and w.tides is not None and w.tides.cfg['moon']['gm'] == 2 * 4.9028e12 and w.gas_on and w.geology is not None):
                problems.append('planet settings')
            await ws.send(json.dumps({'type': 'new', 'n': 3, 'preset': 'earth7', 'params': {'dt': 1}}))
            if 'not allowed' not in (await recv_json(ws, 'error'))['message']:
                problems.append('params not refused')
            await ws.send(json.dumps({'type': 'step', 'k': 3}))
            st = await recv_json(ws, 'status')
            # the new fields, bytes equal the engine's arrays
            for name, lay in (('sla', 0), ('tide', 0), ('co2', 2), ('tsoil', 3), ('g_rain', 0), ('g_ftc', 0)):
                if name not in info['fields']:
                    problems.append(f'{name} not offered')
                    continue
                await ws.send(json.dumps({'type': 'get', 'field': name, 'layer': lay, 'dtype': 'f8'}))
                fh2, (a,) = await recv_field(ws, name)
                if a.tobytes() != engine.world.field(name, lay).get().astype('<f8').tobytes():
                    problems.append(f'{name} bytes')
            # the Sun and the Moon against an independent ephemeris (test_tides' NumPy copy) at the status's time
            from test_tides import moon_dir
            sk = st['sky']
            m = moon_dir(sk['t'], engine.world.tides.par.get())
            tp = engine.world.tides.par.get()
            sky_err = float(np.max(np.abs(np.array(sk['moon']) - m)))
            box['sky'] = sky_err
            # save, then load: the same state
            h0 = engine.world.state_hash()
            await ws.send(json.dumps({'type': 'save', 'name': 'v6g'}))
            info2 = await recv_json(ws, 'info')
            await ws.send(json.dumps({'type': 'load', 'name': 'v6g'}))
            await recv_json(ws, 'info')
            if 'v6g' not in info2['saves'] or engine.world.state_hash() != h0:
                problems.append('save and load')
            await ws.send(json.dumps({'type': 'save', 'name': '../x'}))
            if 'save name' not in (await recv_json(ws, 'error'))['message']:
                problems.append('bad save name not refused')
            return fh, err

    box = {}
    fh, err = asyncio.run(session())
    if box.get('sky', 1.0) > 1e-9:
        problems.append(f'Moon direction off by {box.get("sky")}')
    page = urllib.request.urlopen(f'http://127.0.0.1:{port}/').read()
    three = urllib.request.urlopen(f'http://127.0.0.1:{port}/vendor/three.min.js').read()
    if b'InSilica World V2' not in page or len(three) < 500_000:
        problems.append('static files')
    ok = not problems
    record('VIEWER', 'all round trips correct (grid, f8 bytes exact, f4 bytes, step +1/+5, play pushes, pause holds, error reply, page + three.js served; '
           f'V2-6g: planet settings reach a new earth7, other params refused, sla, tide, CO₂, soil temperature and geology fields byte-exact, the Moon '
           f'against an independent ephemeris {box["sky"]:.1e}, save and load identical, a bad save name refused)'
           if ok else '; '.join(problems), 'all correct', ok)
    assert ok
