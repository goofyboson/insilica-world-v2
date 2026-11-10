"""Each air layer's vapor, cloud-water and air-mass budget by the column's stages (V2-6c, SPEC #136): the air_column kernel compiled with
-DVAPDIAG writes each layer's vapor and cloud water (kg/m^2) at ten points of the column step into the diagnostics array after
its rows; the differences are the stages' tendencies, and what changes between one step's end and the next one's start is the
transport. Means over the tropical sea, the subtropical sea and the globe, kg/m^2/day, layers from the top.
Usage: python tools/vapor_budget.py <saved world .iw2> [days]"""
import sys

import cupy as cp
import numpy as np

from insilica.world import World

STAGES = ['surface evaporation', 'clouds, large-scale rain', 'crossing', 'plume mixing', 'plume lifts base air',
          'lifts entrained, detrains', 'downdraft', 'shallow plume', 'conv. rain evaporation']


def main(path, days=10.0):
    w = World.load(path); g = w.grid; p = w.physics; L = w.L
    p.k['air_column'] = w.dg.kernel('air_column', 'air.cu', (f'-DNL={L}', '-DVAPDIAG'))
    base = 5 * L - 2
    p.dgo = cp.zeros((base + 30 * L + 16, g.N))   # (the shallow plume's four rows, then each layer's mass at the ten points)
    lat = np.degrees(g.lat); A = g.areaCell; land = w.land.get() > 0.5
    masks = {'sea 10S-10N': ~land & (np.abs(lat) < 10), 'sea 10-30': ~land & (np.abs(lat) >= 10) & (np.abs(lat) < 30),
             'sea 30-60': ~land & (np.abs(lat) >= 30) & (np.abs(lat) < 60), 'land 30-60': land & (np.abs(lat) >= 30) & (np.abs(lat) < 60),
             'global': np.ones(g.N, bool)}
    snaps = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    n = int(round(days * 86400 / w.dt))
    acc = np.zeros((3, len(snaps) + 1, L, g.N)); prev = None
    for _ in range(n):
        w.step(1)
        d = p.dgo.get()
        x = np.stack([d[base:base + 10 * L].reshape(10, L, g.N), d[base + 10 * L:base + 20 * L].reshape(10, L, g.N),
                      d[base + 20 * L + 4:base + 30 * L + 4].reshape(10, L, g.N)])
        for j, (a, b) in enumerate(snaps):
            acc[:, j] += x[:, b] - x[:, a]
        if prev is not None:
            acc[:, -1] += x[:, 0] - prev
        prev = x[:, 9].copy()
    s = 86400 / (w.dt * n)
    for nm, m in masks.items():
        av = lambda y: float(np.average(y[m], weights=A[m]))
        print(f'== {nm}: vapor (cloud water) tendency, kg/m2/day, layers top -> bottom')
        for j, st in enumerate(STAGES + ['transport']):
            print(f'  {st:26s}' + ''.join(f'{s*av(acc[0, j, k]):+8.2f} ({s*av(acc[1, j, k]):+6.2f})' for k in range(L)))
        print(f'  {"sum":26s}' + ''.join(f'{s*av(acc[0, :, k].sum(0)):+8.2f} ({s*av(acc[1, :, k].sum(0)):+6.2f})' for k in range(L)))
        print(f'   air mass tendency, kg/m2/day (the layers mean mass, hPa: ' + ' '.join(f'{av(w.h[k].get()) * 9.81 / 100:.0f}' for k in range(L)) + ')')
        for j, st in enumerate(STAGES + ['transport']):
            print(f'  {st:26s}' + ''.join(f'{s*av(acc[2, j, k]):+10.0f}' for k in range(L)))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
