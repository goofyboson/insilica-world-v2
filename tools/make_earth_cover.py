"""Land cover for earth5 (V2-6b, SPEC #132): reduce NCAR's CLM surface data (Lawrence & Chase 2007, from MODIS; the public CESM input
data, lnd/clm2/rawdata: mksrf_pft_0.5x0.5_simyr2005.c090313.nc, mksrf_soilcol_0.5x0.5_simyr2005.c090313.nc) to
src/insilica/data/earth_cover_1deg.npz on 180 x 360 cells from 90 S and 180 W:
  `land`  the land area of each 1-degree cell (cos-lat weighted count of 0.5-degree land points),
  `share` (9, 180, 360) the share of that land under each cover class, COVER below (CLM's 17 plant types mapped to Noah's classes),
  `dry`, `sat` the bare soil's broadband albedo dry and saturated: the mean of CLM's visible and near-infrared values for its soil
  colour class (CLM 5.0 Technical Note, Table 2.3.3), averaged over the land.
The leaf-area file (mksrf_lai_...) is not used. Run once, from the repository:
    python -I tools/make_earth_cover.py cache/clm-rawdata
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncclassic

# cover classes (cases.COVER holds Noah's values for them) and CLM's plant types in each (0 bare; 1-2 needleleaf evergreen temperate
# and boreal; 3 needleleaf deciduous boreal; 4-5 broadleaf evergreen tropical and temperate; 6-8 broadleaf deciduous tropical,
# temperate, boreal; 9-10 broadleaf shrub evergreen and deciduous temperate; 11 broadleaf deciduous boreal shrub; 12 C3 arctic grass;
# 13 C3 grass; 14 C4 grass; 15-16 crops)
COVER = ('bare', 'needleleaf evergreen', 'broadleaf evergreen', 'needleleaf deciduous', 'broadleaf deciduous', 'shrub', 'tundra',
         'grass', 'crop')
PFT_TO_COVER = [0, 1, 1, 3, 2, 2, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
# CLM soil colour classes 1-20: dry and saturated albedo, visible and near-infrared
DRY_VIS = [.36, .34, .32, .31, .30, .29, .28, .27, .26, .25, .24, .23, .22, .20, .18, .16, .14, .12, .10, .08]
DRY_NIR = [.61, .57, .53, .51, .49, .48, .45, .43, .41, .39, .37, .35, .33, .31, .29, .27, .25, .23, .21, .16]
SAT_VIS = [.25, .23, .21, .20, .19, .18, .17, .16, .15, .14, .13, .12, .11, .10, .09, .08, .07, .06, .05, .04]
SAT_NIR = [.50, .46, .42, .40, .38, .36, .34, .32, .30, .28, .26, .24, .22, .20, .18, .16, .14, .12, .10, .08]


def main(d, out='src/insilica/data/earth_cover_1deg.npz'):
    _, _, v = ncclassic.read(d + '/mksrf_pft_0.5x0.5_simyr2005.c090313.nc')
    pct, mask, lat = v['PCT_PFT'].data().astype(float), v['LANDMASK'].data() > 0.5, v['LAT'].data().astype(float)
    _, _, c = ncclassic.read(d + '/mksrf_soilcol_0.5x0.5_simyr2005.c090313.nc')
    col = c['SOIL_COLOR'].data()
    assert pct.shape == (17, 360, 720) and np.all(np.abs(lat - (np.arange(360) * 0.5 - 89.75)) < 1e-4)
    tot = pct.sum(0)
    assert np.all(np.abs(tot[mask] - 100) < 0.01), 'plant types do not sum to 100% on land'
    ok = mask & (col >= 1) & (col <= 20)
    assert ok.sum() > 0.999 * mask.sum(), 'land without a soil colour'
    k = np.clip(col.astype(int), 1, 20) - 1
    dry = 0.5 * (np.take(DRY_VIS, k) + np.take(DRY_NIR, k))
    sat = 0.5 * (np.take(SAT_VIS, k) + np.take(SAT_NIR, k))
    w = np.cos(np.radians(lat))[:, None] * mask                          # land area of each 0.5-degree point
    box = lambda a: a.reshape(180, 2, 360, 2).sum((1, 3))
    land = box(w)
    share = np.zeros((len(COVER), 180, 360))
    for p, cl in enumerate(PFT_TO_COVER):
        share[cl] += box(w * pct[p] / 100)
    has = land > 0
    share[:, has] /= land[has]
    dry1 = np.where(has, box(w * np.where(ok, dry, 0)) / np.maximum(box(w * ok), 1e-30), 0)
    sat1 = np.where(has, box(w * np.where(ok, sat, 0)) / np.maximum(box(w * ok), 1e-30), 0)
    assert np.all(np.abs(share[:, has].sum(0) - 1) < 1e-9)
    np.savez_compressed(out, land=land.astype(np.float32), share=share.astype(np.float32), dry=dry1.astype(np.float32),
                        sat=sat1.astype(np.float32))
    a = land                                                            # (already area-weighted)
    print(out, 'land cells', int(has.sum()), 'shares of land:',
          ', '.join(f'{n} {100 * np.sum(a * s) / a.sum():.1f}%' for n, s in zip(COVER, share)),
          f'soil dry {np.sum(a * dry1) / a.sum():.3f} sat {np.sum(a * sat1) / a.sum():.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
