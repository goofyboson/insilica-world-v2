"""Mars's prescribed dust (V2-7, SPEC #239) from two downloads, to two tables for the model:

1. src/insilica/data/mars_dust_my30.npz: Montabone et al.'s (2015, 2020) kriged dust scenario for Mars Year 30 (LMD,
   dustscenario_MY30_v2-0.nc: daily column dust optical depth, 9.3 um absorption, normalized to 610 Pa, 3 x 3 degrees), as
   5-sol means: `cdod610` (134 x 60 x 120, from 88.5 S and 180 W), `ls` (each mean's solar longitude, degrees), `lat`, `lon`.
   The data are CC BY-SA 3.0 (credit: L. Montabone et al., LMD); this derived table is under the same licence.
2. src/insilica/data/mars_dust_optics.npz: dust's optics in the model's 10 longwave and 10 shortwave bands (lbl.BANDS,
   make_swtables.BANDS) from Wolff et al.'s (2010) per-size tables as the NASA Ames Mars GCM ships them
   (Dust_vis/ir_wolff2010_JD_15bands.dat: for 20 radii from 0.1 to 50 um, each band's Qext, Qscat and g; bands 10-2222 cm-1
   (8) and 2222-41667 cm-1 (7)), averaged over a lognormal of effective radius 1.5 um and variance 0.3, then over each model band
   by the sun's (5778 K) or the ground's (210 K) Planck spectrum: `lw_ext`, `lw_ssa`, `lw_g`, `sw_ext`, `sw_ssa`, `sw_g`, each
   band's extinction relative to the 9.3 um absorption the maps give, scaled so the visible's is LMD's 2.6.
Run once, from the repository:
    python -I tools/make_mars_dust.py cache/mars-dust/dustscenario_MY30_v2-0.nc cache/mars-dust-optics
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc3  # noqa: E402

LW_BANDS = [10, 350, 500, 630, 700, 820, 980, 1180, 1390, 1900, 3000]                     # lbl.BANDS
SW_BANDS = [(2000, 2600), (2600, 4000), (4000, 5250), (5250, 7700), (7700, 9350), (9350, 12850), (12850, 16000), (16000, 29000),
            (29000, 38000), (38000, 50000)]                                                  # make_swtables.BANDS (outer edges)
AMES = [10.0, 300.300, 549.451, 636.943, 704.225, 775.194, 970.874, 1282.051, 2222.222,     # its 15-band set: infrared, then
        3087.37, 4030.63, 5370.57, 7651.11, 12500.00, 25000.00, 41666.67]                  # visible (rtmod_mgcm setspi, setspv)
REFF, VEFF = 1.5e-6, 0.3


def size_average(path, nb):
    """Qext, ssa, g per band averaged over the lognormal (REFF, VEFF) on the tables' 20 radii (Ames's rad_rt grid)."""
    v = np.loadtxt(path).reshape(20, 3, nb)
    qe, qs, g = v[:, 0], v[:, 1], v[:, 2]
    r = 1e-7 * (50e-6 / 1e-7) ** (np.arange(20) / 19)
    s2 = np.log(1 + VEFF)
    rg = REFF / np.exp(2.5 * s2)
    n = np.exp(-np.log(r / rg) ** 2 / (2 * s2)) / r                  # per unit radius (the constant left out)
    w = (n * np.pi * r ** 2 * r)[:, None]                             # times dr ~ r on the log grid
    Qe = (w * qe).sum(0) / w.sum()
    Qs = (w * qs).sum(0) / w.sum()
    G = (w * qs * g).sum(0) / (w * qs).sum(0)
    return Qe, Qs / Qe, G


def planck(nu, T):
    return nu ** 3 / np.expm1(1.4387769 * nu / T)


def to_bands(edges, Qe, ssa, g, T):
    """Ames's band values onto the model's bands (edges: (lo, hi) pairs), weighted by Planck at T."""
    out = []
    for lo, hi in edges:
        nu = np.linspace(lo, hi, 4001)[:-1] + (hi - lo) / 8000
        j = np.clip(np.searchsorted(AMES, nu) - 1, 0, len(Qe) - 1)
        B = planck(nu, T)
        e = (B * Qe[j]).sum() / B.sum()
        s = (B * Qe[j] * ssa[j]).sum() / (B * Qe[j]).sum()
        gg = (B * Qe[j] * ssa[j] * g[j]).sum() / (B * Qe[j] * ssa[j]).sum()
        out.append((e, s, gg))
    return np.array(out)


def main(nc_path, opt_dir, out_dir='src/insilica/data'):
    d = nc3.read(nc_path)['vars']
    c = d['cdod610'][0].astype(float)                                # sol x lat x lon
    la = d['latitude'][0].astype(float)
    if la[0] > la[-1]:                                                # (the file's latitudes run from the north: stored rising)
        c, la = c[:, ::-1], la[::-1]
    ls = d['Ls'][0].astype(float)
    nb = (c.shape[0] + 4) // 5
    cm = np.stack([c[5 * j:5 * j + 5].mean(0) for j in range(nb)])
    lsm = np.array([np.degrees(np.angle(np.exp(1j * np.radians(ls[5 * j:5 * j + 5])).mean())) % 360 for j in range(nb)])
    np.savez_compressed(os.path.join(out_dir, 'mars_dust_my30.npz'), cdod610=cm.astype(np.float32), ls=lsm,
                        lat=la.copy(), lon=d['longitude'][0].astype(float),
                        credit=np.array('Montabone et al. (2015, 2020), LMD dust scenario MY30 v2.0; CC BY-SA 3.0; 5-sol means'))
    Qi, si, gi = size_average(os.path.join(opt_dir, 'Dust_ir_wolff2010_JD_15bands.dat'), 8)
    Qv, sv, gv = size_average(os.path.join(opt_dir, 'Dust_vis_wolff2010_JD_15bands.dat'), 7)
    Qe, ssa, g = np.concatenate([Qi, Qv]), np.concatenate([si, sv]), np.concatenate([gi, gv])
    lw = to_bands(list(zip(LW_BANDS[:-1], LW_BANDS[1:])), Qe, ssa, g, 210.0)
    sw = to_bands(SW_BANDS, Qe, ssa, g, 5778.0)
    # the maps' 9.3 um absorption: the 970-1282 cm-1 band's average lies below the silicate peak at 9.3 um (its visible extinction
    # came out 2.96 times it); scaled so the visible (Ames's 12500-25000 cm-1 band) is LMD's own 2.6 (their extinction / absorption
    # at 9.3 um, 1.3, the band's 1.34)
    qabs = Qv[5] / 2.6
    np.savez_compressed(os.path.join(out_dir, 'mars_dust_optics.npz'), lw_ext=lw[:, 0] / qabs, lw_ssa=lw[:, 1], lw_g=lw[:, 2],
                        sw_ext=sw[:, 0] / qabs, sw_ssa=sw[:, 1], sw_g=sw[:, 2], reff=REFF, veff=VEFF)
    vis = to_bands([(16000, 20000)], Qe, ssa, g, 5778.0)[0]
    w = np.cos(np.radians(la))[None, :, None]
    print(f'maps: {nb} 5-sol means, planet mean {(cm * w).sum() / (w.sum() * cm.shape[0] * cm.shape[2]):.3f} (9.3 um absorption at 610 Pa), '
          f'{cm.min():.4f}-{cm.max():.3f}; {os.path.getsize(os.path.join(out_dir, "mars_dust_my30.npz")) / 1e6:.2f} MB')
    print(f'optics: visible (500-625 nm) extinction {vis[0] / qabs:.2f} x the 9.3 um absorption (LMD gives 2.6), ssa {vis[1]:.3f}, g {vis[2]:.3f}')
    print('  longwave ext', np.round(lw[:, 0] / qabs, 3), 'ssa', np.round(lw[:, 1], 3))
    print('  shortwave ext', np.round(sw[:, 0] / qabs, 3), 'ssa', np.round(sw[:, 1], 3), 'g', np.round(sw[:, 2], 3))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
