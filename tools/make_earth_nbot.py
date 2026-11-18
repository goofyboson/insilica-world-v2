"""The buoyancy frequency just above the sea floor, for the internal-tide drag (V2-6f, decision #228), from the World Ocean Atlas
2023's annual 1991-2020 temperature and salinity on 1-degree cells (cache/woa23/, downloaded with the owner's approval for V2-6h):
src/insilica/data/earth_nbot_1deg.npz with `n` (1/s) on 180 x 360 cells from 90 S and 180 W. The model's abyss is one
well-mixed layer, so its own stratification above the floor is the thermocline's; the drag needs the floor's. N^2 =
g (alpha d(theta)/dz - beta dS/dz) over the deepest 1000 m with data (the deepest level against the one nearest 1000 m above it),
theta the potential temperature (WOA's in-situ T less the adiabatic heating, GAMMA_AD per metre), alpha the expansion of cold
water at abyssal pressure (approximate; the model's surface alpha is a third of it), beta haline contraction; negative or
missing taken as 0. Run once, from the repository:
    python -I tools/make_earth_nbot.py cache/woa23
"""
import os
import sys

import h5py
import numpy as np

G, ALPHA_D, BETA, GAMMA_AD = 9.80616, 1.4e-4, 7.6e-4, 1.2e-4     # 1/K at ~2 C and 4000 dbar (approx.); ocean.py's beta; K/m


def main(d, out='src/insilica/data/earth_nbot_1deg.npz'):
    t = h5py.File(os.path.join(d, 'woa23_decav91C0_t00_01.nc'), 'r')
    s = h5py.File(os.path.join(d, 'woa23_decav91C0_s00_01.nc'), 'r')
    z = t['depth'][:].astype(float)
    T, S = t['t_an'][0].astype(float), s['s_an'][0].astype(float)          # depth x lat x lon
    T[np.abs(T) > 100] = np.nan
    S[np.abs(S) > 100] = np.nan
    ok = np.isfinite(T) & np.isfinite(S)
    nlev = ok.sum(0)                                                     # levels with data, from the surface down
    th = T - GAMMA_AD * z[:, None, None]
    n2 = np.zeros(nlev.shape)
    for j in range(180):
        for i in range(360):
            k = nlev[j, i]
            if k >= 2:
                kb = k - 1
                ka = int(np.argmin(np.abs(z[:kb] - (z[kb] - 1000.0))))
                n2[j, i] = G * (ALPHA_D * (th[ka, j, i] - th[kb, j, i]) - BETA * (S[ka, j, i] - S[kb, j, i])) / (z[kb] - z[ka])
    n = np.sqrt(np.maximum(n2, 0.0))
    np.savez_compressed(out, n=n.astype(np.float32), zbot=np.where(nlev > 0, z[np.maximum(nlev - 1, 0)], 0.0).astype(np.float32))
    deep = (nlev > 0) & (z[np.maximum(nlev - 1, 0)] >= 1000)
    print(f'bottom N over the deep ocean (10/50/90%): {np.percentile(n[deep], [10, 50, 90])} 1/s; {os.path.getsize(out) / 1024:.0f} KB')


if __name__ == '__main__':
    main(sys.argv[1])
