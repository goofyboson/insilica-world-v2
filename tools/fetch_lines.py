"""Fetch the HITRAN line lists V2-6a's k-tables are made from (SPEC #109), through HAPI, into cache/hitran/ (not in git).

    python tools/fetch_lines.py            every gas not yet fetched
    python tools/fetch_lines.py CO2 O3     these

All isotopologues of each gas (HITRAN's intensities include their natural abundances), over the wavenumbers its bands
need: the longwave from 10 cm⁻¹, and for H₂O and CO₂ the near infrared the sunlight crosses.
"""
import sys
from pathlib import Path

import hapi

DB = Path(__file__).resolve().parents[1] / 'cache' / 'hitran'
# gas: (HITRAN molecule number, numin, numax cm⁻¹)
GASES = {'H2O': (1, 10.0, 20000.0), 'CO2': (2, 10.0, 8000.0), 'O3': (3, 10.0, 3000.0), 'N2O': (4, 10.0, 3000.0),
         'CH4': (6, 10.0, 3500.0)}


def main():
    names = sys.argv[1:] or list(GASES)
    DB.mkdir(parents=True, exist_ok=True)
    hapi.db_begin(str(DB))
    for name in names:
        if (DB / f'{name}.data').exists():
            print(name, 'already fetched')
            continue
        m, lo, hi = GASES[name]
        ids = [g for g, v in hapi.ISO_ID.items() if v[0] == m]
        hapi.fetch_by_ids(name, ids, lo, hi)
        f = DB / f'{name}.data'
        print(name, f'{sum(1 for _ in open(f))} lines, {f.stat().st_size / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
