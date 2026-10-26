"""Save files: an uncompressed zip holding `meta.json` and one `.npy` per state array (raw 64-bit bytes).
Entries carry a fixed timestamp and are written in sorted order, so saving the same state twice gives
identical bytes (test S1)."""
import io
import json
import zipfile

import numpy as np

FORMAT_VERSION = 1
_STAMP = (1980, 1, 1, 0, 0, 0)


def write(path, meta, arrays):
    meta = dict(meta, formatVersion=FORMAT_VERSION, arrays=sorted(arrays))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo('meta.json', date_time=_STAMP)
        z.writestr(info, json.dumps(meta, sort_keys=True, indent=1))
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            z.writestr(zipfile.ZipInfo(f'{name}.npy', date_time=_STAMP), buf.getvalue())


def read(path):
    """Returns (meta, arrays)."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read('meta.json'))
        if meta.get('formatVersion') != FORMAT_VERSION:
            raise ValueError(f'save file format {meta.get("formatVersion")}, expected {FORMAT_VERSION}')
        arrays = {}
        for name in meta['arrays']:
            arrays[name] = np.lib.format.read_array(io.BytesIO(z.read(f'{name}.npy')), allow_pickle=False)
    return meta, arrays
