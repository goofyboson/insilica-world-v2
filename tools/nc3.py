"""A minimal reader for classic NetCDF (CDF-1 and CDF-2) files: their dimensions, attributes and variables as NumPy arrays."""
import numpy as np

_TYPES = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}


def read(path):
    b = open(path, 'rb').read()
    assert b[:3] == b'CDF' and b[3] in (1, 2), 'not a classic NetCDF file'
    off_size = 4 if b[3] == 1 else 8
    pos = [4]

    def u32():
        v = int.from_bytes(b[pos[0]:pos[0] + 4], 'big'); pos[0] += 4; return v

    def off():
        v = int.from_bytes(b[pos[0]:pos[0] + off_size], 'big'); pos[0] += off_size; return v

    def name():
        n = u32(); s = b[pos[0]:pos[0] + n].decode('latin-1'); pos[0] += (n + 3) // 4 * 4; return s

    def values(t, n):
        dt, sz = _TYPES[t]
        raw = b[pos[0]:pos[0] + n * sz]; pos[0] += (n * sz + 3) // 4 * 4
        return raw.decode('latin-1') if t == 2 else np.frombuffer(raw, dt)

    def atts():
        tag, n = u32(), u32()
        out = {}
        for _ in range(n):
            k = name(); t = u32(); out[k] = values(t, u32())
        return out
    nrecs = u32()
    tag, n = u32(), u32()
    dims = [(name(), u32()) for _ in range(n)]
    gatts = atts()
    tag, n = u32(), u32()
    vars_ = {}
    for _ in range(n):
        k = name(); nd = u32(); ids = [u32() for _ in range(nd)]
        a = atts(); t = u32(); vsize = u32(); begin = off()
        vars_[k] = (ids, a, t, vsize, begin)
    recsize = sum(v[3] for v in vars_.values() if v[0] and dims[v[0][0]][1] == 0)
    out = {}
    for k, (ids, a, t, vsize, begin) in vars_.items():
        dt, sz = _TYPES[t]
        shape = [dims[j][1] for j in ids]
        if ids and shape[0] == 0:                               # a record variable
            shape[0] = nrecs
            per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            arr = np.stack([np.frombuffer(b[begin + r * recsize:begin + r * recsize + per * sz], dt) for r in range(nrecs)])
            arr = arr.reshape(shape)
        else:
            n = int(np.prod(shape)) if shape else 1
            arr = np.frombuffer(b[begin:begin + n * sz], dt).reshape(shape)
        out[k] = (arr, a)
    return dict(dims=dims, atts=gatts, vars=out)
