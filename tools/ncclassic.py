"""A minimal reader for NetCDF classic files (CDF-1 and CDF-2; the format of NCAR's CLM input data), no dependencies beyond
numpy. read(path) -> (dims {name: len}, attrs {name: value}, vars {name: Var}); Var.data() reads the array (big-endian, as stored).
Record variables (an unlimited first dimension) are read too."""
import numpy as np

_T = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}


class Var:
    def __init__(self, f, name, dims, shape, attrs, typ, begin, rec, recsize, nrec):
        self.f, self.name, self.dims, self.shape, self.attrs = f, name, dims, shape, attrs
        self.typ, self.begin, self.rec, self.recsize, self.nrec = typ, begin, rec, recsize, nrec

    def data(self):
        dt, sz = _T[self.typ]
        with open(self.f, 'rb') as fh:
            if not self.rec:
                fh.seek(self.begin)
                n = int(np.prod(self.shape)) if self.shape else 1
                a = np.frombuffer(fh.read(n * sz), dtype=dt).reshape(self.shape)
            else:
                per = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
                out = []
                for r in range(self.nrec):
                    fh.seek(self.begin + r * self.recsize)
                    out.append(np.frombuffer(fh.read(per * sz), dtype=dt))
                a = np.stack(out).reshape((self.nrec,) + tuple(self.shape[1:]))
        return a.astype(a.dtype.newbyteorder('='))


def read(path):
    b = open(path, 'rb').read(1 << 20)               # headers are small
    assert b[:3] == b'CDF' and b[3] in (1, 2), 'not NetCDF classic'
    off64 = b[3] == 2
    pos = [4]

    def i4():
        v = int.from_bytes(b[pos[0]:pos[0] + 4], 'big', signed=True); pos[0] += 4; return v

    def i8():
        v = int.from_bytes(b[pos[0]:pos[0] + 8], 'big', signed=False); pos[0] += 8; return v

    def name():
        n = i4(); s = b[pos[0]:pos[0] + n].decode('latin1'); pos[0] += (n + 3) // 4 * 4; return s

    def values(t, n):
        dt, sz = _T[t]; raw = b[pos[0]:pos[0] + n * sz]; pos[0] += (n * sz + 3) // 4 * 4
        if t == 2:
            return raw.decode('latin1').rstrip('\x00')
        v = np.frombuffer(raw, dtype=dt)
        return v[0].item() if n == 1 else v

    def attlist():
        tag, n = i4(), i4(); out = {}
        for _ in range(n if tag == 12 else 0):
            k = name(); t = i4(); out[k] = values(t, i4())
        return out

    nrec = i4()
    tag, n = i4(), i4()
    dims = []
    for _ in range(n if tag == 10 else 0):
        k = name(); dims.append((k, i4()))
    gatt = attlist()
    tag, n = i4(), i4()
    raw = []
    for _ in range(n if tag == 11 else 0):
        k = name(); ids = [i4() for _ in range(i4())]; a = attlist(); t = i4(); vsize = i4()
        begin = i8() if off64 else i4()
        raw.append((k, ids, a, t, vsize, begin))
    recsize = sum(r[4] for r in raw if r[1] and dims[r[1][0]][1] == 0)
    vars_ = {}
    for k, ids, a, t, vsize, begin in raw:
        rec = bool(ids) and dims[ids[0]][1] == 0
        shape = tuple(nrec if (j == 0 and rec) else dims[d][1] for j, d in enumerate(ids))
        vars_[k] = Var(path, k, [dims[d][0] for d in ids], shape, a, t, begin, rec, recsize, nrec)
    return dict(dims), gatt, vars_
