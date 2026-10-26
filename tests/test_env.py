import cupy as cp
import numpy as np

from conftest import fmt


def test_F0_environment(record):
    """F0 Environment: CuPy, compute capability 8.9, a 64-bit kernel (exp/log/pow/sin) compiles and matches NumPy"""
    from insilica.gpu import NVRTC_OPTIONS
    cc = cp.cuda.Device(0).compute_capability
    k = cp.RawKernel(r'''extern "C" __global__ void f(const double* x, double* y, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) y[i] = exp(x[i]) + log(x[i] + 2.0) + pow(x[i], 1.5) + sin(x[i]); }''', 'f', options=NVRTC_OPTIONS)
    x = cp.linspace(0, 1, 10000)
    y = cp.empty_like(x)
    k((40,), (256,), (x, y, np.int32(x.size)))
    xh = x.get()
    ref = np.exp(xh) + np.log(xh + 2) + xh ** 1.5 + np.sin(xh)
    err = float(np.max(np.abs(y.get() - ref) / np.abs(ref)))
    ulp = float(np.max(np.abs(y.get() - ref) / np.spacing(np.abs(ref))))
    name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    value = f'{name}, cc {cc}, CuPy {cp.__version__}, CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}; max rel err {fmt(err)} ({ulp:.0f} ulp of the sum)'
    ok = cc == '89' and err < 1e-15
    assert record('F0', value, 'cc 8.9; ≤ a few ulp (rel < 1e-15)', ok)
