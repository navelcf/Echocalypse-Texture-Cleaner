"""The game's ETC2 encoder, in process: Ericsson ETCPACK v2.74, 'fast' speed, 'perceptual' metric.

Encoding a clean original with it reproduces the game's blocks byte for byte (100% on the pair this was measured on), so
training data made with it carries the real damage. That needs the same rand() the game's build machine had (macOS); the
library brings its own, see native/etcpack_lib.cxx. Build it first with  ./build_etcpack.sh  (see README).
"""
import os, ctypes
import numpy as np
from . import blocks as B

_LIB = None
_HAS_FAST = False


def _lib():
    global _LIB, _HAS_FAST
    if _LIB is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.environ.get('ETCPACK_LIB') or os.path.join(here, 'native', 'libetcpack.so')
        if not os.path.exists(path): raise FileNotFoundError(f'{path} not found. Run ./build_etcpack.sh first (or set ETCPACK_LIB).')
        _LIB = ctypes.CDLL(path)
        _LIB.etcpack_rgba8.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        _LIB.etcpack_rgba8.restype = ctypes.c_int
        _HAS_FAST = hasattr(_LIB, 'etcpack_rgba8_fast')
        if _HAS_FAST:
            _LIB.etcpack_rgba8_fast.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            _LIB.etcpack_rgba8_fast.restype = ctypes.c_int
    return _LIB


def has_fast():
    """True if libetcpack.so was built with the fast pthreads + zero/dirty-block-skipping entry point."""
    try:
        _lib()
        return _HAS_FAST
    except FileNotFoundError:
        return False


def encode_blocks(pm, fast=True, perceptual=True, nthreads=1, prev_pm=None, prev_blocks=None):
    """premultiplied RGBA uint8 (H and W multiples of 4) -> blocks (H/4, W/4, 16).
    nthreads=0 uses all logical CPUs minus 2 when built with etcpack_rgba8_fast.
    Optional prev_pm/prev_blocks skip re-encoding 4x4 blocks whose input pixels did not change."""
    h, w = pm.shape[:2]
    if h % 4 or w % 4: raise ValueError('size must be a multiple of 4')
    lib = _lib()
    pm_c = np.ascontiguousarray(pm, np.uint8)
    if _HAS_FAST:
        out = np.empty((h // 4, w // 4, 16), np.uint8)
        pp = np.ascontiguousarray(prev_pm, np.uint8) if prev_pm is not None else None
        pb = np.ascontiguousarray(prev_blocks, np.uint8) if prev_blocks is not None else None
        rc = lib.etcpack_rgba8_fast(
            pm_c.ctypes.data, w, h, out.ctypes.data,
            int(fast), int(perceptual), int(nthreads),
            pp.ctypes.data if pp is not None else 0,
            pb.ctypes.data if pb is not None else 0,
        )
        if rc: raise RuntimeError('etcpack failed')
        return out
    out = ctypes.create_string_buffer(h * w)
    if lib.etcpack_rgba8(pm_c.tobytes(), w, h, out, int(fast), int(perceptual)): raise RuntimeError('etcpack failed')
    return np.frombuffer(out.raw, np.uint8).reshape(h // 4, w // 4, 16)


def encode(pm):
    """clean premultiplied image -> the same image as the game would ship it (decoded ETC2)."""
    return B.decode(encode_blocks(pm))


def _stripe(args): return encode_blocks(args)
def encode_blocks_mp(pm, pool=None, prev_pm=None, prev_blocks=None):
    """Whole pages faster: uses native pthreads when available, otherwise stripes across a multiprocessing pool."""
    if has_fast():
        nthreads = pool._processes if pool is not None else 0
        return encode_blocks(pm, nthreads=nthreads, prev_pm=prev_pm, prev_blocks=prev_blocks)
    if pool is None:
        return encode_blocks(pm)
    h = pm.shape[0]; n = max(1, min(pool._processes, h // 64)); cuts = [round(i * (h // 4) / n) * 4 for i in range(n + 1)]
    return np.concatenate(pool.map(_stripe, [pm[cuts[i]:cuts[i + 1]] for i in range(n) if cuts[i + 1] > cuts[i]]), 0)
