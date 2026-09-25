"""opl_numba.py -- the exact Deriche blur of opl_vr.py, compiled with Numba.

Same operations in the same order as CImg's deriche() (and opl_vr._deriche_1d),
so results are bit-identical to the C++ simulator, about 5x faster than the
pure-Python loop. The compiled code is cached in __pycache__ after the first run.
"""
import math
import numpy as np
import numba


@numba.njit(cache=True)
def _deriche_rows(X, a0, a1, a2, a3, b1, b2, coefp, coefn):
    H, W = X.shape
    out = np.empty_like(X)
    Y = np.empty(W)
    for r in range(H):
        xp = X[r, 0]; yb = coefp * xp; yp = yb                     # causal pass
        for m in range(W):
            xc = X[r, m]
            yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb
            Y[m] = yc; xp = xc; yb = yp; yp = yc
        xn = X[r, W - 1]; xa = xn; yn = coefn * xn; ya = yn        # anti-causal pass
        for m in range(W - 1, -1, -1):
            xc = X[r, m]
            yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya
            xa = xn; xn = xc; ya = yn; yn = yc
            out[r, m] = Y[m] + yc
    return out


def deriche_1d_numba(img, sigma, axis):
    sigma = float(np.float32(sigma))
    if sigma < 0.1:
        return img
    alpha = float(np.float32(1.695)) / sigma
    ema, ema2 = math.exp(-alpha), math.exp(-2 * alpha)
    b1, b2 = -2 * ema, ema2
    k = (1 - ema) ** 2 / (1 + 2 * alpha * ema - ema2)
    a0, a1, a2, a3 = k, k * (alpha - 1) * ema, k * (alpha + 1) * ema, -k * ema2
    cp = (a0 + a1) / (1 + b1 + b2)
    cn = (a2 + a3) / (1 + b1 + b2)
    if axis == 1:                       # along rows (contiguous)
        return _deriche_rows(np.ascontiguousarray(img, dtype=np.float64), a0, a1, a2, a3, b1, b2, cp, cn)
    return _deriche_rows(np.ascontiguousarray(img.T, dtype=np.float64), a0, a1, a2, a3, b1, b2, cp, cn).T
