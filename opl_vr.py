"""
opl_vr.py  --  a faithful Python port of Virtual Retina's Outer Plexiform Layer
(LinearOPL in VirtualRetina/src/retina.cc), written to mirror the C++ structure:

    C++ class / function                      this file
    ------------------------------------      ---------------------------
    BaseRecFilter  (baserecfilter.cc)    ->   RecFilter      : temporal IIR filter
        Exp(tau), ExpCascade(tau,n)      ->   RecFilter.exp(), .exp_cascade()
        operator*=, operator-=           ->   RecFilter.scale(), .delta_minus()
    CImg::deriche(sigma,0,axis)          ->   deriche_blur() : spatial Gaussian
    BaseMapFilter / RetinaMapFilter      ->   MapFilter      : blur + IIR per pixel
    LinearOPL                            ->   LinearOPL      : wires C, undershoot, S

Only the OPL is implemented (no gain control, no ganglion cells), with a
uniform retina (no <log-polar-scheme>), which is what our XML files use.
"""
import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. TEMPORAL FILTER  --  BaseRecFilter
#    A causal IIR difference equation   sum_k a[k] y[n-k] = sum_j b[j] x[n-j]
# ---------------------------------------------------------------------------
class RecFilter:
    def __init__(self, dt):
        self.dt = dt
        self.a = np.array([1.0])      # identity filter: y[n] = x[n]
        self.b = np.array([1.0])

    def exp(self, tau):
        """E_tau(t) = exp(-t/tau)/tau   ->   y[n] = c*y[n-1] + (1-c)*x[n]"""
        if tau > 0:
            c = math.exp(-self.dt / tau)
            self.a = np.array([1.0, -c])
            self.b = np.array([1.0 - c])
        return self

    def exp_cascade(self, tau, n):
        """Gamma kernel E_{n,tau}: n+1 exponential stages of tau/n (peak at tau).
        Built in one go: a = binomial expansion of (1 - c z^-1)^(n+1)."""
        if tau > 0:
            N = n + 1
            c = math.exp(-self.dt / (tau / n if n else tau))
            self.b = np.array([(1 - c) ** N])
            self.a = np.array([(-c) ** i * math.comb(N, i) for i in range(N + 1)])
        return self

    def scale(self, lam):
        """operator*= : multiply the filter by a gain (only b changes)."""
        self.b = self.b * lam
        return self

    def delta_minus(self, other):
        """self = identity - other   (the C++ does  undershoot -= adapTr).
        For two rational filters b1/a1 - b2/a2 = (b1*a2 - b2*a1) / (a1*a2)."""
        a = np.convolve(self.a, other.a)
        n1 = np.convolve(self.b, other.a)
        n2 = np.convolve(other.b, self.a)
        L = max(len(n1), len(n2))
        b = np.pad(n1, (0, L - len(n1))) - np.pad(n2, (0, L - len(n2)))
        self.a, self.b = a, b
        return self

    def gain(self):
        """DC gain (response to a constant input) = sum(b) / sum(a)."""
        return self.b.sum() / self.a.sum()


# ---------------------------------------------------------------------------
# 2. SPATIAL FILTER  --  CImg<T>::deriche(sigma, order=0, axis, boundary=1)
#    Deriche's recursive approximation of a Gaussian: one causal pass plus one
#    anti-causal pass per axis. Borders: Neumann (edge pixel repeated forever).
# ---------------------------------------------------------------------------
def _deriche_1d(img, sigma, axis):
    sigma = float(np.float32(sigma))                 # C++ passes a float
    if sigma < 0.1:                                  # CImg: too small -> no blur
        return img
    alpha = float(np.float32(1.695)) / sigma
    ema, ema2 = math.exp(-alpha), math.exp(-2 * alpha)
    b1, b2 = -2 * ema, ema2
    k = (1 - ema) ** 2 / (1 + 2 * alpha * ema - ema2)
    a0, a1, a2, a3 = k, k * (alpha - 1) * ema, k * (alpha + 1) * ema, -k * ema2
    coefp = (a0 + a1) / (1 + b1 + b2)
    coefn = (a2 + a3) / (1 + b1 + b2)

    X = np.moveaxis(img, axis, 0)                    # filter along axis 0, all lines at once
    n = X.shape[0]
    Y = np.empty_like(X)
    xp = X[0].copy(); yb = coefp * xp; yp = yb.copy()          # causal pass
    for m in range(n):
        xc = X[m]
        yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb
        Y[m] = yc
        xp, yb, yp = xc, yp, yc
    out = np.empty_like(X)
    xn = X[-1].copy(); xa = xn.copy(); yn = coefn * xn; ya = yn.copy()   # anti-causal pass
    for m in range(n - 1, -1, -1):
        xc = X[m]
        yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya
        xa, xn, ya, yn = xn, xc, yn, yc
        out[m] = Y[m] + yc
    return np.moveaxis(out, 0, axis)


def _deriche_1d_fast(img, sigma, axis):
    """Same Deriche recursion and border handling as _deriche_1d, but each pass
    runs in compiled code (scipy.signal.lfilter). Differs only by float rounding
    (~1e-15 relative). The initial filter states reproduce CImg's Neumann border:
    samples beyond the edge equal the edge pixel."""
    from scipy.signal import lfilter, lfiltic
    sigma = float(np.float32(sigma))
    if sigma < 0.1:
        return img
    alpha = float(np.float32(1.695)) / sigma
    ema, ema2 = math.exp(-alpha), math.exp(-2 * alpha)
    b1, b2 = -2 * ema, ema2
    k = (1 - ema) ** 2 / (1 + 2 * alpha * ema - ema2)
    a0, a1, a2, a3 = k, k * (alpha - 1) * ema, k * (alpha + 1) * ema, -k * ema2
    coefp = (a0 + a1) / (1 + b1 + b2)
    coefn = (a2 + a3) / (1 + b1 + b2)
    A = [1.0, b1, b2]
    X = np.moveaxis(img, axis, -1)
    # causal:      y[n] = a0 x[n] + a1 x[n-1] - b1 y[n-1] - b2 y[n-2]
    z1 = lfiltic([a0, a1], A, y=[coefp, coefp], x=[1.0])
    Y1, _ = lfilter([a0, a1], A, X, axis=-1, zi=z1 * X[..., :1])
    # anti-causal: y[n] = a2 x[n+1] + a3 x[n+2] - b1 y[n+1] - b2 y[n+2]
    Xr = X[..., ::-1]
    z2 = lfiltic([0.0, a2, a3], A, y=[coefn, coefn], x=[1.0, 1.0])
    Y2, _ = lfilter([0.0, a2, a3], A, Xr, axis=-1, zi=z2 * Xr[..., :1])
    return np.moveaxis(Y1 + Y2[..., ::-1], -1, axis)


_BLUR_1D = _deriche_1d            # "exact": bit-identical to the C++ simulator


def set_blur_backend(name="auto"):
    """Choose how the Deriche blur is computed (the model is the same in all cases):
      'numba' - compiled loop, bit-identical to Virtual Retina, ~5x faster (needs numba)
      'fast'  - scipy.signal.lfilter, differs only by float rounding (~1e-13), ~3x faster
      'exact' - pure-Python loop, bit-identical to Virtual Retina (reference)
      'auto'  - numba if installed, else fast if scipy is installed, else exact
    Returns the name of the backend actually used."""
    global _BLUR_1D
    if name == "auto":
        for cand in ("numba", "fast"):
            try:
                return set_blur_backend(cand)
            except ImportError:
                pass
        name = "exact"
    if name == "numba":
        from opl_numba import deriche_1d_numba
        deriche_1d_numba(np.zeros((4, 4)), 1.0, 1)   # compile / load the cache now, not mid-stream
        _BLUR_1D = deriche_1d_numba
    elif name == "fast":
        import scipy.signal  # noqa: F401  (fail early if scipy is missing)
        _BLUR_1D = _deriche_1d_fast
    elif name == "exact":
        _BLUR_1D = _deriche_1d
    else:
        raise ValueError("backend must be 'auto', 'numba', 'fast' or 'exact'")
    return name


def deriche_blur(img, sigma_px):
    """2-D blur = x pass then y pass (RadialFilter::radiallyVariantBlur)."""
    return _BLUR_1D(_BLUR_1D(img, sigma_px, axis=1), sigma_px, axis=0)


# ---------------------------------------------------------------------------
# 3. SPACE + TIME FILTER ON A MAP  --  BaseMapFilter + RetinaMapFilter
#    Every pixel runs the same RecFilter. Optional spatial blur of the INPUT
#    (sigmaPool) before the recursion, or leaky-heat blur of the OUTPUT.
# ---------------------------------------------------------------------------
class MapFilter(RecFilter):
    def __init__(self, dt, ppd=1.0):
        super().__init__(dt)
        self.ppd = ppd
        self.sigma_pool_deg = 0.0     # blur applied to each new input
        self.heat_sigma_deg = 0.0     # leaky-heat: blur applied to each new output

    def leaky_heat_equation(self, tau, sigma_deg):
        """RetinaMapFilter::leakyHeatEquation: Exp(tau) + diffusion of the state.
        gCoupling = sigma^2/(2 tau); per step, blur by sqrt(2 gCoupling dt)."""
        g = sigma_deg ** 2 / (2 * tau)
        self.heat_sigma_deg = math.sqrt(2 * g * self.dt)
        return self.exp(tau)

    def allocate(self, shape, initial_input):
        """BaseMapFilter::allocateValues: start in steady state for a constant
        input equal to initial_input (so there is no start-up transient)."""
        M, N = len(self.b), len(self.a) - 1
        self.inputs = [np.zeros(shape)] + [np.full(shape, initial_input) for _ in range(M - 1)]
        self.values = [np.full(shape, initial_input * self.gain()) for _ in range(N + 1)]

    def feed_input(self, x):
        self.inputs[0] = np.array(x, dtype=np.float64)       # stored, not yet processed

    def temp_step(self):
        """One time step dt: blur the input, run the difference equation."""
        if self.sigma_pool_deg:
            self.inputs[0] = deriche_blur(self.inputs[0], self.sigma_pool_deg * self.ppd)
        y = self.b[0] * self.inputs[0]
        for j in range(1, len(self.b)):
            y = y + self.b[j] * self.inputs[j]
        for k in range(1, len(self.a)):
            y = y - self.a[k] * self.values[k - 1]
        y = y / self.a[0]
        self.values = [y] + self.values[:-1]                  # newest output first
        self.inputs = [np.zeros_like(y)] + self.inputs[:-1]   # current input becomes x[n-1]
        if self.heat_sigma_deg:
            self.values[0] = deriche_blur(self.values[0], self.heat_sigma_deg * self.ppd)

    def read(self):
        return self.values[0]


# ---------------------------------------------------------------------------
# 4. THE OPL  --  LinearOPL  (retina.cc)
#    receptor  C = Undershoot( lambda * G_sigmaC * E_{n,tauC} * L )
#    horizontal S = w * G_sigmaS * E_tauS * C
#    output    I_OPL = C - S
# ---------------------------------------------------------------------------
class LinearOPL:
    def __init__(self, dt, ppd, lum_range=255.0,
                 center_sigma=0.03, center_tau=0.01, center_n=0,
                 surround_sigma=0.1, surround_tau=0.01,
                 opl_amplification=10.0, w_surround=1.0, leaky_heat=False,
                 undershoot_w=None, undershoot_tau=None):
        self.amp = opl_amplification / lum_range      # lambda_OPL, as in allocateValues

        # center / photoreceptors: excitCells
        self.excit = MapFilter(dt, ppd)
        if leaky_heat:
            self.excit.leaky_heat_equation(center_tau, center_sigma)   # center_n ignored
        else:
            self.excit.exp_cascade(center_tau, center_n)
            self.excit.sigma_pool_deg = center_sigma
        self.excit.scale(self.amp)

        # undershoot: identity minus w_U * Exp(tau_U)   (no spatial part)
        self.under = None
        if undershoot_w is not None:
            adap = RecFilter(dt).exp(undershoot_tau).scale(undershoot_w)
            self.under = MapFilter(dt, ppd).delta_minus(adap)

        # surround / horizontal cells: inhibCells
        self.inhib = None
        if w_surround:
            self.inhib = MapFilter(dt, ppd)
            if leaky_heat:
                self.inhib.leaky_heat_equation(surround_tau, surround_sigma)
            else:
                self.inhib.exp(surround_tau)
                self.inhib.sigma_pool_deg = surround_sigma
            self.inhib.scale(w_surround)

    def allocate(self, shape, init_value):
        """Everything starts in equilibrium with a uniform screen of init_value
        (Retina uses input_luminosity_range / 2 = 127.5 by default)."""
        cy, cx = shape[0] // 2, shape[1] // 2
        self.excit.allocate(shape, init_value)
        self.receptor = self.excit
        if self.under:
            self.under.allocate(shape, self.excit.read()[cy, cx])
            self.receptor = self.under
        if self.inhib:
            self.inhib.allocate(shape, self.receptor.read()[cy, cx])

    def step(self, image):
        """One dt, in the SAME ORDER as LinearOPL::feedInput + tempStep.
        Note: each stage is fed the PREVIOUS output of the stage before it."""
        self.excit.feed_input(image)
        if self.under:
            self.under.feed_input(self.excit.read())
        if self.inhib:
            self.inhib.feed_input(self.receptor.read())
        self.excit.temp_step()
        if self.under:
            self.under.temp_step()
        if self.inhib:
            self.inhib.temp_step()
            return self.receptor.read() - self.inhib.read()
        return self.receptor.read()


def opl_from_xml(path, dt=None):
    """Build a LinearOPL from a Virtual Retina XML file (linear-version only)."""
    import xml.etree.ElementTree as ET
    r = ET.parse(path).getroot().find("retina")
    lv = r.find("outer-plexiform-layer/linear-version")
    g = lambda e, k, d: float(e.get(k, d))
    us = lv.find("undershoot")
    return LinearOPL(
        dt=dt or g(r, "temporal-step__sec", 0.005), ppd=g(r, "pixels-per-degree", 1),
        lum_range=g(r, "input-luminosity-range", 255),
        center_sigma=g(lv, "center-sigma__deg", 0), center_tau=g(lv, "center-tau__sec", 0.01),
        center_n=int(g(lv, "center-n__uint", 0)),
        surround_sigma=g(lv, "surround-sigma__deg", 1), surround_tau=g(lv, "surround-tau__sec", 0.01),
        opl_amplification=g(lv, "opl-amplification", 1), w_surround=g(lv, "opl-relative-weight", 0),
        leaky_heat=bool(int(g(lv, "leaky-heat-equation", 0))),
        undershoot_w=None if us is None else g(us, "relative-weight", 0),
        undershoot_tau=None if us is None else g(us, "tau__sec", 0.1),
    ), g(r, "input-luminosity-range", 255)
