"""Chapter 32 helpers: running plane curves, samplings, discrete curvature flows (Track 2, GUIDELINES 2.5).

Import from a figure or verify script of this chapter with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ddg32

The discrete curvatures themselves come from tools/dgdec.py (the canonical implementation); this module only adds
what Chapter 32 needs on top of it:

- the running plane curves of GUIDELINES 2.5, taken from dgsym.EXAMPLES: the ellipse (2 cos t, sin t) and
  the flower curve r(t) = 1 + 0.3 cos 5t, as sympy expressions and numpy samplers, with their signed curvature
  computed by dgsym.signed_curvature (n_s = J t, counterclockwise positive);
- fixed test polygons: the regular n-gon, the unevenly sampled circle UNEVEN_DEG, the jittered flower polygon;
- the discrete curvature vectors kappa^X_i of Section 32.4 and the forward Euler flow;
- a high-resolution approximation of the smooth curve shortening flow (for Figure 32.4.1 and its checks);
- code_block(), run_block(), code_lines(): re-exported from tools/dgcode.py (shared by all Track 2 chapters), so
  that verify runs exactly the code printed in an Implementation box.

Conventions (as in dgdec): vertex i is X[i]; edge i runs from X[i] to X[i+1] (indices mod n for a closed curve);
theta_i is the signed turning angle from t_{i-1} to t_i; n_i = J tbar_i is the vertex normal.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import dgdec as dd  # noqa: E402
import dgsym  # noqa: E402
from dgcode import code_block, code_lines, run_block  # noqa: E402,F401  (re-exported for the verify scripts)

J = np.array([[0.0, -1.0], [1.0, 0.0]])

# one style per discrete curvature, shared by every figure of the chapter (Okabe-Ito colors of dgfig.COLORS;
# line style and marker differ as well, so the four are distinguishable without color)
KIND_STYLE = {
    "A": dict(color="#0072B2", ls="-", marker="o"),     # turning angle      (blue)
    "B": dict(color="#009E73", ls="--", marker="s"),    # length variation   (green)
    "C": dict(color="#CC79A7", ls=":", marker="^"),     # Steiner            (purple)
    "D": dict(color="#E69F00", ls="-.", marker="D"),    # osculating circle  (orange)
}

# ---------------------------------------------------------------------------
# running curves (GUIDELINES 2.5): sympy expressions and numeric versions
# ---------------------------------------------------------------------------

# the ellipse and the flower curve are defined once, in dgsym.EXAMPLES (GUIDELINES 3.6); here as plane curves
T_SYM = dgsym.EXAMPLES["ellipse"]["coords"][0]
ELLIPSE_EXPR = sp.Matrix(list(dgsym.EXAMPLES["ellipse"]["expr"][:2]))
FLOWER_R = dgsym.EXAMPLES["flower"]["radius"]
FLOWER_EXPR = sp.Matrix(list(dgsym.EXAMPLES["flower"]["expr"][:2]))
# convex limacon, used only in Section 32.4 to show that the smooth flow can move a centroid (asymmetric curve)
LIMACON_R = 1 + sp.Rational(2, 5) * sp.cos(T_SYM)
LIMACON_EXPR = sp.Matrix([LIMACON_R * sp.cos(T_SYM), LIMACON_R * sp.sin(T_SYM)])

CURVES = {"ellipse": ELLIPSE_EXPR, "flower": FLOWER_EXPR, "limacon": LIMACON_EXPR}


def _lamb(expr):
    f = sp.lambdify(T_SYM, expr, "numpy")
    return lambda t: np.asarray(f(np.asarray(t, float)), float) * np.ones_like(np.asarray(t, float))


_POINT = {k: sp.lambdify(T_SYM, list(v), "numpy") for k, v in CURVES.items()}
_KAPPA = {k: _lamb(dgsym.signed_curvature(v, T_SYM)) for k, v in CURVES.items()}
_SPEED = {k: _lamb(sp.sqrt(v.diff(T_SYM).dot(v.diff(T_SYM)))) for k, v in CURVES.items()}


def point(name, t):
    """(m, 2) points gamma(t) of the named curve."""
    t = np.asarray(t, float)
    x, y = _POINT[name](t)
    return np.stack([x * np.ones_like(t), y * np.ones_like(t)], -1)


def kappa(name, t):
    """signed curvature kappa_s(t) (dgsym.signed_curvature, counterclockwise positive)."""
    return _KAPPA[name](t)


def speed(name, t):
    return _SPEED[name](t)


def sample(name, n, jitter=0.0, seed=0):
    """n vertices gamma(t_i), t_i = 2 pi (i + u_i) / n, u_i uniform in [-jitter, jitter] (fixed seed)."""
    u = np.zeros(n) if jitter == 0 else np.random.default_rng(seed).uniform(-jitter, jitter, n)
    t = 2 * np.pi * (np.arange(n) + u) / n
    return point(name, t), t


def smooth_length(name, m=20000):
    """length of the closed curve by the periodic trapezoid rule (spectrally accurate)."""
    t = 2 * np.pi * np.arange(m) / m
    return float(speed(name, t).sum() * 2 * np.pi / m)


def regular_ngon(n, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return r * np.stack([np.cos(a), np.sin(a)], 1)


# the unevenly sampled unit circle used for the ROUND test (degrees, counterclockwise)
UNEVEN_DEG = np.array([0, 18, 50, 80, 105, 160, 185, 200, 240, 275, 300, 335], float)


def uneven_circle(r=1.0):
    a = np.radians(UNEVEN_DEG)
    return r * np.stack([np.cos(a), np.sin(a)], 1)


def jittered_flower(n=48, jitter=0.3, seed=3):
    return sample("flower", n, jitter, seed)[0]


# ---------------------------------------------------------------------------
# geometry of a vertex
# ---------------------------------------------------------------------------

def circumcenters(X):
    """(n, 2) centers c_i of the circle through x_{i-1}, x_i, x_{i+1} of a closed curve (NaN if collinear)."""
    X = np.asarray(X, float)
    a, b = np.roll(X, 1, 0) - X, np.roll(X, -1, 0) - X
    d = 2 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    aa, bb = (a ** 2).sum(1), (b ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.stack([b[:, 1] * aa - a[:, 1] * bb, a[:, 0] * bb - b[:, 0] * aa], 1) / d[:, None]
    return X + c


def curvature_vectors(X, kind, normal="circumcenter"):
    """discrete curvature vectors kappa^X_i of Definition 32.4.5 for a closed plane curve.

    A, B, C: kappa^X_i n_i with the vertex normal n_i (integrated). D: (c_i - x_i) / r_i^2 toward the circumcenter
    (pointwise, |.| = 1/r_i), or kappa^D_i n_i if normal="bisector" (Exercise 32.4.5)."""
    X = np.asarray(X, float)
    if kind == "D" and normal == "circumcenter":
        c = circumcenters(X) - X
        return c / (c ** 2).sum(1, keepdims=True)
    k = dd.DISCRETE_CURVATURES[kind](X, True)
    return k[:, None] * dd.curve_vertex_normals(X, True)


def flow(X, kind, h, steps, normal="circumcenter", every=1):
    """forward Euler x_i <- x_i + h kappa^X_i; returns the list of iterates 0, every, 2 every, ..."""
    X = np.asarray(X, float).copy()
    out = [X.copy()]
    for m in range(1, steps + 1):
        X = X + h * curvature_vectors(X, kind, normal)
        if m % every == 0:
            out.append(X.copy())
    return out


def own_total(X, kind):
    """total curvature measured by the definition itself: sum kappa^X_i (A, B, C), sum kappa^D_i lbar_i (D)."""
    k = dd.DISCRETE_CURVATURES[kind](X, True)
    if kind == "D":
        k = k * dd.curve_dual_lengths(X, True)
    return float(k.sum())


def fit_circle(X):
    """least-squares circle (Kasa fit): center c, radius r."""
    X = np.asarray(X, float)
    A = np.c_[2 * X, np.ones(len(X))]
    sol = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)[0]
    c = sol[:2]
    return c, float(np.sqrt(sol[2] + c @ c))


def roundness_defect(X):
    """(max - min) / mean of the distances from the vertices to the fitted center; 0 iff all on one circle."""
    c, _ = fit_circle(X)
    d = np.linalg.norm(np.asarray(X) - c, axis=1)
    return float((d.max() - d.min()) / d.mean())


# ---------------------------------------------------------------------------
# a high-resolution approximation of the smooth curve shortening flow
# ---------------------------------------------------------------------------

def resample(X, n):
    """n points equally spaced in arc length along the closed polygon X (linear interpolation)."""
    X = np.asarray(X, float)
    e = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
    s = np.r_[0.0, np.cumsum(e)]
    u = np.linspace(0, s[-1], n, endpoint=False)
    Xc = np.vstack([X, X[:1]])
    return np.stack([np.interp(u, s, Xc[:, 0]), np.interp(u, s, Xc[:, 1])], 1)


def smooth_flow(X0, times, n=400, cfl=0.2, resample_every=None):
    """approximate the smooth flow d gamma/dt = kappa_s n_s from the closed polygon X0 (already fine).

    Pointwise length gradient -grad L / lbar (a consistent discretization of kappa_s n_s), forward Euler with
    h = cfl * min(lbar)^2. X0 is resampled once to n points equally spaced in arc length; optional resampling every
    few steps (off by default: linear resampling cuts corners and loses area, about 5e-3 on the flower curve).
    Returns the curves at the given times."""
    X = resample(X0, n)
    t, step, out = 0.0, 0, []
    for target in times:
        while t < target - 1e-15:
            lb = dd.curve_dual_lengths(X, True)
            h = min(cfl * lb.min() ** 2, target - t)
            X = X - h * dd.length_gradient(X, True) / lb[:, None]
            t += h
            step += 1
            if resample_every and step % resample_every == 0:
                X = resample(X, n)
        out.append(X.copy())
    return out


def polygon_area(X):
    """signed area (shoelace), positive for a counterclockwise polygon."""
    X = np.asarray(X, float)
    Y = np.roll(X, -1, 0)
    return 0.5 * float((X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0]).sum())
