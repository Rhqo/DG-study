"""Chapter 37 helpers: running curves, samplings, angle lifts, rigid alignment (Track 2, GUIDELINES 2.5).

Import from a figure or verify script of this chapter with::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ddg37

The discrete-curve quantities themselves come from tools/dgdec.py (the canonical implementation): turning_angles,
curve_vertex_normals, length_gradient, vertex_binormals, torsion_angles, rotation_between, parallel_transport_frames,
bishop_holonomy. The circle and the helix come from dgsym.EXAMPLES. This module adds what Chapter 37 needs:

- the running plane curves of GUIDELINES 2.5 that are not in dgsym.EXAMPLES (copied from Chapter 32's lib/ddg32.py,
  same expressions): the ellipse (2 cos t, sin t), the flower r = 1 + 0.3 cos 5t, the limacon r = a + cos t
  (a = 0.4 in Section 32.4 is a different, convex limacon; here a is a parameter), and the figure-eight
  (sin t, sin t cos t);
- arc-length resampling of a smooth curve, continuous angle lifts, rotation and winding numbers of samples;
- the closed-form discrete curvature and torsion angles of a uniformly sampled helix (Example 37.5.6);
- proper rigid alignment (rotation + translation, no reflection) for "equal up to rigid motion" checks.

Conventions (as in dgdec): vertex i is X[i]; edge i runs from X[i] to X[i+1] (indices mod n for a closed curve);
theta_i is the turning angle at vertex i, phi the tangent angle; psi_i the torsion angle on edge i
(right-hand rule about t_i, so a right-handed helix has psi > 0, matching b' = -tau n of GUIDELINES 6.2).
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import dgsym  # noqa: E402

J = np.array([[0.0, -1.0], [1.0, 0.0]])

T_SYM = dgsym.EXAMPLES["ellipse"]["coords"][0]
A_SYM = sp.symbols("a", positive=True)
# the ellipse and the flower curve are defined once, in dgsym.EXAMPLES (GUIDELINES 3.6)
ELLIPSE_EXPR = sp.Matrix(list(dgsym.EXAMPLES["ellipse"]["expr"][:2]))
FLOWER_R = dgsym.EXAMPLES["flower"]["radius"]
FLOWER_EXPR = sp.Matrix(list(dgsym.EXAMPLES["flower"]["expr"][:2]))
FIG8_EXPR = sp.Matrix([sp.sin(T_SYM), sp.sin(T_SYM) * sp.cos(T_SYM)])
CIRCLE_EXPR = dgsym.EXAMPLES["circle"]["expr"][:2, :]          # (r cos t, r sin t)


def limacon_expr(a):
    """limacon r(t) = a + cos t (inner loop for 0 < a < 1, a cusp at a = 1)."""
    r = a + sp.cos(T_SYM)
    return sp.Matrix([r * sp.cos(T_SYM), r * sp.sin(T_SYM)])


CURVES = {"ellipse": ELLIPSE_EXPR, "flower": FLOWER_EXPR, "figure8": FIG8_EXPR,
          "circle": CIRCLE_EXPR.subs(dgsym.EXAMPLES["circle"]["params"][0], 1),
          "limacon": limacon_expr(sp.Rational(1, 2))}


def _vec(expr):
    f = sp.lambdify(T_SYM, list(expr), "numpy")

    def g(t):
        t = np.asarray(t, float)
        return np.stack([np.asarray(c, float) * np.ones_like(t) for c in f(t)], -1)
    return g


def _scal(expr):
    f = sp.lambdify(T_SYM, expr, "numpy")
    return lambda t: np.asarray(f(np.asarray(t, float)), float) * np.ones_like(np.asarray(t, float))


_POINT = {k: _vec(v) for k, v in CURVES.items()}
_D1 = {k: _vec(v.diff(T_SYM)) for k, v in CURVES.items()}
_KAPPA = {k: _scal(dgsym.signed_curvature(v, T_SYM)) for k, v in CURVES.items()}


def point(name, t):
    """(m, 2) points gamma(t) of the named closed curve (parameter period 2 pi)."""
    return _POINT[name](t)


def velocity(name, t):
    return _D1[name](t)


def speed(name, t):
    return np.linalg.norm(velocity(name, t), axis=-1)


def kappa(name, t):
    """signed curvature kappa_s(t) (dgsym.signed_curvature: n_s = J t, counterclockwise positive)."""
    return _KAPPA[name](t)


def sample(name, n, phase=0.0):
    """n vertices gamma(t_i), t_i = phase + 2 pi i / n (equal parameter steps, not equal arc length)."""
    t = phase + 2 * np.pi * np.arange(n) / n
    return point(name, t), t


def smooth_length(name, m=20000):
    t = 2 * np.pi * np.arange(m) / m
    return float(speed(name, t).sum() * 2 * np.pi / m)


def arclength_table(name, m=40000):
    """(t_j, s_j), j = 0..m: arc length s(t) on [0, 2 pi] by the trapezoid rule on a fine grid."""
    t = np.linspace(0, 2 * np.pi, m + 1)
    v = speed(name, t)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    return t, s


def t_of_s(name, s, m=40000):
    """parameter values at the arc lengths s (linear interpolation of the inverse of s(t))."""
    tt, ss = arclength_table(name, m)
    return np.interp(np.asarray(s, float), ss, tt)


# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------

def angle_lift(V):
    """continuous angle function of the (m, 2) array of nonzero vectors V (consecutive vectors not opposite)."""
    a = np.arctan2(V[:, 1], V[:, 0])
    d = np.angle(np.exp(1j * np.diff(a)))
    return a[0] + np.concatenate([[0.0], np.cumsum(d)])


def rotation_index_samples(name, m=20000):
    """rotation index of the closed curve from a fine sampling of its unit tangent."""
    t = 2 * np.pi * np.arange(m + 1) / m
    a = angle_lift(velocity(name, t))
    return (a[-1] - a[0]) / (2 * np.pi)


def winding_number_polygon(P, p):
    """(1 / 2 pi) sum of signed angles subtended by the edges of the closed polygon P at the point p."""
    A = P - p
    B = np.roll(P, -1, 0) - p
    return float(np.arctan2(A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0], (A * B).sum(1)).sum() / (2 * np.pi))


def signed_angle_about(u, v, axis):
    """signed angle from u to v measured about the unit vector axis (right-hand rule); u, v orthogonal to axis."""
    return float(np.arctan2(np.dot(np.cross(u, v), axis), np.dot(u, v)))


# ---------------------------------------------------------------------------
# helix (dgsym.EXAMPLES["helix"])
# ---------------------------------------------------------------------------

def helix_point(t, a=1.0, b=0.3):
    ex = dgsym.EXAMPLES["helix"]
    sa, sb = ex["params"]
    f = sp.lambdify(ex["coords"][0], list(ex["expr"].subs({sa: a, sb: b})), "numpy")
    t = np.asarray(t, float)
    return np.stack([np.asarray(c, float) * np.ones_like(t) for c in f(t)], -1)


def helix_kappa_tau(a=1.0, b=0.3):
    ex = dgsym.EXAMPLES["helix"]
    sa, sb = ex["params"]
    return float(ex["expected"]["kappa"].subs({sa: a, sb: b})), float(ex["expected"]["tau"].subs({sa: a, sb: b}))


def helix_discrete_angles(delta, a=1.0, b=0.3):
    """closed forms for the helix sampled at t_j = j delta (Example 37.5.6): edge length l, curvature angle theta,
    torsion angle psi (all edges and vertices alike by the screw symmetry)."""
    c, S, C = 1 - np.cos(delta), np.sin(delta), np.cos(delta)
    ell = np.sqrt(2 * a * a * c + b * b * delta * delta)
    theta = np.arccos((a * a * (S * S - c * c) + b * b * delta * delta) / ell ** 2)
    psi = np.arctan2(b * delta * S * ell, b * b * delta * delta * C + a * a * S * S)
    return ell, theta, psi


# ---------------------------------------------------------------------------
# rigid alignment
# ---------------------------------------------------------------------------

def rigid_align_error(X, Y):
    """max distance between Y and the best proper rigid motion of X (Kabsch without reflection)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    cx, cy = X.mean(0), Y.mean(0)
    H = (X - cx).T @ (Y - cy)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(X.shape[1])
    D[-1, -1] = d
    R = Vt.T @ D @ U.T
    return float(np.max(np.linalg.norm((X - cx) @ R.T + cy - Y, axis=1)))


# ---------------------------------------------------------------------------
# Whitney-Graustein homotopy by averaging tangent angles (Section 37.2, Proof sketch of Theorem 37.2.9)
# ---------------------------------------------------------------------------

def unit_length_tangent_angle(name, m=4000):
    """tangent angle phi(s) of the named curve rescaled to length 2 pi, on s_j = 2 pi j / m (j = 0..m),
    as a continuous lift, together with the rescaled start point."""
    L = smooth_length(name)
    s = 2 * np.pi * np.arange(m + 1) / m
    t = t_of_s(name, s * L / (2 * np.pi), 200000)
    phi = angle_lift(velocity(name, t))
    return s, phi, point(name, t[:1])[0] * 2 * np.pi / L


def averaged_curve(s, phi0, phi1, lam, p0=(0.0, 0.0)):
    """gamma_lam' = t_lam - mean(t_lam), t_lam = (cos, sin) of (1 - lam) phi0 + lam phi1 (trapezoid rule).
    Returns the points and the minimum speed |gamma_lam'|."""
    phi = (1 - lam) * phi0 + lam * phi1
    T = np.stack([np.cos(phi), np.sin(phi)], 1)
    ds = s[1] - s[0]
    mean = (0.5 * (T[1:] + T[:-1])).sum(0) * ds / (s[-1] - s[0])
    V = T - mean
    X = np.asarray(p0, float) + np.vstack([[0.0, 0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * ds, 0)])
    return X, float(np.linalg.norm(V, axis=1).min()), mean


# ---------------------------------------------------------------------------
# curve flows (Section 37.6); the page prints the same code in its Implementation boxes
# ---------------------------------------------------------------------------

def length_laplacian(X):
    """K = D^T diag(1/l) D for a closed polygon, so that grad L = K X (Proposition 37.6.6); also returns l."""
    n = len(X)
    D = np.roll(np.eye(n), 1, 1) - np.eye(n)
    ell = np.linalg.norm(D @ X, axis=1)
    return D.T @ np.diag(1.0 / ell) @ D, ell


def dual_lengths(X):
    ell = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
    return 0.5 * (ell + np.roll(ell, 1))


def explicit_step(X, h, weighted=True):
    """forward Euler for the length gradient flow; weighted: M = diag(lbar), else M = I."""
    K, _ = length_laplacian(X)
    g = K @ X
    return X - h * (g / dual_lengths(X)[:, None] if weighted else g)


def semi_implicit_step(X, h):
    """(M + h K) X_new = M X with K and M = diag(lbar) frozen at X."""
    K, _ = length_laplacian(X)
    M = np.diag(dual_lengths(X))
    return np.linalg.solve(M + h * K, M @ X)


def explicit_critical_step(X):
    """2 / largest eigenvalue of M^{-1} K (the stability bound of Proposition 37.6.5 for the frozen operator)."""
    K, _ = length_laplacian(X)
    return 2.0 / np.linalg.eigvals(np.diag(1.0 / dual_lengths(X)) @ K).real.max()


def elastic_energy(theta, lbar):
    return 0.5 * float(np.sum(theta ** 2 / lbar))


def edge_angles(theta, phi0=0.0):
    return phi0 + np.concatenate([[0.0], np.cumsum(theta[1:])])


def rebuild_closed(ell, theta, p0=(0.0, 0.0), phi0=0.0):
    """vertices gamma_0..gamma_{n-1} of the closed curve with edge lengths ell and turning angles theta."""
    phi = edge_angles(theta, phi0)
    T = np.stack([np.cos(phi), np.sin(phi)], 1)
    return np.asarray(p0, float) + np.vstack([[0.0, 0.0], np.cumsum(ell[:, None] * T, 0)[:-1]])


def isometric_elastic_step(theta, ell, h, k=1, newton=5):
    """theta <- theta / (1 + h), then weighted least-norm Newton steps back onto the closing conditions (37.4.4)
    with the edge lengths ell fixed (Section 37.6; the page prints the same algorithm)."""
    lbar = 0.5 * (ell + np.roll(ell, 1))
    theta = theta / (1 + h)
    for _ in range(newton):
        phi = edge_angles(theta)
        T = np.stack([np.cos(phi), np.sin(phi)], 1)
        c = np.r_[theta.sum() - 2 * np.pi * k, ell @ T]
        JT = (ell[:, None] * T) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
        G = np.zeros((3, len(theta)))
        G[0] = 1.0
        G[1:, 1:] = np.cumsum(JT[::-1], 0)[::-1][1:].T
        W = np.diag(lbar)
        theta = theta - W @ G.T @ np.linalg.solve(G @ W @ G.T, c)
    return theta
