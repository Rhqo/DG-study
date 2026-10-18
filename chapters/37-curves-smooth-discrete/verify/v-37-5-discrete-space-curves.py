"""Verification for Section 37.5 (Discrete Space Curves).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-5-discrete-space-curves.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "37-5-discrete-space-curves.html"
check("Implementation 37.5: at most 30 lines", dgcode.code_lines(PAGE, "code-37-5-1") <= 30)
NS = dgcode.run_block(PAGE, "code-37-5-1")
rot, build = NS["rotate"], NS["space_curve_from_angles"]
rng = np.random.default_rng(375)

# ---------------------------------------------------------------- Lemma 37.5.4 and Implementation rotate
for _ in range(5):
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    b = np.cross(t, rng.normal(size=3))
    b /= np.linalg.norm(b)
    th = rng.uniform(0.1, 3.0)
    t2 = rot(t, b, th)
    check("Lemma 37.5.4: R(b, theta) t = cos t + sin (b x t), angle theta, t x t' / |t x t'| = b",
          np.allclose(t2, np.cos(th) * t + np.sin(th) * np.cross(b, t))
          and abs(np.arccos(np.clip(t @ t2, -1, 1)) - th) < 1e-12
          and np.allclose(np.cross(t, t2) / np.linalg.norm(np.cross(t, t2)), b))

# ---------------------------------------------------------------- Example 37.5.6 (closed forms)
d = sp.symbols("Delta", positive=True)
a, b = sp.symbols("a b", positive=True)
c = 1 - sp.cos(d)
em = sp.Matrix([a * c, a * sp.sin(d), b * d])
e0 = sp.Matrix([-a * c, a * sp.sin(d), b * d])
g = lambda tt: sp.Matrix([a * sp.cos(tt), a * sp.sin(tt), b * tt])  # noqa: E731
sym_equal("Example 37.5.6: e_{-1}, e_0 of the sampled helix", list(g(0) - g(-d)) + list(g(d) - g(0)), list(em) + list(e0))
sym_equal("Example 37.5.6: e_{-1} x e_0 = 2 a c (0, -b Delta, a sin Delta)", list(em.cross(e0)),
          list(2 * a * c * sp.Matrix([0, -b * d, a * sp.sin(d)])), {d: (0.1, 3)})
sym_equal("Example 37.5.6: l^2 = 2 a^2 c + b^2 Delta^2", e0.dot(e0), 2 * a ** 2 * c + b ** 2 * d ** 2, {d: (0.1, 3)})
for n in (8, 16, 32, 64):
    D = 2 * np.pi / n
    X = ddg37.helix_point(D * np.arange(-2, 5))
    ell, th, psi = ddg37.helix_discrete_angles(D)
    close(f"Example 37.5.6: closed forms = dgdec (n = {n})",
          [ell, th, psi], [dd.curve_edge_lengths(X, False)[2], dd.turning_angles(X, False)[2], dd.torsion_angles(X, False)[2]],
          1e-12)
table = {8: (0.80081, 0.74887, 0.24255, 0.93513, 0.30288), 16: (0.40758, 0.37573, 0.11486, 0.92186, 0.28182),
         32: (0.20469, 0.18802, 0.05667, 0.91854, 0.27686), 64: (0.10246, 0.09403, 0.02824, 0.91771, 0.27564)}
for n, row in table.items():
    ell, th, psi = ddg37.helix_discrete_angles(2 * np.pi / n)
    close(f"Example 37.5.6: table row n = {n}", np.round([ell, th, psi, th / ell, psi / ell], 5), row, 1e-12)
check("Example 37.5.6: psi > 0 for b > 0 and psi < 0 for b < 0",
      ddg37.helix_discrete_angles(0.5)[2] > 0 and ddg37.helix_discrete_angles(0.5, 1.0, -0.3)[2] < 0)
k, tau = ddg37.helix_kappa_tau()
e8 = ddg37.helix_discrete_angles(2 * np.pi / 8)
close("Figure 37.5.2: n = 8 relative errors about 2% (curvature) and 10% (torsion)",
      np.round([abs(e8[1] / e8[0] - k) / k, abs(e8[2] / e8[0] - tau) / tau], 2), [0.02, 0.10], 1e-12)

# ---------------------------------------------------------------- Example 37.5.7 (plane polygons)
X, _ = ddg37.sample("flower", 40)
X3 = np.c_[X, np.zeros(40)]
psi = dd.torsion_angles(X3, True)
th2 = dd.turning_angles(X, True)
check("Example 37.5.7: psi in {0, pi} for a plane polygon",
      np.all((np.abs(psi) < 1e-12) | (np.abs(np.abs(psi) - np.pi) < 1e-12)))
check("Example 37.5.7: psi = pi exactly where the turning direction changes (10 edges of the flower)",
      np.array_equal(np.abs(np.abs(psi) - np.pi) < 1e-12, th2 * np.roll(th2, -1) < 0) and np.sum(np.abs(psi) > 1) == 10)
close("Prop 37.5.14: the flower polygon has holonomy 0", dd.bishop_holonomy(X3), 0.0, 1e-12)

# ---------------------------------------------------------------- Theorem 37.5.8 (Implementation round trip)
for name, X in (("helix", ddg37.helix_point(2 * np.pi * np.arange(17) / 10)),
                ("random", np.cumsum(rng.normal(size=(12, 3)), 0))):
    ell = dd.curve_edge_lengths(X, False)
    th = dd.turning_angles(X, False)[1:-1]
    psi = dd.torsion_angles(X, False)[1:-1]
    T = dd.curve_tangents(X, False)
    B = dd.vertex_binormals(X, False)
    G = build(ell, th, psi, X[0], T[0], B[1])
    close(f"Theorem 37.5.8(a): rebuilding the {name} polygon from (l, theta, psi) and (gamma_0, t_0, b_1)", G, X, 1e-11)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    Y = X @ Q.T + rng.normal(size=3)
    close(f"Theorem 37.5.8(b): (l, theta, psi) invariant under a proper rigid motion ({name})",
          np.r_[dd.curve_edge_lengths(Y, False), dd.turning_angles(Y, False)[1:-1], dd.torsion_angles(Y, False)[1:-1]],
          np.r_[ell, th, psi], 1e-10)
    Z = X * np.array([1, 1, -1])
    close(f"Exercise 37.5.2: a reflection keeps theta and negates psi ({name})",
          np.r_[dd.turning_angles(Z, False)[1:-1], dd.torsion_angles(Z, False)[1:-1]], np.r_[th, -psi], 1e-10)

# ---------------------------------------------------------------- Non-example 37.5.9
P = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]], float)
Pt = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1]], float)
close("Non-example 37.5.9: same edge lengths and curvature angles (pi/2, 0, pi/2)",
      np.r_[dd.curve_edge_lengths(P, False), dd.turning_angles(P, False)[1:-1],
            dd.curve_edge_lengths(Pt, False), dd.turning_angles(Pt, False)[1:-1]],
      np.r_[[1, 1, 1, 1], [np.pi / 2, 0, np.pi / 2], [1, 1, 1, 1], [np.pi / 2, 0, np.pi / 2]], 1e-12)
check("Non-example 37.5.9: one polygon is planar, the other is not",
      np.linalg.matrix_rank(P[1:] - P[0], tol=1e-9) == 2 and np.linalg.matrix_rank(Pt[1:] - Pt[0], tol=1e-9) == 3)

# ---------------------------------------------------------------- Proposition 37.5.11, Exercise 37.5.3
for _ in range(5):
    t0 = rng.normal(size=3)
    t0 /= np.linalg.norm(t0)
    t1 = rng.normal(size=3)
    t1 /= np.linalg.norm(t1)
    v = np.cross(t0, t1)
    cc = t0 @ t1
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    P1 = np.eye(3) + K + K @ K / (1 + cc)
    bb = v / np.linalg.norm(v)
    thv = np.arccos(cc)
    Rb = np.array([rot(e, bb, thv) for e in np.eye(3)]).T
    u = np.cross(t0, rng.normal(size=3))
    close("Prop 37.5.11: (37.5.7) = R(b, theta) = dgdec.rotation_between; maps t0 to t1",
          np.r_[P1.ravel(), (P1 @ t0)], np.r_[Rb.ravel(), t1], 1e-12)
    close("Exercise 37.5.3: P u = u - <u, t1> (t0 + t1)/(1 + c) for u orthogonal to t0",
          P1 @ u, u - (u @ t1) / (1 + cc) * (t0 + t1), 1e-12)
    close("Prop 37.5.11: matches dgdec.rotation_between", P1, dd.rotation_between(t0, t1), 1e-12)

# ---------------------------------------------------------------- Prop 37.5.13 (discrete Bishop vs binormals)
X = ddg37.helix_point(2 * np.pi * np.arange(25) / 9) + 0.05 * rng.normal(size=(25, 3))
T, U, W = dd.parallel_transport_frames(X, closed=False)
B = dd.vertex_binormals(X, False)
psi = dd.torsion_angles(X, False)
vt = np.array([ddg37.signed_angle_about(B[i], U[i], T[i]) for i in range(1, len(T))])
close("Prop 37.5.13: vartheta_{i+1} - vartheta_i = -psi_i (mod 2 pi) on a perturbed helix",
      np.angle(np.exp(1j * (np.diff(vt) + psi[1:-1]))), np.zeros(len(vt) - 1), 1e-10)
check("Definition 37.5.12: the discrete Bishop frame is orthonormal and positively oriented",
      np.allclose(np.einsum("ij,ij->i", T, U), 0) and np.allclose(np.linalg.norm(U, axis=1), 1)
      and np.allclose(np.einsum("ij,ij->i", np.cross(T, U), W), 1))


# ---------------------------------------------------------------- Prop 37.5.14, Example 37.5.15 (trefoil)
def trefoil(n):
    s = 2 * np.pi * np.arange(n) / n
    return np.stack([(2 + np.cos(3 * s)) * np.cos(2 * s), (2 + np.cos(3 * s)) * np.sin(2 * s), np.sin(3 * s)], 1)


hols = []
for n in (60, 120, 240, 480):
    Xt = trefoil(n)
    check(f"Example 37.5.15: all curvature angles positive (n = {n})", dd.turning_angles(Xt, True).min() > 0)
    h = dd.bishop_holonomy(Xt)
    close(f"Prop 37.5.14: holonomy = -sum psi (mod 2 pi), n = {n}",
          np.angle(np.exp(1j * (h + dd.torsion_angles(Xt, True).sum()))), 0.0, 1e-9)
    hols.append(round(h, 4))
close("Example 37.5.15: holonomies 3.1307, 3.0533, 3.0336, 3.0286", hols, [3.1307, 3.0533, 3.0336, 3.0286], 1e-12)

# ---------------------------------------------------------------- Exercise 37.5.1, 37.5.4
P = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [2, 1, 3]], float)
close("Exercise 37.5.1: theta = (pi/2, pi/2), b_1 = e3, b_2 = e1, psi_1 = pi/2",
      np.r_[dd.turning_angles(P, False)[1:3], dd.vertex_binormals(P, False)[1], dd.vertex_binormals(P, False)[2],
            dd.torsion_angles(P, False)[1]], np.r_[np.pi / 2, np.pi / 2, 0, 0, 1, 1, 0, 0, np.pi / 2], 1e-12)
for _ in range(5):
    e1, e2 = rng.normal(size=3), rng.normal(size=3)
    kb = 2 * np.cross(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + e1 @ e2)
    thv = np.arccos(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2))
    close("Exercise 37.5.4: |(kappa b)_i| = 2 tan(theta_i / 2) along b_i", kb,
          2 * np.tan(thv / 2) * np.cross(e1, e2) / np.linalg.norm(np.cross(e1, e2)), 1e-10)

# ---------------------------------------------------------------- Exercises 37.5.5, 37.5.6 (printed code and output)
for ex in ("37-5-5", "37-5-6"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dgcode.run_block(PAGE, f"code-ex-{ex}", {})
    check(f"Exercise {ex.replace('-', '.')}: printed output matches",
          buf.getvalue().strip() == dgcode.code_block(PAGE, f"out-ex-{ex}").strip(), buf.getvalue())

summary()
