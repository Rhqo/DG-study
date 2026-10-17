"""Section 34.3 (Covectors, Sharp, and Flat): verification of every computation (GUIDELINES 13).

The sphere example takes its parametrization from dgsym.EXAMPLES (GUIDELINES 3.6); the mesh example uses
dgmesh.icosphere and dgdec.hodge0. The coding exercise is extracted from the HTML and executed as printed.
Run from the project root: PYTHONPATH=tools python3 chapters/34-exterior-algebra/verify/v-34-3-sharp-flat.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from dgcheck import check, close, sym_equal, summary  # noqa: E402
from dgsym import EXAMPLES  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgdec as dd  # noqa: E402
import ch34  # noqa: E402

PAGE = "34-3-covectors-sharp-flat.html"
rng = np.random.default_rng(343)


def flat(u, G):
    return G @ u


def sharp(a, G):
    return np.linalg.solve(G, a)


# ---------------------------------------------------------------------------
# (34.3.1), (34.3.2), Proposition 34.3.2
# ---------------------------------------------------------------------------
A = rng.normal(size=(4, 4))
G = A.T @ A + np.eye(4)
u, w, a = rng.normal(size=(3, 4))
close("(34.3.1): g(alpha^sharp, w) = alpha(w)", sharp(a, G) @ G @ w, a @ w, 1e-10)
close("(34.3.1): u^flat(w) = g(u, w)", flat(u, G) @ w, u @ G @ w, 1e-10)
close("(34.3.2): sharp and flat are inverse", sharp(flat(u, G), G), u, 1e-10)

s = sharp(a, G)
ns = np.sqrt(s @ G @ s)
sh = s / ns
close("Proposition 34.3.2(a): alpha(u) = |a| g(a_hat, u)", a @ u, ns * (sh @ G @ u), 1e-10)
# (b): points c a/|a|^2 lie on {alpha = c}; ker alpha = a-perp
for c in (-1.0, 0.0, 2.5):
    close(f"Proposition 34.3.2(b): alpha(c a/|a|^2) = c (c={c})", a @ (c * s / ns ** 2), c, 1e-10)
x = rng.normal(size=4)
xperp = x - (s @ G @ x) / ns ** 2 * s
close("Proposition 34.3.2(b): ker alpha = a-perp", a @ xperp, 0.0, 1e-10)
close("Proposition 34.3.2(b): spacing of {alpha = c} and {alpha = c+1} is 1/|a|", np.sqrt((s / ns ** 2) @ G @ (s / ns ** 2)),
      1 / ns, 1e-12)
# (c): maximize alpha over the g-unit sphere by sampling
L = np.linalg.cholesky(G)
Z = rng.normal(size=(200000, 4))
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
U = np.linalg.solve(L.T, Z.T).T                 # |U|_g = 1
vals = U @ a
check("Proposition 34.3.2(c): max over the g-unit sphere <= |a| and close to it",
      vals.max() <= ns + 1e-9 and vals.max() > ns * 0.97)
close("Proposition 34.3.2(c): alpha(a_hat) = |a|", a @ sh, ns, 1e-10)

# ---------------------------------------------------------------------------
# Example 34.3.3 and Figure 34.3.1
# ---------------------------------------------------------------------------
G2 = np.array([[2.0, 1.0], [1.0, 1.0]])
al = np.array([1.0, 1.0])
close("Example 34.3.3: Euclidean alpha^sharp = (1, 1)", sharp(al, np.eye(2)), [1, 1], 1e-12)
close("Example 34.3.3: G^{-1} = [[1,-1],[-1,2]]", np.linalg.inv(G2), [[1, -1], [-1, 2]], 1e-12)
close("Example 34.3.3: alpha^sharp_g = (0, 1)", sharp(al, G2), [0, 1], 1e-12)
close("Example 34.3.3: g(e2, d) = 0 for d = (1, -1)", np.array([0, 1]) @ G2 @ np.array([1, -1]), 0, 1e-12)
close("Example 34.3.3: e1^flat_g = 2 e^1 + e^2", flat(np.array([1.0, 0.0]), G2), [2, 1], 1e-12)
uu = np.array([2.2, 0.4])
close("Figure 34.3.1(a): alpha(u) = 2.6 = sqrt2 * (2.6/sqrt2)", [al @ uu, np.sqrt(2) * (uu @ al / np.sqrt(2))], [2.6, 2.6], 1e-12)
close("Figure 34.3.1: alpha(alpha^sharp) = |alpha^sharp|^2 (Euclid 2, g 1)",
      [al @ al, al @ sharp(al, G2), sharp(al, G2) @ G2 @ sharp(al, G2)], [2, 1, 1], 1e-12)

# ---------------------------------------------------------------------------
# Proposition 34.3.4
# ---------------------------------------------------------------------------
b, c = rng.normal(size=(2, 4))
close("Proposition 34.3.4(a): <alpha, beta> = sum g^{ij} alpha_i beta_j", sharp(b, G) @ G @ sharp(c, G),
      b @ np.linalg.inv(G) @ c, 1e-10)
close("Proposition 34.3.4(a): flat is an isometry", sharp(flat(u, G), G) @ G @ sharp(flat(w, G), G), u @ G @ w, 1e-10)
Q = np.linalg.cholesky(np.linalg.inv(G))       # columns: a g-orthonormal basis? check: Q^T G Q = I
Fb = np.linalg.inv(np.linalg.cholesky(G)).T
check("Proposition 34.3.4(b): for a g-orthonormal basis F, F^T G F = I and e_i^flat = e^i (dual basis)",
      np.allclose(Fb.T @ G @ Fb, np.eye(4)) and np.allclose((G @ Fb).T @ Fb, np.eye(4)))
Bm = rng.normal(size=(4, 4))
dualchange = np.linalg.inv(Bm).T
close("Proposition 34.3.4(c): dual change-of-basis matrix has det 1/det B", np.linalg.det(dualchange), 1 / np.linalg.det(Bm),
      1e-10)
Cm = rng.normal(size=(4, 4))
close("Proposition 34.3.4(c): det G = det(C)^2 > 0 for G = C^T C", np.linalg.det(Cm.T @ Cm), np.linalg.det(Cm) ** 2, 1e-8)

# ---------------------------------------------------------------------------
# Example 34.3.5 (sphere) and Figure 34.3.2
# ---------------------------------------------------------------------------
ex = EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
X = ex["expr"]
Xt, Xp = X.diff(th), X.diff(ph)
Gs = sp.Matrix([[Xt.dot(Xt), Xt.dot(Xp)], [Xp.dot(Xt), Xp.dot(Xp)]]).applyfunc(sp.simplify)
sym_equal("Example 34.3.5: first fundamental form diag(r^2, r^2 sin^2 th) (Example 7.1.2)", Gs,
          sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), ex["domain"])
comp2 = Gs.solve(sp.Matrix([0, 1]))
comp1 = Gs.solve(sp.Matrix([1, 0]))
sym_equal("Example 34.3.5: (beta^2)^sharp = x_phi / (r^2 sin^2 th)", comp2, sp.Matrix([0, 1 / (r ** 2 * sp.sin(th) ** 2)]),
          ex["domain"])
sym_equal("Example 34.3.5: (beta^1)^sharp = x_th / r^2", comp1, sp.Matrix([1 / r ** 2, 0]), ex["domain"])
sym_equal("Example 34.3.5: x_th^flat = r^2 beta^1, x_ph^flat = r^2 sin^2 th beta^2", Gs * sp.Matrix([1, 0]).row_join(sp.Matrix([0, 1])),
          sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), ex["domain"])
len2 = (comp2.T * Gs * comp2)[0, 0]
sym_equal("Example 34.3.5: |(beta^2)^sharp|^2 = 1/(r^2 sin^2 th)", len2, 1 / (r ** 2 * sp.sin(th) ** 2), ex["domain"])
for t0, expect in ((sp.pi / 2, 1), (sp.pi / 6, 2)):
    close(f"Example 34.3.5 / Figure 34.3.2: |(beta^2)^sharp| = {expect} at theta = {t0}, r = 1",
          float(sp.sqrt(len2.subs({r: 1, th: t0}))), expect, 1e-12)
close("Figure 34.3.2: |x_phi| = 0.5 at theta = pi/6, r = 1", float(sp.sqrt(Xp.dot(Xp)).subs({r: 1, th: sp.pi / 6, ph: 0.35})),
      0.5, 1e-12)

# ---------------------------------------------------------------------------
# Smooth vs. Discrete box: flat on mesh functions with M = diag(A_i)
# ---------------------------------------------------------------------------
V, F = dm.icosphere(2)
Ai = dd.hodge0(V, F)
check("Discrete box: vertex areas A_i > 0 (barycentric hodge0)", np.all(Ai > 0))
close("Discrete box: sum_i A_i = total mesh area", Ai.sum(), dm.face_areas(V, F).sum(), 1e-12)
M = np.diag(Ai)
one = np.ones(len(V))
close("Discrete box: 1^flat = (A_1, ..., A_n0)", flat(one, M), Ai, 1e-14)
close("Discrete box: sharp of the area covector is 1", sharp(Ai, M), one, 1e-12)
uz = V[:, 2]
close("Discrete box: lumped integral of z over the icosphere is ~ 0 (symmetry)", Ai @ uz, 0.0, 1e-12)

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
a3 = np.array([2.0, 0.0, -1.0])
u3 = np.array([1.0, 2.0, 3.0])
close("Exercise 34.3.1: alpha(u) = -1, |alpha^sharp| = sqrt5, projection -1/sqrt5, spacing 1/sqrt5",
      [a3 @ u3, np.linalg.norm(a3), (a3 @ u3) / np.linalg.norm(a3), 1 / np.linalg.norm(a3)],
      [-1, np.sqrt(5), -1 / np.sqrt(5), 1 / np.sqrt(5)], 1e-12)
close("Exercise 34.3.2(c): sharp for 4g is sharp/4", sharp(a, 4 * G), sharp(a, G) / 4, 1e-12)
lam = rng.uniform(0.5, 3.0, size=4)
close("Exercise 34.3.3(b): alpha^sharp = sum alpha_i/lambda_i e_i", sharp(a, np.diag(lam)), a / lam, 1e-12)
aa, bb = sp.symbols("a b", real=True)
dual_norm2 = (sp.Matrix([[aa, bb]]) * Gs.inv() * sp.Matrix([aa, bb]))[0, 0]
sym_equal("Exercise 34.3.4: |a beta^1 + b beta^2|^2 = a^2/r^2 + b^2/(r^2 sin^2 th)", dual_norm2,
          aa ** 2 / r ** 2 + bb ** 2 / (r ** 2 * sp.sin(th) ** 2), ex["domain"])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns5 = ch34.run_block(PAGE, "code-ex-34-3-5")
check("Exercise 34.3.5: solution output equals out-ex-34-3-5 (True True [0. 4.])",
      buf.getvalue().strip() == ch34.output_block(PAGE, "out-ex-34-3-5") == "True True [0. 4.]", buf.getvalue())
al6 = np.array([2.0, 3.0])
close("Exercise 34.3.6(a): Euclidean max sqrt13 at (2,3)/sqrt13", np.linalg.norm(al6), np.sqrt(13), 1e-12)
s6 = sharp(al6, G2)
close("Exercise 34.3.6(b): alpha^sharp_g = (-1, 4), |.|_g^2 = 10", list(s6) + [s6 @ G2 @ s6, al6 @ s6], [-1, 4, 10, 10], 1e-12)

summary()
