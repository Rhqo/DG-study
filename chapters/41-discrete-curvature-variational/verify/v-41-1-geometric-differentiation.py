"""Verification for Section 41.1 (Geometric Differentiation).

Run from the project root: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/verify/v-41-1-geometric-differentiation.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "41-1-geometric-differentiation.html"
rng = np.random.default_rng(20261020)

# ---------------------------------------------------------------- Implementation 41.1 and Exercise 41.1.6
check("Implementation 41.1: at most 30 lines", dgcode.code_lines(PAGE, "code-41-1-1") <= 30)
ns, out = ddg41.run_printed(PAGE, "code-41-1-1")
check("Implementation 41.1: printed output matches the page", out.strip() == ddg41.printed_block(PAGE, "out-41-1-1").strip(), out)
Vb, Fb = ns["V"], ns["F"]
close("Implementation 41.1: area_gradient = dgdec.area_gradient", ns["G"], dd.area_gradient(Vb, Fb), 1e-12)
close("Implementation 41.1: central_gradient = ddg41.central_gradient",
      ns["central_gradient"](ns["area"], Vb, 1e-6), ddg41.central_gradient(ns["area"], Vb, 1e-6), 1e-12)
ns2, out2 = ddg41.run_printed(PAGE, "code-ex-41-1-6", dict(ns))
check("Exercise 41.1.6: printed output matches the page", out2.strip() == ddg41.printed_block(PAGE, "out-ex-41-1-6").strip(), out2)
X = rng.normal(size=(4, 3))
close("Exercise 41.1.6: bending_gradient = ddg41.bending_angle_gradients", ns2["bending_gradient"](X),
      np.array(ddg41.bending_angle_gradients(*X)), 1e-12)
close("Exercise 41.1.6: bending_angle = dgmesh.edge_bending_angles",
      ns2["bending_angle"](X), list(dm.edge_bending_angles(X, np.array([[0, 1, 2], [1, 0, 3]])).values())[0][0], 1e-12)

# ---------------------------------------------------------------- Proposition 41.1.2 (edge length)
for t in range(5):
    a, b = rng.normal(size=3), rng.normal(size=3)
    P = np.array([a, b])
    num = ddg41.central_gradient(lambda Y: np.linalg.norm(Y[1] - Y[0]), P)
    u = (b - a) / np.linalg.norm(b - a)
    close(f"Proposition 41.1.2: grad of |b - a| = (-u, u) (random pair {t})", num, np.array([-u, u]), 1e-8)
    close(f"Proposition 41.1.2: dgdec.edge_length_gradient (random pair {t})", np.array(dd.edge_length_gradient(a, b)), np.array([-u, u]), 1e-14)

# ---------------------------------------------------------------- Proposition 41.1.4 (triangle area), 41.1.6 (angle), 41.1.3 identities
F1 = np.array([[0, 1, 2]])
for t in range(5):
    P = rng.normal(size=(3, 3))
    G = dd.triangle_area_gradients(P, F1)[0]
    num = ddg41.central_gradient(lambda Y: dm.face_areas(Y, F1).sum(), P)
    close(f"Proposition 41.1.4: (41.1.2) = finite differences (random triangle {t})", G, num, 1e-8)
    N = dm.face_normals(P, F1)[0]
    close(f"Proposition 41.1.4: (41.1.2) explicit formula at x_i (random triangle {t})", G[0], 0.5 * np.cross(N, P[2] - P[1]), 1e-14)
    check(f"Proposition 41.1.4: points toward x_i, length l_jk / 2 (random triangle {t})",
          np.dot(G[0], P[0] - P[1]) > 0 and abs(np.linalg.norm(G[0]) - 0.5 * np.linalg.norm(P[2] - P[1])) < 1e-12)
    A = dm.face_areas(P, F1)[0]
    check(f"Proposition 41.1.3 for the area: sum 0, torque 0, Euler k = 2 (random triangle {t})",
          np.abs(G.sum(0)).max() < 1e-13 and np.abs(np.cross(P, G).sum(0)).max() < 1e-13 and abs(np.sum(P * G) - 2 * A) < 1e-12)
    ga, gb, gc = dd.angle_gradient(P[0], P[1], P[2])
    ang = lambda Y: dm.corner_angles(Y, F1)[0, 0]  # noqa: E731
    num = ddg41.central_gradient(ang, P)
    close(f"Proposition 41.1.6: (41.1.3) = finite differences (random triangle {t})", np.array([ga, gb, gc]), num, 1e-7)
    u, v = P[1] - P[0], P[2] - P[0]
    n = np.cross(u, v) / np.linalg.norm(np.cross(u, v))
    close(f"Proposition 41.1.6: explicit formulas (random triangle {t})", np.array([gb, gc]),
          np.array([-np.cross(n, u) / (u @ u), np.cross(n, v) / (v @ v)]), 1e-13)
    g3 = np.array([ga, gb, gc])
    check(f"Proposition 41.1.3 for the angle: sum 0, torque 0, Euler k = 0 (random triangle {t})",
          np.abs(g3.sum(0)).max() < 1e-12 and np.abs(np.cross(P, g3).sum(0)).max() < 1e-12 and abs(np.sum(P * g3)) < 1e-12)
    CG = dd.corner_angle_gradients(P, F1)[0]               # [c, k]: gradient of corner angle c wrt vertex k
    close(f"Section 41.1: the three angle gradients sum to 0 at each vertex (random triangle {t})", CG.sum(0), np.zeros((3, 3)), 1e-12)
# volume (k = 3) and length (k = 1) Euler identities on a mesh
V, F = dm.icosphere(1)
V = V * (1 + 0.2 * rng.uniform(-1, 1, (len(V), 1)))
close("Proposition 41.1.3(c): volume is homogeneous of degree 3", np.sum(V * dd.signed_volume_gradient(V, F)), 3 * dm.signed_volume(V, F), 1e-12)
E = dm.edges(F)
Ltot = lambda Y: np.linalg.norm(Y[E[:, 1]] - Y[E[:, 0]], axis=1).sum()  # noqa: E731
close("Proposition 41.1.3(c): total edge length is homogeneous of degree 1", np.sum(V * ddg41.central_gradient(Ltot, V)), Ltot(V), 1e-7)

# ---------------------------------------------------------------- Example 41.1.5 and Figure 41.1.1
P = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
G = dd.triangle_area_gradients(P, F1)[0]
c0 = P.mean(0)
check("Example 41.1.5(a): equilateral: |grad A| = 1/2, pointing from the centroid to the vertex",
      np.allclose(np.linalg.norm(G, axis=1), 0.5) and all(np.dot(G[c], P[c] - c0) / (0.5 * np.linalg.norm(P[c] - c0)) > 1 - 1e-12 for c in range(3)))
close("Example 41.1.5(a): third vertex (0, 1/2)", G[2], [0, 0.5, 0], 1e-14)
P = np.array([[0, 0, 0], [4, 0, 0], [1, 1, 0.0]])
G = dd.triangle_area_gradients(P, F1)[0]
close("Example 41.1.5(b): obtuse triangle area gradients", G, [[-0.5, -1.5, 0], [0.5, -0.5, 0], [0, 2, 0]], 1e-14)
close("Example 41.1.5(b): angles 45, 18.43, 116.57 degrees", np.degrees(dm.corner_angles(P, F1)[0]), [45, 18.4349488, 116.5650512], 1e-6)
close("Example 41.1.5(b): |grad A| = 1.581, 0.707, 2", np.linalg.norm(G, axis=1), [np.sqrt(10) / 2, np.sqrt(2) / 2, 2], 1e-12)
close("Example 41.1.5(b): sum <x_c, grad_c A> = 4 = 2A", np.sum(P * G), 4.0, 1e-12)
ga, gb, gc = dd.angle_gradient(P[0], P[1], P[2])
close("Figure 41.1.1(b): angle gradients (0.5, -0.25), (0, -0.25), (-0.5, 0.5)", np.array([ga, gb, gc]),
      [[0.5, -0.25, 0], [0, -0.25, 0], [-0.5, 0.5, 0]], 1e-14)

# ---------------------------------------------------------------- Non-example 41.1.7
ax_, ay, az, bx, by, bz, cx, cy, cz = sp.symbols("a_x a_y a_z b_x b_y b_z c_x c_y c_z", real=True)
A_, B_, C_ = sp.Matrix([ax_, ay, az]), sp.Matrix([bx, by, bz]), sp.Matrix([cx, cy, cz])
cr = (B_ - A_).cross(C_ - A_)
check("Non-example 41.1.7: sympy d(area)/d a_x has 55 operations", sp.count_ops(sp.diff(sp.sqrt(cr.dot(cr)) / 2, ax_)) == 55)
uu, vv = B_ - A_, C_ - A_
check("Non-example 41.1.7: sympy d(alpha)/d a_x has 132 operations",
      sp.count_ops(sp.diff(sp.acos(uu.dot(vv) / sp.sqrt(uu.dot(uu) * vv.dot(vv))), ax_)) == 132)


def naive_grad_b(eps):
    a, b, c = np.zeros(3), np.array([1.0, 0, 0]), np.array([-np.cos(eps), np.sin(eps), 0])
    u, v = b - a, c - a
    cs = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    dcos = (v / np.linalg.norm(v) - cs * u / np.linalg.norm(u)) / np.linalg.norm(u)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -dcos / np.sqrt(1 - cs * cs), dd.angle_gradient(a, b, c)[1]


for eps, lo, hi in [(1e-6, 5e-5, 2e-4), (1e-7, 2e-4, 8e-4)]:
    nv, fv = naive_grad_b(eps)
    rel = np.linalg.norm(nv - np.array([0, -1.0, 0]))
    check(f"Non-example 41.1.7: arccos route at alpha = pi - {eps:g}: relative error {rel:.1e} in [{lo:g}, {hi:g}]", lo < rel < hi)
    close(f"Non-example 41.1.7: (41.1.3) at alpha = pi - {eps:g} is exact", fv, [0, -1, 0], 1e-12)
nv, fv = naive_grad_b(1e-8)
check("Non-example 41.1.7: arccos route fails at alpha = pi - 1e-8 (1 - cos^2 = 0)", not np.all(np.isfinite(nv)))
close("Non-example 41.1.7: (41.1.3) at alpha = pi - 1e-8", fv, [0, -1, 0], 1e-12)
X0 = np.array([[0, 0, 0], [1, 0, 0], [0.3, 1, 0], [0.6, -1, 0.0]])          # a flat hinge
d = np.array([0, 0, 1.0])
unsigned = lambda t: abs(ddg41.bending_angle(X0[0], X0[1], X0[2] + t * d, X0[3]))  # noqa: E731
h = 1e-6
right, left = (unsigned(h) - unsigned(0)) / h, (unsigned(0) - unsigned(-h)) / h
check("Non-example 41.1.7: |theta| has a kink at a flat edge (one-sided slopes +-1/h_k)", abs(right - 1.0) < 1e-4 and abs(left + 1.0) < 1e-4)

# ---------------------------------------------------------------- Proposition 41.1.8 and Example 41.1.9
worst = 0.0
for t in range(50):
    X = rng.normal(size=(4, 3))
    if abs(ddg41.bending_angle(*X)) > 3:
        continue
    g = np.array(ddg41.bending_angle_gradients(*X))
    worst = max(worst, np.abs(ddg41.central_gradient(lambda Y: ddg41.bending_angle(*Y), X) - g).max())
    if not (np.abs(g.sum(0)).max() < 1e-11 and np.abs(np.cross(X, g).sum(0)).max() < 1e-11 and abs(np.sum(X * g)) < 1e-11):
        check(f"Proposition 41.1.8: identities on random hinge {t}", False)
check(f"Proposition 41.1.8: (41.1.4) = finite differences on random hinges (max {worst:.1e})", worst < 1e-8)
X = np.array([[0, 0, 0.0], [2, 0, 0], [0.6, 1.2, 0], [1.3, -1.0, -0.7]])
th = ddg41.bending_angle(*X)
g = np.array(ddg41.bending_angle_gradients(*X))
close("Example 41.1.9: theta = 0.6107 (34.99 degrees)", [th, np.degrees(th)], [0.6107, 34.99], 6e-3)
close("Example 41.1.9: gradients", g, [[0, -0.1644, 0.8182], [0, -0.3054, 0.6862], [0, 0, -0.8333], [0, 0.4698, -0.6711]], 6e-5)
close("Example 41.1.9: A_f = 1.2, s_k = 0.3, s_l = 0.65", [dm.face_areas(X, np.array([[0, 1, 2]]))[0], 0.6 / 2, 1.3 / 2], [1.2, 0.3, 0.65], 1e-12)
close("Example 41.1.9 / Figure 41.1.2: |g_k| = 1/h_k = 0.833, |g_l| = 1/h_l = 0.819", np.linalg.norm(g[2:], axis=1), [0.8333, 0.8192], 5e-5)
check("Example 41.1.9: no gradient has a component along the edge", np.abs(g[:, 0]).max() < 1e-14)
check("Example 41.1.9: convex edge (theta > 0, x_l below the plane of f)", th > 0 and X[3, 2] < 0)

# ---------------------------------------------------------------- Figure 41.1.3 numbers
rng4 = np.random.default_rng(4)
V, F = dm.icosphere(1)
V = V * (1 + 0.2 * rng4.uniform(-1, 1, (len(V), 1)))
W = np.random.default_rng(5).normal(size=V.shape)
W /= np.linalg.norm(W)
exact = np.sum(dd.area_gradient(V, F) * W)
area = lambda Y: dm.face_areas(Y, F).sum()  # noqa: E731
close("Figure 41.1.3: dA(W) = 0.4943", exact, 0.4943, 5e-5)
eps = 10.0 ** -np.arange(1, 12.01, 0.25)
cen = np.array([abs((area(V + e * W) - area(V - e * W)) / (2 * e) - exact) for e in eps])
k = int(np.argmin(cen))
check(f"Figure 41.1.3: central minimum at eps = {eps[k]:.2e} with error {cen[k]:.1e}", abs(eps[k] - 1.78e-5) < 1e-7 and 2e-12 < cen[k] < 5e-12)
check("Figure 41.1.3: rounding error about 1e-4 at eps = 1e-12", 1e-5 < cen[-1] < 1e-3)
check("Figure 41.1.3: R^126 (n0 = 42)", V.size == 126)

# ---------------------------------------------------------------- Exercise 41.1.2
P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
close("Exercise 41.1.2: area gradients", dd.triangle_area_gradients(P, F1)[0], [[-0.5, -0.5, 0], [0.5, 0, 0], [0, 0.5, 0]], 1e-14)
close("Exercise 41.1.2: angle gradients", np.array(dd.angle_gradient(*P)), [[1, 1, 0], [0, -1, 0], [-1, 0, 0]], 1e-14)
summary()
