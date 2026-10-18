"""Verification for Section 36.1 (Discretization: The de Rham Map).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-1-de-rham-map.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch36  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "36-1-de-rham-map.html"
rng = np.random.default_rng(3601)

# ---------------------------------------------------------------- Implementation 36.1
ns, same, printed = ch36.run_printed(PAGE, "code-36-1-1", "out-36-1-1")
check("Implementation 36.1: code-36-1-1 prints exactly out-36-1-1", same, printed)
check("Implementation 36.1: code-36-1-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-1-1") <= 30)
my_derham1 = ns["derham1"]

# ---------------------------------------------------------------- Example 36.1.2
V, F = ch36.grid(2)
E, index = dd.edge_list(F)
check("Example 36.1.2: 9 vertices, 16 edges, 8 counterclockwise faces, vertex 3a+b at (a/2, b/2)",
      (len(V), len(E), len(F)) == (9, 16, 8) and all(ch36.cross2(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) > 0)
      and all(np.allclose(V[3 * a + b], [a / 2, b / 2]) for a in range(3) for b in range(3)))
check("Example 36.1.2: edge order [0,1], [0,3], [0,4], [1,2], ..., [7,8]",
      E[:4].tolist() == [[0, 1], [0, 3], [0, 4], [1, 2]] and E[-1].tolist() == [7, 8])
f = lambda p: p[0] ** 2 - p[1] ** 2  # noqa: E731
df = lambda p: np.array([2 * p[0], -2 * p[1]])  # noqa: E731
xdy = lambda p: np.array([0.0, p[0]])  # noqa: E731
f_hat = dd.derham0(f, V)
close("Example 36.1.2(a): f_hat = (0, -0.25, -1, 0.25, 0, -0.75, 1, 0.75, 0)", f_hat,
      [0, -0.25, -1, 0.25, 0, -0.75, 1, 0.75, 0], tol=1e-15)
df_hat = dd.derham1(df, V, E)
close("Example 36.1.2(b): df_hat = differences of f_hat on every edge", df_hat, f_hat[E[:, 1]] - f_hat[E[:, 0]], 1e-14)
close("Example 36.1.2(b): df_hat on [0,1], [2,4], [4,8] = -0.25, 1, 0",
      df_hat[[index[(0, 1)], index[(2, 4)], index[(4, 8)]]], [-0.25, 1.0, 0.0], tol=1e-14)
a = dd.derham1(xdy, V, E)
close("Example 36.1.2(c): x dy on [i,j] = (x_i + x_j)/2 (y_j - y_i) (affine integrand, midpoint value)", a,
      0.5 * (V[E[:, 0], 0] + V[E[:, 1], 0]) * (V[E[:, 1], 1] - V[E[:, 0], 1]), tol=1e-15)
close("Example 36.1.2(c): x dy on [7,8], [4,8], [2,4] = 0.5, 0.375, -0.125",
      a[[index[(7, 8)], index[(4, 8)], index[(2, 4)]]], [0.5, 0.375, -0.125], tol=1e-15)
horiz = V[E[:, 0], 1] == V[E[:, 1], 1]
check("Example 36.1.2(c): x dy vanishes on the horizontal edges", np.all(np.abs(a[horiz]) < 1e-15))
fi = [n for n, t in enumerate(F.tolist()) if t == [4, 7, 8]][0]
w = dd.derham2(lambda p: p[0], V, F)
close("Example 36.1.2(d): x dx^dy over [4,7,8] = 5/48", w[fi], 5 / 48, tol=1e-15)
close("Example 36.1.2(d): dx^dy over every face = 1/8", dd.derham2(lambda p: 1.0, V, F), np.full(8, 0.125), 1e-15)
close("Figure 36.1.3(a): x dx^dy = area x centroid x on every face", w, 0.125 * V[F].mean(1)[:, 0], tol=1e-15)
close("Figure 36.1.3(a): value / area = 1/6, 1/3, 2/3, 5/6 (sorted distinct)", np.unique(np.round(w / 0.125, 12)),
      [1 / 6, 1 / 3, 2 / 3, 5 / 6], tol=1e-12)

# ---------------------------------------------------------------- Example 36.1.3
Vi, Fi = dm.icosphere(1)
Ei, _ = dd.edge_list(Fi)
check("Example 36.1.3(a): icosphere(1) has 42 vertices on the unit sphere",
      len(Vi) == 42 and np.allclose(np.linalg.norm(Vi, axis=1), 1.0))
close("Example 36.1.3(a): R_1(dz) = z_j - z_i on icosphere(1)", dd.derham1(lambda p: np.array([0, 0, 1.0]), Vi, Ei),
      Vi[Ei[:, 1], 2] - Vi[Ei[:, 0], 2], tol=1e-14)
s, t = sp.symbols("s t", real=True)
X0, X1, X2 = sp.Matrix([1, 0, 0]), sp.Matrix([0, 2, 0]), sp.Matrix([0, 0, 1])
P = X0 + s * (X1 - X0) + t * (X2 - X0)
u, v = X1 - X0, X2 - X0
integrand = P[2] * (u[0] * v[1] - u[1] * v[0])
exact = sp.integrate(sp.integrate(integrand, (s, 0, 1 - t)), (t, 0, 1))
check("Example 36.1.3(b): parametrization (1 - s - t, 2s, t), dx^dy(x1 - x0, x2 - x0) = 2, integral = 1/3 (sympy)",
      sp.simplify(P - sp.Matrix([1 - s - t, 2 * s, t])) == sp.zeros(3, 1) and (u[0] * v[1] - u[1] * v[0]) == 2
      and exact == sp.Rational(1, 3))
T3 = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
close("Example 36.1.3(b): derham2 of z dx^dy over the triangle = 1/3",
      dd.derham2(lambda p: np.array([0.0, 0.0, p[2]]), T3, np.array([[0, 1, 2]]))[0], 1 / 3, tol=1e-14)
close("Example 36.1.3(b): constant dx^dy over the same triangle = 1 (shadow area)",
      dd.derham2(lambda p: np.array([0.0, 0.0, 1.0]), T3, np.array([[0, 1, 2]]))[0], 1.0, tol=1e-14)

# ---------------------------------------------------------------- Proposition 36.1.4
Vg, Fg = ch36.grid(4)
Vg = Vg + 0.04 * rng.normal(size=Vg.shape)
Eg, _ = dd.edge_list(Fg)
al = lambda p: np.array([p[0] * p[1] ** 2, 1 + p[0] ** 3])  # noqa: E731
be = lambda p: np.array([np.sin(p[1]), p[0] - p[1]])  # noqa: E731
close("Proposition 36.1.4(a): R_1 is linear (2 alpha - 3 beta)",
      dd.derham1(lambda p: 2 * al(p) - 3 * be(p), Vg, Eg), 2 * dd.derham1(al, Vg, Eg) - 3 * dd.derham1(be, Vg, Eg),
      tol=1e-13)
rho = lambda p: 1 + p[0] * p[1] - p[1] ** 2  # noqa: E731
base = dd.derham2(rho, Vg, Fg)
cyc = dd.derham2(rho, Vg, Fg[:, [1, 2, 0]])
odd = dd.derham2(rho, Vg, Fg[:, [1, 0, 2]])
close("Proposition 36.1.4(b): cyclic (even) reorder of a face keeps the value", cyc, base, tol=1e-14)
close("Proposition 36.1.4(b): odd reorder of a face flips the sign", odd, -base, tol=1e-14)
flip = rng.random(len(Eg)) < 0.5
Eflip = np.where(flip[:, None], Eg[:, ::-1], Eg)
P1m = np.diag(np.where(flip, -1.0, 1.0))
close("Proposition 36.1.4(c): new orientations give P_1 R_1", dd.derham1(al, Vg, Eflip), P1m @ dd.derham1(al, Vg, Eg),
      tol=1e-14)

# ---------------------------------------------------------------- Non-example 36.1.5
mid = lambda alpha, A, B: float(np.dot(alpha(0.5 * (A + B)), B - A))  # noqa: E731
dcube = lambda p: np.array([3 * p[0] ** 2, 0.0])  # noqa: E731
A_, B_ = np.array([0.0, 0.0]), np.array([1.0, 0.0])
close("Non-example 36.1.5(a): midpoint sample of d(x^3) on [(0,0),(1,0)] is 3/4", mid(dcube, A_, B_), 0.75, 1e-15)
close("Non-example 36.1.5(a): the integral (de Rham value) is f(1) - f(0) = 1",
      dd.derham1(dcube, np.array([A_, B_]), np.array([[0, 1]]))[0], 1.0, tol=1e-14)
check("Non-example 36.1.5(a): midpoint sampling agrees with R_1 for affine coefficients (x dy on the grid)",
      all(abs(mid(xdy, V[i], V[j]) - a[n]) < 1e-15 for n, (i, j) in enumerate(E)))
# (b): rotating the coordinate axes changes vertex samples of (P, Q) but not the de Rham image
Rot = np.array([[0.0, 1.0], [-1.0, 0.0]])          # new coordinates p' = Rot p (axes turned by 90 degrees)
alpha_old = lambda p: np.array([p[0] * p[1], p[0] ** 2 + 1])  # noqa: E731  (P, Q)
alpha_new = lambda q: Rot @ alpha_old(Rot.T @ q)  # noqa: E731  components in the new coordinates
Vnew = Vg @ Rot.T
samples_old = np.array([alpha_old(p) for p in Vg])
samples_new = np.array([alpha_new(q) for q in Vnew])
check("Non-example 36.1.5(b): vertex samples change to (Q, -P) under the 90-degree change of axes",
      np.allclose(samples_new, np.stack([samples_old[:, 1], -samples_old[:, 0]], 1))
      and not np.allclose(samples_new, samples_old))
close("Non-example 36.1.5(b): the de Rham image does not change", dd.derham1(alpha_new, Vnew, Eg),
      dd.derham1(alpha_old, Vg, Eg), tol=1e-13)

# ---------------------------------------------------------------- Example 36.1.9
for n in (3, 5, 8):
    Vn, Fn = ch36.grid(n)
    En, _ = dd.edge_list(Fn)
    h = 1.0 / n
    g = lambda p, h=h: np.sin(np.pi * p[0] / h) * np.sin(np.pi * p[1] / h)  # noqa: E731
    dg = lambda p, h=h: (np.pi / h) * np.array([np.cos(np.pi * p[0] / h) * np.sin(np.pi * p[1] / h),  # noqa: E731
                                                np.sin(np.pi * p[0] / h) * np.cos(np.pi * p[1] / h)])
    gh, dgh = dd.derham0(g, Vn), dd.derham1(dg, Vn, En, order=10)
    check(f"Example 36.1.9 (n = {n}): R_0(g) = 0 and R_1(dg) = 0 although g(h/2, h/2) = 1 and dg(h/4, h/2) != 0",
          np.abs(gh).max() < 1e-12 and np.abs(dgh).max() < 1e-12 and abs(g(np.array([h / 2, h / 2])) - 1) < 1e-12
          and np.linalg.norm(dg(np.array([h / 4, h / 2]))) > 1)

# ---------------------------------------------------------------- Figure 36.1.3(b): scaling h^k
ns_ = np.array([2, 4, 8, 16, 32, 64])
m = [[], [], []]
for n in ns_:
    Vn, Fn = ch36.grid(int(n))
    En, _ = dd.edge_list(Fn)
    m[0].append(np.abs(dd.derham0(f, Vn)).max())
    m[1].append(np.abs(dd.derham1(xdy, Vn, En)).max())
    m[2].append(np.abs(dd.derham2(lambda p: p[0], Vn, Fn)).max())
sl = [np.polyfit(np.log(1.0 / ns_), np.log(mm), 1)[0] for mm in m]
check(f"Figure 36.1.3(b): slopes {sl[0]:.3f}, {sl[1]:.3f}, {sl[2]:.3f} ~ 0, 1, 1.95",
      abs(sl[0]) < 1e-9 and abs(sl[1] - 1) < 1e-6 and abs(sl[2] - 1.95) < 0.01)
close("Figure 36.1.3(b): the largest face value of x dx^dy is (h^2/2)(1 - h/3)", m[2],
      [(1 / n) ** 2 / 2 * (1 - 1 / (3 * n)) for n in ns_], tol=1e-15)

# ---------------------------------------------------------------- Exercises
Vt, Ft = ch36.reference_triangle()
Et = np.array([[0, 1], [1, 2], [2, 0]])
ydx = dd.derham1(lambda p: np.array([p[1], 0.0]), Vt, Et)
close("Exercise 36.1.1: y dx on [x0,x1], [x1,x2], [x2,x0] = 0, -1/2, 0; d(y dx) = -dx^dy integrates to -1/2",
      np.concatenate([ydx, [dd.derham2(lambda p: -1.0, Vt, Ft)[0]]]), [0, -0.5, 0, -0.5], tol=1e-15)
ok = True
for _ in range(20):
    tri = rng.normal(size=(3, 2))
    T = np.array([[tri[1] - tri[0]], [tri[2] - tri[1]], [tri[0] - tri[2]]])[:, 0, :]   # rows: edge vectors
    ok &= np.linalg.matrix_rank(T) == 2 and np.allclose(np.ones(3) @ T, 0)
    c = rng.normal(size=3)
    c -= c.mean()                                        # zero cyclic sum
    ab = np.linalg.lstsq(T, c, rcond=None)[0]
    ok &= np.allclose(T @ ab, c)
check("Exercise 36.1.3: (a, b) -> edge values has rank 2 with image {zero cyclic sum}; every such c is attained", ok)
ns5, same5, printed5 = ch36.run_printed(PAGE, "code-ex-36-1-5", "out-ex-36-1-5")
check("Exercise 36.1.5: code-ex-36-1-5 prints exactly out-ex-36-1-5", same5, printed5)
Vi2, Fi2 = dm.icosphere(2)
N2 = dm.face_normals(Vi2, Fi2)
shadow = 0.5 * dm.face_cross(Vi2, Fi2)[:, 2]
close("Exercise 36.1.5(b): derham2 of dx^dy = signed projected area per face", dd.derham2(
      lambda p: np.array([0.0, 0.0, 1.0]), Vi2, Fi2), shadow, tol=1e-14)
check("Exercise 36.1.5(b): the upward faces cover a polygon inside the unit disk (sum < pi)",
      shadow[N2[:, 2] > 0].sum() < np.pi)

summary()
