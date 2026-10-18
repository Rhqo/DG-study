"""Verification for Section 38.3 (The Gauss Map and Vector-Valued Forms).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-3-gauss-map-forms.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import ddg38  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "38-3-gauss-map-forms.html"
u, v, t, s = sp.symbols("u v t s", real=True)
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Implementation 38.3 and Exercise 38.3.5
check("Implementation 38.3: at most 30 lines", dgcode.code_lines(PAGE, "code-38-3-1") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-3-1")
check("Implementation 38.3: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-38-3-1").strip(),
      out)
vap, vapoly = ns["vector_area_patch"], ns["vector_area_polygon"]
close("Implementation 38.3: saddle patch = pi a^2 e3 (a = 0.8)", vap(ns["saddle_polar"], 0, 0.8, 0, 2 * np.pi),
      [0, 0, np.pi * 0.64], 1e-5)
close("Implementation 38.3: upper half torus = -4 pi R r e3", vap(ns["torus"], 0, np.pi, 0, 2 * np.pi, n=800),
      [0, 0, -4 * np.pi * 1.6], 2e-5)
check("Exercise 38.3.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-38-3-5") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-3-5", ns)
check("Exercise 38.3.5: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-ex-38-3-5").strip(), out)

# ---------------------------------------------------------------- Example 38.3.2, Proposition 38.3.3, Example 38.3.4
to = ex["torus"]
xt = to["expr"]
Rs, rs = to["params"]
ut, vt = to["coords"]
Nt = dgsym.unit_normal(xt, ut, vt, positive=to["positive"])
sym_equal("(38.3.2): torus N = -(cos u cos v, cos u sin v, sin u)", Nt,
          -sp.Matrix([sp.cos(ut) * sp.cos(vt), sp.cos(ut) * sp.sin(vt), sp.sin(ut)]))
xu, xv = xt.diff(ut), xt.diff(vt)
sym_equal("(38.3.4): N_u = -x_u / r", Nt.diff(ut), -xu / rs)
sym_equal("(38.3.4): N_v = -cos u / (R + r cos u) x_v", Nt.diff(vt), -sp.cos(ut) / (Rs + rs * sp.cos(ut)) * xv)
sym_equal("Prop 38.3.3: <N, x_u> = <N, x_v> = 0 (torus)", [Nt.dot(xu), Nt.dot(xv)], [0, 0])
sym_equal("Prop 38.3.3: <N, N_u> = <N, N_v> = 0 (torus)", [Nt.dot(Nt.diff(ut)), Nt.dot(Nt.diff(vt))], [0, 0])
xsa = sp.Matrix([u, v, u ** 2 - v ** 2])
Ns = dgsym.unit_normal(xsa, u, v)
sym_equal("Example 38.3.2: saddle N = (-2u, 2v, 1)/sqrt(1 + 4u^2 + 4v^2)", Ns,
          sp.Matrix([-2 * u, 2 * v, 1]) / sp.sqrt(1 + 4 * u ** 2 + 4 * v ** 2))
sym_equal("Prop 38.3.3: saddle <N, N_u> = <N, N_v> = 0", [Ns.dot(Ns.diff(u)), Ns.dot(Ns.diff(v))], [0, 0])
sp_ = ex["sphere"]
th, ph = sp_["coords"]
r = sp_["params"][0]
xs = sp_["expr"]
Nsp = dgsym.unit_normal(xs, th, ph, positive=sp_["positive"])
sym_equal("Example 38.3.4(a): sphere N = x / r, dN = dx / r", [Nsp - xs / r, Nsp.diff(th) - xs.diff(th) / r],
          [sp.zeros(3, 1), sp.zeros(3, 1)])

# ---------------------------------------------------------------- Proposition 38.3.5 (area normal)
for (X, a, b, pos) in ((xt, ut, vt, to["positive"]), (xs, th, ph, sp_["positive"]), (xsa, u, v, ())):
    E, F, G = dgsym.first_ff(X, a, b, positive=pos)
    Nn = dgsym.unit_normal(X, a, b, positive=pos)
    half_wedge = (X.diff(a).cross(X.diff(b)) - X.diff(b).cross(X.diff(a))) / 2      # (35.5.5) on (d_u, d_v), halved
    sym_equal("Prop 38.3.5: (1/2)(dx ^ dx)(d_u, d_v) = sqrt(EG - F^2) N", half_wedge,
              dgsym.simp(sp.sqrt(E * G - F ** 2), pos) * Nn)
rng = np.random.default_rng(383)
q = rng.uniform(-1, 1, 2)
xu_, xv_ = ddg38.partials(ddg38.torus, np.array(q[0]), np.array(q[1]))
J = np.stack([xu_, xv_], 1)
n_ = np.cross(xu_, xv_)
Nq = n_ / np.linalg.norm(n_)
for _ in range(3):
    X, Y = rng.normal(size=2), rng.normal(size=2)
    close("Prop 38.3.5: dx(X) x dx(Y) = dA(X, Y) N", np.cross(J @ X, J @ Y),
          np.linalg.norm(n_) * (X[0] * Y[1] - X[1] * Y[0]) * Nq, 1e-9)
sym_equal("Prop 38.3.5 / Exercise 35.5.4: torus x_u x x_v = r (R + r cos u) N (N inward)", xu.cross(xv),
          rs * (Rs + rs * sp.cos(ut)) * Nt)

# ---------------------------------------------------------------- Proposition 38.3.6 and Example 38.3.7
def gprime(g, X):
    """+90 degree rotation with respect to g, orientation of (d_u, d_v)."""
    Ginv = np.linalg.inv(g)
    Jg = Ginv @ np.array([[0.0, -1.0], [1.0, 0.0]]) * np.sqrt(np.linalg.det(g))
    return Jg @ X


for name, x, q in (("saddle", ddg38.saddle, (0.5, -0.4)), ("torus", ddg38.torus, (0.7, 1.9)),
                   ("Mercator", ddg38.mercator_inv, (0.3, -0.6))):
    a_, b_ = ddg38.partials(x, np.array(q[0]), np.array(q[1]))
    J = np.stack([a_, b_], 1)
    g = J.T @ J
    n_ = np.cross(a_, b_)
    Nq = n_ / np.linalg.norm(n_)
    for _ in range(2):
        X = rng.normal(size=2)
        Xp = gprime(g, X)
        check(f"Prop 38.3.6: X' is the g-rotation of X ({name})",
              abs(X @ g @ Xp) < 1e-8 and abs(Xp @ g @ Xp - X @ g @ X) < 1e-8 and X[0] * Xp[1] - X[1] * Xp[0] > 0)
        close(f"Prop 38.3.6(a): dx(X') = N x dx(X) ({name})", J @ Xp, np.cross(Nq, J @ X), 1e-6)
    # (b): star on 1-forms of (R^2, g): star(alpha) = sqrt(det g) * (g^{-1} alpha) rotated; compare with -alpha(X')
    Ginv = np.linalg.inv(g)
    for _ in range(2):
        al = rng.normal(size=2)                      # alpha = al_1 du + al_2 dv
        w = Ginv @ al                                # alpha^sharp
        star = np.sqrt(np.linalg.det(g)) * np.array([-w[1], w[0]])   # star alpha = iota_{alpha^sharp} dA
        X = rng.normal(size=2)
        close(f"Prop 38.3.6(b): (star alpha)(X) = -alpha(X') ({name})", star @ X, -(al @ gprime(g, X)), 1e-8)
# sanity of the star formula above in the Euclidean plane: star dx = dy, star dy = -dx (Example 35.1.8)
close("Prop 38.3.6(b): Euclidean plane star dx = dy, star dy = -dx",
      [np.array([-(np.linalg.inv(np.eye(2)) @ e)[1], (np.linalg.inv(np.eye(2)) @ e)[0]]) for e in np.eye(2)],
      [np.array([0.0, 1.0]), np.array([-1.0, 0.0])], 1e-15)
# Figure 38.3.2 numbers
g0 = np.array([[2.0, 0.8], [0.8, 1.64]])
Xp0 = gprime(g0, np.array([1.0, 0.0]))
close("Figure 38.3.2: X' = (-0.4 b, b), b = sqrt(2/1.32) = 1.231", Xp0, [-0.4 * np.sqrt(2 / 1.32), np.sqrt(2 / 1.32)], 1e-12)
J0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.8]])
close("Figure 38.3.2: N x dx(X) = (-0.49, 1.23, 0.49)", J0 @ Xp0, [-0.4924, 1.2309, 0.4924], 1e-4)
# Example 38.3.7 symbolically on the sphere
Ns_ = xs / r
sym_equal("Example 38.3.7: N x x_theta = x_phi / sin(theta)", Ns_.cross(xs.diff(th)), xs.diff(ph) / sp.sin(th))
sym_equal("Example 38.3.7: N x x_phi = -sin(theta) x_theta", Ns_.cross(xs.diff(ph)), -sp.sin(th) * xs.diff(th))
# star dx coefficients (d theta, d phi): star d theta = sin th d phi, star d phi = -d theta / sin th
star_dx = (sp.Matrix.hstack(-xs.diff(ph) / sp.sin(th), sp.sin(th) * xs.diff(th)))
minus_N_dx = sp.Matrix.hstack(-Ns_.cross(xs.diff(th)), -Ns_.cross(xs.diff(ph)))
sym_equal("Example 38.3.7: star dx = -N x dx on the sphere", star_dx, minus_N_dx)

# ---------------------------------------------------------------- Non-example 38.3.9, Proposition 38.3.10
c8 = sp.Matrix([sp.sin(t), sp.sin(t) * sp.cos(t), 0])
sym_equal("Non-example 38.3.9(b): x y' - y x' = -sin^3 t for the figure-eight", (c8[0] * c8[1].diff(t) - c8[1] * c8[0].diff(t)),
          -sp.sin(t) ** 3)
check("Non-example 38.3.9(b): its integral is 0", sp.integrate(-sp.sin(t) ** 3, (t, 0, 2 * sp.pi)) == 0)
p, qv = sp.Matrix(sp.symbols("p1:4", real=True)), sp.Matrix(sp.symbols("q1:4", real=True))
seg = p + s * (qv - p)
sym_equal("Prop 38.3.10(d): integral over a segment of c x c' = p x q",
          sp.integrate(seg.cross(seg.diff(s)), (s, 0, 1)), p.cross(qv))
P = rng.normal(size=(7, 3))
close("Prop 38.3.10(a): translation invariance", vapoly(P + np.array([1.0, -2.0, 0.5])), vapoly(P), 1e-12)
close("Prop 38.3.10(b): reversal flips the sign", vapoly(P[::-1]), -vapoly(P), 1e-12)

# ---------------------------------------------------------------- Theorem 38.3.11: d(x ^ dx) = dx ^ dx in coordinates
for (X, a, b) in ((xt, ut, vt), (xsa, u, v), (xs, th, ph)):
    # x ^ dx = (x x x_u) du + (x x x_v) dv; its d = (d_u (x x x_v) - d_v (x x x_u)) du ^ dv
    lhs = X.cross(X.diff(b)).diff(a) - X.cross(X.diff(a)).diff(b)
    sym_equal("Theorem 38.3.11: d(x ^ dx) = dx ^ dx = 2 x_u x x_v", lhs, 2 * X.diff(a).cross(X.diff(b)))

# ---------------------------------------------------------------- Example 38.3.13 and Corollary 38.3.12
th0, ae = sp.symbols("theta0 a", positive=True)
cap = ns["vector_area_patch"](lambda T, P_: ddg38.sphere(T, P_), 0, 1.1, 0, 2 * np.pi, n=600)
close("(38.3.11): cap theta0 = 1.1 has A = pi sin^2(theta0) e3 (r = 1)", cap, [0, 0, np.pi * np.sin(1.1) ** 2], 1e-5)
lat = sp.Matrix([sp.sin(th0) * sp.cos(s), sp.sin(th0) * sp.sin(s), sp.cos(th0)])
sym_equal("Example 38.3.13(a): latitude circle vector area = pi sin^2 theta0 e3",
          sp.integrate(lat.cross(lat.diff(s)), (s, 0, 2 * sp.pi)) / 2, sp.Matrix([0, 0, sp.pi * sp.sin(th0) ** 2]))
sym_equal("Example 38.3.13(a): |A| / area = (1 + cos theta0)/2",
          sp.pi * sp.sin(th0) ** 2 / (2 * sp.pi * (1 - sp.cos(th0))), (1 + sp.cos(th0)) / 2)
whole = ns["vector_area_patch"](lambda T, P_: ddg38.sphere(T, P_), 0, np.pi, 0, 2 * np.pi, n=400)
close("Corollary 38.3.12(b): sphere vector area = 0", whole, [0, 0, 0], 1e-9)
bd = sp.Matrix([ae * sp.cos(t), ae * sp.sin(t), ae ** 2 * sp.cos(2 * t)])
sym_equal("Example 38.3.13(b): saddle boundary vector area = pi a^2 e3",
          sp.integrate(sp.expand(bd.cross(bd.diff(t))), (t, 0, 2 * sp.pi)) / 2, sp.Matrix([0, 0, sp.pi * ae ** 2]))
sym_equal("Example 38.3.13(b): direct integral of (-2u, 2v, 1) over the disk", sp.integrate(sp.integrate(
          sp.Matrix([-2 * s * sp.cos(t), 2 * s * sp.sin(t), 1]) * s, (s, 0, ae)), (t, 0, 2 * sp.pi)),
          sp.Matrix([0, 0, sp.pi * ae ** 2]))
zc = sp.integrate(sp.integrate(-sp.sin(ut) * rs * (Rs + rs * sp.cos(ut)), (ut, 0, sp.pi)), (vt, 0, 2 * sp.pi))
sym_equal("Example 38.3.13(c): upper half torus z-component = -4 pi R r", zc, -4 * sp.pi * Rs * rs)
sym_equal("Exercise 38.3.3: pi((R - r)^2 - (R + r)^2) = -4 pi R r", sp.pi * ((Rs - rs) ** 2 - (Rs + rs) ** 2),
          -4 * sp.pi * Rs * rs)
# Figure 38.3.3: caps through the unit circle with apex height h
for h in (0.5, 1.0, 1.8):
    Rh = (1 + h * h) / (2 * h)
    c0 = h - Rh
    tm = np.arccos(-c0 / Rh)
    capx = lambda T, P_, Rh=Rh, c0=c0: np.array([Rh * np.sin(T) * np.cos(P_), Rh * np.sin(T) * np.sin(P_),
                                                c0 + Rh * np.cos(T) + 0 * P_])
    close(f"Figure 38.3.3: cap h = {h} has vector area (0, 0, pi)", ns["vector_area_patch"](capx, 0, tm, 0, 2 * np.pi, n=600),
          [0, 0, np.pi], 2e-5)
    close(f"Figure 38.3.3: cap h = {h} area 2 pi R_h h = pi (1 + h^2)", 2 * np.pi * Rh * h, np.pi * (1 + h * h), 1e-12)
close("Figure 38.3.3: area at h = 1.8 is 13.3", np.pi * (1 + 1.8 ** 2), 13.32, 0.01)

# ---------------------------------------------------------------- Exercises 38.3.1, 38.3.2, 38.3.4
rr, hh = sp.symbols("r h", positive=True)
cyl = sp.Matrix([rr * sp.cos(u), rr * sp.sin(u), v])
Nc = dgsym.unit_normal(cyl, u, v)
sym_equal("Exercise 38.3.1: cylinder dN = x_u / r du", [Nc.diff(u), Nc.diff(v)], [cyl.diff(u) / rr, sp.zeros(3, 1)])
sym_equal("Exercise 38.3.1: half cylinder vector area (0, 2rh, 0)",
          sp.integrate(sp.integrate(cyl.diff(u).cross(cyl.diff(v)), (u, 0, sp.pi)), (v, 0, hh)), sp.Matrix([0, 2 * rr * hh, 0]))
pp, qq, ss = (sp.Matrix(sp.symbols(f"{c}1:4", real=True)) for c in "pqs")
sym_equal("Exercise 38.3.2: (p x q + q x s + s x p)/2 = (q - p) x (s - p)/2",
          (pp.cross(qq) + qq.cross(ss) + ss.cross(pp)) / 2, (qq - pp).cross(ss - pp) / 2)
Rot = np.array([[0.0, -1.0], [1.0, 0.0]])
check("Exercise 38.3.4: the rotation squares to -I", np.allclose(Rot @ Rot, -np.eye(2)))

summary()
