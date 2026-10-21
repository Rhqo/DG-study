"""Verification for Section 42.1 (Many Definitions of the Laplacian).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-1-laplacian-definitions.py``
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
import ch42  # noqa: E402

from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE / "42-1-laplacian-definitions.html"
x, y, z = sp.symbols("x y z", real=True)


# ---- a tiny symbolic exterior calculus on R^2 with a given 1-form star -------------------------------------------
def d0(f):                      # function -> 1-form (a, b) meaning a dx + b dy
    return (sp.diff(f, x), sp.diff(f, y))


def d1(a):                      # 1-form -> coefficient of dx^dy
    return sp.diff(a[1], x) - sp.diff(a[0], y)


def star1(a, sign=1):           # our star: *dx = dy, *dy = -dx; Crane section 7.3: the opposite sign
    return (-sign * a[1], sign * a[0])


def lap_stardstard(f, sign=1):  # * d * d f with star on 2-forms *(g dx^dy) = g
    return sp.simplify(d1(star1(d0(f), sign)))


# ---------------------------------------------------------------- Proposition 42.1.1
for f in (sp.exp(x) * sp.sin(2 * y), x ** 3 * y - y ** 2, sp.cos(x * y)):
    lap = sp.diff(f, x, 2) + sp.diff(f, y, 2)
    divgrad = sp.diff(sp.diff(f, x), x) + sp.diff(sp.diff(f, y), y)
    hess = sp.Matrix([[sp.diff(f, x, x), sp.diff(f, x, y)], [sp.diff(f, y, x), sp.diff(f, y, y)]])
    sym_equal(f"Proposition 42.1.1: sum of second partials = div grad = tr Hess = *d*d for {f}",
              [lap, lap, lap], [divgrad, hess.trace(), lap_stardstard(f)])

# ---------------------------------------------------------------- Conventions box: the two stars of Crane
f = sp.exp(x) * sp.sin(2 * y) + x ** 2 * y
sym_equal("Conventions: with our 1-form star, *d*d f = Delta f on R^2", lap_stardstard(f, 1),
          sp.diff(f, x, 2) + sp.diff(f, y, 2))
sym_equal("Conventions: with the 1-form star of Cr section 7.3 (opposite sign), *d*d f = -Delta f, so -*d*d = Delta",
          lap_stardstard(f, -1), -(sp.diff(f, x, 2) + sp.diff(f, y, 2)))
# Green's identity with zero boundary term on the torus [0, 2pi]^2: <<du, dv>> = -<<Delta u, v>>
u_t, v_t = sp.sin(x) * sp.cos(2 * y), sp.sin(x) * sp.cos(2 * y) + sp.cos(x)
lhs = sp.integrate(sp.diff(u_t, x) * sp.diff(v_t, x) + sp.diff(u_t, y) * sp.diff(v_t, y), (x, 0, 2 * sp.pi), (y, 0, 2 * sp.pi))
rhs = sp.integrate((sp.diff(u_t, x, 2) + sp.diff(u_t, y, 2)) * v_t, (x, 0, 2 * sp.pi), (y, 0, 2 * sp.pi))
check("Conventions: <<du, dv>> = -<<Delta u, v>> (periodic square), so Cr Ex 7.6's Delta is our -Delta",
      sp.simplify(lhs + rhs) == 0 and lhs != 0)

# ---------------------------------------------------------------- Proposition 42.1.2 and Example 42.1.3, Figure 42.1.1
p = np.array([0.3, 0.5])
u3 = lambda X, Y: np.exp(X) * np.sin(2 * Y)  # noqa: E731
lap3 = -3 * np.exp(0.3) * np.sin(1.0)
Q = lambda u, e: 4 / e ** 2 * (ch42.circle_mean(u, p, e) - u(*p))  # noqa: E731
close("Example 42.1.3(c): Delta(e^x sin 2y) at (0.3, 0.5) = -3 e^0.3 sin 1 = -3.4076", lap3, -3.4076, tol=5e-5)
close("Figure 42.1.1(b): Q(1) = -2.8195, Q(0.1) = -3.4012", [Q(u3, 1.0), Q(u3, 0.1)], [-2.8195, -3.4012], tol=5e-5)
errs = [abs(Q(u3, e) - lap3) for e in (0.08, 0.04, 0.02)]
check("Proposition 42.1.2: the error of Q is O(eps^2) (ratio 4 per halving)", all(3.9 < a / b < 4.1 for a, b in zip(errs, errs[1:])))
check("Example 42.1.3(a),(b): Q = 4 for x^2 + y^2 and Q = 0 for x^2 - y^2 at every eps",
      all(abs(Q(lambda X, Y: X ** 2 + Y ** 2, e) - 4) < 1e-10 and abs(Q(lambda X, Y: X ** 2 - Y ** 2, e)) < 1e-10
          for e in (0.05, 0.5, 2.0)))
# n = 3: the factor 2n = 6 (Gauss-Legendre in cos(theta) times the trapezoidal rule in phi)
ct, wt = np.polynomial.legendre.leggauss(24)
ph = 2 * np.pi * np.arange(48) / 48
CT, PH = np.meshgrid(ct, ph, indexing="ij")
W = np.outer(wt, np.ones(48)) / (2 * 48)
om = np.stack([np.sqrt(1 - CT ** 2) * np.cos(PH), np.sqrt(1 - CT ** 2) * np.sin(PH), CT], -1)
u3d = lambda P: np.exp(P[..., 0]) * np.cos(P[..., 1]) * (1 + P[..., 2] ** 2)  # noqa: E731
p3 = np.array([0.2, -0.4, 0.5])
lap3d = (np.exp(p3[0]) * np.cos(p3[1]) * (1 + p3[2] ** 2) - np.exp(p3[0]) * np.cos(p3[1]) * (1 + p3[2] ** 2)
         + np.exp(p3[0]) * np.cos(p3[1]) * 2)
check("Proposition 42.1.2 (n = 3): sphere quadrature integrates omega^i omega^j to delta_ij / 3",
      np.allclose(np.einsum("ab,abi,abj->ij", W, om, om), np.eye(3) / 3))
q3 = [6 / e ** 2 * (np.sum(W * u3d(p3 + e * om)) - u3d(p3)) for e in (0.02, 0.01)]
close("Proposition 42.1.2 (n = 3): (6 / eps^2)(sphere mean - u(p)) -> Delta u(p)", q3[1], lap3d, tol=2e-4)
# n = 1: second difference
g1 = lambda X: np.sin(2 * X) + 0.3 * X ** 2  # noqa: E731
close("Proposition 42.1.2 (n = 1) and Figure 42.1.1(a): second differences of sin 2x + 0.3 x^2 at 0.75 and 2.3",
      [(g1(a + 1e-3) - 2 * g1(a) + g1(a - 1e-3)) / 1e-6 for a in (0.75, 2.3)],
      [-4 * np.sin(1.5) + 0.6, -4 * np.sin(4.6) + 0.6], tol=1e-5)
close("Figure 42.1.1(a): u''(0.75) = -3.39, u''(2.3) = 4.58", [-4 * np.sin(1.5) + 0.6, -4 * np.sin(4.6) + 0.6],
      [-3.39, 4.58], tol=6e-3)

# ---------------------------------------------------------------- Non-example 42.1.4
r_, th_ = sp.symbols("r theta", positive=True)
for uf, lap in ((r_ ** 2, 4), (r_ * sp.cos(th_), 0)):
    naive = sp.diff(uf, r_, 2) + sp.diff(uf, th_, 2)
    true = sp.diff(r_ * sp.diff(uf, r_), r_) / r_ + sp.diff(uf, th_, 2) / r_ ** 2
    check(f"Non-example 42.1.4: polar d_r^2 + d_th^2 of {uf} is {sp.simplify(naive)} but Delta = {lap}",
          sp.simplify(true - lap) == 0 and sp.simplify(naive - lap) != 0)

# ---------------------------------------------------------------- Example 42.1.7, Non-example 42.1.8, Remark 42.1.9
hgrid = 0.01
xs = np.arange(-5, 6) * hgrid + 0.7
uu = np.sin(3 * xs)
second = (uu[2:] - 2 * uu[1:-1] + uu[:-2]) / hgrid ** 2
close("Example 42.1.7(a): path graph with w = 1/h^2 gives u''", second, -9 * np.sin(3 * xs[1:-1]), tol=1e-3)
V, F = dm.icosphere(2)
w = dm.cotan_weights(V, F)
LG = np.zeros((len(V), len(V)))
for (i, j), wij in w.items():
    LG[i, j] += wij
    LG[j, i] += wij
    LG[i, i] -= wij
    LG[j, j] -= wij
close("Example 42.1.7(b): the graph Laplacian with cotan weights is dgmesh.cotan_laplacian", LG, dm.cotan_laplacian(V, F),
      tol=1e-13)
ell = 0.37
hexv = np.array([[0.0, 0.0]] + [[ell * np.cos(k * np.pi / 3), ell * np.sin(k * np.pi / 3)] for k in range(6)])
lgu = np.sum(hexv[1:, 0] ** 2 - hexv[0, 0] ** 2)
close("Non-example 42.1.8: d0^T d0 = -L_G (w = 1) applied to x^2 at the hexagon center is -3 l^2", -lgu, -3 * ell ** 2,
      tol=1e-14)
uf3 = lambda P: np.sin(P[0]) * np.exp(0.5 * P[1]) + P[2] ** 3  # noqa: E731
x0 = np.array([0.3, -0.2, 0.4])
lapu = -np.sin(0.3) * np.exp(-0.1) + 0.25 * np.sin(0.3) * np.exp(-0.1) + 6 * 0.4
for hh in (1e-2, 5e-3):
    drift = np.mean([uf3(x0 + s * hh * e) - uf3(x0) for e in np.eye(3) for s in (1, -1)])
    close(f"Remark 42.1.9: random walk drift = h^2/(2n) Delta u + O(h^3) (h = {hh})", drift / (hh ** 2 / 6), lapu,
          tol=3 * hh)

# ---------------------------------------------------------------- Proposition 42.1.10 and Example 42.1.11
uu_, vv_ = sp.symbols("u v", real=True)
mu = sp.Function("mu")(uu_, vv_)
fsym = sp.Function("f")(uu_, vv_)
# (42.1.5) with g = e^{2 mu} I
gmat = sp.exp(2 * mu) * sp.eye(2)
gi = gmat.inv()
sq = sp.sqrt(gmat.det())
coords = (uu_, vv_)
lb = sum(sp.diff(sq * gi[a, b] * sp.diff(fsym, coords[b]), coords[a]) for a in range(2) for b in range(2)) / sq
sym_equal("Proposition 42.1.10: (42.1.5) with g = e^{2 mu}(du^2 + dv^2) reduces to e^{-2 mu}(f_uu + f_vv)", sp.simplify(lb),
          sp.exp(-2 * mu) * (sp.diff(fsym, uu_, 2) + sp.diff(fsym, vv_, 2)))
# the star of Proposition 38.3.6(b) for g = e^{2 mu} I: X' = J X, so *du = dv, *dv = -du (independent of mu)
J = sp.Matrix([[0, -1], [1, 0]])
Xv = sp.Matrix(sp.symbols("X1 X2", real=True))
gX = lambda A, B: (A.T * gmat * B)[0]  # noqa: E731
check("Proposition 42.1.10(b) proof: J X has the same g-length, is g-orthogonal, and (X, JX) is positive",
      sp.simplify(gX(J * Xv, J * Xv) - gX(Xv, Xv)) == 0 and sp.simplify(gX(Xv, J * Xv)) == 0
      and sp.simplify(sp.Matrix.hstack(Xv, J * Xv).det() - (Xv[0] ** 2 + Xv[1] ** 2)) == 0)
t_, ph_ = sp.symbols("t phi", real=True)
for name, fz, ex in (("z", sp.tanh(t_), -2 * sp.tanh(t_)), ("x", sp.sech(t_) * sp.cos(ph_), -2 * sp.sech(t_) * sp.cos(ph_))):
    lapm = sp.cosh(t_) ** 2 * (sp.diff(fz, t_, 2) + sp.diff(fz, ph_, 2))
    sym_equal(f"Example 42.1.11: Mercator Delta {name} = -2 {name}", sp.simplify(lapm), ex)
close("Figure 42.1.2: at t = 1, d_t^2 z = -0.640, cosh^2 = 2.381, Delta z = -1.523",
      [float(sp.diff(sp.tanh(t_), t_, 2).subs(t_, 1)), np.cosh(1) ** 2, -2 * np.tanh(1)], [-0.640, 2.381, -1.523], tol=6e-4)
th, ph = sp.symbols("theta phi", positive=True)
lb_sph = lambda fz: sp.simplify(sp.diff(sp.sin(th) * sp.diff(fz, th), th) / sp.sin(th) + sp.diff(fz, ph, 2) / sp.sin(th) ** 2)  # noqa: E731
sym_equal("Example 42.1.11: (theta, phi) coordinates give the same Delta z = -2z", lb_sph(sp.cos(th)), -2 * sp.cos(th))

# ---------------------------------------------------------------- Proposition 42.1.12 (Green) on the unit disk
rr, tt = sp.symbols("rho tau", positive=True)
pol = {x: rr * sp.cos(tt), y: rr * sp.sin(tt)}
for uf, vf in ((x ** 2, 1 + y), (x * y + x ** 3, x ** 2 * y + 1), (x ** 2 - y ** 2, x)):
    lapu_ = sp.diff(uf, x, 2) + sp.diff(uf, y, 2)
    lhs = sp.integrate(sp.integrate((vf * lapu_).subs(pol) * rr, (rr, 0, 1)), (tt, 0, 2 * sp.pi))
    gg = (sp.diff(uf, x) * sp.diff(vf, x) + sp.diff(uf, y) * sp.diff(vf, y)).subs(pol)
    mid = -sp.integrate(sp.integrate(gg * rr, (rr, 0, 1)), (tt, 0, 2 * sp.pi))
    dn = (sp.diff(uf, x) * x + sp.diff(uf, y) * y).subs(pol).subs(rr, 1)          # n = (cos, sin) on the unit circle
    bd = sp.integrate((vf.subs(pol).subs(rr, 1) * dn), (tt, 0, 2 * sp.pi))
    check(f"Proposition 42.1.12: Green's first identity on the unit disk, u = {uf}, v = {vf}",
          abs(float(sp.N(lhs - mid - bd))) < 1e-10)

# ---------------------------------------------------------------- Corollary 42.1.13 on the unit sphere
area = lambda e: sp.integrate(sp.integrate(e * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))  # noqa: E731
_ct, _wt = np.polynomial.legendre.leggauss(40)
_TH = np.arccos(_ct)
_PH = 2 * np.pi * np.arange(80) / 80


def area_num(e):
    """int over the unit sphere of the sympy expression e(theta, phi) dA (Gauss-Legendre x trapezoid, d cos)."""
    fn = sp.lambdify((th, ph), e, modules="numpy")
    T, P = np.meshgrid(_TH, _PH, indexing="ij")
    vals = np.broadcast_to(np.asarray(fn(T, P), float), T.shape)
    return float(np.sum(_wt[:, None] * vals) * 2 * np.pi / 80)


uf = sp.cos(th) ** 2 + sp.sin(th) * sp.cos(ph) + sp.cos(th) ** 3
vf = sp.sin(th) * sp.sin(ph) + sp.cos(th)
grad_inner = lambda a, b: sp.diff(a, th) * sp.diff(b, th) + sp.diff(a, ph) * sp.diff(b, ph) / sp.sin(th) ** 2  # noqa: E731
lu, lv = lb_sph(uf), lb_sph(vf)
check("Corollary 42.1.13(b): int Delta u dA = 0 on the unit sphere", abs(area_num(lu)) < 1e-12)
check("Corollary 42.1.13(a): int v Delta u = int u Delta v", abs(area_num(vf * lu) - area_num(uf * lv)) < 1e-12)
check("Corollary 42.1.13(c): int u Delta u = -int |grad u|^2 < 0",
      abs(area_num(uf * lu) + area_num(grad_inner(uf, uf))) < 1e-12 and area_num(uf * lu) < 0)
check("Green on the sphere with u = v = z: int z Delta z = -8 pi / 3 = -int |grad z|^2",
      sp.simplify(area(sp.cos(th) * lb_sph(sp.cos(th))) + 8 * sp.pi / 3) == 0
      and sp.simplify(area(grad_inner(sp.cos(th), sp.cos(th))) - 8 * sp.pi / 3) == 0)

# ---------------------------------------------------------------- Proposition 42.1.16, Example 42.1.17
a_, b_ = sp.exp(x) * sp.cos(y) * x, x ** 2 * y ** 3 - sp.sin(x * y)


def lap_H_1(a):                 # -( *d*d + d*d* ) on 1-forms of R^2
    da = d1(a)                  # coefficient of dx^dy
    term1 = star1(d0(da))       # * d * (da dx^dy) = * d(da)
    sa = star1(a)
    dsa = d1(sa)                # d * a = dsa dx^dy, * of it = dsa
    term2 = d0(dsa)
    return (sp.simplify(-(term1[0] + term2[0])), sp.simplify(-(term1[1] + term2[1])))


res = lap_H_1((a_, b_))
sym_equal("Proposition 42.1.16(b): Delta_H(a dx + b dy) = -(Delta a dx + Delta b dy) on R^2", list(res),
          [-(sp.diff(a_, x, 2) + sp.diff(a_, y, 2)), -(sp.diff(b_, x, 2) + sp.diff(b_, y, 2))])
dtheta = (-y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2))
check("Example 42.1.17: d(dtheta) = 0, d*(dtheta) = 0 and Delta_H(dtheta) = 0",
      sp.simplify(d1(dtheta)) == 0 and sp.simplify(d1(star1(dtheta))) == 0 and all(sp.simplify(c) == 0 for c in lap_H_1(dtheta)))
check("Example 42.1.17: d_x^2(x / r^2) = (2x^3 - 6xy^2) / r^6 = -d_y^2(x / r^2)",
      sp.simplify(sp.diff(x / (x ** 2 + y ** 2), x, 2) - (2 * x ** 3 - 6 * x * y ** 2) / (x ** 2 + y ** 2) ** 3) == 0
      and sp.simplify(sp.diff(x / (x ** 2 + y ** 2), x, 2) + sp.diff(x / (x ** 2 + y ** 2), y, 2)) == 0)
# (c): positivity for a compactly supported 1-form, numerically on [-1, 1]^2 with a polynomial bump
bump = (1 - x ** 2) ** 3 * (1 - y ** 2) ** 3
al = (bump * x, bump * (1 + y))
lh = lap_H_1(al)
lhsc = sp.integrate(sp.expand(lh[0] * al[0] + lh[1] * al[1]), (x, -1, 1), (y, -1, 1))
dal = d1(al)
cod = -d1(star1(al))            # d* = -*d* on 1-forms of R^2: a function
rhsc = sp.integrate(sp.expand(dal ** 2 + cod ** 2), (x, -1, 1), (y, -1, 1))
check("Proposition 42.1.16(c): <<Delta_H a, a>> = ||da||^2 + ||d* a||^2 > 0 (a polynomial bump on the square)",
      sp.simplify(lhsc - rhsc) == 0 and rhsc > 0)

# ---------------------------------------------------------------- exercises
s_, tt_ = sp.symbols("s t", real=True)
X = sp.Matrix([sp.exp(s_) * sp.cos(tt_), sp.exp(s_) * sp.sin(tt_)])
Xs, Xt = X.diff(s_), X.diff(tt_)
check("Exercise 42.1.1: log-polar coordinates have E = G = e^{2s}, F = 0",
      sp.simplify(Xs.dot(Xs) - sp.exp(2 * s_)) == 0 and sp.simplify(Xs.dot(Xt)) == 0 and sp.simplify(Xt.dot(Xt) - sp.exp(2 * s_)) == 0)
check("Exercise 42.1.1: Delta(r^2) = 4 and Delta(log r) = 0 via e^{-2s}(d_s^2 + d_t^2)",
      sp.simplify(sp.exp(-2 * s_) * sp.diff(sp.exp(2 * s_), s_, 2) - 4) == 0 and sp.diff(s_, s_, 2) == 0)
a2, b2, c2, e2 = sp.symbols("a b c epsilon", real=True)
p1, p2, tt2 = sp.symbols("p1 p2 t", real=True)
quad = a2 * x ** 2 + b2 * x * y + c2 * y ** 2 + 3 * x - y + 2
circ = {x: p1 + e2 * sp.cos(tt2), y: p2 + e2 * sp.sin(tt2)}
mean = lambda e: sp.integrate(sp.expand(e.subs(circ, simultaneous=True)), (tt2, 0, 2 * sp.pi)) / (2 * sp.pi)  # noqa: E731
check("Exercise 42.1.2: (4/eps^2)(mean - u(p)) = 2(a + c) exactly for quadratics",
      sp.simplify(4 / e2 ** 2 * (mean(quad) - quad.subs({x: p1, y: p2})) - 2 * (a2 + c2)) == 0)
check("Exercise 42.1.2: and = 6 p1 exactly for x^3", sp.simplify(4 / e2 ** 2 * (mean(x ** 3) - p1 ** 3) - 6 * p1) == 0)
ns3, same3, printed3 = ch42.run_printed(PAGE, "code-ex-42-1-3", "out-ex-42-1-3")
check("Exercise 42.1.3: code-ex-42-1-3 prints exactly out-ex-42-1-3", same3, printed3)
cth = sp.cos(th)
sym_equal("Exercise 42.1.4: Delta(z^2 - 1/3) = -6 (z^2 - 1/3) on the unit sphere", lb_sph(cth ** 2 - sp.Rational(1, 3)),
          -6 * (cth ** 2 - sp.Rational(1, 3)))
check("Exercise 42.1.4: int z (z^2 - 1/3) dA = 0", abs(area_num(cth * (cth ** 2 - sp.Rational(1, 3)))) < 1e-14)
fz = sp.exp(x) * sp.cos(2 * y)
# 2-form f dx^dy: Delta_H = d d* = -d * d *; * (f dx^dy) = f; d * d f = (f_xx + f_yy) dx^dy
sym_equal("Exercise 42.1.5: Delta_H(f dx^dy) = -(Delta f) dx^dy", -d1(star1(d0(fz))), -(sp.diff(fz, x, 2) + sp.diff(fz, y, 2)))
Rs, rs, us, vs = sp.symbols("R r u v", positive=True)
rho = Rs + rs * sp.cos(us)
fz6 = rs * sp.sin(us)
lap6 = sp.diff(rho / rs * sp.diff(fz6, us), us) / (rs * rho) + sp.diff(fz6, vs, 2) / rho ** 2
Hs = -(Rs + 2 * rs * sp.cos(us)) / (2 * rs * rho)
sym_equal("Exercise 42.1.6: Delta z = -sin u (R + 2r cos u) / (r (R + r cos u)) = 2 H N_z (outward N)", lap6,
          -sp.sin(us) * (Rs + 2 * rs * sp.cos(us)) / (rs * rho), {Rs: (2.5, 3.0), rs: (0.3, 1.0), us: (0, 6.2)})
sym_equal("Exercise 42.1.6: 2 H N_z with H = -(R + 2r cos u)/(2 r rho), N_z = sin u", lap6, 2 * Hs * sp.sin(us),
          {Rs: (2.5, 3.0), rs: (0.3, 1.0), us: (0, 6.2)})
ns6, same6, printed6 = ch42.run_printed(PAGE, "code-ex-42-1-6", "out-ex-42-1-6")
check("Exercise 42.1.6: code-ex-42-1-6 prints exactly out-ex-42-1-6", same6, printed6)
for nu6, nv6 in ((12, 24), (24, 48)):
    V6, F6 = dm.torus_mesh(2.0, 0.8, nu6, nv6)
    E6, _ = dd.edge_list(F6)
    w6 = dd.hodge1(V6, F6)
    iu6, iv6 = E6 // nv6, E6 % nv6
    diag6 = ((iu6[:, 1] - iu6[:, 0]) % nu6 != 0) & ((iv6[:, 1] - iv6[:, 0]) % nv6 != 0)
    c6 = V6[F6].mean(1)
    tube6 = np.arctan2(c6[:, 2], np.hypot(c6[:, 0], c6[:, 1]) - 2.0)
    obt6 = dm.corner_angles(V6, F6).max(1) > np.pi / 2
    rows6 = np.histogram(tube6[obt6], bins=nu6, range=(-np.pi, np.pi))[0]
    check(f"Exercise 42.1.6 ({nu6}, {nv6}): the weight is 0 (round-off) exactly on the nu*nv diagonal edges, "
          "all other weights are positive, half of the triangles are obtuse, spread evenly around the tube",
          np.array_equal(np.abs(w6) < 1e-12, diag6) and diag6.sum() == nu6 * nv6 and np.all(w6[~diag6] > 0.5)
          and obt6.sum() * 2 == len(F6) and rows6.min() == rows6.max())
    # isosceles trapezoid cell -> cyclic -> opposite angles of a diagonal sum to pi
    ang6 = dm.corner_angles(V6, F6)
    pair6 = {}
    for t6 in range(len(F6)):
        for k6 in range(3):
            a6, b6 = sorted((F6[t6][(k6 + 1) % 3], F6[t6][(k6 + 2) % 3]))
            pair6.setdefault((a6, b6), []).append(ang6[t6, k6])
    sums6 = np.array([sum(pair6[(a6, b6)]) for (a6, b6), dg in zip(map(tuple, np.sort(E6, 1)), diag6) if dg])
    check(f"Exercise 42.1.6 ({nu6}, {nv6}): the two angles opposite each diagonal sum to pi",
          np.abs(sums6 - np.pi).max() < 1e-12, np.abs(sums6 - np.pi).max())
check("Exercise 42.1.3, 42.1.6: code blocks are at most 30 lines",
      dgcode.code_lines(PAGE, "code-ex-42-1-3") <= 30 and dgcode.code_lines(PAGE, "code-ex-42-1-6") <= 30)

summary()
