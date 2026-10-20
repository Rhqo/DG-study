"""Verification for Section 43.4 (Conformal Energy on a Riemann Surface).

Pointwise linear algebra: at a point a 1-form is a row vector of its values on a positively oriented orthonormal basis
(e1, e2 = J e1); a complex 1-form is a complex row vector. Then (star a)(e1) = -a(e2), (star a)(e2) = a(e1)
(Proposition 43.4.1(a)), (a ^ b)(e1, e2) = a1 b2 - a2 b1, and the pointwise inner product <a, b> = a . conj(b).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/verify/v-43-4-conformal-energy.py``
"""

import pathlib
import sys

import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch43  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "43-4-conformal-energy.html"
rng = np.random.default_rng(4)

star = lambda a: np.array([-a[1], a[0]])                # our star on 1-forms (values on e1, e2)  # noqa: E731
star_cr = lambda a: np.array([a[1], -a[0]])             # Crane's star: alpha o J  # noqa: E731
wedge = lambda a, b: a[0] * b[1] - a[1] * b[0]          # (a ^ b)(e1, e2)  # noqa: E731
ip = lambda a, b: wedge(a, star(np.conj(b)))            # density of <<a, b>> = a ^ star conj(b)  # noqa: E731
crand = lambda: rng.standard_normal(2) + 1j * rng.standard_normal(2)  # noqa: E731

# ---------------------------------------------------------------- Proposition 43.4.1
dx, dy = np.array([1.0, 0]), np.array([0, 1.0])
check("Proposition 43.4.1(a), Exercise 43.4.2: star dx = dy, star dy = -dx, star star = -1",
      np.allclose(star(dx), dy) and np.allclose(star(dy), -dx) and np.allclose(star(star(dx)), -dx))
ok = True
for _ in range(20):
    a, b = crand(), crand()
    ok &= abs(wedge(star(a), star(b)) - wedge(a, b)) < 1e-12
check("Proposition 43.4.1(c): star a ^ star b = a ^ b for 1-forms", ok)
# sphere: -dtheta o J = sin(theta) dphi, -dphi o J = -dtheta / sin(theta)  (Example 38.3.7)
th = 1.1
Jmat = np.array([[0, -np.sin(th)], [1 / np.sin(th), 0]])     # columns: J d_theta, J d_phi in (d_theta, d_phi) coordinates
dth, dph = np.array([1.0, 0]), np.array([0, 1.0])
check("Proposition 43.4.1(a): on the sphere -d theta o J = sin(theta) d phi, -d phi o J = -d theta / sin(theta)",
      np.allclose(-dth @ Jmat, np.sin(th) * dph) and np.allclose(-dph @ Jmat, -dth / np.sin(th)))
# (b): omega(X, JX) = c for omega = c dA and every unit X
c = 2.7
okb = all(abs(c * wedge(np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])) - c) < 1e-12 for t in np.linspace(0, 6, 7))
check("Proposition 43.4.1(b): (c dA)(X, JX) = c for every unit X", okb)

# ---------------------------------------------------------------- Conventions box (Crane comparison) in R^2
dz_id = np.array([1, 1j])                                   # dz of z = x + i y: dz(e1) = 1, dz(e2) = i
check("Conventions: our star dz = -i dz, Crane's star dz = i dz for the identity map",
      np.allclose(star(dz_id), -1j * dz_id) and np.allclose(star_cr(dz_id), 1j * dz_id) and np.allclose(star_cr(dx), -dy))
okE = okI = True
for _ in range(20):
    a, b = crand(), crand()
    ours = 0.25 * ip(star(a) + 1j * a, star(a) + 1j * a).real
    cr = 0.25 * wedge(star_cr(np.conj(star_cr(a) - 1j * a)), star_cr(a) - 1j * a).real
    okE &= abs(ours - cr) < 1e-12
    okI &= abs(wedge(star_cr(np.conj(a)), b).real - ip(a, b).real) < 1e-12
check("Conventions: Crane's (1/4)|star_Cr dz - i dz|^2 (inner product Re int star_Cr conj(a) ^ b) equals ours", okE)
check("Conventions, Exercise 43.4.3: Re int star_Cr conj(a) ^ b = Re <<a, b>>", okI)
xs, ys = sp.symbols("x y", real=True)
u = sp.cos(sp.pi * xs)                                       # zero normal derivative on the unit square
lhs = sp.integrate(sp.integrate(sp.diff(u, xs) ** 2 + sp.diff(u, ys) ** 2, (xs, 0, 1)), (ys, 0, 1))
lap = sp.diff(u, xs, 2) + sp.diff(u, ys, 2)
rhs_minus = sp.integrate(sp.integrate(-lap * u, (xs, 0, 1)), (ys, 0, 1))
check("Conventions: <<du, du>> = <<-Delta u, u>> with Neumann data (so Crane's Delta in Exercise 7.6 is our -Delta)",
      sp.simplify(lhs - rhs_minus) == 0 and sp.simplify(lhs - sp.pi ** 2 / 2) == 0)
# the literal formula -star_Cr d star_Cr d with Crane's 1-form star gives +Delta
uu = sp.Function("u")(xs, ys)
du = [sp.diff(uu, xs), sp.diff(uu, ys)]
scr = [du[1], -du[0]]                                        # star_Cr du = du o J: coefficients of dx, dy
d_scr = sp.diff(scr[1], xs) - sp.diff(scr[0], ys)           # d(a dx + b dy) = (b_x - a_y) dx ^ dy
check("Conventions: -star_Cr d star_Cr d u = +Delta u (Crane's formula read with his 1-form star)",
      sp.simplify(-d_scr - (sp.diff(uu, xs, 2) + sp.diff(uu, ys, 2))) == 0)
ux, uy, vx, vy = sp.symbols("u_x u_y v_x v_y", real=True)
dzs = sp.Matrix([ux + sp.I * vx, uy + sp.I * vy])
dzb = sp.Matrix([ux - sp.I * vx, uy - sp.I * vy])
w_ = lambda a, b: a[0] * b[1] - a[1] * b[0]  # noqa: E731
check("(43.4.5), Conventions: (i/2) dz ^ dzbar = -(i/2) dzbar ^ dz = du ^ dv = det",
      sp.simplify(sp.I / 2 * w_(dzs, dzb) - (ux * vy - uy * vx)) == 0
      and sp.simplify(-sp.I / 2 * w_(dzb, dzs) - (ux * vy - uy * vx)) == 0)

# ---------------------------------------------------------------- Proposition 43.4.2
okH = okP = okS = True
for _ in range(20):
    a, b, lam = crand(), crand(), complex(*rng.standard_normal(2))
    okH &= abs(ip(b, a) - np.conj(ip(a, b))) < 1e-12 and abs(ip(lam * a, b) - lam * ip(a, b)) < 1e-12
    okP &= abs(ip(a, a) - (np.sum(a.real ** 2) + np.sum(a.imag ** 2))) < 1e-12
    okS &= abs(ip(star(a), star(a)) - ip(a, a)) < 1e-12
check("Proposition 43.4.2(a): Hermitian and linear in the first slot", okH)
check("Proposition 43.4.2(b): |a + ib|^2 = |a|^2 + |b|^2 (real and positive)", okP)
check("Proposition 43.4.2(c): |star alpha| = |alpha|", okS)

# ---------------------------------------------------------------- Proposition 43.4.3, (43.4.3)
for f, name in ((lambda w: w ** 3, "z^3"), (np.exp, "exp")):
    w0, h = 0.3 + 0.2j, 1e-6
    d = np.array([(f(w0 + h) - f(w0 - h)) / (2 * h), (f(w0 + 1j * h) - f(w0 - 1j * h)) / (2 * h)])
    check(f"Proposition 43.4.3: star dz = -i dz for {name}; (43.4.3) star du = dv",
          np.allclose(star(d), -1j * d, atol=1e-6) and np.allclose(star(d.real), d.imag, atol=1e-6))
d = np.array([1, -1j])                                       # conj(w)
check("Proposition 43.4.3: conj(w) satisfies star dz = i dz", np.allclose(star(d), 1j * d))

# ---------------------------------------------------------------- Theorem 43.4.6 and Corollary 43.4.8 (pointwise)
st_s = sp.Matrix([-dzs[1], dzs[0]])
v_ = st_s + sp.I * dzs
ec = sp.Rational(1, 4) * sum(sp.expand(v_[k] * sp.conjugate(v_[k])) for k in range(2))
check("Theorem 43.4.6 (density): (1/4)|star dz + i dz|^2 = (1/2)|dz|^2 - det = ((u_x - v_y)^2 + (u_y + v_x)^2)/2",
      sp.simplify(ec - (sp.Rational(1, 2) * (ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2) - (ux * vy - uy * vx))) == 0
      and sp.simplify(ec - ((ux - vy) ** 2 + (uy + vx) ** 2) / 2) == 0)
okC = True
for _ in range(40):
    D = rng.standard_normal((2, 2))
    s1, s2 = np.linalg.svd(D, compute_uv=False)
    dens = 0.5 * ((D[0, 0] - D[1, 1]) ** 2 + (D[0, 1] + D[1, 0]) ** 2)
    okC &= abs(dens - (0.5 * (s1 - s2) ** 2 if np.linalg.det(D) >= 0 else 0.5 * (s1 + s2) ** 2)) < 1e-12
check("Corollary 43.4.8: e_C = (s1 - s2)^2/2 (det >= 0) and (s1 + s2)^2/2 (det < 0)", okC)
# a nonlinear map on the unit disk: E_C = E_D - A and A = boundary integral, by quadrature
f = lambda w: w + 0.3 * np.conj(w) ** 2 + 0.2 * w ** 2  # noqa: E731
nr, nt = 300, 600
rr = (np.arange(nr) + 0.5) / nr
tt = 2 * np.pi * (np.arange(nt) + 0.5) / nt
R_, T_ = np.meshgrid(rr, tt, indexing="ij")
W = R_ * np.exp(1j * T_)
dA = (1 / nr) * (2 * np.pi / nt) * R_
h = 1e-6
fx, fy = (f(W + h) - f(W - h)) / (2 * h), (f(W + 1j * h) - f(W - 1j * h)) / (2 * h)
ED = 0.5 * np.sum((np.abs(fx) ** 2 + np.abs(fy) ** 2) * dA)
AA = np.sum((fx.real * fy.imag - fy.real * fx.imag) * dA)
EC = np.sum(0.25 * (np.abs(-fy + 1j * fx) ** 2 + np.abs(fx + 1j * fy) ** 2) * dA)
tb = 2 * np.pi * np.arange(20000) / 20000
zb = f(np.exp(1j * tb))
dzb_ = (f(np.exp(1j * (tb + h))) - f(np.exp(1j * (tb - h)))) / (2 * h)
Ab = 0.5 * np.sum((np.conj(zb) * dzb_).imag) * (2 * np.pi / 20000)
close("Theorem 43.4.6: E_C = E_D - A for w + 0.3 conj(w)^2 + 0.2 w^2 on the unit disk", EC, ED - AA, 1e-8)
close("Proposition 43.4.7(b): A = (1/2) Im oint conj(z) dz (same map)", AA, Ab, 2e-5)

# ---------------------------------------------------------------- Example 43.4.5, Figure 43.4.2
wr, wa = sp.symbols("rho alpha", positive=True)
eps, a = sp.symbols("epsilon a", positive=True)
for zexpr, EDv, Av, ECv, dom in ((lambda X, Y: (X + sp.I * Y) + eps * (X - sp.I * Y), sp.pi * (1 + eps ** 2), sp.pi * (1 - eps ** 2), 2 * sp.pi * eps ** 2, "disk"),
                                 (lambda X, Y: a * X + sp.I * Y, (a ** 2 + 1) / 2, a, (a - 1) ** 2 / 2, "square")):
    Z = zexpr(xs, ys)
    U, V = sp.re(Z), sp.im(Z)
    dens_D = sp.Rational(1, 2) * (sp.diff(U, xs) ** 2 + sp.diff(U, ys) ** 2 + sp.diff(V, xs) ** 2 + sp.diff(V, ys) ** 2)
    dens_A = sp.diff(U, xs) * sp.diff(V, ys) - sp.diff(U, ys) * sp.diff(V, xs)
    area = sp.pi if dom == "disk" else 1
    check(f"Example 43.4.5 ({dom}): E_D, A, E_C = {EDv}, {Av}, {ECv}",
          sp.simplify(dens_D * area - EDv) == 0 and sp.simplify(dens_A * area - Av) == 0 and sp.simplify(EDv - Av - ECv) == 0)
close("Figure 43.4.2: eps = 0.5 gives E_D = 3.927, A = 2.356, E_C = 1.571", (np.pi * 1.25, np.pi * 0.75, 2 * np.pi * 0.25), (3.927, 2.356, 1.571), 6e-4)
close("Example 43.4.5(c): a = 2 and a = 1/2 give E_C = 1/2 and 1/8", ((2 - 1) ** 2 / 2, (0.5 - 1) ** 2 / 2), (0.5, 0.125), 1e-15)

# ---------------------------------------------------------------- Figure 43.4.1
al, be = 1 + 0.2j, 0.35
dzf = lambda X: al * X + be * np.conj(X)  # noqa: E731
defs = [abs(dzf(1j * X) - 1j * dzf(X)) for X in np.exp(1j * np.linspace(0, 6, 9))]
s1, s2 = abs(al) + abs(be), abs(al) - abs(be)
close("Figure 43.4.1: |defect| = 0.70 for all X, sigma1 = 1.370, sigma2 = 0.670, density 0.245",
      (max(defs), min(defs), s1, s2, 0.5 * (s1 - s2) ** 2), (0.7, 0.7, 1.370, 0.670, 0.245), 6e-4)

# ---------------------------------------------------------------- Proposition 43.4.9, Theorem 43.4.10, Example 43.4.11
Z3 = sp.expand((xs + sp.I * ys) ** 3)
check("Proposition 43.4.9: Re and Im of z^3 are harmonic",
      all(sp.simplify(sp.diff(q, xs, 2) + sp.diff(q, ys, 2)) == 0 for q in (sp.re(Z3), sp.im(Z3))))
EC11 = sp.Rational(1, 2) * ((2 - 1) ** 2 + 0) * sp.pi
check("Example 43.4.11: z = 2x + iy has E_C = pi/2, E_D = 5 pi/2, A = 2 pi on the unit disk",
      EC11 == sp.pi / 2 and sp.Rational(1, 2) * (4 + 1) * sp.pi - 2 * sp.pi == EC11)
# discrete check: the Dirichlet minimizer of E_C with boundary values 2x + iy on a planar disk mesh is 2x + iy
Vh, Fh = ch43.hemisphere(2)
Vp = np.c_[Vh[:, 0], Vh[:, 1], np.zeros(len(Vh))]
loop = dm.boundary_loop(Fh)
EC_m = ch43.conformal_energy_matrix(Vp, Fh)
inner = np.setdiff1d(np.arange(len(Vp)), loop)
zt = 2 * Vp[:, 0] + 1j * Vp[:, 1]
zsol = zt.copy()
zsol[inner] = np.linalg.solve(EC_m[np.ix_(inner, inner)], -EC_m[np.ix_(inner, loop)] @ zt[loop])
check("Theorem 43.4.10 (discrete): fixing the boundary, the E_C minimizer on a planar disk mesh is the harmonic 2x + iy",
      np.abs(zsol - zt).max() < 1e-10 and (np.conj(zsol) @ EC_m @ zsol).real > 0.1)

# ---------------------------------------------------------------- Remark 43.4.12, Exercise 43.4.5
T = 0.9
tq, pq = sp.symbols("t phi", real=True)
EDm = sp.Rational(1, 2) * sp.integrate(sp.integrate(2 * sp.cosh(tq) ** 2 * sp.sech(tq) ** 2, (pq, 0, 2 * sp.pi)), (tq, -T, T))
check("Exercise 43.4.5: E_D = A = 4 pi T for the Mercator band", abs(float(EDm) - 4 * np.pi * T) < 1e-10)
close("Exercise 43.4.5: area of the band = 4 pi tanh T", float(sp.integrate(2 * sp.pi * sp.sech(tq) ** 2, (tq, -T, T))), 4 * np.pi * np.tanh(T), 1e-10)

# ---------------------------------------------------------------- Exercise 43.4.1
check("Exercise 43.4.1: boundary integral of (u dv - v du)/2 for z = a x + i y on the unit square is a",
      sp.simplify(sp.Rational(1, 2) * (sp.integrate(a, (ys, 0, 1)) + sp.integrate(-a, (xs, 1, 0)))) == a)

# ---------------------------------------------------------------- Exercise 43.4.4
okP4 = True
for _ in range(20):
    d = crand()
    Pp, Pm = 0.5 * (d + 1j * star(d)), 0.5 * (d - 1j * star(d))
    nD = 0.5 * ip(d, d).real
    nC = 0.25 * ip(star(d) + 1j * d, star(d) + 1j * d).real
    nA = (0.5j * wedge(d, np.conj(d))).real
    okP4 &= np.allclose(star(Pp), -1j * Pp) and np.allclose(star(Pm), 1j * Pm) and abs(ip(Pp, Pm)) < 1e-12
    okP4 &= abs(nC - ip(Pm, Pm).real) < 1e-12 and abs(nD - 0.5 * (ip(Pp, Pp) + ip(Pm, Pm)).real) < 1e-12
    okP4 &= abs(nA - 0.5 * (ip(Pp, Pp) - ip(Pm, Pm)).real) < 1e-12
    okP4 &= abs(0.25 * ip(star(d) - 1j * d, star(d) - 1j * d).real - (nD + nA)) < 1e-12
check("Exercise 43.4.4: P+ / P- decomposition, E_C = |P- dz|^2, E_D, A, and the antiholomorphic energy E_D + A", okP4)

# ---------------------------------------------------------------- Exercise 43.4.6
ns, same, printed = ch43.run_printed(PAGE, "code-ex-43-4-6", "out-ex-43-4-6")
check("Exercise 43.4.6: code-ex-43-4-6 prints exactly out-ex-43-4-6", same, printed)
Tq = np.log(1 / np.tan(np.pi / 8))
close("Exercise 43.4.6: quadrature in (theta, phi) matches the Mercator hand values",
      (ns["EC"], ns["ED"], ns["A"]), (2 * np.pi * Tq * 0.25, 2 * np.pi * Tq * 3.25, 4 * np.pi * 1.5 * Tq), 1e-4)
close("Exercise 43.4.6: T = log cot(pi/8) = 0.8814", Tq, 0.8814, 5e-5)

summary()
