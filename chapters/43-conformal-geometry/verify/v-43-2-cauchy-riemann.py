"""Verification for Section 43.2 (Complex Numbers and the Cauchy-Riemann Equations).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/verify/v-43-2-cauchy-riemann.py``
"""

import pathlib
import sys

import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch43  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "43-2-cauchy-riemann.html"
x, y = sp.symbols("x y", real=True)
z = x + sp.I * y


def uv(expr):
    e = sp.expand(sp.expand_complex(expr))
    return sp.simplify(sp.re(e)), sp.simplify(sp.im(e))


def cr(expr):
    u, v = uv(expr)
    return sp.simplify(sp.diff(u, x) - sp.diff(v, y)), sp.simplify(sp.diff(u, y) + sp.diff(v, x))


# ---------------------------------------------------------------- Lemma 43.2.1
rng = np.random.default_rng(0)
ok_a = ok_c = True
for _ in range(20):
    T = rng.standard_normal((2, 2))
    Tc = lambda w: complex(*(T @ [w.real, w.imag]))  # noqa: E731
    al = 0.5 * (Tc(1) - 1j * Tc(1j))
    be = 0.5 * (Tc(1) + 1j * Tc(1j))
    for w in rng.standard_normal(5) + 1j * rng.standard_normal(5):
        ok_c &= abs(Tc(w) - (al * w + be * np.conj(w))) < 1e-12
    s = np.linalg.svd(T, compute_uv=False)
    ok_c &= abs(np.linalg.det(T) - (abs(al) ** 2 - abs(be) ** 2)) < 1e-12
    ok_c &= np.allclose(s, [abs(al) + abs(be), abs(abs(al) - abs(be))])
    a, b = rng.standard_normal(2)
    M = np.array([[a, -b], [b, a]])
    ok_a &= np.allclose(M @ [0, 1], [(1j * (a + 1j * b)).real, (1j * (a + 1j * b)).imag])
check("Lemma 43.2.1(a): [[a,-b],[b,a]] commutes with i (= multiplication by a + ib)", ok_a)
check("Lemma 43.2.1(c): T = alpha w + beta conj(w), det = |alpha|^2 - |beta|^2, singular values |alpha| +- |beta|", ok_c)
a, b = sp.symbols("a b", real=True)
check("Lemma 43.2.1(b): w -> c conj(w) has matrix [[a, b], [b, -a]]",
      uv((a + sp.I * b) * sp.conjugate(1 + 0 * z).subs(x, 1)) == (a, b)
      and uv((a + sp.I * b) * sp.conjugate(sp.I)) == (b, -a))

# ---------------------------------------------------------------- Proposition 43.2.4, Example 43.2.5
for name, f in (("z^2", z ** 2), ("z^3", z ** 3), ("exp(z)", sp.exp(x) * (sp.cos(y) + sp.I * sp.sin(y))),
                ("1/z", 1 / z)):
    r1, r2 = cr(f)
    u, v = uv(f)
    fprime = sp.simplify(sp.diff(u, x) + sp.I * sp.diff(v, x))
    J = sp.Matrix([[sp.diff(u, x), sp.diff(u, y)], [sp.diff(v, x), sp.diff(v, y)]])
    check(f"Proposition 43.2.4 / Example 43.2.5: {name} satisfies Cauchy-Riemann and det Df = |f'|^2",
          r1 == 0 and r2 == 0 and sp.simplify(J.det() - (sp.re(fprime) ** 2 + sp.im(fprime) ** 2)) == 0)
check("Example 43.2.5: (z^2)' = 2z, (e^z)' = e^z, (1/z)' = -1/z^2",
      sp.simplify(sp.diff(uv(z ** 2)[0], x) + sp.I * sp.diff(uv(z ** 2)[1], x) - 2 * z) == 0
      and sp.simplify(sp.diff(sp.exp(x) * sp.cos(y), x) + sp.I * sp.diff(sp.exp(x) * sp.sin(y), x)
                      - sp.exp(x) * (sp.cos(y) + sp.I * sp.sin(y))) == 0
      and sp.simplify(sp.diff(uv(1 / z)[0], x) + sp.I * sp.diff(uv(1 / z)[1], x) + 1 / z ** 2) == 0)
# directional difference quotients of conj(z)
h = 1e-7
check("Non-example 43.2.6(a): the difference quotient of conj(z) is 1 along R and -1 along iR",
      abs(np.conj(h) / h - 1) < 1e-12 and abs(np.conj(1j * h) / (1j * h) + 1) < 1e-12)
eps = sp.Rational(3, 10)
fz = z + eps * sp.conjugate(z)
u, v = uv(fz)
fx, fy = sp.diff(u + sp.I * v, x), sp.diff(u + sp.I * v, y)
check("Non-example 43.2.6(b): z + eps conj(z) has f_y - i f_x = -2 i eps", sp.simplify(fy - sp.I * fx + 2 * sp.I * eps) == 0)
close("Non-example 43.2.6(b): sigma1/sigma2 = (1 + 0.3)/(1 - 0.3) = 1.857", 1.3 / 0.7, 1.857, 5e-4)
r1, r2 = cr(x ** 2 + y ** 2)
check("Non-example 43.2.6(c): |z|^2 satisfies Cauchy-Riemann only at 0", sp.solve([r1, r2], [x, y], dict=True) == [{x: 0, y: 0}])

# ---------------------------------------------------------------- Proposition 43.2.7 (e^z, z^3 + 2z conformal)
for f in (sp.exp(x) * (sp.cos(y) + sp.I * sp.sin(y)), z ** 3 + 2 * z):
    u, v = uv(f)
    J = sp.Matrix([[sp.diff(u, x), sp.diff(u, y)], [sp.diff(v, x), sp.diff(v, y)]])
    G = sp.simplify(J.T * J)
    check("Proposition 43.2.7: Df^T Df = |f'|^2 I for a holomorphic f", sp.simplify(G[0, 1]) == 0 and sp.simplify(G[0, 0] - G[1, 1]) == 0
          and sp.simplify(G[0, 0] - J.det()) == 0)
u, v = uv(sp.conjugate(z) ** 2)
J = sp.Matrix([[sp.diff(u, x), sp.diff(u, y)], [sp.diff(v, x), sp.diff(v, y)]])
check("Proposition 43.2.7(b): conj(z)^2 is orientation-reversing conformal away from 0",
      sp.simplify((J.T * J)[0, 1]) == 0 and sp.simplify((J.T * J)[0, 0] - (J.T * J)[1, 1]) == 0
      and sp.simplify(J.det() + 4 * (x ** 2 + y ** 2)) == 0)

# ---------------------------------------------------------------- Non-example 43.2.8, Figure 43.2.2
for t1, t2 in ((0.0, np.pi / 2), (0.3, 1.1)):
    w1, w2 = np.exp(1j * t1) ** 2, np.exp(1j * t2) ** 2
    close("Non-example 43.2.8: z^2 doubles the angle between rays at 0", abs(np.angle(w2 / w1)), 2 * (t2 - t1), 1e-12)
p = 0.8 + 0.6j
close("Figure 43.2.2: f(p) = 0.28 + 0.96 i, |f'(p)| = 2, angle of f_x, f_y = 90 deg",
      (p ** 2, abs(2 * p), np.degrees(np.angle((2j * p) / (2 * p)))), (0.28 + 0.96j, 2.0, 90.0), 1e-12)

# ---------------------------------------------------------------- Figure 43.2.1
c = 1.25 * np.exp(1j * np.radians(40))
close("Figure 43.2.1: matrix of c = [[0.958, -0.804], [0.804, 0.958]]", (c.real, c.imag), (0.958, 0.804), 6e-4)
M2 = np.array([[c.real, c.imag], [c.imag, -c.real]])
close("Figure 43.2.1: det of w -> c conj(w) is -|c|^2 = -1.5625", np.linalg.det(M2), -1.5625, 1e-12)
v1, v2 = np.array([1.2, 0]), np.array([0.7, 0.9])
close("Figure 43.2.1: the real-linear image has angle 52.1 deg", np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2))), 52.1, 0.05)

# ---------------------------------------------------------------- Remark 43.2.9(b)
F = z ** 2 + 3 * sp.conjugate(z) + z * sp.conjugate(z)
u, v = uv(F)
fx, fy = sp.diff(u + sp.I * v, x), sp.diff(u + sp.I * v, y)
dzf, dzbf = (fx - sp.I * fy) / 2, (fx + sp.I * fy) / 2
check("Remark 43.2.9(b): for z^2 + 3 conj(z) + |z|^2, d_z f = 2z + conj(z) and d_zbar f = 3 + z",
      sp.simplify(dzf - (2 * z + sp.conjugate(z))) == 0 and sp.simplify(dzbf - (3 + z)) == 0)
check("Remark 43.2.9(b): df = d_z f dz + d_zbar f dzbar (dx and dy coefficients) and f_y - i f_x = -2i d_zbar f",
      sp.simplify(dzf + dzbf - fx) == 0 and sp.simplify(sp.I * dzf - sp.I * dzbf - fy) == 0
      and sp.simplify(fy - sp.I * fx + 2 * sp.I * dzbf) == 0)

# ---------------------------------------------------------------- Implementation 43.2
ns, same, printed = ch43.run_printed(PAGE, "code-43-2-1", "out-43-2-1")
check("Implementation 43.2: code-43-2-1 prints exactly out-43-2-1", same, printed)
import dgcode  # noqa: E402
check("Implementation 43.2: at most 30 lines", dgcode.code_lines(PAGE, "code-43-2-1") <= 30)

# ---------------------------------------------------------------- Proposition 43.2.11 and Example 43.2.12
A, B, C, D = sp.symbols("A B C D")
w = sp.symbols("w")
mob = (A * w + B) / (C * w + D)
check("Proposition 43.2.11(a): f' = (ad - bc)/(cw + d)^2", sp.simplify(sp.diff(mob, w) - (A * D - B * C) / (C * w + D) ** 2) == 0)
check("Proposition 43.2.11(b): (aw + b)/(cw + d) = a/c - (ad - bc)/c * 1/(cw + d)",
      sp.simplify(mob - (A / C - (A * D - B * C) / C / (C * w + D))) == 0)
ok = True
for _ in range(30):
    al, ga = rng.standard_normal(2)
    be = complex(*rng.standard_normal(2)) * 2
    if abs(be) ** 2 <= al * ga:
        continue
    # sample points of the generalized circle and their images under 1/z
    if abs(al) > 1e-6:
        cen, rad = -be / al, np.sqrt(abs(be) ** 2 - al * ga) / abs(al)
        pts = cen + rad * np.exp(1j * np.linspace(0, 2 * np.pi, 50, endpoint=False))
    else:
        continue
    pts = pts[np.abs(pts) > 1e-3]
    q = 1 / pts
    resid = ga * np.abs(q) ** 2 + 2 * (np.conj(np.conj(be)) * q).real + al
    ok &= np.abs(resid).max() < 1e-8 * max(1, np.abs(q).max() ** 2)
check("Proposition 43.2.11(c): 1/z maps alpha|z|^2 + 2Re(conj(beta) z) + gamma = 0 to (gamma, conj(beta), alpha)", ok)
a0 = 0.5
phi = lambda zz: (zz - a0) / (1 - np.conj(a0) * zz)  # noqa: E731
tt = np.linspace(0, 2 * np.pi, 200)
check("Example 43.2.12: |phi_a| = 1 on the unit circle, phi_a(a) = 0, phi_{-a} o phi_a = id",
      np.allclose(np.abs(phi(np.exp(1j * tt))), 1) and phi(a0) == 0
      and np.allclose(((phi(0.3 + 0.2j)) + a0) / (1 + np.conj(a0) * phi(0.3 + 0.2j)), 0.3 + 0.2j))
for r in (0.2, 0.4, 0.6, 0.8):
    img = phi(r * np.exp(1j * tt))
    cen = -a0 * (1 - r * r) / (1 - a0 * a0 * r * r)
    rad = r * (1 - a0 * a0) / (1 - a0 * a0 * r * r)
    check(f"Example 43.2.12: image of |z| = {r} is the circle with centre {cen:.4f}, radius {rad:.4f}",
          np.allclose(np.abs(img - cen), rad))
close("Example 43.2.12: r = 0.4 gives centre -0.4375 and radius 0.3125",
      (-a0 * (1 - 0.16) / (1 - 0.04), 0.4 * 0.75 / 0.96), (-0.4375, 0.3125), 1e-12)

# ---------------------------------------------------------------- Proposition 43.2.13 (inversion)
X = sp.Matrix(sp.symbols("x1 x2 x3", real=True))
r2 = (X.T * X)[0]
Ix = X / r2
DI = Ix.jacobian(X)
check("(43.2.5): DI = (I - 2 x x^T/|x|^2)/|x|^2", sp.simplify(DI - (sp.eye(3) - 2 * X * X.T / r2) / r2) == sp.zeros(3, 3))
check("Proposition 43.2.13: DI^T DI = |x|^-4 I and det DI = -|x|^-6",
      sp.simplify(DI.T * DI - sp.eye(3) / r2 ** 2) == sp.zeros(3, 3) and sp.simplify(DI.det() + 1 / r2 ** 3) == 0)
check("Proposition 43.2.13: I o I = id", sp.simplify(Ix.subs({X[i]: Ix[i] for i in range(3)}, simultaneous=True) - X) == sp.zeros(3, 1))
check("Proposition 43.2.13: in the plane z/|z|^2 = 1/conj(z)", sp.simplify(z / (x ** 2 + y ** 2) - 1 / sp.conjugate(z)) == 0)

# ---------------------------------------------------------------- Exercise 43.2.1
check("Exercise 43.2.1(a,b): 1/z and z^3 satisfy Cauchy-Riemann", cr(1 / z) == (0, 0) and cr(z ** 3) == (0, 0))
r1, r2 = cr(x + 0 * y)
check("Exercise 43.2.1(c): Re z never satisfies Cauchy-Riemann (u_x - v_y = 1)", r1 == 1)
r1, r2 = cr(sp.exp(x) * (sp.cos(y) - sp.I * sp.sin(y)))
check("Exercise 43.2.1(d): e^{conj z}: u_x - v_y = 2 e^x cos y and u_y + v_x = -2 e^x sin y (never both 0)",
      sp.simplify(r1 - 2 * sp.exp(x) * sp.cos(y)) == 0 and sp.simplify(r2 + 2 * sp.exp(x) * sp.sin(y)) == 0)

# ---------------------------------------------------------------- Exercise 43.2.4 (log |f'| harmonic)
for f in (z ** 3 + 2 * z + 1, sp.exp(x) * (sp.cos(y) + sp.I * sp.sin(y)) * (z + 2)):
    u, v = uv(f)
    p_, q_ = sp.diff(u, x), sp.diff(v, x)
    mu = sp.log(p_ ** 2 + q_ ** 2) / 2
    lap = sp.diff(mu, x, 2) + sp.diff(mu, y, 2)
    vals = [float(lap.subs({x: xv, y: yv})) for xv, yv in ((0.3, 0.7), (-0.4, 0.2), (1.1, -0.5))]
    check("Exercise 43.2.4: log|f'| is harmonic (numerical values at 3 points)", max(abs(t) for t in vals) < 1e-10)
P, Q = sp.symbols("p q", real=True)
check("Exercise 43.2.4: (ap + bq)^2 + (aq - bp)^2 = (a^2 + b^2)(p^2 + q^2)",
      sp.expand((a * P + b * Q) ** 2 + (a * Q - b * P) ** 2 - (a ** 2 + b ** 2) * (P ** 2 + Q ** 2)) == 0)

# ---------------------------------------------------------------- Exercise 43.2.5 (1/z is a rotation of the sphere)
th, ph = sp.symbols("theta phi", positive=True)
S = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
sig = lambda P_: (P_[0] + sp.I * P_[1]) / (1 - P_[2])  # noqa: E731
RS = sp.Matrix([S[0], -S[1], -S[2]])
vals = [complex(sp.N((sig(RS) - 1 / sig(S)).subs({th: tv, ph: pv}))) for tv, pv in ((0.4, 1.0), (1.3, 2.5), (2.6, 4.0))]
check("Exercise 43.2.5: sigma(R x) = 1/sigma(x) on the sphere", max(abs(t) for t in vals) < 1e-12)
sigt = (S[0] + sp.I * S[1]) / (1 + S[2])
vals = [complex(sp.N((sigt - sp.conjugate(sig(RS))).subs({th: tv, ph: pv}))) for tv, pv in ((0.4, 1.0), (1.3, 2.5))]
check("Exercise 43.2.5: tilde sigma(x) = conj(sigma(R x))", max(abs(t) for t in vals) < 1e-12)

# ---------------------------------------------------------------- Exercise 43.2.6
ns6, same6, printed6 = ch43.run_printed(PAGE, "code-ex-43-2-6", "out-ex-43-2-6")
check("Exercise 43.2.6: code-ex-43-2-6 prints exactly out-ex-43-2-6", same6, printed6)
F6 = z + sp.Rational(1, 4) * sp.conjugate(z) ** 2
u, v = uv(F6)
fx, fy = sp.diff(u + sp.I * v, x), sp.diff(u + sp.I * v, y)
check("Exercise 43.2.6: d_z f = 1 and d_zbar f = conj(z)/2",
      sp.simplify((fx - sp.I * fy) / 2 - 1) == 0 and sp.simplify((fx + sp.I * fy) / 2 - sp.conjugate(z) / 2) == 0)

summary()
