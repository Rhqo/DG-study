"""Verification for Section 43.3 (Conformal Maps of Surfaces and Uniformization).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/verify/v-43-3-conformal-surfaces.py``
"""

import pathlib
import sys

import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch43  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "43-3-conformal-surfaces.html"
rng = np.random.default_rng(1)


def num_jac(x, q, h=1e-6):
    q = np.asarray(q, float)
    return np.stack([(x(q + h * e) - x(q - h * e)) / (2 * h) for e in np.eye(2)], 1)


def J_param(x, q):
    """J_q on the parameter plane: d x(J X) = N x d x(X), N = x_u x x_v / |.|"""
    D = num_jac(x, q)
    N = np.cross(D[:, 0], D[:, 1])
    N /= np.linalg.norm(N)
    rot = np.stack([np.cross(N, D[:, 0]), np.cross(N, D[:, 1])], 1)
    return np.linalg.lstsq(D, rot, rcond=None)[0], D, N


sphere = lambda q: np.array([np.sin(q[0]) * np.cos(q[1]), np.sin(q[0]) * np.sin(q[1]), np.cos(q[0])])  # noqa: E731
saddle = lambda q: np.array([q[0], q[1], q[0] ** 2 - q[1] ** 2])  # noqa: E731
torus = lambda q: np.array([(2 + 0.8 * np.cos(q[0])) * np.cos(q[1]), (2 + 0.8 * np.cos(q[0])) * np.sin(q[1]), 0.8 * np.sin(q[0])])  # noqa: E731

# ---------------------------------------------------------------- Proposition 43.3.2
ok_a = ok_b = ok_c = True
for x, qs in ((sphere, [(1.0, 0.6), (0.4, 2.0)]), (saddle, [(0.5, -0.4), (-0.3, 0.7)]), (torus, [(0.3, 1.0), (2.5, 4.0)])):
    for q in qs:
        Jq, D, N = J_param(x, q)
        g = D.T @ D
        ok_a &= np.allclose(Jq @ Jq, -np.eye(2), atol=1e-7)
        for _ in range(5):
            X = rng.standard_normal(2)
            w, Jw = D @ X, D @ (Jq @ X)
            ok_b &= abs(np.linalg.norm(Jw) - np.linalg.norm(w)) < 1e-7 and abs(w @ Jw) < 1e-7 and np.linalg.det(np.c_[w, Jw, N]) > 0
            # (c): J X is the g-rotation: g(X, JX) = 0, g(JX, JX) = g(X, X), det[X, JX] > 0 in the parameter plane
            ok_c &= abs(X @ g @ (Jq @ X)) < 1e-7 and abs((Jq @ X) @ g @ (Jq @ X) - X @ g @ X) < 1e-7 and np.linalg.det(np.c_[X, Jq @ X]) > 0
check("Proposition 43.3.2(a): J^2 = -id (sphere, saddle, torus)", ok_a)
check("Proposition 43.3.2(b): |Jw| = |w|, <w, Jw> = 0, det[w, Jw, N] > 0", ok_b)
check("Proposition 43.3.2(c): in the parameter plane J_q is the g_q-rotation by +90 deg", ok_c)
# (d): scaling g by c^2 keeps the g-rotation; in conformal coordinates J is the standard rotation
gq = np.array([[2.0, 0.8], [0.8, 1.64]])
Lc = np.linalg.cholesky(gq)                                  # g = Lc Lc^T; rotation = Lc^{-T} R Lc^T
Rg = np.linalg.inv(Lc.T) @ np.array([[0, -1], [1, 0]]) @ Lc.T
Rg2 = np.linalg.inv((np.sqrt(3.7) * Lc).T) @ np.array([[0, -1], [1, 0]]) @ (np.sqrt(3.7) * Lc).T
check("Proposition 43.3.2(d): the g-rotation is unchanged by g -> c^2 g", np.allclose(Rg, Rg2))
merc = lambda q: np.array([np.cos(q[0]) / np.cosh(q[1]), np.sin(q[0]) / np.cosh(q[1]), np.tanh(q[1])])  # noqa: E731
Jm, _, _ = J_param(merc, (0.6, 0.604))
check("Proposition 43.3.2(d), Example 43.3.3: in Mercator coordinates J is the standard rotation", np.allclose(Jm, [[0, -1], [1, 0]], atol=1e-7))

# ---------------------------------------------------------------- Example 43.3.3 (sphere)
th0, ph0 = 1.0, 0.6
Js, D, N = J_param(sphere, (th0, ph0))
check("Example 43.3.3: J d_phi = -sin(theta) d_theta, J d_theta = d_phi / sin(theta)",
      np.allclose(Js @ [0, 1], [-np.sin(th0), 0], atol=1e-7) and np.allclose(Js @ [1, 0], [0, 1 / np.sin(th0)], atol=1e-7))
eE = np.array([-np.sin(ph0), np.cos(ph0), 0])
eN = np.array([-np.cos(th0) * np.cos(ph0), -np.cos(th0) * np.sin(ph0), np.sin(th0)])
check("Example 43.3.3: N x east = north, N x north = -east", np.allclose(np.cross(N, eE), eN) and np.allclose(np.cross(N, eN), -eE))
close("Figure 43.3.1: JX = north = (-0.446, -0.305, 0.841)", np.cross(N, eE), (-0.446, -0.305, 0.841), 6e-4)

# ---------------------------------------------------------------- Definition 43.3.4, Proposition 43.3.5, Example 43.3.6


def holo_residual(x, z, q, h=1e-6):
    """max over the basis of |dz(J X) - i dz(X)| (z: parameter -> C)"""
    Jq, _, _ = J_param(x, q)
    q = np.asarray(q, float)
    dz = np.array([(z(q + h * e) - z(q - h * e)) / (2 * h) for e in np.eye(2)])
    return max(abs(dz @ (Jq @ e) - 1j * (dz @ e)) for e in np.eye(2)), dz


zm = lambda q: q[0] + 1j * q[1]  # noqa: E731
res = max(holo_residual(merc, zm, q)[0] for q in ((0.3, 0.2), (2.0, -1.1), (4.0, 0.7)))
check("Example 43.3.6(a): Mercator z = phi + i t is holomorphic (outward N)", res < 1e-6)
_, dz = holo_residual(sphere, lambda q: q[1] + 1j * np.log(1 / np.tan(q[0] / 2)), (th0, ph0))
close("Figure 43.3.1: dz(X) = 1/sin(1) = 1.188 for the unit east vector X", dz @ np.array([0, 1 / np.sin(th0)]), 1 / np.sin(th0), 1e-6)
close("Figure 43.3.1: 1/sin(1.0) = 1.188, t(p) = log cot(0.5) = 0.604", (1 / np.sin(1.0), np.log(1 / np.tan(0.5))), (1.188, 0.604), 6e-4)
# Proposition 43.3.5: Mercator dz preserves angles (dz^T dz proportional to the metric)
Jm, Dm, _ = J_param(merc, (1.3, 0.4))
check("Proposition 43.3.5: for Mercator the metric is proportional to |dz|^2 (conformal)", np.allclose((Dm.T @ Dm) / (Dm.T @ Dm)[0, 0], np.eye(2), atol=1e-7))
# (b) stereographic: sigma antiholomorphic for outward N, conj(sigma) holomorphic
st = lambda q: 1 / (1 + q[0] ** 2 + q[1] ** 2) * np.array([2 * q[0], 2 * q[1], q[0] ** 2 + q[1] ** 2 - 1])  # noqa: E731
Jst, Dst, Nst = J_param(st, (0.3, -0.5))
P = st((0.3, -0.5))
check("Example 43.3.6(b): stereographic coordinates are negatively oriented for the outward normal", Nst @ P < 0)
# use the sphere parametrization (outward) and z = sigma(x(theta, phi))
sig = lambda q: (sphere(q)[0] + 1j * sphere(q)[1]) / (1 - sphere(q)[2])  # noqa: E731
Jsph, _, _ = J_param(sphere, (2.0, 1.0))
hres_sig, dz_sig = holo_residual(sphere, sig, (2.0, 1.0))
hres_bar, _ = holo_residual(sphere, lambda q: np.conj(sig(q)), (2.0, 1.0))
anti = max(abs(dz_sig @ (Jsph @ e) + 1j * (dz_sig @ e)) for e in np.eye(2))
check("Example 43.3.6(b): sigma is antiholomorphic, conj(sigma) holomorphic (outward N)", anti < 1e-6 and hres_bar < 1e-6 and hres_sig > 0.1)
# (c) cylinder
r = 0.7
cyl = lambda q: np.array([r * np.cos(q[0]), r * np.sin(q[0]), q[1]])  # noqa: E731
res_c, _ = holo_residual(cyl, lambda q: q[0] + 1j * q[1] / r, (0.4, 0.9))
res_c2, _ = holo_residual(cyl, lambda q: np.exp(1j * (q[0] + 1j * q[1] / r)), (0.4, 0.9))
res_bad, _ = holo_residual(cyl, lambda q: q[0] + 1j * q[1], (0.4, 0.9))
check("Example 43.3.6(c), Exercise 43.3.2: u + iv/r and exp(i(u + iv/r)) holomorphic on the cylinder, u + iv not (r = 0.7)",
      res_c < 1e-6 and res_c2 < 1e-6 and res_bad > 0.1)
Jc, _, _ = J_param(cyl, (0.4, 0.9))
check("Exercise 43.3.2: J d_u = r d_v, J d_v = -d_u / r", np.allclose(Jc @ [1, 0], [0, r], atol=1e-7) and np.allclose(Jc @ [0, 1], [-1 / r, 0], atol=1e-7))
uu = np.linspace(0, 2 * np.pi, 13)[:-1]
vals = np.exp(-0.5 / r) * np.exp(1j * uu)
check("Example 43.3.6(c): exp(iz) = e^{-v/r} e^{iu} is injective on a circle of the cylinder", len(np.unique(np.round(vals, 12))) == 12)

# ---------------------------------------------------------------- Non-example 43.3.7
_, dzs = holo_residual(sphere, lambda q: q[1] - 1j * q[0], (th0, ph0))
close("Non-example 43.3.7: dz(J d_phi) = i sin(theta) for z = phi - i theta", dzs @ (Js @ [0, 1]), 1j * np.sin(th0), 1e-6)

# ---------------------------------------------------------------- Example 43.3.11, Figure 43.3.2
x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
z2 = ((x1 ** 2 + x2 ** 2) / (1 - x3) ** 2).subs(x1 ** 2, 1 - x2 ** 2 - x3 ** 2)
check("Example 43.3.11: |zeta|^2 = (1 + x3)/(1 - x3) on S^2", sp.simplify(z2 - (1 + x3) / (1 - x3)) == 0)
A = 0.45
s_ = np.conj(A)
c_b = np.array([2 * s_.real, 2 * s_.imag, abs(s_) ** 2 - 1]) / (1 + abs(s_) ** 2)
close("Example 43.3.11: zeta^{-1}(0.45) = (0.748, 0, -0.663)", c_b, (0.748, 0.0, -0.663), 6e-4)
check("Example 43.3.11: zeta(zeta^{-1}(a)) = a", abs((c_b[0] - 1j * c_b[1]) / (1 - c_b[2]) - A) < 1e-12)
t = np.linspace(0, 2 * np.pi, 241)
phi_inv = lambda w: (w + A) / (1 + A * w)  # noqa: E731
plan = []
for rr in (0.2, 0.4, 0.6, 0.8):
    s2 = np.conj(phi_inv(rr * np.exp(1j * t)))
    d = 1 + np.abs(s2) ** 2
    C = np.stack([2 * s2.real / d, 2 * s2.imag / d, (np.abs(s2) ** 2 - 1) / d], -1)
    plan.append(np.linalg.svd(C - C.mean(0), compute_uv=False)[-1])
check("Figure 43.3.2: the circles of panel (b) are planar (circles on the sphere) to 1e-12", max(plan) < 1e-12)

# ---------------------------------------------------------------- Proposition 43.3.13
g0 = np.array([[1.3, 0.2], [0.2, 0.7]])
X, Y = rng.standard_normal(2), rng.standard_normal(2)
cosang = lambda G: X @ G @ Y / np.sqrt((X @ G @ X) * (Y @ G @ Y))  # noqa: E731
close("Proposition 43.3.13(a): angles agree for g and e^{2 mu} g", cosang(g0), cosang(np.exp(2 * 0.37) * g0), 1e-12)

# ---------------------------------------------------------------- Example 43.3.14
u, v = sp.symbols("u v", real=True)
mu = sp.log(2) - sp.log(1 + u ** 2 + v ** 2)
check("Example 43.3.14(a): e^{2 mu} = 4/(1 + u^2 + v^2)^2", sp.simplify(sp.exp(2 * mu) - 4 / (1 + u ** 2 + v ** 2) ** 2) == 0)
R_, r_ = 2.0, 0.8
W = 2 * np.pi * r_ / np.sqrt(R_ ** 2 - r_ ** 2)
close("Example 43.3.14(c): W = 2 pi r / sqrt(R^2 - r^2) = 2.7422 and W / 2 pi = 0.4364", (W, W / (2 * np.pi)), (2.7422, 0.4364), 6e-5)
U = sp.symbols("U", real=True)
Rs, rs = sp.symbols("R r", positive=True)
wexpr = 2 * rs / sp.sqrt(Rs ** 2 - rs ** 2) * sp.atan(sp.sqrt((Rs - rs) / (Rs + rs)) * sp.tan(U / 2))
dw = sp.diff(wexpr, U)
vals = [float((dw - rs / (Rs + rs * sp.cos(U))).subs({Rs: 2, rs: 0.8, U: uv_})) for uv_ in (-2.9, -1.0, 0.3, 2.2)]
check("(43.3.3), Exercise 43.3.4: dw/du = r / (R + r cos u)", max(abs(a) for a in vals) < 1e-12)
check("Exercise 43.3.4: int ds/(a + b s^2) = arctan(sqrt(b/a) s)/sqrt(ab)",
      sp.simplify(sp.diff(sp.atan(sp.sqrt(sp.Symbol("b", positive=True) / sp.Symbol("a", positive=True)) * U)
                          / sp.sqrt(sp.Symbol("a", positive=True) * sp.Symbol("b", positive=True)), U)
                  - 1 / (sp.Symbol("a", positive=True) + sp.Symbol("b", positive=True) * U ** 2)) == 0)
close("Exercise 43.3.4: w(u) -> pi r/sqrt(R^2 - r^2) as u -> pi", ch43.torus_w(np.pi - 1e-9), np.pi * r_ / np.sqrt(R_ ** 2 - r_ ** 2), 1e-7)
ok = True
for uv_ in (0.3, 2.0, -1.4):
    hh = 1e-6
    Xu = (torus((uv_ + hh, 1.0)) - torus((uv_ - hh, 1.0))) / (2 * hh)
    Xv = (torus((uv_, 1.0 + hh)) - torus((uv_, 1.0 - hh))) / (2 * hh)
    rho = R_ + r_ * np.cos(uv_)
    Xw = Xu * rho / r_
    Nout = torus((uv_, 1.0)) - R_ * np.array([np.cos(1.0), np.sin(1.0), 0])
    ok &= abs(Xv @ Xv - rho ** 2) < 1e-6 and abs(Xw @ Xw - rho ** 2) < 1e-6 and abs(Xv @ Xw) < 1e-8
    ok &= np.cross(Xv, Xw) @ Nout > 0
check("Example 43.3.14(c): g = rho^2 (dv^2 + dw^2) and (v, w) is positively oriented for the outward normal", ok)
close("Exercise 43.3.4: ratio of the conformal factor rho at the outer and inner equator = 7/3", (R_ + r_) / (R_ - r_), 7 / 3, 1e-12)
res_t, _ = holo_residual(torus, lambda q: q[1] + 1j * ch43.torus_w(q[0]), (0.7, 1.3))
J_t, _, _ = J_param(torus, (0.7, 1.3))
# the torus parametrization (u, v) has inward x_u x x_v, so J_param uses the inward normal: z = v + i w is
# antiholomorphic for it and holomorphic for the outward one
check("Example 43.3.14(c): z = v + i w is antiholomorphic for the inward normal of x(u, v), i.e. holomorphic outward",
      res_t > 0.1)
q0 = np.array([0.7, 1.3])
hh = 1e-6
dzt = np.array([((q0[1] + hh * e[1]) + 1j * ch43.torus_w(q0[0] + hh * e[0]) - ((q0[1] - hh * e[1]) + 1j * ch43.torus_w(q0[0] - hh * e[0]))) / (2 * hh)
                for e in np.eye(2)])
check("Example 43.3.14(c): dz(J X) = -i dz(X) for the inward J (holomorphic for the outward J = -J_inward)",
      max(abs(dzt @ (J_t @ e) + 1j * (dzt @ e)) for e in np.eye(2)) < 1e-6)

# ---------------------------------------------------------------- Figure 43.3.3 numbers
dv_, dw_ = 2 * np.pi / 23, W / 10
close("Figure 43.3.3: cells 0.2732 x 0.2742 (aspect 1.004)", (dv_, dw_, dw_ / dv_), (0.2732, 0.2742, 1.004), 6e-4)


def u_of_w(w):
    k = np.sqrt(R_ ** 2 - r_ ** 2)
    return 2 * np.arctan(np.tan(w * k / (2 * r_)) / np.sqrt((R_ - r_) / (R_ + r_)))


sides = []
for w0 in (0.0, W / 2 - 2 * dw_):
    ws = np.linspace(w0, w0 + dw_ - 1e-12, 4001)
    along_w = np.trapezoid(R_ + r_ * np.cos(u_of_w(ws)), ws)
    along_v = (R_ + r_ * np.cos(u_of_w(w0 + dw_ / 2))) * dv_
    sides += [along_v, along_w]
close("Figure 43.3.3: cell sides 0.741, 0.737 (outer) and 0.372, 0.375 (inner)", sides, (0.741, 0.737, 0.372, 0.375), 6e-4)

# ---------------------------------------------------------------- Looking ahead: K = -e^{-2mu} Delta_0 mu
lap0 = lambda f: sp.diff(f, u, 2) + sp.diff(f, v, 2)  # noqa: E731
check("Looking ahead: stereographic mu gives K = 1", sp.simplify(-sp.exp(-2 * mu) * lap0(mu) - 1) == 0)
yy = sp.symbols("y", positive=True)
muh = -sp.log(yy)
check("Looking ahead: mu = -log y gives K = -1", sp.simplify(-sp.exp(-2 * muh) * sp.diff(muh, yy, 2) + 1) == 0)
check("Looking ahead: Delta_0 mu = -4/(1 + |w|^2)^2 for the stereographic mu", sp.simplify(lap0(mu) + 4 / (1 + u ** 2 + v ** 2) ** 2) == 0)

# ---------------------------------------------------------------- Exercise 43.3.1 (saddle)
Jsd, Dsd, Nsd = J_param(saddle, (0.5, -0.4))
close("Exercise 43.3.1: J_q d_u = (-0.492, 1.231) at q = (0.5, -0.4)", Jsd @ [1, 0], (-0.8 / np.sqrt(2.64), 2 / np.sqrt(2.64)), 1e-7)
close("Exercise 43.3.1: -0.8/sqrt(2.64) = -0.492, 2/sqrt(2.64) = 1.231 = sqrt(2/1.32)", (-0.8 / np.sqrt(2.64), 2 / np.sqrt(2.64), np.sqrt(2 / 1.32)),
      (-0.492, 1.231, 1.231), 6e-4)

# ---------------------------------------------------------------- Exercise 43.3.5
ok5 = True
for _ in range(10):
    B = rng.standard_normal((2, 2))
    if np.linalg.det(B) < 0:
        B[:, 0] *= -1
    Jr = B @ np.array([[0, -1], [1, 0]]) @ np.linalg.inv(B)       # a general complex structure (positive)
    e = rng.standard_normal(2)
    Bb = np.c_[e, Jr @ e]
    G = np.linalg.inv(Bb).T @ np.linalg.inv(Bb)                 # makes (e, J e) orthonormal
    ok5 &= abs(e @ G @ (Jr @ e)) < 1e-10 and abs((Jr @ e) @ G @ (Jr @ e) - e @ G @ e) < 1e-10
    ok5 &= np.allclose(Jr.T @ G @ Jr, G)                        # J is a G-isometry, i.e. the G-rotation
check("Exercise 43.3.5: every positive J is the rotation of the metric making (e, Je) orthonormal", ok5)

# ---------------------------------------------------------------- Exercise 43.3.6
ns, same, printed = ch43.run_printed(PAGE, "code-ex-43-3-6", "out-ex-43-3-6")
check("Exercise 43.3.6: code-ex-43-3-6 prints exactly out-ex-43-3-6", same, printed)
check("Exercise 43.3.6: naive ratio rho/r lies in [1.5, 3.5]", abs((R_ + r_) / r_ - 3.5) < 1e-12 and abs((R_ - r_) / r_ - 1.5) < 1e-12)

summary()
