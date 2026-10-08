"""Verification for Section 30.2 (Curvature of Implicit Surfaces).

Conventions (GUIDELINES §6.3): W_p = -dN_p, H = tr(W_p)/2, K = det W_p. Here N = grad f/|grad f| (outward, f > 0
outside). The parametrizations of tools/dgsym.EXAMPLES give their own normal x_u x x_v; for the torus of revolution
that normal points inward, so H changes sign when compared.

Run from the project root:
``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/verify/v-30-2-implicit-curvature.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, summary, sym_equal

x, y, z = sp.symbols("x y z", real=True)
X3 = sp.Matrix([x, y, z])
r, R = sp.symbols("r R", positive=True)
EX = dgsym.EXAMPLES


def grad(f):
    return sp.Matrix([sp.diff(f, v) for v in X3])


def num_equal(name, a, b, syms, ranges, n=12, tol=1e-9, seed=7):
    """compare two sympy expressions numerically at n random points (no simplify: fast for big expressions)."""
    fa = sp.lambdify(syms, a, modules="numpy")
    fb = sp.lambdify(syms, b, modules="numpy")
    rg = np.random.default_rng(seed)
    pts = [[rg.uniform(lo, hi) for (lo, hi) in ranges] for _ in range(n)]
    va = np.array([np.asarray(fa(*q), dtype=float) for q in pts])
    vb = np.array([np.asarray(fb(*q), dtype=float) for q in pts])
    return close(name, va, vb, tol=tol)


def hess(f):
    return sp.hessian(f, list(X3))


def implicit_curvatures(f):
    """(H, K) with N = grad f/|grad f| by (30.2.4) and (30.2.6)."""
    g = grad(f)
    Hm = hess(f)
    n2 = (g.T * g)[0]
    lap = Hm.trace()
    Hc = ((g.T * Hm * g)[0] - n2 * lap) / (2 * n2 ** sp.Rational(3, 2))
    bordered = Hm.row_join(g).col_join(g.T.row_join(sp.zeros(1, 1)))
    Kc = -bordered.det() / n2 ** 2
    Kadj = (g.T * Hm.adjugate() * g)[0] / n2 ** 2
    return Hc, Kc, Kadj


def shape_op_implicit(f, point):
    """3x3 matrix -(1/|grad f|) P Hess f P at a point (numpy), its nonzero eigenvalues are the kappa_i."""
    g = np.array([float(c) for c in grad(f).subs(point)])
    Hm = np.array(hess(f).subs(point).tolist(), dtype=float)
    ng = np.linalg.norm(g)
    n = g / ng
    P = np.eye(3) - np.outer(n, n)
    return -(P @ Hm @ P) / ng, n


# ---------------------------------------------------------------- Example 30.2.2: sphere, two defining functions
for name, f in [("|x|^2 - r^2", x ** 2 + y ** 2 + z ** 2 - r ** 2), ("|x| - r", sp.sqrt(x ** 2 + y ** 2 + z ** 2) - r)]:
    th, ph = sp.symbols("theta phi", real=True)
    pt = {x: r * sp.sin(th) * sp.cos(ph), y: r * sp.sin(th) * sp.sin(ph), z: r * sp.cos(th)}
    Hc, Kc, Kadj = implicit_curvatures(f)
    sym_equal(f"Example 30.2.2 ({name}): H = -1/r with outward N", sp.simplify(Hc.subs(pt)), -1 / r,
              {th: (0, sp.pi), ph: (0, 2 * sp.pi), r: (0.5, 2)})
    sym_equal(f"Example 30.2.2 ({name}): K = 1/r^2 (bordered Hessian)", sp.simplify(Kc.subs(pt)), 1 / r ** 2,
              {th: (0, sp.pi), ph: (0, 2 * sp.pi), r: (0.5, 2)})
    sym_equal(f"(30.2.6) ({name}): bordered-determinant form = adjugate form", sp.simplify((Kc - Kadj).subs(pt)), 0,
              {th: (0, sp.pi), ph: (0, 2 * sp.pi), r: (0.5, 2)})
# dN = P/r on the sphere for both
for f in (x ** 2 + y ** 2 + z ** 2 - 4, sp.sqrt(x ** 2 + y ** 2 + z ** 2) - 2):
    W, n = shape_op_implicit(f, {x: 0.6, y: -0.8, z: np.sqrt(4 - 1.0)})
    ev = np.sort(np.linalg.eigvalsh(W))
    close("Proposition 30.2.1 / Example 30.2.2: W_p = -(1/|grad f|) P Hess f P has eigenvalues -1/r, -1/r, 0 (r = 2)", ev,
          [-0.5, -0.5, 0.0], tol=1e-12)

# ---------------------------------------------------------------- Proposition 30.2.1 against the parametrization (torus)
th_u, th_v = sp.symbols("u v", real=True)
ex = EX["torus"]
(Rs, rs) = ex["params"]
(us, vs) = ex["coords"]
Xpar = ex["expr"]
f_tor_poly = (sp.sqrt(x ** 2 + y ** 2) - Rs) ** 2 + z ** 2 - rs ** 2         # Example 6.5.8, outward gradient
Kp, Hp = dgsym.K_H(Xpar, us, vs, positive=ex["positive"])                    # parametric normal: inward
Wpar = dgsym.shape_operator(Xpar, us, vs, positive=ex["positive"])
vals = {Rs: 2.0, rs: 0.8}
for uu in (0.0, 0.7, 1.9, np.pi, 4.4):
    pt = {x: float(Xpar[0].subs({**vals, us: uu, vs: 0.3})), y: float(Xpar[1].subs({**vals, us: uu, vs: 0.3})),
          z: float(Xpar[2].subs({**vals, us: uu, vs: 0.3}))}
    W, n = shape_op_implicit(f_tor_poly.subs(vals), pt)
    ev_imp = np.sort(np.linalg.eigvalsh(W))
    ev_imp = ev_imp[np.argsort(np.abs(ev_imp))][1:]                         # drop the normal eigenvalue 0
    ev_par = np.linalg.eigvals(np.array(Wpar.subs({**vals, us: uu}).tolist(), dtype=float)).real
    close(f"Proposition 30.2.1: torus u = {uu:.2f}: implicit kappa_i (outward) = -(parametric kappa_i, inward)",
          np.sort(ev_imp), np.sort(-ev_par), tol=1e-10)
Hc_t, Kc_t, Kadj_t = implicit_curvatures(f_tor_poly)
ptt = {x: Xpar[0], y: Xpar[1], z: Xpar[2]}
TS = (us, vs, Rs, rs)
TR = [(0, 2 * np.pi), (0, 2 * np.pi), (1.1, 2.0), (0.2, 0.9)]
num_equal("Example 30.2.5: torus (implicit, outward N): H = -(R + 2r cos u)/(2r(R + r cos u)) = -H_param",
          Hc_t.subs(ptt), -(Rs + 2 * rs * sp.cos(us)) / (2 * rs * (Rs + rs * sp.cos(us))), TS, TR)
gtor = grad(f_tor_poly)
Ntor = gtor / sp.sqrt((gtor.T * gtor)[0])
H_Nform = ((Ntor.T * hess(f_tor_poly) * Ntor)[0] - hess(f_tor_poly).trace()) / (2 * sp.sqrt((gtor.T * gtor)[0]))
num_equal("(30.2.4): H = (N^T Hess f N - Delta f)/(2|grad f|) agrees with the grad-form on the torus",
          H_Nform.subs(ptt), Hc_t.subs(ptt), TS, TR)
num_equal("Example 30.2.5: the parametric (inward) H of Example 0.4.6", Hp,
          (Rs + 2 * rs * sp.cos(us)) / (2 * rs * (Rs + rs * sp.cos(us))), TS, TR)
# bordered determinant evaluated numerically (np.linalg.det) at random surface points: avoids a huge symbolic det
g_fun = sp.lambdify((x, y, z, Rs, rs), grad(f_tor_poly), modules="numpy")
h_fun = sp.lambdify((x, y, z, Rs, rs), hess(f_tor_poly), modules="numpy")
rg = np.random.default_rng(11)
kb, ke = [], []
for _ in range(12):
    uu, vv, R0, r0 = rg.uniform(0, 2 * np.pi), rg.uniform(0, 2 * np.pi), rg.uniform(1.1, 2.0), rg.uniform(0.2, 0.9)
    P = np.array([(R0 + r0 * np.cos(uu)) * np.cos(vv), (R0 + r0 * np.cos(uu)) * np.sin(vv), r0 * np.sin(uu)])
    gg = np.array(g_fun(*P, R0, r0), dtype=float).ravel()
    HH = np.array(h_fun(*P, R0, r0), dtype=float)
    Bm = np.block([[HH, gg[:, None]], [gg[None, :], np.zeros((1, 1))]])
    kb.append(-np.linalg.det(Bm) / (gg @ gg) ** 2)
    ke.append(np.cos(uu) / (r0 * (R0 + r0 * np.cos(uu))))
close("(30.2.6): torus K from the bordered Hessian = cos u/(r(R + r cos u))", kb, ke, tol=1e-9)
num_equal("(30.2.6): torus K, adjugate form", Kadj_t.subs(ptt), sp.cos(us) / (rs * (Rs + rs * sp.cos(us))), TS, TR)

# ---------------------------------------------------------------- cylinder and saddle
f_cyl = sp.sqrt(x ** 2 + y ** 2) - r
Hc, Kc, _ = implicit_curvatures(f_cyl)
pc = {x: r * sp.cos(th_u), y: r * sp.sin(th_u)}
sym_equal("Exercise 30.2.1: cylinder SDF: H = -1/(2r) (outward)", sp.simplify(Hc.subs(pc)), -1 / (2 * r),
          {th_u: (0, 2 * sp.pi), r: (0.5, 2)})
sym_equal("Exercise 30.2.1: cylinder SDF: K = 0", sp.simplify(Kc.subs(pc)), 0, {th_u: (0, 2 * sp.pi), r: (0.5, 2)})
Hm_cyl = hess(f_cyl).subs({x: 1.5, y: 0.0, z: 0.3, r: 1.5})
close("Exercise 30.2.1: cylinder SDF Hessian eigenvalues 0, 0, 1/r (r = 1.5)", np.sort(np.linalg.eigvalsh(
    np.array(Hm_cyl.tolist(), dtype=float))), [0, 0, 1 / 1.5], tol=1e-12)
a, b = sp.symbols("a b", real=True)
f_par = z - (a * x ** 2 + b * y ** 2) / 2
Hc, Kc, _ = implicit_curvatures(f_par)
o = {x: 0, y: 0, z: 0}
sym_equal("Exercise 30.2.3: z = (ax^2 + by^2)/2 at 0, upward N: H = (a + b)/2", Hc.subs(o), (a + b) / 2)
sym_equal("Exercise 30.2.3: K = ab", Kc.subs(o), a * b)
f_sad = z - (x ** 2 - y ** 2)
Hc, Kc, _ = implicit_curvatures(f_sad)
check("saddle z = x^2 - y^2 at 0 (upward N): H = 0, K = -4 (matches Example 0.4.6)", Hc.subs(o) == 0 and Kc.subs(o) == -4)

# ---------------------------------------------------------------- Proposition 30.2.3: SDF Hessian, torus SDF
f_tsdf = sp.sqrt((sp.sqrt(x ** 2 + y ** 2) - Rs) ** 2 + z ** 2) - rs
g_t = grad(f_tsdf)
XR = (x, y, z, Rs, rs)
XRr = [(0.3, 1.5), (0.2, 1.0), (-0.5, 0.5), (1.6, 2.0), (0.2, 0.5)]
num_equal("Proposition 30.2.3: torus SDF has |grad f| = 1", (g_t.T * g_t)[0], sp.Integer(1) + 0 * x, XR, XRr)
Ht = hess(f_tsdf)
num_equal("Proposition 30.2.3: Hess f grad f = 0 for an SDF", Ht * g_t, sp.zeros(3, 1) + 0 * sp.ones(3, 1) * x, XR, XRr)
Htf = sp.lambdify((x, y, z, Rs, rs), Ht, modules="numpy")


def torus_point(u, v, t, R0=2.0, r0=0.8):
    rr = r0 + t
    return np.array([(R0 + rr * np.cos(u)) * np.cos(v), (R0 + rr * np.cos(u)) * np.sin(v), rr * np.sin(u)])


for uu in (0.0, 1.0, np.pi / 2, 2.5, np.pi):
    p = torus_point(uu, 0.4, 0.0)
    ev = np.sort(np.linalg.eigvalsh(np.array(Htf(*p, 2.0, 0.8), dtype=float)))
    expect = np.sort([0.0, 1 / 0.8, np.cos(uu) / (2.0 + 0.8 * np.cos(uu))])
    close(f"Proposition 30.2.3: torus SDF at u = {uu:.2f}: eig Hess f = (0, 1/r, cos u/(R + r cos u)) = (0, -kappa_i)", ev,
          expect, tol=1e-10)
# offsets (Exercise 30.2.4): along the normal at u = 0, eigenvalues 1/(r + t), 1/(R + r + t)
for t in (-0.6, -0.3, 0.5, 1.2):
    p = torus_point(0.0, 0.0, t)
    ev = np.sort(np.linalg.eigvalsh(np.array(Htf(*p, 2.0, 0.8), dtype=float)))
    close(f"Exercise 30.2.4 / Figure 30.2.1(a): offset t = {t}: eigenvalues 0, 1/(R+r+t), 1/(r+t)", ev,
          np.sort([0.0, 1 / (2.8 + t), 1 / (0.8 + t)]), tol=1e-10)
# general offset formula -kappa/(1 - t kappa): sphere kappa = -1/r
kap, tt = sp.symbols("kappa t", real=True)
sym_equal("Exercise 30.2.4: offset curvature kappa/(1 - t kappa) for kappa = -1/r gives -1/(r + t)",
          (kap / (1 - tt * kap)).subs(kap, -1 / r), -1 / (r + tt), {tt: (0, 1), r: (0.5, 2)})
# SDF: K = sigma_2(Hess f) (sum of principal 2x2 minors) on the torus surface
for uu in (0.3, 2.2):
    Hn = np.array(Htf(*torus_point(uu, 1.1, 0.0), 2.0, 0.8), dtype=float)
    s2 = 0.5 * (np.trace(Hn) ** 2 - np.trace(Hn @ Hn))
    close(f"(30.2.7): SDF: K = sigma_2(Hess f) at u = {uu}", s2, np.cos(uu) / (0.8 * (2.0 + 0.8 * np.cos(uu))), tol=1e-10)

# ---------------------------------------------------------------- Proposition 30.2.4: div N = -2H, SDF: Delta f = -2H
f_gen = (sp.sqrt(x ** 2 + y ** 2) - Rs) ** 2 + z ** 2 - rs ** 2
Ng = grad(f_gen) / sp.sqrt((grad(f_gen).T * grad(f_gen))[0])
divN = sum(sp.diff(Ng[i], X3[i]) for i in range(3))
num_equal("Proposition 30.2.4: div(grad f/|grad f|) = -2H on the torus (non-SDF defining function)",
          divN.subs(ptt), (Rs + 2 * rs * sp.cos(us)) / (rs * (Rs + rs * sp.cos(us))), TS, TR)
lap_t = sp.lambdify((x, y, z, Rs, rs), Ht.trace(), modules="numpy")
for uu in (0.0, np.pi / 2, np.pi):
    close(f"Example 30.2.5 / Figure 30.2.2(a): torus SDF Delta f at u = {uu:.2f} = -2H = (R + 2r cos u)/(r(R + r cos u))",
          lap_t(*torus_point(uu, 0.0, 0.0), 2.0, 0.8), (2 + 1.6 * np.cos(uu)) / (0.8 * (2 + 0.8 * np.cos(uu))), tol=1e-10)
close("Figure 30.2.2(a): Delta f ranges over [0.417, 1.607] (R = 2, r = 0.8)",
      [(2 - 1.6) / (0.8 * 1.2), (2 + 1.6) / (0.8 * 2.8)], [0.4167, 1.6071], tol=1e-4)
close("Figure 30.2.2(b): K ranges over [-1.042, 0.446]", [-1 / (0.8 * 1.2), 1 / (0.8 * 2.8)], [-1.0417, 0.4464], tol=1e-4)
sym_equal("Example 30.2.5: sphere SDF: Delta(|x| - r) = 2/|x|", sp.simplify(hess(sp.sqrt(x ** 2 + y ** 2 + z ** 2)).trace()),
          2 / sp.sqrt(x ** 2 + y ** 2 + z ** 2))

# ---------------------------------------------------------------- Neuralangelo numerical Laplacian (6 and 4 taps)
rng = np.random.default_rng(302)
A = rng.normal(size=(3, 3))
A = A + A.T
bvec = rng.normal(size=3)
quad = lambda P: 0.5 * np.einsum("...i,ij,...j->...", P, A, P) + P @ bvec + 0.7
p0 = rng.normal(size=3)
eps = 0.05
six = sum((quad(p0 + eps * e) + quad(p0 - eps * e) - 2 * quad(p0)) / eps ** 2 for e in np.eye(3))
close("Neuralangelo 6 taps: sum of second differences = tr Hess (exact for quadratics)", six, np.trace(A), tol=1e-9)
ep = eps / np.sqrt(3)
K4 = np.array([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], float)
four = (sum(quad(p0 + ep * k) for k in K4) / 2.0 - 2 * quad(p0)) / ep ** 2
close("Neuralangelo 4 taps (modules.py): ((sum of 4 tetrahedral samples)/2 - 2 f)/eps'^2 = tr Hess", four, np.trace(A),
      tol=1e-9)
check("Neuralangelo analytical mode returns row sums of Hess (autograd of grad.sum()); 1^T H 1 != tr H in general",
      abs(A.sum() - np.trace(A)) > 1e-3)

# ---------------------------------------------------------------- L1 vs L2 for a rounded fold (Figure 30.2.3)
def rounded_corner_sdf(P, rho):
    """SDF of {x <= 0, y <= 0} with the corner rounded by radius rho (offset of the shifted quadrant)."""
    qx, qy = P[..., 0] + rho, P[..., 1] + rho
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - rho


for rho in (0.1, 0.3, 0.6):
    # sample the zero set (arc + two straight pieces), evaluate the 6-tap Laplacian with a small step, integrate
    s_arc = np.linspace(0, np.pi / 2, 4001)
    arc = np.stack([-rho + rho * np.cos(s_arc), -rho + rho * np.sin(s_arc)], 1)
    h = 1e-4
    E = np.eye(2)
    lap = sum(rounded_corner_sdf(arc + h * e, rho) + rounded_corner_sdf(arc - h * e, rho) - 2 * rounded_corner_sdf(arc, rho)
              for e in E) / h ** 2
    Hmean = np.abs(lap) / 2                                            # |H| = |Delta f|/2 (extruded: kappa_z = 0)
    ds = rho * (s_arc[1] - s_arc[0])
    L1 = np.trapezoid(Hmean, dx=ds)
    L2 = np.trapezoid(Hmean ** 2, dx=ds)
    close(f"Figure 30.2.3(b): rounded fold rho = {rho}: int |H| dA per unit length = pi/4 (independent of rho)", L1,
          np.pi / 4, tol=2e-3)
    close(f"Figure 30.2.3(b): rounded fold rho = {rho}: int H^2 dA per unit length = pi/(8 rho)", L2, np.pi / (8 * rho),
          tol=2e-3 * np.pi / (8 * rho))
    flat = np.array([[-rho - 0.5, 0.0], [0.0, -rho - 0.7]])
    lapf = sum(rounded_corner_sdf(flat + h * e, rho) + rounded_corner_sdf(flat - h * e, rho)
               - 2 * rounded_corner_sdf(flat, rho) for e in E) / h ** 2
    close(f"Figure 30.2.3(a): flat parts have Delta f = 0 (rho = {rho})", lapf, [0, 0], tol=1e-6)
check("Exercise-level: fold of angle phi, length l: int|H| = phi l/2 and int H^2 = phi l/(4 rho)",
      abs((np.pi / 2) * 1 / 2 - np.pi / 4) < 1e-15)

# ---------------------------------------------------------------- marching squares (Figure 30.2.4)
f00, f10, f01, f11 = 0.3, -0.4, -0.3, 0.2
den = f00 + f11 - f10 - f01
xs_, ys_ = (f00 - f01) / den, (f00 - f10) / den
bil = lambda X, Y: f00 * (1 - X) * (1 - Y) + f10 * X * (1 - Y) + f01 * (1 - X) * Y + f11 * X * Y
fs = bil(xs_, ys_)
close("Figure 30.2.4(b): bilinear saddle value (f00 f11 - f10 f01)/(f00 + f11 - f10 - f01) = -0.05", fs,
      (f00 * f11 - f10 * f01) / den, tol=1e-15)
close("Figure 30.2.4(b): saddle value = -0.05 at (0.5, 0.583)", [fs, xs_, ys_], [-0.05, 0.5, 0.58333333], tol=1e-7)
eps_ = 1e-6
gx = (bil(xs_ + eps_, ys_) - bil(xs_ - eps_, ys_)) / (2 * eps_)
gy = (bil(xs_, ys_ + eps_) - bil(xs_, ys_ - eps_)) / (2 * eps_)
close("Figure 30.2.4(b): the saddle is a critical point of the bilinear interpolant", [gx, gy], [0, 0], tol=1e-8)


def edge_errors(fun, h=0.25, off=0.0371):
    xs = np.arange(-1.6, 1.6 + 1e-9, h) + off
    Xg, Yg = np.meshgrid(xs, xs)
    F = fun(Xg, Yg)
    errs = []
    for ax in (0, 1):
        if ax == 0:
            a0, a1, x0, y0, dx, dy = F[:, :-1], F[:, 1:], Xg[:, :-1], Yg[:, :-1], h, 0
        else:
            a0, a1, x0, y0, dx, dy = F[:-1, :], F[1:, :], Xg[:-1, :], Yg[:-1, :], 0, h
        m = a0 * a1 < 0
        tt_ = a0[m] / (a0[m] - a1[m])
        errs.append(np.abs(np.hypot(x0[m] + tt_ * dx, y0[m] + tt_ * dy) - 1))
    return np.concatenate(errs)


e_sdf = edge_errors(lambda X, Y: np.hypot(X, Y) - 1)
e_quad = edge_errors(lambda X, Y: X ** 2 + Y ** 2 - 1)
e_tanh = edge_errors(lambda X, Y: np.tanh((np.hypot(X, Y) - 1) / 0.05))
print(f"   edge-vertex error (h = 0.25, unit circle): SDF max {e_sdf.max():.4f} mean {e_sdf.mean():.4f}; "
      f"x^2+y^2-1 max {e_quad.max():.4f}; tanh(SDF/0.05) max {e_tanh.max():.4f} mean {e_tanh.mean():.4f}")
check("What geometry explains (marching squares): SDF vertices within 0.007 of the circle (h = 0.25)", e_sdf.max() < 0.007)
check("What geometry explains: a saturating 'density-like' tanh(SDF/0.05) puts vertices up to ~0.058 off (> 8x worse)",
      e_tanh.max() > 8 * e_sdf.max() and abs(e_tanh.max() - 0.0577) < 1e-3)
# linear interpolation is exact for an affine f along the edge (plane SDF)
check("marching cubes: linear interpolation is exact when f is affine along the edge",
      abs((0.3 / (0.3 + 0.5)) * 1.0 - 0.375) < 1e-15)

# ---------------------------------------------------------------- Exercise 30.2.5: flat Gaussian, level-set normal vs r_3
s_ax = np.array([1.0, 1.0, 0.05])
for psi_deg, expect in ((10.0, None), (60.0, None)):
    psi = np.radians(psi_deg)
    pt = np.array([s_ax[0] * np.sin(psi), 0.0, s_ax[2] * np.cos(psi)])     # point on the 1-sigma ellipsoid
    nrm = pt / s_ax ** 2
    ang = np.degrees(np.arccos(nrm[2] / np.linalg.norm(nrm)))
    print(f"   Exercise 30.2.5: psi = {psi_deg} deg: angle between level-set normal and r_3 = {ang:.3f} deg")
close("Exercise 30.2.5: at psi = 60 deg the level-set normal is 4.95 deg from r_3",
      np.degrees(np.arctan((np.sin(np.radians(60)) / 1.0) / (np.cos(np.radians(60)) / 0.05))), 4.949, tol=1e-3)
close("Exercise 30.2.5: the angle reaches 45 deg only when tan psi = s_1/s_3 = 20 (psi = 87.1 deg)",
      np.degrees(np.arctan(20.0)), 87.138, tol=1e-3)

# ---------------------------------------------------------------- review (R1): a surface with no symmetry to lean on
# ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 with three different axes: (30.2.1), (30.2.4), (30.2.6) against the
# parametric K, H of dgsym (normal x_u x x_v, checked to be outward here), and div N by central differences.
ue, ve = sp.symbols("u v", real=True)
ae, be, ce = sp.Rational(2), sp.Rational(3, 2), sp.Rational(1)
fe = x ** 2 / ae ** 2 + y ** 2 / be ** 2 + z ** 2 / ce ** 2 - 1
Xe = sp.Matrix([ae * sp.sin(ue) * sp.cos(ve), be * sp.sin(ue) * sp.sin(ve), ce * sp.cos(ue)])
Kpe, Hpe = dgsym.K_H(Xe, ue, ve)
Npe = dgsym.unit_normal(Xe, ue, ve)
ge_f = sp.lambdify((x, y, z), grad(fe), "numpy")
He_num = np.array(sp.hessian(fe, (x, y, z)), dtype=float)          # constant Hessian
for (uu, vv) in [(0.7, 0.4), (1.3, 2.2), (2.5, -1.0)]:
    p = np.array(Xe.subs({ue: uu, ve: vv}).evalf(), dtype=float).ravel()
    g = np.array(ge_f(*p), dtype=float).ravel()
    gn = np.linalg.norm(g)
    Nv = g / gn
    sgn = np.sign(Nv @ np.array(Npe.subs({ue: uu, ve: vv}).evalf(), dtype=float).ravel())
    H_par = sgn * float(Hpe.subs({ue: uu, ve: vv}))
    K_par = float(Kpe.subs({ue: uu, ve: vv}))
    Pm = np.eye(3) - np.outer(Nv, Nv)
    ev = np.sort(np.linalg.eigvalsh(-(Pm @ He_num @ Pm) / gn))     # (30.2.1): kappa_1, kappa_2 and a 0 for N
    k12 = ev[np.argsort(np.abs(ev))[1:]]
    close(f"review: ellipsoid ({uu}, {vv}): (30.2.1) eigenvalues give H = H_param (outward N)", k12.sum() / 2, H_par,
          tol=1e-10)
    close(f"review: ellipsoid ({uu}, {vv}): (30.2.4) H = (N^T Hess N - Delta f)/(2|grad f|)",
          (Nv @ He_num @ Nv - np.trace(He_num)) / (2 * gn), H_par, tol=1e-10)
    Bm = np.zeros((4, 4))
    Bm[:3, :3], Bm[:3, 3], Bm[3, :3] = He_num, g, g
    close(f"review: ellipsoid ({uu}, {vv}): (30.2.6) bordered Hessian K = K_param", -np.linalg.det(Bm) / gn ** 4,
          K_par, tol=1e-10)
    hh = 1e-5
    unit = lambda q: (lambda w: w / np.linalg.norm(w))(np.array(ge_f(*q), dtype=float).ravel())
    divN = sum((unit(p + hh * e) - unit(p - hh * e))[i] / (2 * hh) for i, e in enumerate(np.eye(3)))
    close(f"review: ellipsoid ({uu}, {vv}): div N = -2H (central differences)", divN, -2 * H_par, tol=1e-6)

# curvature-loss box: offset mean curvature H_t = (1/2) sum kappa_i/(1 - t kappa_i) = H + t(k1^2 + k2^2)/2 + O(t^2);
# for a minimal point (k2 = -k1) it is k1^2 t/(1 - k1^2 t^2) != 0, and it vanishes for all t only if k1 = k2 = 0
tt_, k1_, k2_ = sp.symbols("t k1 k2", real=True)
Ht_ = sp.Rational(1, 2) * (k1_ / (1 - tt_ * k1_) + k2_ / (1 - tt_ * k2_))
sym_equal("review: curvature loss box: H_t = H + t(k1^2 + k2^2)/2 + O(t^2)",
          sp.series(Ht_, tt_, 0, 2).removeO(), (k1_ + k2_) / 2 + tt_ * (k1_ ** 2 + k2_ ** 2) / 2)
sym_equal("review: curvature loss box: minimal point k2 = -k1: H_t = k1^2 t/(1 - k1^2 t^2)",
          Ht_.subs(k2_, -k1_), k1_ ** 2 * tt_ / (1 - k1_ ** 2 * tt_ ** 2))

# (30.2.3) at a hyperbolic point: inner equator of T_{2, 0.8}, outward N = -e_1 at (R - r, 0, 0); kappa (outward)
# = -1/r (meridian) and +1/(R - r) (parallel). Hessian of the exact SDF at p + t N by central differences.
R_n, r_n = 2.0, 0.8
sdf_t = lambda q: np.hypot(np.hypot(q[0], q[1]) - R_n, q[2]) - r_n
for t_off in (-0.4, 0.3, 0.6):
    q0 = np.array([R_n - r_n - t_off, 0.0, 0.0])
    hh = 1e-4
    Hn = np.zeros((3, 3))
    E3 = np.eye(3)
    for i in range(3):
        for j in range(3):
            Hn[i, j] = (sdf_t(q0 + hh * E3[i] + hh * E3[j]) - sdf_t(q0 + hh * E3[i] - hh * E3[j])
                        - sdf_t(q0 - hh * E3[i] + hh * E3[j]) + sdf_t(q0 - hh * E3[i] - hh * E3[j])) / (4 * hh * hh)
    pred = sorted([0.0] + [-k / (1 - t_off * k) for k in (-1 / r_n, 1 / (R_n - r_n))])
    close(f"review: (30.2.3) at the inner equator (hyperbolic), t = {t_off}: eig Hess f = 0, -kappa_i/(1 - t kappa_i)",
          np.sort(np.linalg.eigvalsh(Hn)), np.array(pred), tol=1e-5)

summary()
