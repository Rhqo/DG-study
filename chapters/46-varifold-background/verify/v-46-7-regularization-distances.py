"""Verification for Section 46.7 (Regularization and Distances Between Varifolds).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-7-regularization-distances.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(467)

# ---------------------------------------------------------------- kernel (46.7.1) and C_rho
c2 = v.normalisation(v.rho_exp, 2)
close("(46.7.1): normalising constant of exp(-1/(1-r^2)) in R^2 is c = 2.1436", c2, 2.1436, 5e-5)
g = np.linspace(-0.31, 0.31, 621)
GX, GY = np.meshgrid(g, g)
pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
close("(46.7.1): int rho_eps = 1 over R^2 (eps = 0.3, grid)", v.kernel_eps(pts, 0.3).sum() * (g[1] - g[0]) ** 2, 1.0, 1e-6)
C_RHO = c2 * v.kernel_constant(v.rho_exp, 1)
close("(46.7.3): C_rho = int_R rho(|z|) dz = 0.9517 (d = 1, kernel normalised in R^2)", C_RHO, 0.9517, 5e-5)
close("(46.7.3): C_rho = d omega_d int rho r^(d-1) dr with d = 1 (omega_1 = 2)", C_RHO, c2 * 2 * v.radial_moment(v.rho_exp, 0), 1e-12)


# ---------------------------------------------------------------- Proposition 46.7.2 and Example 46.7.3
def circle_mass(y, eps, r=1.0):
    return float(v.integrate_1d(lambda t: v.kernel_eps(np.asarray(y)[None, :] - v.circle(t, r), eps) * r, 0, 2 * np.pi,
                                n=32, pieces=256))


for eps, peak in ((0.1, 9.519), (0.2, 4.762), (0.4, 2.387)):
    val = circle_mass([1.0, 0.0], eps)
    close(f"Example 46.7.3(a): (mu * rho_eps)(1, 0) = {peak} (eps = {eps})", val, peak, 5e-4)
close("Example 46.7.3(a): eps (mu * rho_eps)(1, 0) = 0.9519, 0.9525, 0.9548",
      [e * circle_mass([1.0, 0.0], e) for e in (0.1, 0.2, 0.4)], [0.9519, 0.9525, 0.9548], 5e-5)
check("Example 46.7.3(a): zero at distance >= eps from the circle", circle_mass([1.1001, 0.0], 0.1) == 0.0
      and circle_mass([0.8999, 0.0], 0.1) == 0.0)
check("Prop 46.7.2(c): eps (mu * rho_eps)(x) -> C_rho as eps -> 0 (errors decrease)",
      abs(0.05 * circle_mass([1, 0], 0.05) - C_RHO) < abs(0.1 * circle_mass([1, 0], 0.1) - C_RHO)
      < abs(0.4 * circle_mass([1, 0], 0.4) - C_RHO))
# slice integral across the circle is about 1 (exactly 1 for a straight line); vectorised periodic trapezoid in t
TQ = 2 * np.pi * np.arange(8192) / 8192
CQ = v.circle(TQ)


def circle_mass_many(Y, eps):
    out = np.empty(len(Y))
    for i0 in range(0, len(Y), 512):
        Yc = Y[i0:i0 + 512]
        out[i0:i0 + 512] = v.kernel_eps(Yc[:, None, :] - CQ[None, :, :], eps).sum(axis=1) * (2 * np.pi / 8192)
    return out


close("trapezoid density agrees with Gauss-Legendre (eps = 0.1 at (1, 0))", circle_mass_many(np.array([[1.0, 0.0]]), 0.1)[0],
      circle_mass([1.0, 0.0], 0.1), 1e-9)
ss = np.linspace(0.55, 1.45, 1801)
for eps in (0.1, 0.2, 0.4):
    sl = np.trapezoid(circle_mass_many(np.stack([ss, np.zeros_like(ss)], axis=1), eps), ss)
    close(f"Figure 46.7.1(d): area under the slice ~ 1 (eps = {eps})", sl, 1.0, 0.03)
xs = np.linspace(-0.4, 0.4, 801)
line_slice = np.trapezoid([float(v.integrate_1d(lambda t: v.kernel_eps(np.stack([t, np.full_like(t, s)], axis=1), 0.2),
                                            -0.3, 0.3, n=64, pieces=16)) for s in xs], xs)
close("Exercise 46.7.5: for a straight line the slice integrates to exactly 1", line_slice, 1.0, 1e-6)
close("Exercise 46.7.5: g(0) = C_rho for the line (eps (mu * rho_eps) on the line)",
      0.2 * float(v.integrate_1d(lambda t: v.kernel_eps(np.stack([t, np.zeros_like(t)], axis=1), 0.2), -0.3, 0.3, n=64, pieces=16)),
      C_RHO, 1e-9)

N = 24
t24, X24, P24, m24 = v.circle_atoms(N)


def cloud_mass(y, eps, X=X24, m=m24):
    return float(np.sum(m * v.kernel_eps(np.asarray(y)[None, :] - X, eps)))


on_atom = np.array([math.cos(math.pi / N), math.sin(math.pi / N)])
between = np.array([1.0, 0.0])
close("Example 46.7.3(b): N = 24, eps = 0.15: 1.376 on an atom, 0.115 between", [0.15 * cloud_mass(on_atom, 0.15),
                                                                                   0.15 * cloud_mass(between, 0.15)],
      [1.3763, 0.1151], 5e-4)
close("Example 46.7.3(b): eps = 0.3: 0.749 on an atom, 1.088 between", [0.3 * cloud_mass(on_atom, 0.3), 0.3 * cloud_mass(between, 0.3)],
      [0.749, 1.0884], 5e-4)
ang = np.linspace(0, 2 * np.pi / N, 81)
vals = [0.5 * cloud_mass([math.cos(a), math.sin(a)], 0.5) for a in ang]
close("Example 46.7.3(b): eps = 0.5: values in [0.941, 0.981]", [min(vals), max(vals)], [0.9412, 0.9806], 5e-4)
gg = np.linspace(-1.7, 1.7, 341)
HX, HY = np.meshgrid(gg, gg)
hg = np.stack([HX.ravel(), HY.ravel()], axis=1)
tot = sum(np.sum(m24[i] * v.kernel_eps(hg - X24[i][None, :], 0.5)) for i in range(N)) * (gg[1] - gg[0]) ** 2
close("Prop 46.7.2(a): total mass 2 pi is preserved by smoothing (N = 24, eps = 0.5)", tot, 2 * math.pi, 2e-3)
# (b): int phi (mu * rho_eps) dx -> int phi d mu, via int (phi * rho_eps) d mu
phi = lambda x: np.exp(-((x[..., 0] - 0.6) ** 2 + x[..., 1] ** 2))
exact = float(v.integrate_1d(lambda t: phi(v.circle(t)), 0, 2 * np.pi, n=64, pieces=16))
errs = []
xg = np.linspace(-1.6, 1.6, 321)
QX, QY = np.meshgrid(xg, xg)
q = np.stack([QX.ravel(), QY.ravel()], axis=1)
for eps in (0.4, 0.2, 0.1):
    band = np.abs(np.linalg.norm(q, axis=1) - 1) < eps
    dens = np.zeros(len(q))
    dens[band] = circle_mass_many(q[band], eps)
    errs.append(abs(np.sum(phi(q) * dens) * (xg[1] - xg[0]) ** 2 - exact))
check("Prop 46.7.2(b): |int phi (mu * rho_eps) - int phi d mu| decreases as eps -> 0 (eps = 0.4, 0.2, 0.1)",
      errs[0] > errs[1] > errs[2], str(np.round(errs, 5)))


# ---------------------------------------------------------------- Proposition 46.7.4
def dV_conv(x, X, P, m, eps):
    """(delta V * rho_eps)(x) = sum_i m_i P_i grad rho_eps(x_i - x)."""
    gr = v.grad_kernel_eps(X - np.asarray(x)[None, :], eps)
    return np.einsum("i,ijk,ik->j", m, P, gr)


# (i) duality delta V(X * rho_eps) = int <X, delta V * rho_eps> dx for a three-atom cloud and a polynomial field
Xa = np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.25]])
Pa = v.line_proj(np.array([0.3, 1.2, 2.0]))
ma = np.array([0.5, 1.0, 0.7])
eps = 0.25
Xf = lambda x: np.stack([x[..., 0] ** 2 - x[..., 1], x[..., 0] * x[..., 1] + 2 * x[..., 1]], axis=-1)
DXf = lambda x: np.stack([np.stack([2 * x[..., 0], -np.ones_like(x[..., 0])], -1),
                          np.stack([x[..., 1], x[..., 0] + 2.0], -1)], -2)
h = 0.004
u = np.arange(-eps, eps + h / 2, h)
UX, UY = np.meshgrid(u, u)
loc = np.stack([UX.ravel(), UY.ravel()], axis=1)
wk = v.kernel_eps(loc, eps) * h * h
lhs = 0.0
for xi, Pi, mi in zip(Xa, Pa, ma):
    DXs = np.einsum("k,kab->ab", wk, DXf(xi[None, :] - loc))       # (DX * rho_eps)(x_i)
    lhs += mi * np.trace(Pi @ DXs)
w = np.arange(-0.6, 0.65, h)
WX, WY = np.meshgrid(w, w)
big = np.stack([WX.ravel(), WY.ravel()], axis=1)
F = np.array([dV_conv(y, Xa, Pa, ma, eps) for y in big])
rhs = float(np.sum(np.einsum("kj,kj->k", Xf(big), F)) * h * h)
close("Prop 46.7.4: delta V(X * rho_eps) = int <X, delta V * rho_eps> dx (three atoms, polynomial X)", lhs, rhs, 2e-4)
# (ii) smooth closed curve: delta V * rho_eps = -(H ||V||) * rho_eps (unit circle, H = -x)
for y in ([0.95, 0.1], [1.05, -0.2], [0.0, 0.9]):
    y = np.array(y)
    a = v.integrate_1d(lambda t: np.einsum("tjk,tk->tj", v.line_proj(t + np.pi / 2),
                                           v.grad_kernel_eps(v.circle(t) - y[None, :], 0.2)), 0, 2 * np.pi, n=32, pieces=256)
    b = v.integrate_1d(lambda t: v.circle(t) * v.kernel_eps(y[None, :] - v.circle(t), 0.2)[:, None], 0, 2 * np.pi, n=32,
                       pieces=256)
    close(f"Prop 46.7.4: circle: delta V * rho_eps = -(H||V||) * rho_eps at {tuple(y)}", a, b, 1e-8)

# ---------------------------------------------------------------- Example 46.7.5: ellipse
L = v.ellipse_length_total()
close("Example 46.7.5: ellipse length L = 9.6884", L, 9.6884, 5e-5)
tg = np.linspace(0, 2 * np.pi, 200001)
sg = np.concatenate([[0.0], np.cumsum(0.5 * (v.ellipse_speed(tg[1:]) + v.ellipse_speed(tg[:-1])) * np.diff(tg))])
sg *= L / sg[-1]


def cloud(Nc):
    t = np.interp(np.arange(Nc) * L / Nc, sg, tg)
    return t, v.ellipse(t), v.dir_proj(v.ellipse_tangent(t)), np.full(Nc, L / Nc)


def ratio(x, X, P, m, e):
    d = X - x[None, :]
    sel = np.linalg.norm(d, axis=1) < e
    if not np.any(np.linalg.norm(d[sel], axis=1) > 0):
        return np.zeros(2)
    num = np.einsum("i,ijk,ik->j", m[sel], P[sel], v.grad_kernel_eps(d[sel], e))
    den = np.sum(m[sel] * v.kernel_eps(d[sel], e))
    return -num / den


def ratio_smooth(t0, e):
    x = v.ellipse(t0)

    def fnum(t):
        T = v.ellipse_tangent(t)
        gr = v.grad_kernel_eps(v.ellipse(t) - x[None, :], e)
        return T * np.sum(T * gr, axis=1, keepdims=True) * v.ellipse_speed(t)[:, None]

    num = v.integrate_1d(fnum, 0, 2 * np.pi, n=64, pieces=512)
    den = v.integrate_1d(lambda t: v.kernel_eps(v.ellipse(t) - x[None, :], e) * v.ellipse_speed(t), 0, 2 * np.pi, n=64, pieces=512)
    return -num / den


close("Example 46.7.5: smooth ellipse, vertex value for eps = 0.05, 0.1, 0.3 is 1.995, 1.980, 1.840",
      [np.linalg.norm(ratio_smooth(0.0, e)) for e in (0.05, 0.1, 0.3)], [1.9949, 1.9798, 1.8400], 5e-4)
t1, X1, P1, m1 = cloud(1024)
k1 = v.ellipse_kappa(t1)
res = {}
for e in (0.03, 0.1, 0.4):
    R = np.array([ratio(X1[k], X1, P1, m1, e) for k in range(1024)])
    res[e] = R
check("Example 46.7.5: every ratio vector points inward (N = 1024, eps = 0.03, 0.1, 0.4)",
      all(np.all(np.einsum("ij,ij->i", R, v.ellipse_normal(t1)) > 0) for R in res.values()))
rel = lambda R: np.max(np.abs(np.linalg.norm(R, axis=1) - k1) / k1)
close("Example 46.7.5: N = 1024, eps = 0.1: largest relative error 1.5 percent", rel(res[0.1]), 0.0154, 5e-4)
close("Example 46.7.5: N = 1024, eps = 0.03: vertex value 1.448 (28 percent low)", np.linalg.norm(res[0.03][0]), 1.4478, 5e-4)
close("Example 46.7.5: N = 1024, eps = 0.4: vertex value 1.742 (13 percent low), largest error 0.129",
      [np.linalg.norm(res[0.4][0]), rel(res[0.4])], [1.7416, 0.1292], 5e-4)
t2, X2, P2, m2 = cloud(256)
close("Example 46.7.5: N = 256 spacing h = 0.0378", L / 256, 0.0378, 5e-5)
check("Example 46.7.5: N = 256, eps < h: no neighbour, the ratio is 0 everywhere",
      all(np.allclose(ratio(X2[k], X2, P2, m2, 0.035), 0) for k in range(0, 256, 7)))

# ---------------------------------------------------------------- Example 46.7.8: Dirac masses
for hh in (0.3, 0.8, 2.0, 3.0):
    phi_opt = lambda x, hh=hh: np.clip(hh / 2 - x, -1, 1)
    close(f"Example 46.7.8(a): phi(0) - phi(h) = min(h, 2) for h = {hh}", phi_opt(0.0) - phi_opt(hh), min(hh, 2.0), 1e-12)
    # no admissible test function does better: the Lipschitz and sup bounds
    ab = rng.normal(size=(40, 2))
    ab[:, 1] = np.clip(ab[:, 1], -1, 1)                      # slope at most 1: admissible
    tests = [lambda x, a=a, b=b: np.clip(a + b * np.sin(x) * 0.999, -1, 1) for a, b in ab]
    best = max(abs(f(0.0) - f(hh)) for f in tests)
    check(f"Example 46.7.8(a): random admissible tests stay below min(h, 2) (h = {hh})", best <= min(hh, 2.0) + 1e-12)
close("Example 46.7.8(b): Delta(a delta_0, b delta_0) = |a - b| (phi = 1)", abs(2.0 - 0.5), 1.5, 0)
for th in (0.2, 0.7, 1.3):
    Pq, Qq = v.line_proj(np.array(th)), v.line_proj(np.array(0.0))
    close(f"Example 46.7.8(c): ||P - Q|| = |sin theta| (theta = {th})", v.op_dist(Pq, Qq), abs(math.sin(th)), 1e-12)
Pn = v.plane_proj(np.array([0.0, 0.0, 1.0]))
Qn = v.plane_proj(np.array([math.sin(math.radians(10)), 0.0, math.cos(math.radians(10))]))
close("Exercise 46.7.2: planes 10 degrees apart: operator 0.1736, Frobenius 0.2456",
      [v.op_dist(Pn, Qn), v.fro_dist(Pn, Qn)], [0.1736, 0.2456], 5e-5)
close("Exercise 46.7.1: Delta(delta_0, (delta_0 + delta_h)/2) = min(h, 2)/2 via the same test function (h = 0.8)",
      0.5 * (np.clip(0.4 - 0.0, -1, 1) - np.clip(0.4 - 0.8, -1, 1)), 0.4, 1e-12)

# ---------------------------------------------------------------- Proposition 46.7.9 and Example 46.7.10
for Nn, val in ((8, 2.4477), (16, 1.2312), (32, 0.6165), (64, 0.3084)):
    close(f"Example 46.7.10: exact value N = {Nn}: {val}", v.circle_cell_bound_exact(Nn), val, 5e-5)
    close(f"Example 46.7.10: closed form = quadrature of (46.7.6) (N = {Nn})", v.circle_cell_bound(Nn), v.circle_cell_bound_exact(Nn), 1e-12)
close("Example 46.7.10: N Delta -> 2 pi^2 (N = 128: 19.738)", 128 * v.circle_cell_bound_exact(128), 2 * math.pi ** 2, 2e-3)
check("Example 46.7.10: the nearest-atom value stays below 1 for N >= 7 (and not for N = 6)",
      2 * math.sin(math.pi / 14) + math.sin(math.pi / 7) < 1 < 2 * math.sin(math.pi / 12) + math.sin(math.pi / 6))
for Nn in (8, 16, 32):
    t, X, P, m = v.circle_atoms(Nn)

    def phi_star(x, Pm, X=X, P=P):
        d = np.linalg.norm(x[:, None, :] - X[None, :, :], axis=2)
        op = np.abs(np.linalg.eigvalsh(Pm[:, None, :, :] - P[None, :, :, :])).max(axis=2)
        return np.minimum(1.0, (d + op).min(axis=1))

    gap = v.circle_varifold_integral(phi_star, nq=16, pieces=Nn * 8) - v.atoms_integral(phi_star, X, P, m)
    close(f"Example 46.7.10: phi* attains the bound (N = {Nn})", gap, v.circle_cell_bound_exact(Nn), 1e-9)
    tests = v.random_lipschitz_tests(12, seed=7)
    worst = max(abs(v.circle_varifold_integral(f, nq=16, pieces=Nn * 4) - v.atoms_integral(f, X, P, m)) for f in tests)
    check(f"Example 46.7.10: random affine tests see far less (N = {Nn}: {worst:.4f} < 0.05)", worst < 0.05)
# Prop 46.7.9 on a non-uniform partition of the ellipse with wrong masses: random Lipschitz tests stay below the bound
edges = np.sort(np.concatenate([[0.0], rng.uniform(0, 2 * np.pi, size=29)]))
edges = np.append(edges, 2 * np.pi)
mid = 0.5 * (edges[:-1] + edges[1:])
Xc, Pc = v.ellipse(mid), v.dir_proj(v.ellipse_tangent(mid))
lens = np.array([v.ellipse_length(a, b) for a, b in zip(edges[:-1], edges[1:])])
mc = lens * (1 + 0.05 * rng.normal(size=lens.size))
bound = 0.0
for a, b, xi, Pi in zip(edges[:-1], edges[1:], Xc, Pc):
    bound += float(v.integrate_1d(lambda t: (np.linalg.norm(v.ellipse(t) - xi, axis=1)
                                             + np.array([v.op_dist(Q, Pi) for Q in v.dir_proj(v.ellipse_tangent(t))]))
                                  * v.ellipse_speed(t), a, b, n=32, pieces=4))
bound += float(np.sum(np.abs(mc - lens)))
tests = v.random_lipschitz_tests(16, seed=11)
ell_int = lambda f: float(v.integrate_1d(lambda t: f(v.ellipse(t), v.dir_proj(v.ellipse_tangent(t))) * v.ellipse_speed(t),
                                         0, 2 * np.pi, n=32, pieces=128))
gaps = [abs(ell_int(f) - v.atoms_integral(f, Xc, Pc, mc)) for f in tests]
check("Prop 46.7.9: random Lipschitz tests on a non-uniform ellipse partition stay below (46.7.6)", max(gaps) <= bound,
      f"max gap {max(gaps):.4f}, bound {bound:.4f}")
close("Exercise 46.7.3: chord masses add N(2 pi/N - 2 sin(pi/N)) = 0.0403 (N = 16)", 2 * math.pi - 32 * math.sin(math.pi / 16), 0.0403,
      5e-5)
close("Exercise 46.7.3: about pi^3 / (3 N^2) (N = 16)", 2 * math.pi - 32 * math.sin(math.pi / 16), math.pi ** 3 / (3 * 256), 2e-4)
close("Exercise 46.7.3: bound 1.2312 + 0.0403 = 1.2715", v.circle_cell_bound_exact(16) + 2 * math.pi - 32 * math.sin(math.pi / 16), 1.2715,
      5e-5)

# ---------------------------------------------------------------- Example 46.7.11: staircase
for n in (4, 16):
    S = v.staircase(n)
    distD = lambda x, Pm: np.minimum(1.0, np.abs(x[..., 0] - x[..., 1]) / math.sqrt(2))
    close(f"Example 46.7.11: int dist(x, D) over S_n = 1/(sqrt2 n) (n = {n})", v.polyline_integral(distD, S, closed=False, nq=16),
          1 / (math.sqrt(2) * n), 1e-12)
    PD = v.line_proj(np.array(math.pi / 4))
    tiltD = lambda x, Pm: np.array([v.op_dist(Q, PD) for Q in Pm])
    close(f"Example 46.7.11: int ||S - P_D|| over S_n = sqrt2 (n = {n})", v.polyline_integral(tiltD, S, closed=False, nq=4),
          math.sqrt(2), 1e-12)
    close(f"Example 46.7.11: the upper bound: sum of displacements = 1/n (n = {n})", 2 * n * (1 / n) ** 2 / 2, 1 / n, 1e-15)
    # random tests: |v(S_n)(phi) - V_inf(phi)| <= 1/n
    tests = v.random_lipschitz_tests(10, seed=13 + n)
    worst = max(abs(v.polyline_integral(f, S, closed=False, nq=16) - v.staircase_limit_integral(f)) for f in tests)
    check(f"Example 46.7.11: random Lipschitz tests: |v(S_n)(phi) - V_inf(phi)| <= 1/n (n = {n})", worst <= 1 / n + 1e-12,
          f"{worst:.4f}")
close("Example 46.7.11: Hausdorff distance of S_n to the diagonal is 1/(sqrt2 n) (n = 4)",
      v.hausdorff_polyline_to_segment(v.staircase(4), np.array([0.0, 0.0]), np.array([1.0, 1.0])), 1 / (4 * math.sqrt(2)), 1e-9)
close("Example 46.7.11: mass test phi = 1 gives only 2 - sqrt2 = 0.5858 against v(D, 1)", 2 - math.sqrt(2), 0.5858, 5e-5)

summary()
