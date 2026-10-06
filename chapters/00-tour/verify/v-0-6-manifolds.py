"""Verification for Section 0.6 (Manifolds: Spaces That Look Flat Up Close).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-6-manifolds.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, summary, sym_equal

# ---------------------------------------------------------------- Example 0.6.2: stereographic charts of S²
st = dgsym.stereographic(2)
X, U = st["coords"], st["chart_coords"]
u1, u2 = U
sub_inv = dict(zip(X, list(st["sigma_inv"])))
check("Example 0.6.2: sigma^{-1}(u) lies on S^2",
      sp.simplify(sum(c ** 2 for c in st["sigma_inv"]) - 1) == 0)
sym_equal("Example 0.6.2: sigma(sigma^{-1}(u)) = u", st["sigma"].subs(sub_inv), sp.Matrix(U))
sym_equal("(0.6.2): transition sigma~ o sigma^{-1}(u) = u/|u|^2", st["sigma_south"].subs(sub_inv), st["transition"])
sym_equal("(0.6.2): sigma(x) = (x, y)/(1 - z)", st["sigma"],
          sp.Matrix([X[0] / (1 - X[2]), X[1] / (1 - X[2])]))

# Figure 0.6.2: a circle of latitude with colatitude θ has radius cot(θ/2) under σ and tan(θ/2) under σ̃
th, lon = sp.symbols("theta lon", real=True)
pt = {X[0]: sp.sin(th) * sp.cos(lon), X[1]: sp.sin(th) * sp.sin(lon), X[2]: sp.cos(th)}
ru2 = sp.simplify(sum(c ** 2 for c in st["sigma"].subs(pt)))
rv2 = sp.simplify(sum(c ** 2 for c in st["sigma_south"].subs(pt)))
sym_equal("Figure 0.6.2: |sigma| = cot(theta/2)", ru2, sp.cot(th / 2) ** 2, {th: (0, sp.pi)})
sym_equal("Figure 0.6.2: |sigma~| = tan(theta/2)", rv2, sp.tan(th / 2) ** 2, {th: (0, sp.pi)})
xpt = {X[0]: sp.sqrt(3) / 2 * sp.sqrt(3) / 2, X[1]: sp.sqrt(3) / 2 / 2, X[2]: sp.Rational(1, 2)}
sym_equal("Figure 0.6.2: sigma(x) = (3/2, sqrt3/2)", st["sigma"].subs(xpt), sp.Matrix([sp.Rational(3, 2), sp.sqrt(3) / 2]))
sym_equal("Figure 0.6.2: sigma~(x) = (1/2, sqrt3/6)", st["sigma_south"].subs(xpt),
          sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 6]))

# (0.6.4): components of a tangent vector change by the Jacobian of the transition map
t = sp.symbols("t", real=True)
gamma = sp.Matrix([sp.sin(1 + t) * sp.cos(2 * t), sp.sin(1 + t) * sp.sin(2 * t), sp.cos(1 + t)])  # a curve on S²
cg = dict(zip(X, list(gamma)))
uc, vc = st["sigma"].subs(cg), st["sigma_south"].subs(cg)
Jtr = st["transition"].jacobian(sp.Matrix(U))
lhs = sp.diff(vc, t).subs(t, 0)
rhs = Jtr.subs(dict(zip(U, list(uc.subs(t, 0))))) * sp.diff(uc, t).subs(t, 0)
close("(0.6.4): v-components = D(transition) u-components", np.array(sp.N(lhs - rhs), dtype=float).ravel(),
      np.zeros(2), tol=1e-12)

# Exercise 0.6.1: S¹ stereographic projections at (3/5, 4/5)
xs, ys = sp.Rational(3, 5), sp.Rational(4, 5)
check("Exercise 0.6.1: sigma(3/5, 4/5) = 3", xs / (1 - ys) == 3)
check("Exercise 0.6.1: sigma~(3/5, 4/5) = 1/3", xs / (1 + ys) == sp.Rational(1, 3))
check("Exercise 0.6.1: product = 1", (xs / (1 - ys)) * (xs / (1 + ys)) == 1)

# ---------------------------------------------------------------- Example 0.6.4: RP² charts
rp = dgsym.rpn_charts(2)
# φ_3 (index 2) and φ_1 (index 0): φ_1 ∘ φ_3^{-1}(u) = (u²/u¹, 1/u¹)
sym_equal("(0.6.3): phi_1 o phi_3^{-1}(u) = (u2/u1, 1/u1)", rp["transition"](2, 0), sp.Matrix([u2 / u1, 1 / u1]))
sym_equal("(0.6.3): phi_3[x:y:w] = (x/w, y/w)", rp["phi"](2), sp.Matrix([X[0] / X[2], X[1] / X[2]]))
check("Exercise 0.6.2: phi_3[2:4:2] = (1, 2)", list(rp["phi"](2).subs(dict(zip(X, (2, 4, 2))))) == [1, 2])
check("Exercise 0.6.2: phi_3[1:2:1] = (1, 2)", list(rp["phi"](2).subs(dict(zip(X, (1, 2, 1))))) == [1, 2])
check("Exercise 0.6.2: phi_1[1:2:0] = (2, 0)", list(rp["phi"](0).subs(dict(zip(X, (1, 2, 0))))) == [2, 0])
# a homography is well defined on RP²: H(λx) = λ Hx
H = sp.Matrix([[1, 2, 0], [0, 1, 3], [1, 0, 1]])
lam = sp.symbols("lambda", nonzero=True)
xv = sp.Matrix([1, 2, 1])
sym_equal("Remark (homography): phi_3[H(lambda x)] = phi_3[Hx]",
          rp["phi"](2).subs(dict(zip(X, list(H * (lam * xv))))), rp["phi"](2).subs(dict(zip(X, list(H * xv)))))

# ---------------------------------------------------------------- (0.6.5) chain rule; Exercise 0.6.4
x, y = sp.symbols("x y", real=True)
uu, vv = sp.symbols("u v", real=True)
F = sp.Matrix([x ** 2, x * y])
G = sp.Matrix([uu + vv])
DF = F.jacobian([x, y])
DG = G.jacobian([uu, vv]).subs({uu: F[0], vv: F[1]})
comp = G.subs({uu: F[0], vv: F[1]})
sym_equal("(0.6.5): D(G o F) = DG(F) DF", comp.jacobian([x, y]), DG * DF)
check("Exercise 0.6.4: D(G o F)(1,2) = [4 1]", list((DG * DF).subs({x: 1, y: 2})) == [4, 1])
check("Exercise 0.6.4: DF(1,2) = [[2,0],[2,1]]", DF.subs({x: 1, y: 2}) == sp.Matrix([[2, 0], [2, 1]]))
vjp = sp.Matrix([[1, 1]]) * DF.subs({x: 1, y: 2})
jvp = DF.subs({x: 1, y: 2}) * sp.Matrix([1, 0])
check("Exercise 0.6.4: VJP [1 1] DF = [4 1]", list(vjp) == [4, 1])
check("Exercise 0.6.4: JVP DF e1 = (2, 2)", list(jvp) == [2, 2])

# ---------------------------------------------------------------- Example 0.6.6: flow of X = (-y, x, 0) on S²
tt, ss = sp.symbols("t s", real=True)


def Rz(a):
    return sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])


def Ry(a):
    return sp.Matrix([[sp.cos(a), 0, sp.sin(a)], [0, 1, 0], [-sp.sin(a), 0, sp.cos(a)]])


def Rx(a):
    return sp.Matrix([[1, 0, 0], [0, sp.cos(a), -sp.sin(a)], [0, sp.sin(a), sp.cos(a)]])


p0 = sp.Matrix(sp.symbols("p1:4", real=True))
traj = Rz(tt) * p0
Xf = lambda q: sp.Matrix([-q[1], q[0], 0])  # noqa: E731
sym_equal("Example 0.6.6: d/dt theta_t(p) = X(theta_t(p))", sp.diff(traj, tt), Xf(traj))
sym_equal("Example 0.6.6: theta_s o theta_t = theta_{s+t}", Rz(ss) * Rz(tt), Rz(ss + tt))

# ---------------------------------------------------------------- Section on SO(3)
def hat(w):
    return sp.Matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


w = sp.Matrix(sp.symbols("w1:4", real=True))
vv3 = sp.Matrix(sp.symbols("v1:4", real=True))
sym_equal("(0.6.6): [w]_x v = w x v", hat(w) * vv3, w.cross(vv3))

# tangent space at I: d/dt R(t)^T R(t) = 0 at t = 0 gives a skew-symmetric matrix; example R(t) = R_z(t)
dR0 = sp.diff(Rz(tt), tt).subs(tt, 0)
check("Exercise 0.6.3: R_z'(0) = [e3]_x", dR0 == hat(sp.Matrix([0, 0, 1])))
check("Section SO(3): R_z'(0) is skew-symmetric", dR0.T + dR0 == sp.zeros(3))

# dimension 3: the map R -> R^T R from R^9 to symmetric matrices (R^6) has rank 6 at points of SO(3)
rng = np.random.default_rng(6)


def rand_rot():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


ranks = []
for _ in range(5):
    R0 = rand_rot()
    cols = []
    for k in range(9):
        E = np.zeros(9)
        E[k] = 1
        Ed = E.reshape(3, 3)
        Dm = Ed.T @ R0 + R0.T @ Ed   # derivative of R^T R in the direction Ed
        cols.append(Dm[np.triu_indices(3)])
    ranks.append(np.linalg.matrix_rank(np.array(cols).T))
check("Section SO(3): rank of d(R^T R) is 6 on SO(3), so dim SO(3) = 9 - 6 = 3", all(r == 6 for r in ranks))


# Rodrigues formula (0.6.7) against the matrix exponential series
def rodrigues(wv):
    thv = np.linalg.norm(wv)
    K = np.array(hat(sp.Matrix(wv)), dtype=float)
    return np.eye(3) + np.sin(thv) / thv * K + (1 - np.cos(thv)) / thv ** 2 * K @ K


def expm(A, terms=60):
    out, term = np.eye(3), np.eye(3)
    for k in range(1, terms):
        term = term @ A / k
        out = out + term
    return out


worst = 0.0
for _ in range(20):
    wv = rng.uniform(-3, 3, 3)
    Rr = rodrigues(wv)
    worst = max(worst, np.max(np.abs(Rr - expm(np.array(hat(sp.Matrix(wv)), dtype=float)))))
    check_rot = np.allclose(Rr.T @ Rr, np.eye(3)) and np.isclose(np.linalg.det(Rr), 1) and np.allclose(Rr @ wv, wv)
    if not check_rot:
        break
check("(0.6.7): Rodrigues gives rotations fixing the axis", check_rot)
close("(0.6.7): Rodrigues formula = matrix exponential series", worst, 0.0, tol=1e-10)
close("(0.6.7): rotation angle of exp([w]_x) is |w|",
      np.arccos((np.trace(rodrigues(np.array([0.3, -1.1, 0.7]))) - 1) / 2), np.linalg.norm([0.3, -1.1, 0.7]), tol=1e-12)
th0 = sp.symbols("theta0", real=True)
e3 = sp.Matrix([0, 0, 1])
K3 = hat(e3)
sym_equal("(0.6.7): exp(t[e3]_x) = R_z(t) via Rodrigues", sp.eye(3) + sp.sin(tt) * K3 + (1 - sp.cos(tt)) * K3 * K3, Rz(tt))
close("Figure 0.6.4: exp(pi w^) = exp(-pi w^)", rodrigues(np.pi * np.array([0.6, 0, 0.8])),
      rodrigues(-np.pi * np.array([0.6, 0, 0.8])), tol=1e-12)

# Euler angles R_z(a) R_y(b) R_x(c): rank of the differential is 3 for generic b and 2 at b = ±π/2 (gimbal lock)
a, b, c = sp.symbols("alpha beta gamma", real=True)
Re = Rz(a) * Ry(b) * Rx(c)


def omega_vec(M):
    return sp.Matrix([M[2, 1], M[0, 2], M[1, 0]])


Jcols = [omega_vec(sp.simplify(sp.diff(Re, q) * Re.T)) for q in (a, b, c)]
Jeu = sp.Matrix.hstack(*Jcols)
for q in (a, b, c):
    Sk = sp.simplify(sp.diff(Re, q) * Re.T)
    check(f"CV box (Euler angles): dR/d{q} R^T is skew-symmetric", sp.simplify(Sk + Sk.T) == sp.zeros(3))
check("CV box (Euler angles): rank 3 at beta = 0.4", Jeu.subs({a: 0.3, b: 0.4, c: -0.2}).rank() == 3)
check("CV box (Euler angles): rank 2 at beta = pi/2 (gimbal lock)", sp.simplify(Jeu.subs(b, sp.pi / 2)).rank() == 2)
check("CV box (Euler angles): det of the angular-velocity matrix is -cos(beta) up to sign",
      sp.simplify(Jeu.det() ** 2 - sp.cos(b) ** 2) == 0)
sym_equal("CV box (Euler angles): at beta = pi/2 only alpha - gamma matters",
          Re.subs(b, sp.pi / 2), Rz(a - c) * Ry(sp.pi / 2))
check("CV box (Euler angles): rank 2 at beta = -pi/2 as well", sp.simplify(Jeu.subs(b, -sp.pi / 2)).rank() == 2)
sym_equal("CV box (Euler angles): at beta = -pi/2 only alpha + gamma matters (review R2)",
          Re.subs(b, -sp.pi / 2), Rz(a + c) * Ry(-sp.pi / 2))

summary()
