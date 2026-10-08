"""Verification for Section 28.1 (SO(3), SE(3), and Sim(3)).

Run from the project root: ``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/verify/v-28-1-so3-se3-sim3.py``

The closed forms are compared with an independent matrix exponential (scaling and squaring of the Taylor series,
written here, not the one in lib/dglie.py) and, for the series coefficients, with sympy.
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import dglie as L  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

rng = np.random.default_rng(281)


def expm_ref(A, terms=40):
    """Independent matrix exponential: scaling and squaring with a long Taylor series."""
    A = np.asarray(A, float)
    k = 0
    while np.linalg.norm(A, 1) / 2 ** k > 0.25:
        k += 1
    B = A / 2 ** k
    E, term = np.eye(len(A)), np.eye(len(A))
    for n in range(1, terms):
        term = term @ B / n
        E = E + term
    for _ in range(k):
        E = E @ E
    return E


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


# ---------------------------------------------------------------- series coefficients (sympy)
th = sp.symbols("theta", positive=True)
N = 14
c1 = sum((-1) ** j * th ** (2 * j) / sp.factorial(2 * j + 1) for j in range(N))
c2 = sum((-1) ** j * th ** (2 * j) / sp.factorial(2 * j + 2) for j in range(N))
c3 = sum((-1) ** j * th ** (2 * j) / sp.factorial(2 * j + 3) for j in range(N))
for name, ser, closed in [("Rodrigues: coefficient of Omega = sin(th)/th", c1, sp.sin(th) / th),
                          ("(28.1.4): coefficient of Omega in V = (1 - cos th)/th^2", c2, (1 - sp.cos(th)) / th ** 2),
                          ("(28.1.4): coefficient of Omega^2 in V = (th - sin th)/th^3", c3, (th - sp.sin(th)) / th ** 3)]:
    ok = all(abs(float((ser - closed).subs(th, x))) < 1e-12 for x in [0.3, 1.0, 2.0, 3.0])
    check(name + " (series vs closed form, 14 terms)", ok)
sym_equal("Omega^3 = -theta^2 Omega (symbolic)",
          (lambda W: W ** 3 + (W[2, 1] ** 2 + W[0, 2] ** 2 + W[1, 0] ** 2) * W)(
              sp.Matrix([[0, -sp.Symbol("c"), sp.Symbol("b")], [sp.Symbol("c"), 0, -sp.Symbol("a")],
                         [-sp.Symbol("b"), sp.Symbol("a"), 0]])), sp.zeros(3, 3))
check("Taylor: th/(2 sin th) = 1/2 + th^2/12 + O(th^4)",
      sp.simplify(sp.series(th / (2 * sp.sin(th)), th, 0, 4).removeO() - (sp.Rational(1, 2) + th ** 2 / 12)) == 0)
check("Taylor: sin(th/2)/th = 1/2 - th^2/48 + th^4/3840 + O(th^6) (Sophus SO3::expAndTheta)",
      sp.simplify(sp.series(sp.sin(th / 2) / th, th, 0, 6).removeO()
                  - (sp.Rational(1, 2) - th ** 2 / 48 + th ** 4 / 3840)) == 0)
check("Taylor: (1 - (th/2) cot(th/2))/th^2 -> 1/12 (V^{-1}, Sophus leftJacobianInverse small-angle form)",
      sp.limit((1 - th / 2 * sp.cot(th / 2)) / th ** 2, th, 0) == sp.Rational(1, 12))

# ---------------------------------------------------------------- exp closed forms vs matrix exponential
angles = [1e-9, 1e-6, 1e-3, 0.4, 1.3, 2.7, np.pi - 1e-6, 3.0 * np.pi / 2]
worst = [0.0, 0.0, 0.0]
for a in angles:
    for _ in range(3):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        w = a * n
        v = rng.normal(size=3)
        for sig in [0.0, 1e-9, 0.3, -1.2, 2.5]:
            z = np.concatenate([v, w, [sig]])
            worst[2] = max(worst[2], np.abs(L.expSim3(z) - expm_ref(L.hat7(z))).max())
        worst[0] = max(worst[0], np.abs(L.expSO3(w) - expm_ref(hat(w))).max())
        xi = np.concatenate([v, w])
        worst[1] = max(worst[1], np.abs(L.expSE3(xi) - expm_ref(L.hat6(xi))).max())
close("(0.6.7) Rodrigues = matrix exponential (theta 1e-9 .. 3pi/2)", worst[0], 0.0, tol=1e-13)
close("(28.1.4) SE(3) exp = matrix exponential", worst[1], 0.0, tol=1e-12)
close("(28.1.5) Sim(3) exp (stable evaluation of W) = matrix exponential (theta, sigma down to 1e-9)", worst[2], 0.0,
      tol=1e-12)
x = 1e-9
close("Section 28.1: computing e^sigma - 1 directly at sigma = 1e-9 has relative error ~ 1e-7",
      (np.exp(x) - 1) / np.expm1(x) - 1, 8.2e-8, tol=1e-8)
for _ in range(5):
    w, sig = rng.normal(size=3), rng.normal()
    close("(28.1.5): paper formula (Strasdat10 (22)) = stable evaluation for generic theta, sigma",
          L.W_Sim3_formula(w, sig), L.W_Sim3(w, sig), tol=1e-13)


def W_strasdat(w, sig):
    """[Strasdat10] (22) written out independently."""
    t = np.linalg.norm(w)
    K = hat(w)
    a, b, c = np.exp(sig) * np.sin(t), np.exp(sig) * np.cos(t), (np.exp(sig) - 1) / sig
    return ((a * sig + (1 - b) * t) / (t * (sig ** 2 + t ** 2)) * K
            + (c - ((b - 1) * sig + a * t) / (sig ** 2 + t ** 2)) * K @ K / t ** 2 + c * np.eye(3))


for _ in range(5):
    w = rng.normal(size=3)
    sig = rng.normal()
    Wnum = np.column_stack([expm_ref(L.hat7(np.concatenate([e, w, [sig]])))[:3, 3] for e in np.eye(3)])
    close("(28.1.5): Strasdat10 (22) = top-right block of expm (columns = images of e_k)", W_strasdat(w, sig), Wnum,
          tol=1e-12)
    close("(28.1.5): W -> V as sigma -> 0", L.W_Sim3(w, 1e-10), L.Jl(w), tol=1e-8)
# V = int_0^1 exp(u Omega) du  (midpoint rule)
w = np.array([0.4, -1.1, 0.7])
us = (np.arange(4000) + 0.5) / 4000
close("Derivation of (28.1.4): V = int_0^1 exp(u Omega) du", np.mean([L.expSO3(u * w) for u in us], axis=0), L.Jl(w),
      tol=1e-7)

# ---------------------------------------------------------------- group laws, dimensions
R1, R2 = L.random_rotation(rng), L.random_rotation(rng)
t1, t2 = rng.normal(size=3), rng.normal(size=3)
T1, T2 = L.make_T(R1, t1), L.make_T(R2, t2)
close("SE(3) product (R1R2, R1 t2 + t1)", T1 @ T2, L.make_T(R1 @ R2, R1 @ t2 + t1), tol=1e-14)
close("SE(3) inverse (R^T, -R^T t)", L.inv_T(T1) @ T1, np.eye(4), tol=1e-14)
s1, s2 = 1.7, 0.4
S1 = np.eye(4); S1[:3, :3] = s1 * R1; S1[:3, 3] = t1
S2 = np.eye(4); S2[:3, :3] = s2 * R2; S2[:3, 3] = t2
P = S1 @ S2
close("Exercise 28.1.1: S1 S2 = (s1 s2 R1 R2, s1 R1 t2 + t1)", P[:3, :3], s1 * s2 * R1 @ R2, tol=1e-14)
close("Exercise 28.1.1: translation of S1 S2", P[:3, 3], s1 * R1 @ t2 + t1, tol=1e-14)
S1inv = np.eye(4); S1inv[:3, :3] = R1.T / s1; S1inv[:3, 3] = -R1.T @ t1 / s1
close("Exercise 28.1.1: S1^{-1} = (R1^T/s1, -R1^T t1/s1)", S1 @ S1inv, np.eye(4), tol=1e-14)
# Lie algebra dimensions: rank of the hat maps
H6 = np.array([L.hat6(e).ravel() for e in np.eye(6)])
H7 = np.array([L.hat7(e).ravel() for e in np.eye(7)])
check("dim se(3) = 6 (hat is injective)", np.linalg.matrix_rank(H6) == 6)
check("dim sim(3) = 7 (hat is injective)", np.linalg.matrix_rank(H7) == 7)
# Exercise 28.1.1: similarity of the whole scene leaves projections unchanged
Rc, tc = L.random_rotation(rng), rng.normal(size=3) + np.array([0, 0, 5.0])
X = rng.normal(size=(6, 3))
Q, s, u = L.random_rotation(rng), 2.3, rng.normal(size=3)
Xn = (s * (Q @ X.T)).T + u
Rn, tn = Rc @ Q.T, s * tc - Rc @ Q.T @ u
xc, xcn = (Rc @ X.T).T + tc, (Rn @ Xn.T).T + tn
close("Exercise 28.1.1: camera coordinates scale by s under a scene similarity", xcn, s * xc, tol=1e-12)
close("Exercise 28.1.1: projections are unchanged", xcn[:, :2] / xcn[:, 2:], xc[:, :2] / xc[:, 2:], tol=1e-12)

# ---------------------------------------------------------------- twist = velocity field (28.1.3)
xi = rng.normal(size=6)
x = rng.normal(size=3)
h = 1e-6
num = (L.expSE3(h * xi)[:3, :3] @ x + L.expSE3(h * xi)[:3, 3] - L.expSE3(-h * xi)[:3, :3] @ x
       - L.expSE3(-h * xi)[:3, 3]) / (2 * h)
close("(28.1.3): d/dtau Exp(tau xi) x = w x x + v", num, np.cross(xi[3:], x) + xi[:3], tol=1e-8)

# ---------------------------------------------------------------- screw motion (Example 28.1.4, Figure 28.1.1)
a, b = 1.0, 0.3
xi = np.array([0, a, b, 0, 0, 1.0])
for tau in np.linspace(0, 2 * np.pi, 13):
    close(f"Example 28.1.4: c({tau:.2f}) = helix - (a,0,0)", L.expSE3(tau * xi)[:3, 3],
          [a * np.cos(tau) - a, a * np.sin(tau), b * tau], tol=1e-12)
close("Example 28.1.4: c(pi) = (-2, 0, 0.942)", L.expSE3(np.pi * xi)[:3, 3], [-2, 0, 0.3 * np.pi], tol=1e-12)
close("Example 28.1.4: 0.3 pi = 0.942", 0.3 * np.pi, 0.9425, tol=5e-5)
q = np.cross(xi[3:], xi[:3]) / (xi[3:] @ xi[3:])
close("Example 28.1.4: axis point q = w x v/|w|^2 = (-a, 0, 0)", q, [-a, 0, 0], tol=1e-15)
close("Example 28.1.4: pitch h = <w, v>/|w|^2 = b", xi[3:] @ xi[:3], b, tol=1e-15)
for _ in range(4):
    w, v = rng.normal(size=3), rng.normal(size=3)
    qq, hh = np.cross(w, v) / (w @ w), (w @ v) / (w @ w)
    close("Chasles decomposition: v = q x w + h w", np.cross(qq, w) + hh * w, v, tol=1e-12)
    close("Chasles: the axis point q moves only along the axis", np.cross(w, qq) + v, hh * w, tol=1e-12)
    for tau in [0.7, 2.0]:
        T = L.expSE3(tau * np.concatenate([v, w]))
        p_ax = qq + 0.37 * w
        close("Chasles: points on the axis slide by tau h w", T[:3, :3] @ p_ax + T[:3, 3], p_ax + tau * hh * w,
              tol=1e-11)
# Exercise 28.1.4
for tau in [0.5, 2.0, 5.0]:
    T = L.expSE3(tau * np.array([0, 0, b, 0, 0, 1.0]))
    close("Exercise 28.1.4: Exp(tau((0,0,b),(0,0,1))) = (R_z(tau), (0,0,b tau))", T,
          L.make_T(L.expSO3([0, 0, tau]), [0, 0, b * tau]), tol=1e-12)
w = rng.normal(size=3)
close("Exercise 28.1.4: V v = v when v || w", L.Jl(w) @ (2.5 * w), 2.5 * w, tol=1e-13)

# ---------------------------------------------------------------- log (Proposition 28.1.6, Figures 28.1.2, 28.1.3)
for _ in range(20):
    R = L.random_rotation(rng)
    wv = L.logSO3(R)
    close("Prop 28.1.6: Exp(Log R) = R", L.expSO3(wv), R, tol=1e-13)
    check("Prop 28.1.6: |Log R| <= pi", np.linalg.norm(wv) <= np.pi + 1e-14)
    tr_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    close("Prop 28.1.6: cos theta = (tr R - 1)/2", np.linalg.norm(wv), tr_angle, tol=1e-7)
    if 0.1 < tr_angle < np.pi - 0.1:
        close("(28.1.6): [w]_x = theta/(2 sin theta)(R - R^T)", hat(wv),
              tr_angle / (2 * np.sin(tr_angle)) * (R - R.T), tol=1e-10)
n = rng.normal(size=3)
n /= np.linalg.norm(n)
close("Prop 28.1.6: Exp(pi n) = Exp(-pi n)", L.expSO3(np.pi * n), L.expSO3(-np.pi * n), tol=1e-14)
close("Prop 28.1.6: Exp(pi n) = 2 n n^T - I", L.expSO3(np.pi * n), 2 * np.outer(n, n) - np.eye(3), tol=1e-14)
close("Prop 28.1.6: (R + I)/2 = n n^T at theta = pi", (L.expSO3(np.pi * n) + np.eye(3)) / 2, np.outer(n, n),
      tol=1e-14)
nr = np.array([np.cos(np.radians(35)), 0, np.sin(np.radians(35))])
for t in [0.5, 2.0, 3.1, 3.2, 4.5, 6.0]:
    expected = t * nr if t < np.pi else (t - 2 * np.pi) * nr
    close(f"Figure 28.1.2: Log(Exp({t} n))", L.logSO3(L.expSO3(t * nr)), expected, tol=1e-12)
# naive formula failure (Figure 28.1.3)
Rs = L.expSO3(1e-9 * n)
check("Figure 28.1.3: textbook log returns 0 for theta = 1e-9", np.allclose(L.logSO3_naive(Rs), 0))
close("Figure 28.1.3: robust log at theta = 1e-9", L.logSO3(Rs), 1e-9 * n, tol=1e-23)
check("Figure 28.1.3: cos(theta) rounds to exactly 1 for theta < sqrt(eps/2) = 1.05e-8 (1.05e-8 yes, 1.06e-8 no)",
      np.cos(1.05e-8) == 1.0 and np.cos(1.06e-8) != 1.0)
close("Figure 28.1.3: sqrt(eps/2) = 1.05e-8", np.sqrt(np.finfo(float).eps / 2), 1.054e-8, tol=1e-11)
Rs2 = L.expSO3(3e-8 * n)
check("Figure 28.1.3: once arccos is nonzero the textbook formula is accurate again (theta = 3e-8)",
      np.linalg.norm(L.logSO3_naive(Rs2) - 3e-8 * n) < 1e-14 * 3e-8)
# SE(3) log and V^{-1}, det V (Exercise 28.1.5)
for _ in range(10):
    w = rng.normal(size=3)
    w *= min(1.0, 3.0 / np.linalg.norm(w))
    th_ = np.linalg.norm(w)
    K = hat(w)
    Vinv = np.eye(3) - 0.5 * K + (1 - th_ / 2 / np.tan(th_ / 2)) / th_ ** 2 * K @ K
    close("(28.1.7): V^{-1} V = I", Vinv @ L.Jl(w), np.eye(3), tol=1e-12)
    close("(28.1.7) / Exercise 28.1.5: det V = 2(1 - cos th)/th^2", np.linalg.det(L.Jl(w)),
          2 * (1 - np.cos(th_)) / th_ ** 2, tol=1e-12)
    xi = np.concatenate([rng.normal(size=3), w])
    close("Prop 28.1.6: Log(Exp(xi)) = xi on SE(3) (|w| < pi)", L.logSE3(L.expSE3(xi)), xi, tol=1e-11)
    z = np.concatenate([rng.normal(size=3), w, [rng.normal()]])
    close("Log(Exp(zeta)) = zeta on Sim(3)", L.logSim3(L.expSim3(z)), z, tol=1e-10)
for k in [1, 2]:
    w = 2 * np.pi * k * n
    check(f"Exercise 28.1.5: V is singular at theta = {2 * k}pi", abs(np.linalg.det(L.Jl(w))) < 1e-12)
ths = np.linspace(1e-3, 2 * np.pi - 1e-3, 500)
check("Exercise 28.1.5: det V > 0 on (0, 2pi)", np.all(2 * (1 - np.cos(ths)) / ths ** 2 > 0))

# ---------------------------------------------------------------- adjoint (Definition 28.1.7, Figure 28.1.4)
for _ in range(10):
    T = L.expSE3(rng.normal(size=6) * 1.5)
    xi = rng.normal(size=6)
    close("(28.1.8): T Exp(xi) T^{-1} = Exp(Ad_T xi)", T @ L.expSE3(xi) @ L.inv_T(T), L.expSE3(L.Ad_SE3(T) @ xi),
          tol=1e-11)
    close("(28.1.8): T xi^ T^{-1} = (Ad_T xi)^", T @ L.hat6(xi) @ L.inv_T(T), L.hat6(L.Ad_SE3(T) @ xi), tol=1e-12)
    T2 = L.expSE3(rng.normal(size=6))
    close("Def 28.1.7: Ad_{T1 T2} = Ad_T1 Ad_T2", L.Ad_SE3(T @ T2), L.Ad_SE3(T) @ L.Ad_SE3(T2), tol=1e-12)
    close("Def 28.1.7: Ad_{T^{-1}} = Ad_T^{-1}", L.Ad_SE3(L.inv_T(T)), np.linalg.inv(L.Ad_SE3(T)), tol=1e-11)
    R = T[:3, :3]
    w = rng.normal(size=3)
    close("Def 28.1.7: R [w]_x R^T = [R w]_x (Ad_R = R)", R @ hat(w) @ R.T, hat(R @ w), tol=1e-13)
    S = L.expSim3(rng.normal(size=7))
    z = rng.normal(size=7) * 0.6
    close("Sim(3) Ad (Sophus Sim3::Adj): S Exp(z) S^{-1} = Exp(Ad_S z)", S @ L.expSim3(z) @ np.linalg.inv(S),
          L.expSim3(L.Ad_Sim3(S) @ z), tol=1e-10)
S = L.expSim3(rng.normal(size=7))
sig = 0.7
zw = L.Ad_Sim3(S) @ np.array([0, 0, 0, 0, 0, 0, sig])
close("Sim(3) Ad third column: (0,0,sigma) in body -> world twist (-sigma t, 0, sigma)", zw,
      np.concatenate([-sig * S[:3, 3], np.zeros(3), [sig]]), tol=1e-12)
c = np.array([2.0, 1.0, 0.0])
Rb = L.expSO3(np.radians(30) * np.array([0, 0, 1.0]))
xw = L.Ad_SE3(L.make_T(Rb, c)) @ np.array([0, 0, 0, 0, 0, 0.5])
close("Figure 28.1.4: xi_w = ((0.5, -1, 0), (0, 0, 0.5))", xw, [0.5, -1, 0, 0, 0, 0.5], tol=1e-14)
close("Figure 28.1.4: |v_w| = 0.5 sqrt5 = 1.12", np.linalg.norm(xw[:3]), 0.5 * np.sqrt(5), tol=1e-14)
xt = L.Ad_SE3(L.make_T(np.eye(3), [1, 2, 3])) @ np.array([0, 0, 0, 0, 0, 1.0])
close("Exercise 28.1.3: Ad_(I,t) (0, e3) = ((2, -1, 0), e3)", xt, [2, -1, 0, 0, 0, 1], tol=1e-14)
close("Exercise 28.1.3: the point t has velocity 0", np.cross([0, 0, 1.0], [1, 2, 3]) + xt[:3], 0 * xt[:3], tol=1e-14)
# COLMAP Rigid3d::Adjoint ordering (w, v): [[R, 0], [t x R, R]] is a block permutation of (28.1.8)
T = L.expSE3(rng.normal(size=6))
R, t = T[:3, :3], T[:3, 3]
colmap_adj = np.block([[R, np.zeros((3, 3))], [hat(t) @ R, R]])
Pm = np.block([[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]])
close("COLMAP Adjoint in (w, v) order = P Ad_T P", colmap_adj, Pm @ L.Ad_SE3(T) @ Pm, tol=1e-14)
# COLMAP's code computes the lower-left block as R.colwise().cross(-t): column k is R_k x (-t) = t x R_k
lower = np.column_stack([np.cross(R[:, k], -t) for k in range(3)])
close("COLMAP rigid3.h: colwise().cross(-translation) = [t]_x R", lower, hat(t) @ R, tol=1e-14)

# ---------------------------------------------------------------- pose conventions (table and insight box)
qc = np.array([0.851773, 0.0165051, 0.503764, -0.142941])
Rc = L.q_to_R(qc)
tc = np.array([-0.737434, 1.02973, 3.74354])
cc = -Rc.T @ tc
close("Exercise 28.1.2: camera center c = (3.797, -0.361, -1.035)", cc, [3.7969, -0.3614, -1.0348], tol=1e-4)
close("Exercise 28.1.2: viewing direction R^T e3 = (-0.863, -0.116, 0.492)", Rc.T @ np.array([0, 0, 1.0]),
      [-0.8629, -0.1159, 0.4919], tol=1e-4)
close("Exercise 28.1.2: |t| = |c| = 3.952", [np.linalg.norm(tc), np.linalg.norm(cc)], [3.9520, 3.9520], tol=1e-4)
ang = np.degrees(np.arccos(tc @ cc / np.linalg.norm(tc) / np.linalg.norm(cc)))
close("Insight box: angle between t and c = 116.8 deg", ang, 116.82, tol=0.01)
close("Insight box: rotation angle of the COLMAP example = 63.2 deg", np.degrees(L.rot_angle(Rc)), 63.19, tol=0.01)
close("Exercise 28.1.2: R (3 decimals)", Rc, [[0.452, 0.260, 0.853], [-0.227, 0.959, -0.172], [-0.863, -0.116, 0.492]],
      tol=5e-4)


# 3DGS getWorld2View2 (utils/graphics_utils.py), reproduced: R stored as R_cw^T, T = t_cw
def getWorld2View2(Rg, tg, translate=np.zeros(3), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = Rg.T
    Rt[:3, 3] = tg
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    C2W[:3, 3] = (C2W[:3, 3] + translate) * scale
    return np.linalg.inv(C2W)


W2V = getWorld2View2(Rc.T, tc)
close("3DGS: getWorld2View2(R_cw^T, t_cw) = T_cw", W2V, L.make_T(Rc, tc), tol=1e-12)
wvt = W2V.T                                    # world_view_transform = getWorld2View2(...).transpose(0, 1)
close("3DGS: camera_center = inverse(world_view_transform)[3, :3] = c", np.linalg.inv(wvt)[3, :3], cc, tol=1e-12)
# OpenGL <-> OpenCV: flipping columns 1:3 of c2w = right-multiplying by diag(1, -1, -1, 1)
c2w = L.inv_T(L.make_T(Rc, tc))
flip = c2w.copy()
flip[:3, 1:3] *= -1
close("c2w[:3, 1:3] *= -1 equals c2w @ diag(1,-1,-1,1)", flip, c2w @ np.diag([1, -1, -1, 1.0]), tol=1e-15)
close("diag(1,-1,-1) is the rotation by 180 deg about x", L.expSO3([np.pi, 0, 0]), np.diag([1, -1, -1.0]), tol=1e-15)
check("the axis flip keeps the camera center", np.allclose(flip[:3, 3], c2w[:3, 3]))
# NeRF get_rays: dirs (x, -y, -1) in the OpenGL camera = (x, y, 1) in the OpenCV camera after the flip
d_cv = np.array([0.2, -0.1, 1.0])
d_gl = np.diag([1, -1, -1.0]) @ d_cv
close("NeRF get_rays: OpenGL dirs (x, -y, -1) = D (x, y, 1)", d_gl, [0.2, 0.1, -1.0], tol=1e-15)
close("NeRF/OpenCV world ray directions agree", flip[:3, :3] @ d_gl, c2w[:3, :3] @ d_cv, tol=1e-14)
# nerfstudio colmap_to_json (default keep_original_world_coordinate=False): after the column flip,
# c2w = c2w[[0, 2, 1, 3], :]; c2w[2, :] *= -1  ==  left multiplication by the world rotation (x, y, z) -> (x, z, -y)
ns = flip.copy()
ns = ns[np.array([0, 2, 1, 3]), :]
ns[2, :] *= -1
Aw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
close("nerfstudio colmap_to_json: row permutation + sign = world change (x, y, z) -> (x, z, -y)", ns, Aw @ flip,
      tol=1e-15)
close("nerfstudio world change is a rotation (det = 1)", np.linalg.det(Aw[:3, :3]), 1.0, tol=1e-15)

summary()
