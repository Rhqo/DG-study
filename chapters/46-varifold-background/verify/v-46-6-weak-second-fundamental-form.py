"""Verification for Section 46.6 (The Second Fundamental Form, Weakly).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-6-weak-second-fundamental-form.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(466)


# ---------------------------------------------------------------- parametrised submanifolds with analytic J and Hessian
class Sub:
    """x = f(u), u in R^d, x in R^n. J(u): (n, d); Hs(u): (d, d, n) second derivatives; Hvec(u): known mean curvature vector."""

    def __init__(self, f, J, Hs, Hvec, d, n):
        self.f, self.J, self.Hs, self.Hvec, self.d, self.n = f, J, Hs, Hvec, d, n

    def P(self, u):
        J = self.J(u)
        G = J.T @ J
        return J @ np.linalg.solve(G, J.T)

    def coeff(self, u):
        """c[:, i] with P e_i = J c[:, i]."""
        J = self.J(u)
        return np.linalg.solve(J.T @ J, J.T)

    def A(self, u, h=1e-5):
        """A_ijk = nabla^M P_jk . e_i by central differences of P along the parameter directions."""
        c = self.coeff(u)
        dP = np.array([(self.P(u + h * e) - self.P(u - h * e)) / (2 * h) for e in np.eye(self.d)])  # (d, n, n)
        return np.einsum("ai,ajk->ijk", c, dP)

    def B(self, u):
        """B^k_ij = II(P e_i, P e_j) . e_k, II(J a, J b) = sum a_alpha b_beta (f_{alpha beta})^perp."""
        c = self.coeff(u)
        Q = np.eye(self.n) - self.P(u)
        Hs = self.Hs(u) @ Q.T  # (d, d, n): normal parts
        return np.einsum("ai,bj,abk->ijk", c, c, Hs)


def sphere_sub(r):
    def f(u):
        th, ph = u
        return r * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])

    def J(u):
        th, ph = u
        return r * np.array([[math.cos(th) * math.cos(ph), -math.sin(th) * math.sin(ph)],
                             [math.cos(th) * math.sin(ph), math.sin(th) * math.cos(ph)],
                             [-math.sin(th), 0.0]])

    def Hs(u):
        th, ph = u
        x = f(u) / r
        tt = -x
        tp = r * np.array([-math.cos(th) * math.sin(ph), math.cos(th) * math.cos(ph), 0.0])
        pp = r * np.array([-math.sin(th) * math.cos(ph), -math.sin(th) * math.sin(ph), 0.0])
        return np.array([[r * tt, tp], [tp, pp]])

    return Sub(f, J, Hs, lambda u: -2 * f(u) / r ** 2, 2, 3)


R_T, r_T = 2.0, 0.8


def torus_N(u):
    a, b = u
    return np.array([math.cos(b) * math.cos(a), math.cos(b) * math.sin(a), math.sin(b)])


def torus_sub(R=R_T, r=r_T):
    def f(u):
        a, b = u
        return np.array([(R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)])

    def J(u):
        a, b = u
        return np.array([[-(R + r * math.cos(b)) * math.sin(a), -r * math.sin(b) * math.cos(a)],
                         [(R + r * math.cos(b)) * math.cos(a), -r * math.sin(b) * math.sin(a)],
                         [0.0, r * math.cos(b)]])

    def Hs(u):
        a, b = u
        aa = np.array([-(R + r * math.cos(b)) * math.cos(a), -(R + r * math.cos(b)) * math.sin(a), 0.0])
        ab = np.array([r * math.sin(b) * math.sin(a), -r * math.sin(b) * math.cos(a), 0.0])
        bb = np.array([-r * math.cos(b) * math.cos(a), -r * math.cos(b) * math.sin(a), -r * math.sin(b)])
        return np.array([[aa, ab], [ab, bb]])

    def Hvec(u):
        a, b = u
        k1 = -1.0 / r
        k2 = -math.cos(b) / (R + r * math.cos(b))
        return (k1 + k2) * torus_N(u)

    return Sub(f, J, Hs, Hvec, 2, 3)


def helix_sub(a=1.0, b=0.5):
    def f(u):
        t = u[0]
        return np.array([a * math.cos(t), a * math.sin(t), b * t])

    def J(u):
        t = u[0]
        return np.array([[-a * math.sin(t)], [a * math.cos(t)], [b]])

    def Hs(u):
        t = u[0]
        return np.array([[[-a * math.cos(t), -a * math.sin(t), 0.0]]])

    def Hvec(u):
        t = u[0]
        return a / (a * a + b * b) * np.array([-math.cos(t), -math.sin(t), 0.0])

    return Sub(f, J, Hs, Hvec, 1, 3)


def ellipse_sub():
    return Sub(lambda u: v.ellipse(u[0]), lambda u: v.ellipse_velocity(u[0])[:, None],
               lambda u: np.array([[-v.ellipse(u[0])]]),
               lambda u: v.ellipse_kappa(u[0]) * v.ellipse_normal(u[0]), 1, 2)


def cylinder_sub(r=0.7):
    def f(u):
        a, z = u
        return np.array([r * math.cos(a), r * math.sin(a), z])

    def J(u):
        a, z = u
        return np.array([[-r * math.sin(a), 0.0], [r * math.cos(a), 0.0], [0.0, 1.0]])

    def Hs(u):
        a, z = u
        return np.array([[[-r * math.cos(a), -r * math.sin(a), 0.0], np.zeros(3)], [np.zeros(3), np.zeros(3)]])

    return Sub(f, J, Hs, lambda u: -f(u) * np.array([1, 1, 0]) / r ** 2, 2, 3)


SUBS = {
    "sphere r = 1.5": (sphere_sub(1.5), lambda: np.array([rng.uniform(0.3, 2.8), rng.uniform(0, 2 * math.pi)])),
    "torus T_{2,0.8}": (torus_sub(), lambda: rng.uniform(0, 2 * math.pi, size=2)),
    "helix (cos t, sin t, t/2)": (helix_sub(), lambda: rng.uniform(0, 6, size=1)),
    "ellipse (2 cos t, sin t)": (ellipse_sub(), lambda: rng.uniform(0, 2 * math.pi, size=1)),
}

# ---------------------------------------------------------------- Proposition 46.6.1
for name, (M, sample) in SUBS.items():
    worst = {"AB": 0, "BA": 0, "H": 0, "tr": 0, "symA": 0, "symB": 0, "Bnormal": 0}
    for _ in range(6):
        u = sample()
        A, B, P = M.A(u), M.B(u), M.P(u)
        worst["AB"] = max(worst["AB"], np.abs(A - (B + np.transpose(B, (0, 2, 1)))).max())
        Bfrom = 0.5 * (A + np.transpose(A, (1, 0, 2)) - np.transpose(A, (1, 2, 0)))
        worst["BA"] = max(worst["BA"], np.abs(Bfrom - B).max())
        worst["H"] = max(worst["H"], np.abs(np.einsum("qiq->i", A) - M.Hvec(u)).max())
        worst["tr"] = max(worst["tr"], np.abs(np.einsum("iqq->i", A)).max())
        worst["symA"] = max(worst["symA"], np.abs(A - np.transpose(A, (0, 2, 1))).max())
        worst["symB"] = max(worst["symB"], np.abs(B - np.transpose(B, (1, 0, 2))).max())
        worst["Bnormal"] = max(worst["Bnormal"], np.abs(np.einsum("lk,ijk->ijl", P, B)).max())
    check(f"Prop 46.6.1(a) {name}: A_ijk = B^k_ij + B^j_ik", worst["AB"] < 1e-7, f"{worst['AB']:.2e}")
    check(f"Prop 46.6.1(b) {name}: B^k_ij = (A_ijk + A_jik - A_kij)/2", worst["BA"] < 1e-7, f"{worst['BA']:.2e}")
    check(f"Prop 46.6.1(c) {name}: sum_q A_qiq = mean curvature vector (sum convention)", worst["H"] < 1e-7, f"{worst['H']:.2e}")
    check(f"Prop 46.6.1(c) {name}: sum_q A_iqq = 0 and A_ijk = A_ikj", worst["tr"] < 1e-7 and worst["symA"] < 1e-9)
    check(f"Prop 46.6.1 {name}: B symmetric in ij, normal-valued", worst["symB"] < 1e-9 and worst["Bnormal"] < 1e-9)

# hypersurface formula D_U P = (W U) N^T + N (W U)^T, |D_U P|_op = |W U|, |D_U P|_F = sqrt2 |W U| (torus)
T = SUBS["torus T_{2,0.8}"][0]
worst_hs, worst_op = 0.0, 0.0
for _ in range(6):
    u = rng.uniform(0, 2 * math.pi, size=2)
    N = torus_N(u)
    J = T.J(u)
    a = rng.normal(size=2)
    U = J @ a
    U /= np.linalg.norm(U)
    c = np.linalg.lstsq(J, U, rcond=None)[0]
    h = 1e-6
    DUN = (torus_N(u + h * c) - torus_N(u - h * c)) / (2 * h)
    WU = -DUN
    DUP = np.einsum("i,ijk->jk", U, T.A(u))
    worst_hs = max(worst_hs, np.abs(DUP - (np.outer(WU, N) + np.outer(N, WU))).max())
    worst_op = max(worst_op, abs(np.linalg.norm(DUP, 2) - np.linalg.norm(WU)),
                   abs(np.linalg.norm(DUP) - math.sqrt(2) * np.linalg.norm(WU)))
check("Prop 46.6.1(d) torus: D_U P = (WU) N^T + N (WU)^T with W = -dN", worst_hs < 1e-7, f"{worst_hs:.2e}")
check("Exercise 46.6.2: |D_U P|_op = |WU|, |D_U P|_F = sqrt2 |WU| (torus, random unit tangent U)", worst_op < 1e-7)

# ---------------------------------------------------------------- Example 46.6.2
# (a) circle of radius r: dP/ds = kappa (n t^T + t n^T), eigenvalues +-kappa, (dP/ds) t = kappa n
for r0 in (0.5, 2.0):
    s0 = 0.7
    tt = lambda s: np.array([-math.sin(s / r0), math.cos(s / r0)])
    h = 1e-6
    dP = (v.dir_proj(tt(s0 + h)) - v.dir_proj(tt(s0 - h))) / (2 * h)
    n0 = -np.array([math.cos(s0 / r0), math.sin(s0 / r0)])
    t0 = tt(s0)
    close(f"Example 46.6.2(a) r = {r0}: dP/ds = (1/r)(n t^T + t n^T)", dP, (np.outer(n0, t0) + np.outer(t0, n0)) / r0, 1e-8)
    close(f"Example 46.6.2(a) r = {r0}: eigenvalues of dP/ds are -1/r, 1/r", np.sort(np.linalg.eigvalsh(dP)), [-1 / r0, 1 / r0], 1e-8)
    close(f"Example 46.6.2(a) r = {r0}: (dP/ds) t = kappa n", dP @ t0, n0 / r0, 1e-8)
# (b) sphere: A_ijk = -(P_ij x_k + P_ik x_j)/r^2, B^k_ij = -P_ij x_k / r^2
S15 = SUBS["sphere r = 1.5"][0]
worst = 0.0
for _ in range(5):
    u = np.array([rng.uniform(0.3, 2.8), rng.uniform(0, 2 * math.pi)])
    x, P = S15.f(u), S15.P(u)
    Af = -(np.einsum("ij,k->ijk", P, x) + np.einsum("ik,j->ijk", P, x)) / 1.5 ** 2
    Bf = -np.einsum("ij,k->ijk", P, x) / 1.5 ** 2
    worst = max(worst, np.abs(S15.A(u) - Af).max(), np.abs(S15.B(u) - Bf).max())
check("Example 46.6.2(b): sphere A_ijk = -(P_ij x_k + P_ik x_j)/r^2 and B^k_ij = -P_ij x_k/r^2", worst < 1e-7, f"{worst:.2e}")
# (c) ellipse: |dP/ds|_op = kappa, and the G_{1,2} coordinates move with speed 2 kappa
E = SUBS["ellipse (2 cos t, sin t)"][0]
worst = 0.0
for t0 in rng.uniform(0, 2 * math.pi, size=8):
    h = 1e-6
    sp = v.ellipse_speed(t0)
    dP = (E.P(np.array([t0 + h])) - E.P(np.array([t0 - h]))) / (2 * h * sp)
    dxy = (v.proj_coords(E.P(np.array([t0 + h]))) - v.proj_coords(E.P(np.array([t0 - h])))) / (2 * h * sp)
    k = v.ellipse_kappa(t0)
    worst = max(worst, abs(np.linalg.norm(dP, 2) - k), abs(np.linalg.norm(dxy) - 2 * k))
check("Example 46.6.2(c): ellipse |dP/ds|_op = kappa and |d(P11 - P22, 2 P12)/ds| = 2 kappa", worst < 1e-7, f"{worst:.2e}")
# Figure 46.6.1 numbers: 48 equal arc-length points; angular gaps on G_{1,2} at the vertex and at the top
L = v.ellipse_length_total()
tg = np.linspace(0, 2 * np.pi, 200001)
sg = np.concatenate([[0.0], np.cumsum(0.5 * (v.ellipse_speed(tg[1:]) + v.ellipse_speed(tg[:-1])) * np.diff(tg))])
sg *= L / sg[-1]
s48 = np.arange(48) * L / 48
t48 = np.interp(s48, sg, tg)
xy = v.proj_coords(v.dir_proj(v.ellipse_tangent(t48)))
ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
gaps = np.abs(np.diff(np.append(ang, ang[0] + 4 * np.pi)))
close("Figure 46.6.1: ds = L/48 = 0.2018", L / 48, 0.2018, 5e-5)
close("Figure 46.6.1: largest angular gap on G_{1,2} (next to the vertex, kappa = 2) = 0.7634", max(gaps), 0.7634, 5e-5)
close("Figure 46.6.1: smallest angular gap (next to the top, kappa = 1/4) = 0.1013", min(gaps), 0.1013, 5e-5)
close("Figure 46.6.1: 2 kappa ds at the vertex = 0.8074, at the top = 0.1009", [4 * L / 48, 0.5 * L / 48], [0.8074, 0.1009], 5e-5)
check("Figure 46.6.1: antipodal points have the same tangent line (48 points -> 24 points of G_{1,2})",
      np.allclose(xy[:24], xy[24:], atol=1e-9))


# ---------------------------------------------------------------- Proposition 46.6.3: the two integral identities
def phi_x(x):
    return math.exp(0.3 * x[0] - 0.2 * x[1] + 0.25 * x[2]) * math.cos(0.7 * x[0] + 0.4 * x[2])


def grad_phi_x(x):
    e = math.exp(0.3 * x[0] - 0.2 * x[1] + 0.25 * x[2])
    c, s = math.cos(0.7 * x[0] + 0.4 * x[2]), math.sin(0.7 * x[0] + 0.4 * x[2])
    return e * np.array([0.3 * c - 0.7 * s, -0.2 * c, 0.25 * c - 0.4 * s])


def surface_terms(M, u):
    x, P, A, H = M.f(u), M.P(u), M.A(u), M.Hvec(u)
    gM = P @ grad_phi_x(x)
    return x, P, A, H, gM


# torus (closed): trapezoid rule in both periodic variables
nu, nv = 96, 64
lhs = np.zeros((3, 3, 3))
rhs = np.zeros((3, 3, 3))
lhs2 = np.zeros(3)
rhs2 = np.zeros(3)


def g_S(S):
    return math.sin(S[0, 2] + 2 * S[1, 1]) + S[0, 1] ** 2


def Dg_S(S):
    D = np.zeros((3, 3))
    c = math.cos(S[0, 2] + 2 * S[1, 1])
    D[0, 2] = c
    D[1, 1] = 2 * c
    D[0, 1] = 2 * S[0, 1]
    return D


for a in 2 * np.pi * np.arange(nu) / nu:
    for b in 2 * np.pi * np.arange(nv) / nv:
        u = np.array([a, b])
        x, P, A, H, gM = surface_terms(T, u)
        w = (2 * np.pi / nu) * (2 * np.pi / nv) * r_T * (R_T + r_T * math.cos(b))
        ph = phi_x(x)
        lhs += w * np.einsum("jk,i->ijk", P, gM)
        rhs -= w * ph * (A + np.einsum("jk,i->ijk", P, H))
        # Hutchinson's identity (4.1) with phi(y, S) = phi_x(y) g(S), nonlinear in S
        lhs2 -= w * gM * g_S(P)
        rhs2 += w * ph * (np.einsum("ijk,jk->i", A, Dg_S(P)) + g_S(P) * H)
err = np.abs(lhs - rhs).max()
check("Prop 46.6.3(a) torus (closed): int P_jk nabla^M phi . e_i = -int phi (A_ijk + P_jk H_i), all 27 entries",
      err < 1e-6, f"{err:.2e}, scale {np.abs(lhs).max():.3f}")
err2 = np.abs(lhs2 - rhs2).max()
check("Prop 46.6.3(b) torus: Hutchinson's identity with phi(y, S) = phi(y)(sin(S_13 + 2 S_22) + S_12^2)",
      err2 < 1e-6, f"{err2:.2e}, scale {np.abs(lhs2).max():.3f}")

# hemisphere z >= 0 of the unit sphere (with boundary): Gauss-Legendre in theta, trapezoid in phi, boundary eta = -e3
S1 = sphere_sub(1.0)
xg, wg = v.gauss_legendre(40)
ths = 0.5 * np.pi * xg          # gauss_legendre gives nodes and weights on [0, 1]
wth = 0.5 * np.pi * wg
nph = 96
lhs = np.zeros((3, 3, 3))
rhs = np.zeros((3, 3, 3))
for th, wt in zip(ths, wth):
    for ph in 2 * np.pi * np.arange(nph) / nph:
        u = np.array([th, ph])
        x, P, A, H, gM = surface_terms(S1, u)
        w = wt * (2 * np.pi / nph) * math.sin(th)
        lhs += w * np.einsum("jk,i->ijk", P, gM)
        rhs -= w * phi_x(x) * (A + np.einsum("jk,i->ijk", P, H))
eta = np.array([0.0, 0.0, -1.0])
for ph in 2 * np.pi * np.arange(nph) / nph:
    x = np.array([math.cos(ph), math.sin(ph), 0.0])
    P = np.eye(3) - np.outer(x, x)
    rhs += (2 * np.pi / nph) * phi_x(x) * np.einsum("jk,i->ijk", P, eta)
err = np.abs(lhs - rhs).max()
check("Prop 46.6.3(a) hemisphere (with boundary term phi P_jk eta_i, eta = -e3)", err < 1e-8,
      f"{err:.2e}, scale {np.abs(lhs).max():.3f}")
rhs_nob = rhs.copy()
for ph in 2 * np.pi * np.arange(nph) / nph:
    x = np.array([math.cos(ph), math.sin(ph), 0.0])
    P = np.eye(3) - np.outer(x, x)
    rhs_nob -= (2 * np.pi / nph) * phi_x(x) * np.einsum("jk,i->ijk", P, eta)
check("Prop 46.6.3(a) hemisphere: without the boundary term the identity fails", np.abs(lhs - rhs_nob).max() > 0.1)


# ---------------------------------------------------------------- Proposition 46.6.5 and Figure 46.6.2: junctions of half-lines
def units(deg):
    a = np.radians(np.asarray(deg, float))
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def psi(x, rho=0.9):
    s = (x[..., 0] ** 2 + x[..., 1] ** 2) / rho ** 2
    return np.where(s < 1, (1 - s) ** 2, 0.0)


def grad_psi(x, rho=0.9):
    s = (x[..., 0] ** 2 + x[..., 1] ** 2) / rho ** 2
    return np.where((s < 1)[..., None], (-4 * (1 - s) / rho ** 2)[..., None] * x, 0.0)


def chi_factory(seed):
    r = np.random.default_rng(seed)
    C0 = r.normal(size=(2, 2))
    C0 = C0 + C0.T
    K = r.normal(size=(2, 2))

    def chi(S):
        return math.exp(-np.sum((S - 0.3 * C0) ** 2)) * (1 + 0.5 * math.sin(np.sum(K * S)))

    return chi


def hutchinson_zero_A(U, chi, nq=48):
    """sum_l int_0^1 (P_l grad_x phi(t u_l, P_l)) dt for phi = psi(x) chi(S), i.e. the left side of Definition 46.6.4 with A = 0."""
    t, w = v.gauss_legendre(nq)    # nodes and weights on [0, 1]; psi vanishes for t >= 0.9
    t, w = 0.9 * t, 0.9 * w
    tot = np.zeros(2)
    for u in U:
        Pl = np.outer(u, u)
        pts = t[:, None] * u[None, :]
        tot += chi(Pl) * (w[:, None] * (grad_psi(pts) @ Pl.T)).sum(axis=0)
    return tot


CASES = {"X": [30, 150, 210, 330], "Y": [90, 210, 330], "T": [0, 90, 180], "nine": [40 * l for l in range(9)]}
for name, deg in CASES.items():
    U = units(deg)
    worst = 0.0
    for seed in range(4):
        chi = chi_factory(seed)
        num = hutchinson_zero_A(U, chi)
        pred = -sum(chi(np.outer(u, u)) * u for u in U)
        worst = max(worst, np.abs(num - pred).max())
    check(f"Prop 46.6.5 proof ({name}): with A = 0 the functional equals -sum_l u_l phi(0, P_l)", worst < 1e-10, f"{worst:.2e}")
X4 = units(CASES["X"])
worstX = max(np.abs(-sum(chi_factory(s)(np.outer(u, u)) * u for u in X4)).max() for s in range(10))
check("Prop 46.6.5 (X): the functional vanishes for every test function (A = 0 works)", worstX < 1e-12)
Y3 = units(CASES["Y"])
sig = 0.3
P1 = np.outer(Y3[0], Y3[0])
chi_loc = lambda S: math.exp(-np.sum((S - P1) ** 2) / sig ** 2)
val = hutchinson_zero_A(Y3, chi_loc)
close("Prop 46.6.5 (Y): chi localised at P_1 gives -u_1 = (0, -1) (up to e^{-1.5/0.09})", val, -Y3[0], 1e-6)
close("Prop 46.6.5 (Y): |P_1 - P_2|_F^2 = 2 sin^2(120 deg) = 1.5", np.sum((P1 - np.outer(Y3[1], Y3[1])) ** 2), 1.5, 1e-12)


def paired(U):
    return np.array([np.any(np.all(np.isclose(U, -u, atol=1e-12), axis=1)) for u in U])


def third(U):
    return np.einsum("li,lj,lk->ijk", U, U, U)


table = {"X": (True, True, True), "Y": (True, False, False), "T": (False, False, False), "nine": (True, False, True)}
for name, (st, cv, gl) in table.items():
    U = units(CASES[name])
    check(f"Figure 46.6.2 ({name}): stationary = {st}, curvature varifold = {cv}, G-linear variation zero = {gl}",
          bool(np.allclose(U.sum(axis=0), 0, atol=1e-12)) == st and bool(paired(U).all()) == cv
          and bool(np.allclose(third(U), 0, atol=1e-12)) == gl)

# ---------------------------------------------------------------- exercises
Cy = cylinder_sub(0.7)
u = np.array([0.9, 0.3])
A = Cy.A(u)
e_th = np.array([-math.sin(0.9), math.cos(0.9), 0.0])
e_z = np.array([0.0, 0.0, 1.0])
close("Exercise 46.6.1: cylinder r = 0.7: |D_U P|_op = 1/r along the circle", np.linalg.norm(np.einsum("i,ijk->jk", e_th, A), 2), 1 / 0.7, 1e-7)
close("Exercise 46.6.1: cylinder: D_U P = 0 along the ruling", np.abs(np.einsum("i,ijk->jk", e_z, A)).max(), 0.0, 1e-7)
for al in (0.4, 1.1):
    U = math.cos(al) * e_th + math.sin(al) * e_z
    close(f"Exercise 46.6.1: cylinder: |D_U P|_op = |cos a|/r (a = {al})", np.linalg.norm(np.einsum("i,ijk->jk", U, A), 2),
          abs(math.cos(al)) / 0.7, 1e-7)
close("Exercise 46.6.1: cylinder: sum_q A_qiq = -x_perp / r^2 (|H| = 1/r)", np.einsum("qiq->i", A), Cy.Hvec(u), 1e-7)

ex4 = {"(a) 0,90,180,270": ([0, 90, 180, 270], True, True), "(b) 0,60,180,240": ([0, 60, 180, 240], True, True),
       "(c) 0,90,180,225": ([0, 90, 180, 225], False, False), "(d) 0,60,...,300": ([60 * l for l in range(6)], True, True),
       "(e) regular pentagon": ([72 * l for l in range(5)], True, False)}
for name, (deg, st, cv) in ex4.items():
    U = units(deg)
    check(f"Exercise 46.6.4 {name}: stationary = {st}, curvature varifold = {cv}",
          bool(np.allclose(U.sum(axis=0), 0, atol=1e-12)) == st and bool(paired(U).all()) == cv)

Ty = third(Y3)
close("Exercise 46.6.5: Y junction (90, 210, 330): sum (u.e2)^3 = 3/4, sum (u.e1)^2 (u.e2) = -3/4", [Ty[1, 1, 1], Ty[0, 0, 1]],
      [0.75, -0.75], 1e-12)
close("Exercise 46.6.5: Y junction: sum (u.e1)^3 = sum (u.e1)(u.e2)^2 = 0", [Ty[0, 0, 0], Ty[0, 1, 1]], [0, 0], 1e-12)
for N in (3, 4, 5, 6, 7, 8, 9):
    U = units([360 * l / N for l in range(N)])
    zero = bool(np.allclose(third(U), 0, atol=1e-12))
    check(f"Exercise 46.6.5: regular N = {N}: third moment zero iff N not in (1, 3)", zero == (N not in (1, 3)))
al = np.radians([40 * l for l in range(9)])
close("Exercise 46.6.5: nine half-lines: sum e^{i a} = sum e^{3 i a} = 0", [abs(np.exp(1j * al).sum()), abs(np.exp(3j * al).sum())],
      [0, 0], 1e-12)
# cos^3 = (3 cos + cos 3a)/4, sin^3 = (3 sin - sin 3a)/4 (the reduction used in the solution)
aa = rng.uniform(0, 2 * math.pi, size=5)
close("Exercise 46.6.5: cos^3 a = (3 cos a + cos 3a)/4 and sin^3 a = (3 sin a - sin 3a)/4",
      [np.cos(aa) ** 3 - (3 * np.cos(aa) + np.cos(3 * aa)) / 4, np.sin(aa) ** 3 - (3 * np.sin(aa) - np.sin(3 * aa)) / 4],
      np.zeros((2, 5)), 1e-12)

summary()
