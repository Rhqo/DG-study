"""Verification for Section 48.3 (The Weak Second Fundamental Form).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-3-weak-sff.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import contextlib
import io
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "48-3-weak-sff.html"
rng = np.random.default_rng(7)


def mean_projection(n, d, K, rng):
    Qs = [np.linalg.qr(rng.normal(size=(n, d)))[0] for _ in range(K)]
    w = rng.dirichlet(np.ones(K))
    return sum(wk * Qk @ Qk.T for wk, Qk in zip(w, Qs))


# ---------------------------------------------------------------- Lemma 48.3.5
ok_det = ok_sol = ok_bound = ok_inv = True
for n, d in [(2, 1), (3, 1), (3, 2), (4, 2)]:
    for _ in range(5):
        c = mean_projection(n, d, 3, rng)
        b = rng.normal(size=(n, n, n))
        L = v.system_matrix(c)
        a = v.solve_system(b, c)
        ok_det &= abs(np.linalg.det(L) - np.linalg.det(np.eye(n) + c)) < 1e-10
        ok_sol &= np.allclose(L @ a.ravel(), b.ravel(), atol=1e-12) and np.allclose(
            np.linalg.solve(L, b.ravel()), a.ravel(), atol=1e-12)
        ok_bound &= np.abs(a).max() <= (1 + n ** 1.5) * np.abs(b).max()
        ok_inv &= np.linalg.norm(np.linalg.inv(np.eye(n) + c), 2) <= 1 + 1e-12
check("Lemma 48.3.5(a): det L = det(I + c) (n = 2, 3, 4)", ok_det)
check("Lemma 48.3.5(b): the closed form (48.3.6) is the unique solution", ok_sol)
check("Lemma 48.3.5(c): ||(I + c)^{-1}|| <= 1 and sup|a| <= (1 + n^{3/2}) sup|b|", ok_bound and ok_inv)
c2 = mean_projection(2, 1, 3, rng)
L2 = v.system_matrix(c2)
ind = [0 * 4 + 0 * 2 + 0, 0 * 4 + 1 * 2 + 0, 1 * 4 + 0 * 2 + 1, 1 * 4 + 1 * 2 + 1]   # 111, 121, 212, 222
Lp = np.array([[1 + c2[0, 0], 0, c2[0, 0], 0], [c2[1, 0], 1, c2[1, 0], 0],
               [0, c2[0, 1], 1, c2[0, 1]], [0, c2[1, 1], 0, 1 + c2[1, 1]]])
close("(48.3.8): the (l, m, l) block of L for n = 2", L2[np.ix_(ind, ind)], Lp, 1e-14)
rest = [k for k in range(8) if k not in ind]
check("Lemma 48.3.5 proof: the columns of L outside (l, m, l) are unit vectors",
      np.allclose(L2[np.ix_(range(8), rest)], np.eye(8)[:, rest]))
c3 = mean_projection(3, 2, 3, rng)
L3 = v.system_matrix(c3)
r123 = (0 * 3 + 1) * 3 + 2
check("Warning (Lemma 3.4): row (1, 2, 3) of L has c_23 in the columns (q, 1, q), so rows are not unit vectors",
      all(abs(L3[r123, (q * 3 + 0) * 3 + q] - c3[1, 2]) < 1e-14 for q in range(3)) and abs(c3[1, 2]) > 1e-3)
lml = [(l * 3 + m) * 3 + l for l in range(3) for m in range(3)]
close("Lemma 48.3.5 proof: det L = det L' (n = 3)", np.linalg.det(L3), np.linalg.det(L3[np.ix_(lml, lml)]), 1e-12)

# ---------------------------------------------------------------- Proposition 48.3.6
ok_a = ok_b = True
for n, d in [(2, 1), (3, 1), (3, 2), (5, 3)]:
    for _ in range(5):
        c = mean_projection(n, d, 4, rng)
        ev = np.linalg.eigvalsh(c)
        ok_a &= ev.min() >= -1e-12 and ev.max() <= 1 + 1e-12 and abs(np.trace(c) - d) < 1e-12 and np.abs(c).max() <= 1
        ok_b &= np.linalg.det(np.eye(n) + c) >= 2 ** d - 1e-12
check("Proposition 48.3.6(a): eigenvalues in [0, 1], tr c = d, |c_jk| <= 1", ok_a)
check("Proposition 48.3.6(b): det(I + c) >= 2^d", ok_b)
T = mean_projection(3, 2, 1, rng)
c_mix = 0.5 * (T + mean_projection(3, 2, 1, rng))
check("Proposition 48.3.6(c): a Dirac mass gives a projection, a mixture of two planes does not",
      np.allclose(T @ T, T) and not np.allclose(c_mix @ c_mix, c_mix))
# the inequality tr(cS) <= d with equality iff S = T used in the proof
S = mean_projection(3, 2, 1, rng)
check("Proposition 48.3.6(c): tr(T S) < d for S != T and = d for S = T",
      np.trace(T @ S) < 2 - 1e-6 and abs(np.trace(T @ T) - 2) < 1e-12)

# ---------------------------------------------------------------- Example 48.3.4 and Proposition 48.3.9 (smooth torus)
R, r = 2.0, 0.8


def torus_A(u, w, h=1e-4):
    xf = lambda a, b: v.torus_point(a, b, R, r)  # noqa: E731
    Pf = lambda a, b: v.proj_from_normal(v.torus_outward(a, b))  # noqa: E731
    xu = (xf(u + h, w) - xf(u - h, w)) / (2 * h)
    xw = (xf(u, w + h) - xf(u, w - h)) / (2 * h)
    Pu = (Pf(u + h, w) - Pf(u - h, w)) / (2 * h)
    Pw = (Pf(u, w + h) - Pf(u, w - h)) / (2 * h)
    gi = np.linalg.inv(np.array([[xu @ xu, xu @ xw], [xw @ xu, xw @ xw]]))
    A = np.zeros((3, 3, 3))
    for a_, Da in enumerate((Pu, Pw)):
        for b_, xb in enumerate((xu, xw)):
            A += gi[a_, b_] * np.einsum("jk,i->ijk", Da, xb)
    return A, Pf(u, w)


ok_inv = ok_310 = True
for u, w in rng.uniform(-np.pi, np.pi, size=(6, 2)):
    A, P = torus_A(u, w)
    H = np.einsum("qiq->i", A)
    beta = A + np.einsum("jk,i->ijk", P, H)                  # (48.3.4)
    AV = v.solve_system(beta, P)
    ok_inv &= np.allclose(AV, A, atol=1e-7)
    ok_310 &= np.allclose(np.einsum("qiq->i", AV), H, atol=1e-7) and np.allclose(np.einsum("iqq->i", AV), 0, atol=1e-7)
    ok_310 &= np.allclose(np.einsum("iqq->i", beta), 2 * H, atol=1e-7) and np.allclose(P @ H, 0, atol=1e-7)
check("Example 48.3.4: on T_{2,0.8} the system with c = P and beta = A + P_jk H_i returns the classical A", ok_inv)
check("Proposition 48.3.9: sum_q A_q.q = H, sum_q A_.qq = 0, sum_q beta_iqq = d H, P H = 0 on T_{2,0.8}", ok_310)

# ---------------------------------------------------------------- Example 48.3.10: a curve with a varying density
# ellipse (2 cos t, sin t), theta(t) = 1 + 0.4 sin(2 t); G-linear variation by quadrature against test functions
M = 20000
t = 2 * np.pi * (np.arange(M) + 0.5) / M
G = np.stack([-2 * np.sin(t), np.cos(t)], 1)
sp = np.linalg.norm(G, axis=1)
tau = G / sp[:, None]
X = np.stack([2 * np.cos(t), np.sin(t)], 1)
ds = sp * 2 * np.pi / M
th = 1 + 0.4 * np.sin(2 * t)
dth = 0.8 * np.cos(2 * t) / sp                              # d theta / ds
kap_t = np.stack([-2 * np.cos(t), -np.sin(t)], 1)            # gamma'' in t
dtau = (kap_t - np.einsum("li,li->l", kap_t, tau)[:, None] * tau) / sp[:, None] ** 2   # tau' = d tau / ds
P = v.proj_from_tangent(tau)
dP = np.einsum("li,lj->lij", dtau, tau) + np.einsum("li,lj->lij", tau, dtau)
Acl = np.einsum("li,ljk->lijk", tau, dP)                     # classical A_ijk = tau_i P'_jk
HV = dtau + (dth / th)[:, None] * tau
beta = Acl + np.einsum("ljk,li->lijk", P, HV)
ok_q = True
for _ in range(4):
    x0 = rng.uniform(-1.5, 1.5, 2) * np.array([1.0, 0.7])
    s2 = rng.uniform(0.3, 0.8)
    phi = np.exp(-np.sum((X - x0) ** 2, 1) / s2)
    gphi = -2 * (X - x0) / s2 * phi[:, None]
    lhs = np.einsum("ljk,li,l->ijk", P, np.einsum("lab,lb->la", P, gphi), th * ds)   # (48.3.1)
    rhs = -np.einsum("l,lijk->ijk", phi * th * ds, beta)
    ok_q &= np.allclose(lhs, rhs, atol=1e-6)
check("Example 48.3.10: delta_ijk V = -(A_ijk + P_jk H^V_i) ||V|| for v(ellipse, theta) (quadrature)", ok_q)
ok_e = ok_c = ok_perp = ok_BN = True
for l in range(0, M, 997):
    AV = v.solve_system(beta[l], P[l])
    pred = Acl[l] + (dth[l] / (2 * th[l])) * np.einsum("jk,i->ijk", P[l], tau[l])
    ok_e &= np.allclose(AV, pred, atol=1e-9)
    ok_c &= np.allclose(np.einsum("qiq->i", AV), dtau[l] + dth[l] / (2 * th[l]) * tau[l], atol=1e-9) and np.allclose(
        np.einsum("iqq->i", AV), dth[l] / (2 * th[l]) * tau[l], atol=1e-9)
    bperp = beta[l] - np.einsum("jk,i->ijk", P[l], P[l] @ np.einsum("qiq->i", beta[l]))   # (48.3.13)
    ok_perp &= np.allclose(bperp, Acl[l] + np.einsum("jk,i->ijk", P[l], dtau[l]), atol=1e-9) and np.allclose(
        v.solve_system(bperp, P[l]), Acl[l], atol=1e-9)
    Nout = np.array([tau[l, 1], -tau[l, 0]])
    ok_BN &= abs(tau[l] @ np.einsum("ijk,k->ij", v.B_from_A(AV), Nout) @ tau[l]
                 - tau[l] @ np.einsum("ijk,k->ij", v.B_from_A(Acl[l]), Nout) @ tau[l]) < 1e-9
check("Example 48.3.10: A^V = A + (theta'/2 theta) P_jk tau_i", ok_e)
check("Example 48.3.10: sum_q A^V_q.q = kappa + (theta'/2theta) tau, sum_q A^V_.qq = (theta'/2theta) tau", ok_c)
check("Exercise 48.3.5: beta-perp = A + P_jk kappa_i and the WSFF-perp is the classical A", ok_perp)
check("Exercise 48.3.4: the normal contraction tau^T B_N tau does not see theta", ok_BN)

# ---------------------------------------------------------------- Figure 48.3.1
AMP, N, EPS = 0.5, 4000, 0.1
thf = lambda s: 1 + AMP * np.cos(s)                     # noqa: E731
gf = lambda s: -AMP * np.sin(s) / (1 + AMP * np.cos(s))  # noqa: E731
y = 2 * np.pi * (np.arange(N) + 0.5) / N
s = y.copy()
for _ in range(60):
    s -= (s + AMP * np.sin(s) - y) / thf(s)
Xc = np.stack([np.cos(s), np.sin(s)], 1)
Tc = np.stack([-np.sin(s), np.cos(s)], 1)
Pc = v.proj_from_tangent(Tc)
mc = np.full(N, 2 * np.pi / N)
idx = np.arange(0, N, N // 48)[:48]
tW, tP, tH, nP = [], [], [], []
for i in idx:
    nb = v.ball(Xc, Xc[i], EPS)
    tW.append(np.einsum("qiq->i", v.wsff(Xc[i], Xc[nb], Pc[nb], mc[nb], EPS, 2, 1)) @ Tc[i])
    AP = v.wsff_perp(Xc[i], Pc[i], Xc[nb], Pc[nb], mc[nb], EPS, 2, 1)
    tP.append(np.einsum("qiq->i", AP) @ Tc[i])
    nP.append(np.einsum("qiq->i", AP) @ Xc[i])
    tH.append(v.H_from_beta(v.beta_tensor(Xc[i], Xc[nb], Pc[nb], mc[nb], EPS, 2, 1)) @ Tc[i])
tW, tP, tH, nP = map(np.array, (tW, tP, tH, nP))
gi_ = gf(s[idx])
close("Figure 48.3.1: max |WSFF tangential - theta'/2theta| = 0.0015", np.abs(tW - gi_ / 2).max(), 0.0015, 1e-4)
close("Figure 48.3.1: max |H tangential - theta'/theta| = 0.0026", np.abs(tH - gi_).max(), 0.0026, 1e-4)
check("Figure 48.3.1: WSFF-perp tangential component < 1e-14", np.abs(tP).max() < 1e-14)
check("Figure 48.3.1: WSFF-perp normal component within 0.002 of -1", np.abs(nP + 1).max() < 0.002)
ss = np.linspace(0, 2 * np.pi, 200001)
close("Figure 48.3.1: max |theta'/theta| = 0.577 at s = 2pi/3, 4pi/3", [np.abs(gf(ss)).max(), ss[np.argmax(gf(ss))]],
      [1 / np.sqrt(3), 4 * np.pi / 3], 1e-4)
check("Figure 48.3.1: points are denser at s = 0 than at s = pi (spacing ratio 1/3)",
      abs((s[1] - s[0]) / (s[N // 2] - s[N // 2 - 1]) - 1 / 3) < 1e-3)

# ---------------------------------------------------------------- Exercise 48.3.1: point cloud identities
n, d, K = 3, 2, 7
Xp = rng.normal(size=(K, n))
Pp = np.array([mean_projection(n, d, 1, rng) for _ in range(K)])
mp = rng.uniform(0.5, 2, K)
x0 = rng.normal(size=n)
gphi = -2 * (Xp - x0) * np.exp(-np.sum((Xp - x0) ** 2, 1))[:, None]
dijk = np.einsum("l,ljk,li->ijk", mp, Pp, np.einsum("lab,lb->la", Pp, gphi))
dV = np.einsum("l,li->i", mp, np.einsum("lab,lb->la", Pp, gphi))
check("Exercise 48.3.1: sum_j delta_ijj V = d delta V . e_i and sum_q delta_qiq V = delta V . e_i",
      np.allclose(np.einsum("ijj->i", dijk), d * dV) and np.allclose(np.einsum("qiq->i", dijk), dV))

# ---------------------------------------------------------------- Exercise 48.3.2
al = 0.83
cl = v.proj_from_tangent(np.array([np.cos(al), np.sin(al)]))
Lpl = v.system_matrix(cl)[np.ix_(ind, ind)]
close("Exercise 48.3.2: det L' = det(I + c) = 2 for a line", [np.linalg.det(Lpl), np.linalg.det(np.eye(2) + cl)],
      [2.0, 2.0], 1e-12)

# ---------------------------------------------------------------- Exercise 48.3.3
ok = True
for al in (0.4, 1.1, np.pi / 2):
    u1, u2 = np.array([1.0, 0.0]), np.array([np.cos(al), np.sin(al)])
    c = 0.5 * (np.outer(u1, u1) + np.outer(u2, u2))
    ok &= np.allclose(np.linalg.eigvalsh(c), np.sort([(1 - np.cos(al)) / 2, (1 + np.cos(al)) / 2]))
    ok &= abs(np.linalg.det(np.eye(2) + c) - (9 - np.cos(al) ** 2) / 4) < 1e-12
check("Exercise 48.3.3: eigenvalues (1 +- cos a)/2, det(I + c) = (9 - cos^2 a)/4", ok)
close("Exercise 48.3.3: a = 90 degrees: det(I + c) = 9/4", (9 - np.cos(np.pi / 2) ** 2) / 4, 2.25, 1e-12)

# ---------------------------------------------------------------- Implementation box
check("Implementation 48.3: at most 30 lines", dgcode.code_lines(PAGE, "code-48-3-1") <= 30)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-48-3-1")
printed = dgcode.code_block(PAGE, "out-48-3-1").strip()
check("Implementation 48.3: printed output matches the page", buf.getvalue().strip() == printed,
      f"got:\n{buf.getvalue()}")
check("Implementation 48.3: solve_wsff and system_matrix agree with lib/var48.py",
      np.allclose(ns["solve_wsff"](ns["b"], ns["c"]), v.solve_system(ns["b"], ns["c"]))
      and np.allclose(ns["system_matrix"](ns["c"]), v.system_matrix(ns["c"])))
check("Implementation 48.3: c is not a projection and det(I + c) > 4", not np.allclose(ns["c"] @ ns["c"], ns["c"])
      and np.linalg.det(np.eye(3) + ns["c"]) > 4)

summary()
