"""Verification for Section 48.5 (Approximate Weak Second Fundamental Form).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-5-approximate-wsff.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(21)

# ---------------------------------------------------------------- Proposition 48.5.2 (n = 2, d = 1, quadrature)
eps, K = 0.3, 9
Xl = rng.uniform(-0.4, 0.4, (K, 2))
ang = rng.uniform(0, np.pi, K)
Pl = v.proj_from_tangent(np.stack([np.cos(ang), np.sin(ang)], 1))
ml = rng.uniform(0.5, 1.5, K)
g = np.linspace(-1.2, 1.2, 961)
hq = g[1] - g[0]
G1, G2 = np.meshgrid(g, g, indexing="ij")
Xg = np.stack([G1, G2], -1)
phi = np.exp(-np.sum((Xg - np.array([0.1, -0.05])) ** 2, -1) / 0.2)
Cn = 1.0 / (v.radial_moment(v.rho, 1) * 2 * np.pi)          # rho_eps normalised on R^2


def grad_rho_eps(Z):
    r = np.linalg.norm(Z, axis=-1)
    s = r / eps
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(r[..., None] > 0, Z / r[..., None], 0.0)
    return Cn / eps ** 3 * v.drho(s)[..., None] * u


lhs = np.zeros((2, 2, 2))                                   # delta_ijk V(phi * rho_eps)
for l in range(K):
    gconv = np.einsum("xy,xya->a", phi, grad_rho_eps(Xl[l] - Xg)) * hq * hq   # grad(phi * rho_eps)(x_l)
    lhs += ml[l] * np.einsum("jk,i->ijk", Pl[l], Pl[l] @ gconv)
field = np.zeros(Xg.shape[:2] + (2, 2, 2))                  # (48.5.2) evaluated on the grid
for l in range(K):
    gr = grad_rho_eps(Xl[l] - Xg)
    field += ml[l] * np.einsum("jk,xyi->xyijk", Pl[l], np.einsum("ab,xyb->xya", Pl[l], gr))
rhs = np.einsum("xy,xyijk->ijk", phi, field) * hq * hq
close("Proposition 48.5.2: delta_ijk V(phi * rho_eps) = int phi (delta_ijk V * rho_eps) dx (point cloud, n = 2)",
      lhs, rhs, 1e-6)

# ---------------------------------------------------------------- Proposition 48.5.4, c^{V,eps}, (48.5.6)
n, d = 3, 2
Xs = v.fib_sphere(3000)
Ns = Xs + 0.05 * rng.normal(size=Xs.shape)
Ps = v.proj_from_normal(Ns / np.linalg.norm(Ns, axis=1)[:, None])   # perturbed planes
ms = rng.uniform(0.5, 1.5, len(Xs)) * 4 * np.pi / len(Xs)
ok4 = okc = oks = okp = True
for i0 in (0, 777, 2100):
    x = Xs[i0]
    nb = v.ball(Xs, x, 0.35)
    b = v.beta_tensor(x, Xs[nb], Ps[nb], ms[nb], 0.35, n, d)
    a, w, den = v.local_terms(x, Xs[nb], Ps[nb], ms[nb], 0.35, n, d)
    Heps = np.einsum("l,li->i", a, w) / den                       # (47.4.3) with C_xi/C_rho = d/n
    ok4 &= np.allclose(np.einsum("qiq->i", b), Heps) and np.allclose(np.einsum("iqq->i", b), d * Heps)
    c = v.c_eps(x, Xs[nb], Ps[nb], ms[nb], 0.35)
    ev = np.linalg.eigvalsh(c)
    okc &= ev.min() > -1e-12 and ev.max() < 1 - 1e-6 and abs(np.trace(c) - d) < 1e-12
    A = v.wsff(x, Xs[nb], Ps[nb], ms[nb], 0.35, n, d)
    oks &= np.allclose(v.system_matrix(c) @ A.ravel(), b.ravel()) and np.allclose(
        A, b - np.einsum("jk,i->ijk", c, np.linalg.solve(np.eye(3) + c, Heps)))
    Ap = v.wsff_perp(x, Ps[i0], Xs[nb], Ps[nb], ms[nb], 0.35, n, d)
    bperp = b - np.einsum("jk,i->ijk", Ps[i0], Ps[i0] @ Heps)
    okp &= np.allclose(Ap, v.solve_system(bperp, Ps[i0])) and np.allclose(Ap, b - np.einsum("jk,i->ijk", Ps[i0], Heps))
check("Proposition 48.5.4: sum_q beta_qiq = H^{V,eps}, sum_q beta_iqq = d H^{V,eps}", ok4)
check("Definition 48.5.3 / Exercise 48.5.2: c^{V,eps} is a mean projection (eigenvalues in [0, 1), tr = d)", okc)
check("Definition 48.5.5: (48.5.6) solves (48.5.5)", oks)
check("Definition 48.5.7: (48.5.10) = solution with beta-perp of (48.5.9) = centered form (48.5.11)", okp)
close("natural pair: C_xi/C_rho = d/n = 2/3", v.ratio(2, 3), 2 / 3, 1e-9)

# ---------------------------------------------------------------- Exercise 48.5.3: a flat plane, uneven sampling
K = 4000
Y = rng.uniform(-1, 1, (K, 2)) * np.sqrt(rng.uniform(0, 1, (K, 1)))   # uneven density in the plane z = 0
Xf = np.concatenate([Y, np.zeros((K, 1))], 1)
Pf = np.broadcast_to(np.diag([1.0, 1.0, 0.0]), (K, 3, 3)).copy()
mf = np.full(K, 1.0 / K)
x0 = np.array([0.35, 0.1, 0.0])
nb = v.ball(Xf, x0, 0.3)
Ap = v.wsff_perp(x0, Pf[0], Xf[nb], Pf[nb], mf[nb], 0.3, 3, 2)
Aw = v.wsff(x0, Xf[nb], Pf[nb], mf[nb], 0.3, 3, 2)
H = v.H_from_beta(v.beta_tensor(x0, Xf[nb], Pf[nb], mf[nb], 0.3, 3, 2))
check("Exercise 48.5.3: flat plane: epsilon-WSFF-perp = 0 exactly", np.abs(Ap).max() < 1e-14)
check("Exercise 48.5.3: uneven sampling gives an in-plane H^{V,eps} != 0 and epsilon-WSFF = P_jk H_i / 2",
      np.linalg.norm(H) > 0.05 and abs(H[2]) < 1e-14 and np.allclose(Aw, 0.5 * np.einsum("jk,i->ijk", Pf[0], H)))

# ---------------------------------------------------------------- Figure 48.5.1
EPS = np.array([0.04, 0.06, 0.1, 0.15, 0.25, 0.4])
NP = 400000
Xp = v.fib_sphere(NP)
Pp = v.proj_from_normal(Xp)
mp = np.full(NP, 4 * np.pi / NP)


def exact_A(P, x):
    return -(np.einsum("ij,k->ijk", P, x) + np.einsum("ik,j->ijk", P, x))


def experiment(N):
    Z = np.random.default_rng(N).normal(size=(N, 3))
    X = Z / np.linalg.norm(Z, axis=1)[:, None]
    P = v.proj_from_normal(X)
    m = np.full(N, 4 * np.pi / N)
    idx = np.random.default_rng(1).choice(N, 60, replace=False)
    out = np.zeros((len(EPS), 5))
    for a_, e in enumerate(EPS):
        rows = []
        for i in idx:
            x = X[i]
            nb_, nbp = v.ball(X, x, e), v.ball(Xp, x, e)
            Aw_ = v.wsff(x, X[nb_], P[nb_], m[nb_], e, 3, 2)
            Awp = v.wsff(x, Xp[nbp], Pp[nbp], mp[nbp], e, 3, 2)
            Ap_ = v.wsff_perp(x, P[i], X[nb_], P[nb_], m[nb_], e, 3, 2)
            App = v.wsff_perp(x, P[i], Xp[nbp], Pp[nbp], mp[nbp], e, 3, 2)
            rows.append([np.abs(Aw_ - Awp).max(), np.abs(Ap_ - App).max(), np.abs(Ap_ - exact_A(P[i], x)).max(),
                         np.abs(Awp - exact_A(P[i], x)).max(), np.abs(App - exact_A(P[i], x)).max()])
        out[a_] = np.mean(rows, axis=0)
    return out


R20, R80 = experiment(20000), experiment(80000)
bias = []
for e in EPS:
    bb = []
    for i in range(0, NP, NP // 60)[:60]:
        nbp = v.ball(Xp, Xp[i], e)
        bb.append(np.abs(v.wsff_perp(Xp[i], Pp[i], Xp[nbp], Pp[nbp], mp[nbp], e, 3, 2) - exact_A(Pp[i], Xp[i])).max())
    bias.append(np.mean(bb))
bias = np.array(bias)
sl = lambda y, e=EPS: np.polyfit(np.log(e), np.log(y), 1)[0]  # noqa: E731
close("Figure 48.5.1: slopes -2.07 (epsilon-WSFF) and -1.00 (epsilon-WSFF-perp), N = 20000",
      [sl(R20[:, 0]), sl(R20[:, 1])], [-2.07, -1.00], 6e-3)
close("Figure 48.5.1: epsilon-WSFF discrepancy 12.1 -> 0.11, WSFF-perp 0.36 -> 0.036 (N = 20000)",
      [R20[0, 0], R20[-1, 0], R20[0, 1], R20[-1, 1]], [12.1, 0.11, 0.357, 0.0357], 0.06)
close("Figure 48.5.1: ratio of the two discrepancies at eps = 0.04 is 34", R20[0, 0] / R20[0, 1], 34.0, 0.6)
close("Figure 48.5.1: N = 80000 halves the discrepancies (mean ratio 0.50)", np.mean(R80[:, :2] / R20[:, :2]), 0.495,
      0.006)
check("Figure 48.5.1: total error minimum 0.055 at eps = 0.15-0.25 (N = 80000), 0.075 at 0.25 (N = 20000)",
      abs(R80[3, 2] - 0.055) < 1e-3 and abs(R80[4, 2] - 0.055) < 1e-3 and R80[:, 2].min() >= R80[4, 2] - 1e-12
      and abs(R20[4, 2] - 0.075) < 1e-3 and np.argmin(R20[:, 2]) == 4, str(R80[:, 2]) + str(R20[:, 2]))
close("Figure 48.5.1: bias of the proxy grows like eps^2 (slope 2 for eps >= 0.1)", sl(bias[2:], EPS[2:]), 2.0, 0.25)
print("proxy error at eps = 0.04 (WSFF, WSFF-perp):", R20[0, 3:], R80[0, 3:])
check("Figure 48.5.1: the proxy's own error at eps = 0.04 is 0.08-0.09 (WSFF) and about 0.003 (WSFF-perp)",
      np.all(np.abs(np.array([R20[0, 3], R80[0, 3]]) - 0.083) < 0.008) and max(R20[0, 4], R80[0, 4]) < 0.0035)
summary()
