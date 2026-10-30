"""Verification for Section 48.4 (Comparison with Hutchinson and Structural Properties).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-4-hutchinson-properties.py
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

rng = np.random.default_rng(11)


def junction(deg):
    U = np.stack([np.cos(np.radians(deg)), np.sin(np.radians(deg))], 1)
    return U, np.einsum("li,lj,lk->ijk", U, U, U)


# ---------------------------------------------------------------- Proposition 48.4.3: (48.4.2) by quadrature
U, T = junction([90, 210, 330, 17])
ok = True
for _ in range(3):
    x0 = rng.normal(size=2) * 0.3
    phi = lambda X: np.exp(-np.sum((X - x0) ** 2, -1) / 0.4)                       # noqa: E731
    gphi = lambda X: -2 * (X - x0) / 0.4 * phi(X)[..., None]                         # noqa: E731
    t = (np.arange(200000) + 0.5) * (12.0 / 200000)
    val = np.zeros((2, 2, 2))
    for u in U:
        X = t[:, None] * u
        Pl = np.outer(u, u)
        val += np.einsum("jk,i->ijk", Pl, np.einsum("ab,lb->la", Pl, gphi(X)).sum(0) * (12.0 / 200000))
    ok &= np.allclose(val, -T * phi(np.zeros(2)), atol=1e-6)
check("Proposition 48.4.3: delta_ijk V(phi) = -T_ijk phi(0) for a 4-junction (quadrature)", ok)
check("Proposition 48.4.3: the trace of T is the first variation: sum_q T_qiq = sum_l u_l",
      np.allclose(np.einsum("qiq->i", T), U.sum(0)) and np.allclose(np.einsum("ijj->i", T), U.sum(0)))

# ---------------------------------------------------------------- Proposition 48.4.4 and Table 48.4.1
rows = []
for N in range(2, 10):
    a = 2 * np.pi * np.arange(N) / N
    U, T = junction(np.degrees(a))
    stat = np.abs(U.sum(0)).max() < 1e-12
    zero = np.abs(T).max() < 1e-12
    cond = abs(np.exp(1j * a).sum()) < 1e-12 and abs(np.exp(3j * a).sum()) < 1e-12
    lines = all(np.min(np.linalg.norm(U + u, axis=1)) < 1e-12 for u in U)
    rows.append((N, stat, zero, lines, round(float(np.abs(T).max()), 2), cond))
expected = [(2, True, True, True, 0.0), (3, True, False, False, 0.75), (4, True, True, True, 0.0),
            (5, True, True, False, 0.0), (6, True, True, True, 0.0), (7, True, True, False, 0.0),
            (8, True, True, True, 0.0), (9, True, True, False, 0.0)]
check("Table 48.4.1: stationary / delta V = 0 / union of lines / max|T| for N = 2..9",
      [r[:5] for r in rows] == expected, str(rows))
check("Proposition 48.4.4: delta_ijk V = 0 iff sum e^{ia} = sum e^{3ia} = 0 (regular junctions)",
      all(r[2] == r[5] for r in rows))
ok = True
for _ in range(200):
    a = rng.uniform(0, 2 * np.pi, rng.integers(3, 8))
    U, T = junction(np.degrees(a))
    c, s = np.cos(a), np.sin(a)
    four = [np.sum(c ** 3), np.sum(s ** 3), np.sum(c ** 2 * s), np.sum(c * s ** 2)]
    ok &= np.allclose([T[0, 0, 0], T[1, 1, 1], T[0, 0, 1], T[0, 1, 1]], four)
    ok &= np.isclose(T[0, 0, 0] + T[0, 1, 1], c.sum()) and np.isclose(T[0, 0, 1] + T[1, 1, 1], s.sum())
check("Proposition 48.4.4 proof: T_111 + T_122 = sum cos, T_112 + T_222 = sum sin (random junctions)", ok)

# ---------------------------------------------------------------- Figure 48.4.1 and Exercise 48.4.1
p = np.linspace(0, 2 * np.pi, 361)
W = np.stack([np.cos(p), np.sin(p)], 1)
fY = np.einsum("ijk,pi,pj,pk->p", junction([90, 210, 330])[1], W, W, W)
fT = np.einsum("ijk,pi,pj,pk->p", junction([0, 90, 180])[1], W, W, W)
check("Exercise 48.4.1: Y junction cubic form = -(3/4) sin 3p, max 0.75", np.allclose(fY, -0.75 * np.sin(3 * p))
      and abs(np.abs(fY).max() - 0.75) < 1e-12)
check("Exercise 48.4.1: T junction cubic form = sin^3 p, max 1", np.allclose(fT, np.sin(p) ** 3))
check("Figure 48.4.1: X junction and regular pentagon have T = 0",
      np.abs(junction([30, 150, 210, 330])[1]).max() < 1e-12
      and np.abs(junction([90 + 72 * k for k in range(5)])[1]).max() < 1e-12)
check("Figure 48.4.1: the pentagon junction is not a union of lines",
      min(np.linalg.norm(junction([90 + 72 * k for k in range(5)])[0] + junction([90])[0][0], axis=1)) > 0.1)

# ---------------------------------------------------------------- Exercise 48.4.3: N = 3 and N = 4
ok3 = ok4 = True
for _ in range(300):
    a1 = rng.uniform(0, 2 * np.pi)
    U3 = np.array([[np.cos(a1 + k * 2 * np.pi / 3), np.sin(a1 + k * 2 * np.pi / 3)] for k in range(3)])
    ok3 &= abs(np.linalg.norm(np.einsum("li,lj,lk->ijk", U3, U3, U3)) - 1.5) < 1e-12      # |T|_F^2 = 3 - 6/8
    u1, u2 = rng.normal(size=2), rng.normal(size=2)
    u1 /= np.linalg.norm(u1)
    u2 /= np.linalg.norm(u2)
    sv = u1 + u2
    if np.linalg.norm(sv) > 1e-6:                                    # the other two unit vectors with sum -s
        hdir = np.array([-sv[1], sv[0]]) / np.linalg.norm(sv)
        hl = np.sqrt(max(0.0, 1 - np.dot(sv, sv) / 4))
        u3, u4 = -sv / 2 + hl * hdir, -sv / 2 - hl * hdir
        U4 = np.stack([u1, u2, u3, u4])
        ok4 &= np.allclose(U4.sum(0), 0) and all(np.min(np.linalg.norm(U4 + u, axis=1)) < 1e-9 for u in U4)
check("Exercise 48.4.3: a stationary 3-junction (120 degrees) always has |T|_F = 1.5", ok3)
check("Exercise 48.4.3: four unit vectors with zero sum form two opposite pairs", ok4)

# ---------------------------------------------------------------- Proposition 48.4.7, 48.4.8 on random beta, c
ok7 = ok8 = True
for n, d in [(2, 1), (3, 2), (3, 1)]:
    for _ in range(5):
        Qs = [np.linalg.qr(rng.normal(size=(n, d)))[0] for _ in range(3)]
        w = rng.dirichlet(np.ones(3))
        c = sum(wk * Qk @ Qk.T for wk, Qk in zip(w, Qs))
        beta = rng.normal(size=(n, n, n))
        beta = 0.5 * (beta + np.transpose(beta, (0, 2, 1)))     # beta_ijk = beta_ikj
        Hb = np.einsum("qiq->i", beta)
        A = v.solve_system(beta, c)
        ok7 &= np.allclose(np.einsum("qiq->i", A), np.linalg.solve(np.eye(n) + c, Hb))
        ok7 &= np.allclose(np.einsum("iqq->i", A), np.einsum("iqq->i", beta) - d * np.linalg.solve(np.eye(n) + c, Hb))
        bperp = beta - np.einsum("jk,i->ijk", c, c @ Hb)
        Ap = v.solve_system(bperp, c)
        ok8 &= np.allclose(Ap, beta - np.einsum("jk,i->ijk", c, Hb))
        ok8 &= np.allclose(np.einsum("qiq->i", Ap), (np.eye(n) - c) @ Hb)
        ok8 &= np.allclose(np.einsum("iqq->i", Ap), np.einsum("iqq->i", beta) - d * Hb)
check("Proposition 48.4.7(c): sum_q A_q.q = (I + c)^{-1} h, sum_q A_.qq = sum_q beta_.qq - d (I + c)^{-1} h", ok7)
check("Proposition 48.4.8: A-perp = beta - c_jk h_i and sum_q A-perp_q.q = (I - c) h", ok8)
c = np.diag([0.7, 0.3])
H = np.array([0.4, -1.1])
check("Exercise 48.4.2: (I + c)^{-1} H = H iff c H = 0",
      not np.allclose(np.linalg.solve(np.eye(2) + c, H), H)
      and np.allclose(np.linalg.solve(np.eye(2) + np.diag([1.0, 0.0]), [0.0, 2.0]), [0.0, 2.0]))

# ---------------------------------------------------------------- Example 48.4.9 and Exercise 48.4.4 (quadrature)
L, M = 5.0, 801
g = np.linspace(-L, L, M)
h = g[1] - g[0]
X1, X2 = np.meshgrid(g, g, indexing="ij")
f = np.exp(-(X1 ** 2 + X2 ** 2))
E1, E2 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
x0 = np.array([0.3, -0.2])
phi = np.exp(-((X1 - x0[0]) ** 2 + (X2 - x0[1]) ** 2) / 0.5)
gphi = np.stack([-2 * (X1 - x0[0]) / 0.5 * phi, -2 * (X2 - x0[1]) / 0.5 * phi], -1)


def gl_var(planes, wts):
    """delta_ijk V(phi) for V = f L^2 (x) sum_w w delta_E."""
    out = np.zeros((2, 2, 2))
    for E, wt in zip(planes, wts):
        out += wt * np.einsum("jk,i->ijk", E, np.einsum("ab,xyb->a", E, gphi[..., :] * f[..., None]) * h * h)
    return out


Hx = np.stack([-X1, -X2], -1)                                  # H = grad f / (2 f) = -x
beta_c = np.zeros((M, M, 2, 2, 2))
beta_c[..., 0, 0, 0] = Hx[..., 0]
beta_c[..., 1, 1, 1] = Hx[..., 1]
rhs = -np.einsum("xy,xyijk->ijk", phi * f, beta_c) * h * h
check("Example 48.4.9: delta_ijk V = -beta ||V|| with beta_111 = H_1, beta_222 = H_2, H = -x (quadrature)",
      np.allclose(gl_var([E1, E2], [0.5, 0.5]), rhs, atol=1e-8))
dV = 0.5 * np.einsum("xya->a", gphi * f[..., None]) * h * h
check("Example 48.4.9: delta V = -H ||V|| with H = grad f/(2f) = -x",
      np.allclose(dV, -np.einsum("xy,xya->a", phi * f, Hx) * h * h, atol=1e-8))
Hs = np.array([0.37, -0.81])
b = np.zeros((2, 2, 2))
b[0, 0, 0], b[1, 1, 1] = Hs
cI = np.eye(2) / 2
A = v.solve_system(b, cI)
check("Example 48.4.9: WSFF A = beta - (1/3) delta_jk H_i", np.allclose(A, b - np.einsum("jk,i->ijk", np.eye(2), Hs) / 3))
check("Example 48.4.9: sum_q A_q.q = 2H/3, sum_q A_.qq = H/3",
      np.allclose(np.einsum("qiq->i", A), 2 * Hs / 3) and np.allclose(np.einsum("iqq->i", A), Hs / 3))
Ap = v.solve_system(b - np.einsum("jk,i->ijk", cI, cI @ Hs), cI)
check("Example 48.4.9: WSFF-perp = beta - (1/2) delta_jk H_i, contractions H/2 and 0",
      np.allclose(Ap, b - np.einsum("jk,i->ijk", np.eye(2), Hs) / 2) and np.allclose(np.einsum("qiq->i", Ap), Hs / 2)
      and np.allclose(np.einsum("iqq->i", Ap), 0))
# Exercise 48.4.4: horizontal sticks
dV1 = np.einsum("ab,xyb->a", E1, gphi * f[..., None]) * h * h
check("Exercise 48.4.4: delta V = -H ||V|| with H = -2 x_1 e_1",
      np.allclose(dV1, -np.einsum("xy,xya->a", phi * f, np.stack([-2 * X1, 0 * X2], -1)) * h * h, atol=1e-8))
b1 = np.zeros((2, 2, 2))
b1[0, 0, 0] = -2 * 0.37
gv = gl_var([E1], [1.0])
beta1 = np.zeros((M, M, 2, 2, 2))
beta1[..., 0, 0, 0] = -2 * X1
check("Exercise 48.4.4: beta_111 = d_1 f / f = H_1 is the only nonzero density (quadrature)",
      np.allclose(gv, -np.einsum("xy,xyijk->ijk", phi * f, beta1) * h * h, atol=1e-8))
A1 = v.solve_system(b1, E1)
A1p = v.solve_system(b1 - np.einsum("jk,i->ijk", E1, E1 @ np.einsum("qiq->i", b1)), E1)
check("Exercise 48.4.4: WSFF A_111 = H_1/2 = -x_1, WSFF-perp = 0",
      np.isclose(A1[0, 0, 0], b1[0, 0, 0] / 2) and np.abs(np.delete(A1.ravel(), 0)).max() < 1e-15
      and np.abs(A1p).max() < 1e-15)

# ---------------------------------------------------------------- Example 48.4.12: inscribed polygons
x0 = np.array([0.8, 0.3])
phi_ = lambda X: np.exp(-np.sum((X - x0) ** 2, -1) / 0.3)  # noqa: E731
Mq = 200000
s = 2 * np.pi * (np.arange(Mq) + 0.5) / Mq
Xc = np.stack([np.cos(s), np.sin(s)], 1)
tau = np.stack([-np.sin(s), np.cos(s)], 1)
nn = -Xc
beta = (np.einsum("li,lj,lk->lijk", tau, nn, tau) + np.einsum("li,lj,lk->lijk", tau, tau, nn)
        + np.einsum("li,lj,lk->lijk", nn, tau, tau))
exact = -np.einsum("l,lijk->ijk", phi_(Xc) * 2 * np.pi / Mq, beta)
close("Example 48.4.12: circle: |beta|_F = sqrt 3, int |beta|_F ds = 2 pi sqrt 3 = 10.88",
      [np.linalg.norm(beta[0]), 2 * np.pi * np.sqrt(3)], [np.sqrt(3), 10.8828], 1e-4)
tv, err, mass = {}, {}, {}
for N in (8, 16, 32, 64, 256):
    gk = 2 * np.pi * np.arange(N) / N
    Vk = np.stack([np.cos(gk), np.sin(gk)], 1)
    Ed = np.roll(Vk, -1, 0) - Vk
    tk = Ed / np.linalg.norm(Ed, axis=1)[:, None]
    tp = np.roll(tk, 1, 0)
    D = np.einsum("li,lj,lk->lijk", tp, tp, tp) - np.einsum("li,lj,lk->lijk", tk, tk, tk)
    tv[N] = np.linalg.norm(D.reshape(N, -1), axis=1).sum()
    err[N] = np.abs(np.einsum("l,lijk->ijk", phi_(Vk), D) - exact).max()
    mass[N] = np.linalg.norm(Ed, axis=1).sum()
    th = 2 * np.pi / N
    check(f"Example 48.4.12: |t^3 - s^3|_F^2 = 2 - 2 cos^3 theta (N = {N})",
          np.allclose(np.linalg.norm(D.reshape(N, -1), axis=1) ** 2, 2 - 2 * np.cos(th) ** 3))
close("Example 48.4.12: total variation 9.10, 10.85, 10.88 for N = 8, 64, 256", [tv[8], tv[64], tv[256]],
      [9.10, 10.85, 10.88], 6e-3)
close("Example 48.4.12: errors 0.135, 0.018, 0.0045, 0.0011 for N = 8, 16, 32, 64",
      [err[8], err[16], err[32], err[64]], [0.135, 0.018, 0.0045, 0.0011], 6e-4)
check("Example 48.4.12: mass 2 N sin(pi/N) < 2 pi", all(np.isclose(mass[N], 2 * N * np.sin(np.pi / N)) and mass[N] < 2 * np.pi
                                                       for N in mass))

summary()
