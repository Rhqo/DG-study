"""Figure 47.3.1: the regularized first variation of point clouds on the circle of radius 0.5 ([BLM17] Def 3.1,
Prop 3.2 (3.4), Thm 3.6 (3.7)); the exp kernel is normalised in R^2.

Point clouds: N points x_j = 0.5 (cos t_j, sin t_j) with exact tangent lines and equal masses pi/N. "uniform":
t_j = 2 pi j / N; "jittered": t_j = 2 pi (j + n_j) / N with n_j standard normal (seed 3), so the spacing is uneven
while the masses stay equal. h = pi / N is the mean spacing.
(a) eps (delta V * rho_eps)(x) at x = 0.5 (cos s, sin s), 0 <= s <= 72 degrees, N = 250, eps = 4h: outward normal
component (uniform: black, jittered: blue) and tangential component (jittered: vermilion; uniform is ~0, grey).
The dashed line is the value for the smooth circle, |H| C_rho = 2 * 0.9517.
(b) || delta V * rho_eps ||_{L^1(R^2)} against N for four choices of eps; the smooth circle has |delta V|(R^2) = 2 pi.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-3-1-regularized-first-variation.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var47 as v  # noqa: E402

C = dgfig.COLORS
R, L = 0.5, math.pi
c_exp = v.normaliser(v.rho_exp, 2)
C_RHO = c_exp * v.C_const(v.rho_exp, 1)


def cloud(N, jitter, seed=3):
    j = np.arange(N, dtype=float)
    if jitter:
        j = j + np.random.default_rng(seed).standard_normal(N)
    X, T, _ = v.circle(2 * np.pi * j / N, R)
    return X, v.proj_from_tangent(T), np.full(N, L / N)


def field(Q, X, P, m, eps):
    """(delta V * rho_eps)(q) = sum_j m_j P_j grad rho_eps(x_j - q), formula (3.4)."""
    D = X[None, :, :] - Q[:, None, :]
    r = np.linalg.norm(D, axis=2)
    g = c_exp * v.drho_exp(r / eps) / eps ** 3
    u = D / np.where(r > 0, r, 1.0)[:, :, None]
    Pu = np.einsum("jab,qjb->qja", P, u)
    return np.einsum("j,qj,qja->qa", m, g, Pu)


N0 = 250
h0 = L / N0
eps0 = 4 * h0
s = np.linspace(0, np.radians(20), 1200)
Q = R * np.stack([np.cos(s), np.sin(s)], 1)
nu = np.stack([np.cos(s), np.sin(s)], 1)
tg = np.stack([-np.sin(s), np.cos(s)], 1)
Xu, Pu_, mu = cloud(N0, False)
Xj, Pj, mj = cloud(N0, True)
Fu = eps0 * field(Q, Xu, Pu_, mu, eps0)
Fj = eps0 * field(Q, Xj, Pj, mj, eps0)
# smooth circle: a very fine uniform cloud
Fs = eps0 * field(Q[:3], *cloud(20000, False), eps0)
smooth = float(np.einsum("qa,qa->q", Fs, nu[:3]).mean())
assert abs(smooth - 2 * C_RHO) < 0.01, smooth
un, ut = np.einsum("qa,qa->q", Fu, nu), np.einsum("qa,qa->q", Fu, tg)
jn, jt = np.einsum("qa,qa->q", Fj, nu), np.einsum("qa,qa->q", Fj, tg)
assert np.all(un > 0) and np.abs(un - smooth).max() < 0.06 * smooth
assert 2.5 < np.abs(ut).max() < 3.5 and np.abs(jt).max() > 15
# at the uniform data points the tangential part vanishes by symmetry
Fd = eps0 * field(Xu[:3], Xu, Pu_, mu, eps0)
assert np.abs(np.einsum("qa,qa->q", Fd, np.stack([-Xu[:3, 1], Xu[:3, 0]], 1) / R)).max() < 1e-10

fig = plt.figure(figsize=(6.4, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3, height_ratios=[1, 1.15])
deg = np.degrees(s)
for k, (nn, tt_, X_, ttl, yl) in enumerate(((un, ut, Xu, "(a) uniform", 4.5), (jn, jt, Xj, "(b) jittered", 28))):
    ax = fig.add_subplot(gs[0, k])
    ax.set_ylim(-yl, yl * 1.15)
    ax.axhline(smooth, color=C["aux"], ls="--", lw=0.9)
    ax.axhline(0, color=C["aux"], lw=0.5)
    ax.plot(deg, nn, color=C["main"], lw=1.1, label="normal")
    ax.plot(deg, tt_, color=C["normal"], lw=1.0, label="tangential")
    ang = np.degrees(np.arctan2(X_[:, 1], X_[:, 0]))
    ang = ang[(ang >= 0) & (ang <= 20)]
    ax.plot(ang, np.full_like(ang, -0.9 * yl), "|", color=C["tangent"], ms=6)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r"position (degrees)")
    ax.set_title(ttl + r", $N = 250$, $\varepsilon = 4h$", fontsize=10)
    if k == 0:
        ax.set_ylabel(r"$\varepsilon\,(\delta V * \rho_\varepsilon)$")
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

Ns = np.array([250, 500, 1000, 2000, 4000])
rules = {
    r"uniform, $\varepsilon = 4h$": (False, lambda N: 4 * L / N, C["main"], "o-"),
    r"jittered, $\varepsilon = 4h$": (True, lambda N: 4 * L / N, C["normal"], "s-"),
    r"jittered, $\varepsilon \propto N^{-1/3}$": (True, lambda N: 4 * h0 * (N / N0) ** (-1 / 3), C["tangent"], "^-"),
    r"uniform, $\varepsilon = 16h$": (False, lambda N: 16 * L / N, C["third"], "D-"),
}
vals = {}
for name, (jit, epsf, col, mk) in rules.items():
    vals[name] = np.array([v.field_L1_circle(*cloud(N, jit), epsf(N), R)[0] for N in Ns])
ax = fig.add_subplot(gs[1, :])
for name, (jit, epsf, col, mk) in rules.items():
    ax.loglog(Ns, vals[name], mk, color=col, ms=3.5, lw=1.1, label=name)
ax.axhline(2 * np.pi, color=C["aux"], ls="--", lw=0.9)
ax.text(4400, 2 * np.pi * 0.95, r"$2\pi$", fontsize=10, color=C["aux"], va="top")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.minorticks_off()
ax.set_xlim(220, 5200)
ax.set_xlabel(r"$N$")
ax.set_title(r"(c) $\|\delta V * \rho_\varepsilon\|_{L^1(\mathbb{R}^2)}$", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_ylim(3, 3000)

V = list(vals.values())
assert V[0][-1] / V[0][0] > 8 and V[1][-1] / V[1][0] > 8                   # grows like N
assert np.all(np.diff(V[2]) < 0) and V[2][-1] < 2 * V[2][0] / 2            # decreases
assert np.all(np.abs(V[3] - 2 * np.pi) < 0.1)                               # ~ 2 pi
dgfig.save(fig, __file__)
print("smooth eps*|dV*rho| =", smooth, " 2 C_rho =", 2 * C_RHO)
print("max |tangential| jittered:", np.abs(jt).max(), " normal range jittered:", jn.min(), jn.max())
print("max |tangential| uniform:", np.abs(ut).max())
print("uniform normal range:", un.min(), un.max(), " max |ut|", np.abs(ut).max())
for name, val in vals.items():
    print(name, np.round(val, 3))
