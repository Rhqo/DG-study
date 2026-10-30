"""Figure 48.1.1: the key idea of [BLM22] on a point cloud of the ellipse (2 cos t, sin t).

N = 200 points x_l = (2 cos t_l, sin t_l), t_l = 2 pi l / N, exact tangent lines P_l = tau_l tau_l^T, masses
m_l = |gamma'(t_l)| 2 pi / N. At x_0 = x_25 (t = pi/4, curvature kappa = 0.5060) with eps = 0.3 we evaluate the
epsilon-WSFF-perp of [BLM22] (Definition 6.9, (6.22), with the 1/eps of (6.4)) and contract it with the outward
normal n_0: tau_0^T (sum_k B^k_ij (n_0)_k) tau_0 = -kappa_hat.
(a) zoom on x_0: points, tangent lines, the ball B_eps(x_0), the angle theta_l of one neighbour.
(b) for every neighbour the difference quotient theta_l / s_l of the tangent angle (s_l = <x_l - x_0, tau_0>),
    marker area proportional to the weight m_l xi(|x_l - x_0| / eps); their xi-weighted mean is the leading-order
    value of kappa_hat (Section 48.1); dashed: kappa(x_0); grey: theta(s)/s of the ellipse itself.

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-1-1-key-idea.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import var48 as v  # noqa: E402

C = dgfig.COLORS
N, I0, EPS, n, d = 200, 25, 0.3, 2, 1
t = 2 * np.pi * np.arange(N) / N
X = np.stack([2 * np.cos(t), np.sin(t)], axis=1)
G = np.stack([-2 * np.sin(t), np.cos(t)], axis=1)
speed = np.linalg.norm(G, axis=1)
T = G / speed[:, None]
P = v.proj_from_tangent(T)
m = speed * 2 * np.pi / N
x0, tau0 = X[I0], T[I0]
n0 = np.array([tau0[1], -tau0[0]])                      # outward normal (counterclockwise curve)
kappa = lambda tt: 2.0 / (4 * np.sin(tt) ** 2 + np.cos(tt) ** 2) ** 1.5  # noqa: E731


def angle_from(tau, U):
    """signed angle from the unit vector tau to the unit vectors U (rows)."""
    return np.arctan2(tau[0] * U[:, 1] - tau[1] * U[:, 0], U @ tau)


nb = v.ball(X, x0, EPS)
A = v.wsff_perp(x0, P[I0], X[nb], P[nb], m[nb], EPS, n, d)
B = v.B_from_A(A)
kappa_hat = -float(tau0 @ np.einsum("ijk,k->ij", B, n0) @ tau0)

oth = nb[nb != I0]
s = (X[oth] - x0) @ tau0
theta = angle_from(tau0, T[oth])
wts = m[oth] * v.xi(np.linalg.norm(X[oth] - x0, axis=1) / EPS, n)
quot = theta / s
mean_quot = float(np.sum(wts * quot) / np.sum(wts))

# self-checks
assert len(nb) == 12 and I0 in nb
assert abs(t[I0] - np.pi / 4) < 1e-12 and abs(kappa(t[I0]) - 0.5060) < 1e-4
assert np.allclose(A, np.transpose(A, (0, 2, 1)))                                # A_ijk = A_ikj
assert abs(kappa_hat - 0.5074) < 1e-4
assert abs(mean_quot - 0.5128) < 1e-4
assert np.all(theta * s > 0)                                                      # tangents turn inwards
assert abs(quot.min() - 0.445) < 1e-3 and abs(quot.max() - 0.611) < 1e-3

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(2 * np.cos(tt), np.sin(tt), color=C["aux"], lw=0.6, ls="--")
ax.plot(X[:, 0], X[:, 1], "o", color=C["main"], ms=1.2, zorder=4)
ax.add_patch(Circle(x0, EPS, facecolor=C["region"], alpha=0.3, edgecolor=C["region"], lw=0.8))
for k in nb:
    a_, b_ = X[k] - 0.1 * T[k], X[k] + 0.1 * T[k]
    ax.plot([a_[0], b_[0]], [a_[1], b_[1]], color=C["tangent"], lw=1.0, zorder=3)
ax.plot(*x0, "o", color=C["normal"], ms=4.5, zorder=5)
ax.annotate(r"$x_0$", x0, xytext=(-14, -16), textcoords="offset points", fontsize=12)
ax.annotate(r"$B_\varepsilon(x_0)$", (x0[0] + 0.21, x0[1] + 0.21), xytext=(6, 10), textcoords="offset points",
            fontsize=11, arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
ax.set_aspect("equal")
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.1, 1.25)
ax.set_axis_off()
ax.set_title(r"(a) $N = 200$, $\varepsilon = 0.3$", fontsize=11)

ax = axes[1]
tc = np.linspace(t[I0] - 0.3, t[I0] + 0.3, 600)
xc = np.stack([2 * np.cos(tc), np.sin(tc)], axis=1)
gc = np.stack([-2 * np.sin(tc), np.cos(tc)], axis=1)
gc /= np.linalg.norm(gc, axis=1)[:, None]
sc = (xc - x0) @ tau0
thc = angle_from(tau0, gc)
ok = (np.abs(sc) > 1e-4) & (np.linalg.norm(xc - x0, axis=1) < EPS)
ax.plot(sc[ok], (thc / sc)[ok], color=C["aux"], lw=1.0)
ax.scatter(s, quot, s=70 * wts / wts.max() + 6, color=C["accent"], edgecolor=C["main"], lw=0.5, zorder=4)
ax.axhline(kappa(t[I0]), color=C["main"], lw=0.8, ls="--")
ax.axhline(mean_quot, color=C["normal"], lw=1.3)
ax.text(-0.29, kappa(t[I0]) - 0.004, r"$\kappa(x_0) = 0.506$", fontsize=10, va="top", ha="left")
ax.text(-0.29, mean_quot + 0.004, r"weighted mean $%.3f$" % mean_quot, fontsize=10, va="bottom", ha="left",
        color=C["normal"])
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(0.42, 0.63)
ax.set_xlabel(r"$s_l = \langle x_l - x_0, \tau_0\rangle$")
ax.set_ylabel(r"$\theta_l \,/\, s_l$")
ax.set_title(r"(b) difference quotients", fontsize=11)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

print(f"neighbours {len(nb)}, kappa_hat = {kappa_hat:.4f}, weighted mean of quotients = {mean_quot:.4f}")
dgfig.save(fig, __file__)
