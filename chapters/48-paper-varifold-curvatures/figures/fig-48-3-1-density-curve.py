"""Figure 48.3.1: a curve with varying density separates the WSFF from the WSFF-perp.

Unit circle with multiplicity theta(s) = 1 + 0.5 cos s, realised as a point cloud: N = 4000 points
s_l = F^{-1}(2 pi (l + 1/2)/N) with F(s) = s + 0.5 sin s (so the points are denser where theta is larger),
exact tangent lines P_l, equal masses m_l = 2 pi / N. At 48 of the points (every 83rd) we evaluate, with eps = 0.1
and the kernels of [BLM22] Section 7.1 (eta = rho in c^{V,eps}):
  the epsilon-WSFF (Definition 6.5, solved with (3.5)), the epsilon-WSFF-perp ((6.22), c^V(x_l) = P_l),
  and the approximate mean curvature H^{V,eps} = sum_q beta_qiq ((6.6)).
For the smooth varifold v(C, theta) the limits are (Example 48.3.10):
  sum_q A_{q i q}: WSFF -> kappa + (theta'/2 theta) tau, WSFF-perp -> kappa, H^V -> kappa + (theta'/theta) tau.
(a) the cloud (every 20th point drawn) and, at 12 points, the vectors sum_q A_{q.q} of the WSFF and of the WSFF-perp.
(b) their tangential components <., tau> against s, with the limits theta'/theta, theta'/(2 theta), 0.

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-3-1-density-curve.py
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

import var48 as v  # noqa: E402

C = dgfig.COLORS
AMP, N, EPS, n, d = 0.5, 4000, 0.1, 2, 1
theta = lambda s: 1 + AMP * np.cos(s)                       # noqa: E731
g = lambda s: -AMP * np.sin(s) / (1 + AMP * np.cos(s))      # noqa: E731  theta'/theta


def F_inv(y):
    s = np.array(y, float)
    for _ in range(60):
        s -= (s + AMP * np.sin(s) - y) / theta(s)
    return s


s = F_inv(2 * np.pi * (np.arange(N) + 0.5) / N)
X = np.stack([np.cos(s), np.sin(s)], 1)
T = np.stack([-np.sin(s), np.cos(s)], 1)
P = v.proj_from_tangent(T)
m = np.full(N, 2 * np.pi / N)

idx = np.arange(0, N, N // 48)[:48]
HW, HP, HH = [], [], []
for i in idx:
    nb = v.ball(X, X[i], EPS)
    HW.append(np.einsum("qiq->i", v.wsff(X[i], X[nb], P[nb], m[nb], EPS, n, d)))
    HP.append(np.einsum("qiq->i", v.wsff_perp(X[i], P[i], X[nb], P[nb], m[nb], EPS, n, d)))
    HH.append(v.H_from_beta(v.beta_tensor(X[i], X[nb], P[nb], m[nb], EPS, n, d)))
HW, HP, HH = map(np.array, (HW, HP, HH))
tW, tP, tH = (np.einsum("li,li->l", Z, T[idx]) for Z in (HW, HP, HH))
gi = g(s[idx])

# self-checks
assert abs(F_inv(np.array([np.pi]))[0] - np.pi) < 1e-12
assert np.abs(tW - 0.5 * gi).max() < 0.002                 # WSFF: tangential part theta'/(2 theta)
assert np.abs(tH - gi).max() < 0.003                       # H^V: tangential part theta'/theta
assert np.abs(tP).max() < 1e-12                            # WSFF-perp: no tangential part
assert np.all(np.abs(np.einsum("li,li->l", HP, X[idx]) + 1) < 0.002)   # normal part: curvature vector -x
assert abs(np.abs(gi).max() - 0.5773) < 2e-3

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.35})
ax = axes[0]
ax.plot(X[::20, 0], X[::20, 1], "o", color=C["main"], ms=1.3)
for i_, l in enumerate(idx[::4]):
    k = i_ * 4
    for Z, col, z in ((HW, C["normal"], 4), (HP, C["tangent"], 3)):
        ax.annotate("", xy=X[l] + 0.42 * Z[k], xytext=X[l],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=7), zorder=z)
ax.plot([], [], color=C["normal"], lw=1.2, label=r"WSFF")
ax.plot([], [], color=C["tangent"], lw=1.2, label=r"WSFF$^\perp$")
ax.legend(loc="center", fontsize=8.5, frameon=False, handlelength=1.2)
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_axis_off()
ax.set_title(r"(a) $\sum_q A_{q\cdot q}$, scaled by $0.42$", fontsize=10.5)

ax = axes[1]
ss = np.linspace(0, 2 * np.pi, 400)
ax.plot(ss, g(ss), color=C["aux"], lw=1.0, ls="--", label=r"$\theta'/\theta$")
ax.plot(ss, 0.5 * g(ss), color=C["aux"], lw=1.0, label=r"$\theta'/2\theta$")
ax.axhline(0, color=C["aux"], lw=0.6, ls=":")
ax.plot(s[idx], tH, "s", ms=3.0, color=C["third"], label=r"$\vec H^{V,\varepsilon}$")
ax.plot(s[idx], tW, "o", ms=3.0, color=C["normal"], label=r"WSFF")
ax.plot(s[idx], tP, "^", ms=3.0, color=C["tangent"], label=r"WSFF$^\perp$")
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"tangential component")
ax.set_title(r"(b) $\langle \sum_q A_{q\cdot q},\, \tau\rangle$", fontsize=10.5)
ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02), fontsize=8.5, frameon=False, ncol=1, handlelength=1.6, labelspacing=0.3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

print(f"max |t_W - g/2| = {np.abs(tW - 0.5 * gi).max():.4f}, max |t_H - g| = {np.abs(tH - gi).max():.4f}, "
      f"max |t_perp| = {np.abs(tP).max():.1e}")
dgfig.save(fig, __file__)
