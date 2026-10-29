"""Figure 46.8.2: what clone and split do to three different "amounts" of a flat Gaussian.

A flat Gaussian seen along its normal has the footprint g(x) = exp(-x^T Sigma^-1 x / 2) in its plane, Sigma = diag(s1^2, s2^2),
s1 = 0.3, s2 = 0.15, opacity alpha = 0.5. The rendered coverage of several footprints is int (1 - prod_i (1 - alpha g_i)) dx
(alpha compositing along the normal direction).
(a) coverage of the parent, of a clone (two identical copies at the same place, as in the official implementation) and of a
split (children with Sigma / 1.6^2 at mu +- 0.8 s1 r1, a typical draw from N(mu, Sigma)); 1-sigma ellipses in blue.
(b) after clone and after split, relative to the parent: the sum of 1-sigma ellipse areas pi s1 s2 (grey), the rendered
coverage (blue; for split the bar is the fixed draw of (a) and the whisker is the range over 400 random draws), and the
geometric mass of a conservative ledger (vermilion, kept at 1 by Proposition 46.8.6).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-8-2-split-clone-ledger.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

C = dgfig.COLORS
rng = np.random.default_rng(4682)
s1, s2, alpha = 0.3, 0.15, 0.5
SIG = np.diag([s1 ** 2, s2 ** 2])
g = np.linspace(-1.6, 1.6, 481)
GX, GY = np.meshgrid(g, g)
PTS = np.stack([GX, GY], axis=-1)
dA = (g[1] - g[0]) ** 2


def foot(mu, S):
    d = PTS - np.asarray(mu)
    return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, np.linalg.inv(S), d))


def coverage_map(items):
    T = np.ones(GX.shape)
    for mu, S in items:
        T *= 1 - alpha * foot(mu, S)
    return 1 - T


def coverage(items):
    return float(coverage_map(items).sum() * dA)


parent = [((0, 0), SIG)]
clone = [((0, 0), SIG), ((0, 0), SIG)]
SC = SIG / 1.6 ** 2
split = [((0.8 * s1, 0), SC), ((-0.8 * s1, 0), SC)]
c0 = coverage(parent)
# self-checks: one footprint covers alpha 2 pi s1 s2; a clone covers (2 - alpha/2) times as much; far-apart split 2/2.56
assert abs(c0 - alpha * 2 * math.pi * s1 * s2) < 1e-6
assert abs(coverage(clone) / c0 - (2 - alpha / 2)) < 1e-6
far = coverage([((0.6, 0), SC), ((-0.6, 0), SC)]) / c0
assert abs(far - 2 / 2.56) < 1e-3
draws = []
for _ in range(400):
    x1, x2 = rng.multivariate_normal([0, 0], SIG, size=2)
    draws.append(coverage([(x1, SC), (x2, SC)]) / c0)
draws = np.array(draws)
assert draws.max() <= 2 / 2.56 + 1e-6

fig = plt.figure(figsize=(6.4, 5.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45, wspace=0.08)
for k, (name, items) in enumerate((("parent", parent), ("clone", clone), ("split", split))):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(coverage_map(items), origin="lower", extent=(g[0], g[-1], g[0], g[-1]), cmap="Greys", vmin=0, vmax=1)
    for mu, S in items:
        w, h = 2 * math.sqrt(S[0, 0]), 2 * math.sqrt(S[1, 1])
        ax.add_patch(Ellipse(mu, w, h, facecolor="none", edgecolor=C["tangent"], lw=1.0))
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) {name}" if k == 0 else name, fontsize=11)

ax = fig.add_subplot(gs[1, :])
labels = ["clone", "split"]
area = [2.0, 2 / 2.56]
cov = [coverage(clone) / c0, coverage(split) / c0]
ledger = [1.0, 1.0]
x = np.arange(2)
wdt = 0.25
ax.bar(x - wdt, area, wdt, color=C["aux"], label=r"sum of $\pi s_1 s_2$")
ax.bar(x, cov, wdt, color=C["tangent"], label="rendered coverage")
ax.errorbar([1], [cov[1]], yerr=[[cov[1] - draws.min()], [draws.max() - cov[1]]], color=C["main"], capsize=3, lw=1)
ax.bar(x + wdt, ledger, wdt, color=C["normal"], label=r"mass ledger $\sum q_j$")
ax.axhline(1.0, color=C["main"], lw=0.6, ls=":")
for xi, vals in zip(x, zip(area, cov, ledger)):
    for dx, val in zip((-wdt, 0, wdt), vals):
        ax.text(xi + dx, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("relative to the parent")
ax.set_ylim(0, 2.45)
ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.4)
ax.set_title(r"(b) three amounts after refinement ($\alpha = 0.5$)", fontsize=11)

dgfig.save(fig, __file__)
print("coverage parent", round(c0, 4), "clone", round(cov[0], 4), "split", round(cov[1], 4),
      "draws", round(draws.min(), 4), round(draws.mean(), 4), round(draws.max(), 4))
