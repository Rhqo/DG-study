"""Figure 46.8.1: the plane of a Gaussian is read from its covariance, and it is only as stable as the eigenvalue gap.

(a) In R^2 (a cross-section): four Gaussians with s1 = 0.4 and s2 = 0.05, 0.2, 0.32, 0.38 (1-sigma ellipses). Blue: the
line P = r1 r1^T of the major axis. Grey: the major-axis lines of 25 perturbed covariances Sigma + E with random symmetric
E, ||E|| = 0.01 (operator norm). The fan opens as s2 approaches s1 (gap s1^2 - s2^2 = 0.1575, 0.12, 0.0576, 0.0156).
(b) In R^3: s = (0.3, 0.2, s3); the largest observed ||P(Sigma + E) - P(Sigma)|| / ||E|| over 300 random symmetric E with
||E|| = 1e-6, against the first-order bound 1 / (s2^2 - s3^2) of Proposition 46.8.2 (dashed).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-8-1-plane-from-covariance.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

C = dgfig.COLORS
rng = np.random.default_rng(468)


def rand_sym(n, size):
    E = rng.normal(size=(n, n))
    E = 0.5 * (E + E.T)
    return E * size / np.linalg.norm(E, 2)


def plane(S, k):
    """projection onto the span of the k eigenvectors with the largest eigenvalues."""
    w, U = np.linalg.eigh(S)
    V = U[:, -k:]
    return V @ V.T


# (b) data, with self-check against the first-order bound
s1, s2 = 0.3, 0.2
s3s = np.linspace(0.01, 0.19, 19)
ratio = []
for s3 in s3s:
    S = np.diag([s1 ** 2, s2 ** 2, s3 ** 2])
    P0 = plane(S, 2)
    worst = 0.0
    for _ in range(300):
        E = rand_sym(3, 1e-6)
        worst = max(worst, np.linalg.norm(plane(S + E, 2) - P0, 2) / 1e-6)
    ratio.append(worst)
ratio = np.array(ratio)
bound = 1 / (s2 ** 2 - s3s ** 2)
assert np.all(ratio <= bound * (1 + 1e-3))
assert np.all(ratio >= 0.5 * bound)

fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.9), gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.25})
ax = axes[0]
for k, minor in enumerate((0.05, 0.2, 0.32, 0.38)):
    c = np.array([1.0 * k, 0.0])
    S = np.diag([0.4 ** 2, minor ** 2])
    ax.add_patch(Ellipse(c, 2 * 0.4, 2 * minor, facecolor=C["region"], alpha=0.35, edgecolor=C["tangent"], lw=0.8))
    for _ in range(25):
        P = plane(S + rand_sym(2, 0.01), 1)
        u = P[:, 0] / np.linalg.norm(P[:, 0]) if np.linalg.norm(P[:, 0]) > 1e-9 else P[:, 1] / np.linalg.norm(P[:, 1])
        ax.plot(*np.stack([c - 0.48 * u, c + 0.48 * u]).T, color=C["aux"], lw=0.6, alpha=0.9)
    ax.plot([c[0] - 0.48, c[0] + 0.48], [0, 0], color=C["tangent"], lw=1.6)
    ax.text(c[0], -0.55, rf"$s_2 = {minor}$", fontsize=10, ha="center", va="top")
ax.set_aspect("equal")
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(-0.75, 0.55)
ax.set_axis_off()
ax.set_title(r"(a) in $\mathbb{R}^2$: $s_1 = 0.4$, $\|E\| = 0.01$", fontsize=11)

ax = axes[1]
ax.semilogy(s3s, ratio, "o", color=C["normal"], ms=4, label="observed")
ss = np.linspace(0.005, 0.195, 200)
ax.semilogy(ss, 1 / (s2 ** 2 - ss ** 2), "--", color=C["main"], lw=1.1, label=r"$1/(s_2^2 - s_3^2)$")
ax.set_xlabel(r"$s_3$  ($s_1 = 0.3$, $s_2 = 0.2$)")
ax.set_ylabel(r"$\|\delta P\|\,/\,\|E\|$")
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
ax.set_yticks([30, 100, 300, 1000])
ax.set_yticklabels(["30", "100", "300", "1000"])
ax.minorticks_off()
ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.4, handletextpad=0.3)
ax.set_title(r"(b) in $\mathbb{R}^3$: plane sensitivity", fontsize=11)

dgfig.save(fig, __file__)
print("ratio/bound", np.round(ratio / bound, 3))
