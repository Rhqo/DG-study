"""Figure 38.1.3: pushing the unit sphere through itself in the naive way is not a regular homotopy.

x_lambda(theta, phi) = (sin theta cos phi, sin theta sin phi, (1 - 2 lambda) cos theta), lambda in [0, 1]:
lambda = 0 is the sphere, lambda = 1 its reflection in the plane z = 0 (the same set, normal x_theta x x_phi inward).
x_theta x x_phi = ((1 - 2 lambda) sin^2 theta cos phi, (1 - 2 lambda) sin^2 theta sin phi, sin theta cos theta),
so |x_theta x x_phi| = |1 - 2 lambda| on the equator theta = pi/2 and the family fails to be an immersion exactly
at lambda = 1/2, along the whole equator.

(a) meridian sections in the xz-plane: the right half (phi = 0) for lambda = 0, 1/4, the left half (phi = pi) for
lambda = 3/4, 1 (the same sets as lambda = 1/4, 0), the segment lambda = 1/2, with unit normals
N = x_theta x x_phi / |...| at five points of each half curve. (b) |x_theta x x_phi| at the equator against
lambda.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-1-3-naive-eversion.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg38  # noqa: E402

C = dgfig.COLORS


def cross_exact(th, ph, lam):
    s, c = np.sin(th), np.cos(th)
    return np.array([(1 - 2 * lam) * s * s * np.cos(ph), (1 - 2 * lam) * s * s * np.sin(ph), s * c + 0 * ph])


# self-checks: the closed form agrees with finite differences; equator value |1 - 2 lambda|; inward for lambda > 1/2
for lam in (0.0, 0.3, 0.5, 0.8, 1.0):
    for th, ph in ((0.4, 0.2), (1.3, 2.0), (np.pi / 2, 4.0)):
        a, b = ddg38.partials(ddg38.eversion, np.array(th), np.array(ph), lam=lam)
        assert np.allclose(np.cross(a, b), cross_exact(th, ph, lam), atol=1e-6)
    assert np.isclose(np.linalg.norm(cross_exact(np.pi / 2, 0.7, lam)), abs(1 - 2 * lam))
    th = np.linspace(0.05, np.pi - 0.05, 50)
    out = np.sum(cross_exact(th, 0.3, lam) * ddg38.eversion(th, 0.3, lam), 0)
    if lam < 0.5:
        assert np.all(out > 0)
    elif lam > 0.5:
        assert np.all(out < 0)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
th = np.linspace(0, np.pi, 300)
# right half (phi = 0): lambda = 0, 0.25 with outward normals; left half (phi = pi): lambda = 0.75, 1 (the same
# sets as lambda = 0.25, 0) with inward normals; the orange segment is lambda = 1/2.
halves = {0.0: (0.0, C["main"]), 0.25: (0.0, C["tangent"]), 0.75: (np.pi, C["tangent"]), 1.0: (np.pi, C["main"])}
for lam, (ph, col) in halves.items():
    p = ddg38.eversion(th, ph + 0 * th, lam)
    ax.plot(p[0], p[2], color=col, lw=1.5)
    for t in np.deg2rad([20, 55, 90, 125, 160]):
        q = ddg38.eversion(t, ph, lam)
        n = cross_exact(t, ph, lam)
        n = 0.22 * n / np.linalg.norm(n)
        ax.annotate("", xy=(q[0] + n[0], q[2] + n[2]), xytext=(q[0], q[2]),
                    arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.0, mutation_scale=8))
ax.plot([-1, 1], [0, 0], color=C["accent"], lw=2.4)
ax.plot([0, 0], [-1.25, 1.25], color=C["aux"], lw=0.6, ls=(0, (3, 2.5)))
ax.text(0.62, 1.02, r"$\lambda = 0$", ha="center", fontsize=11)
ax.text(0.36, 0.30, r"$\lambda = \frac{1}{4}$", ha="center", fontsize=11, color=C["tangent"])
ax.text(-0.62, 1.02, r"$\lambda = 1$", ha="center", fontsize=11)
ax.text(-0.36, 0.30, r"$\lambda = \frac{3}{4}$", ha="center", fontsize=11, color=C["tangent"])
ax.text(0.0, -0.16, r"$\lambda = \frac{1}{2}$", ha="center", fontsize=11, color=C["accent"])
ax.text(0.75, -1.22, r"$\varphi = 0$", ha="center", fontsize=10, color=C["aux"])
ax.text(-0.75, -1.22, r"$\varphi = \pi$", ha="center", fontsize=10, color=C["aux"])
ax.set_aspect("equal")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.3, 1.25)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$z$", rotation=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", loc="left", fontsize=12)

lam = np.linspace(0, 1, 201)
ax2.plot(lam, np.abs(1 - 2 * lam), color=C["main"], lw=1.6)
ax2.plot([0.5], [0.0], "o", color=C["accent"], ms=6)
for l0 in (0.0, 0.25, 0.75, 1.0):
    ax2.plot([l0], [abs(1 - 2 * l0)], "o", color=C["tangent"] if 0 < l0 < 1 else C["main"], ms=4)
ax2.annotate("crease", xy=(0.5, 0.0), xytext=(0.58, 0.32), fontsize=11, color=C["accent"],
             arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=0.9, mutation_scale=9))
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel(r"$|\mathbf{x}_\theta\times\mathbf{x}_\varphi|$ at $\theta = \pi/2$")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.08)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("(b)", loc="left", fontsize=12)
fig.tight_layout(w_pad=2.0)
dgfig.save(fig, __file__)
