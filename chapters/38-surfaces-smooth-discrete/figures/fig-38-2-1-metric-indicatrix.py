"""Figure 38.2.1: unit circles of the induced metric, drawn in the parameter plane (unit sphere, r = 1).

At grid points q of the parameter plane we draw {q + X : g_q(X, X) = eps^2} with eps = 0.16: the vectors X that
dx maps to vectors of length eps.
(a) standard coordinates (theta, phi): g = d theta^2 + sin^2 theta d phi^2, an ellipse with half-axes eps (theta
direction) and eps / sin theta (phi direction); phi is drawn horizontally.
(b) Mercator coordinates (phi, t): g = sech^2 t (d phi^2 + dt^2), a circle of radius eps cosh t.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-2-1-metric-indicatrix.py
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
eps = 0.16
a = np.linspace(0, 2 * np.pi, 120)


def indicatrix(x, q, order):
    """points q + X with g_q(X, X) = eps^2; ``order`` maps (first, second) parameter to the plot (horizontal, vertical)."""
    xu, xv = ddg38.partials(x, np.array(q[0]), np.array(q[1]))
    g = np.array([[xu @ xu, xu @ xv], [xu @ xv, xv @ xv]])
    w, Q = np.linalg.eigh(g)
    pts = Q @ (eps * np.stack([np.cos(a), np.sin(a)]) / np.sqrt(w)[:, None])
    assert np.allclose(np.einsum("in,ij,jn->n", pts, g, pts), eps ** 2)
    P = np.array(q)[:, None] + pts
    return P[order[0]], P[order[1]]


# self-checks: E, F, G of the two coordinate systems (r = 1)
xu, xv = ddg38.partials(ddg38.sphere, np.array(0.7), np.array(1.1))
assert np.allclose(ddg38.gram(xu, xv), (1.0, 0.0, np.sin(0.7) ** 2), atol=1e-8)
xu, xv = ddg38.partials(ddg38.mercator_inv, np.array(1.1), np.array(-0.8))
assert np.allclose(ddg38.gram(xu, xv), (1 / np.cosh(0.8) ** 2, 0.0, 1 / np.cosh(0.8) ** 2), atol=1e-8)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1]))
for th in np.linspace(np.pi / 8, 7 * np.pi / 8, 7):
    for ph in np.linspace(np.pi / 4, 7 * np.pi / 4, 7):
        X, Y = indicatrix(ddg38.sphere, (th, ph), (1, 0))
        ax.plot(X, Y, color=C["tangent"], lw=1.0)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(np.pi, 0)
ax.set_aspect("equal")
ax.set_xticks([0, np.pi, 2 * np.pi])
ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, np.pi / 2, np.pi])
ax.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_xlabel(r"$\varphi$")
ax.set_ylabel(r"$\theta$", rotation=0, labelpad=10)
ax.set_title("(a)", loc="left", fontsize=12)

for t in np.linspace(-1.5, 1.5, 7):
    for ph in np.linspace(np.pi / 4, 7 * np.pi / 4, 7):
        X, Y = indicatrix(ddg38.mercator_inv, (ph, t), (0, 1))
        ax2.plot(X, Y, color=C["tangent"], lw=1.0)
ax2.set_xlim(0, 2 * np.pi)
ax2.set_ylim(-2.1, 2.1)
ax2.set_aspect("equal")
ax2.set_xticks([0, np.pi, 2 * np.pi])
ax2.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax2.set_xlabel(r"$\varphi$")
ax2.set_ylabel(r"$t$", rotation=0)
ax2.set_title("(b)", loc="left", fontsize=12)
for axx in (ax, ax2):
    for s in ("top", "right"):
        axx.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.5)
dgfig.save(fig, __file__)
