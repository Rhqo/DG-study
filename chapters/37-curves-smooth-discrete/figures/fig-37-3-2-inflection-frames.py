"""Figure 37.3.2: the Frenet frame flips at an inflection point, the Bishop frame does not.

The plane cubic gamma(t) = (t, t^3, 0) viewed as a space curve (Non-example 5.1.11). (a) the principal normal
n = t'/kappa = sgn(t) (-3t^2, 1, 0)/sqrt(1 + 9t^4), undefined at t = 0; (b) the Bishop vector u = J t =
(-3t^2, 1, 0)/sqrt(1 + 9t^4) (Example 37.3.10), continuous through t = 0.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-3-2-inflection-frames.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
tt = np.linspace(-1.05, 1.05, 400)
curve = np.stack([tt, tt ** 3], 1)
tj = np.array([-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0])
P = np.stack([tj, tj ** 3], 1)
T = np.stack([np.ones_like(tj), 3 * tj ** 2], 1) / np.sqrt(1 + 9 * tj ** 4)[:, None]
U = np.stack([-3 * tj ** 2, np.ones_like(tj)], 1) / np.sqrt(1 + 9 * tj ** 4)[:, None]   # J t
Nf = np.sign(tj)[:, None] * U                                                         # principal normal
# self-checks: n is t'/|t'| (finite differences), u = J t, kappa = 6|t|/(1+9t^4)^{3/2}
for t0, n0 in zip(tj, Nf):
    h = 1e-6
    tp = lambda s: np.array([1.0, 3 * s * s]) / np.sqrt(1 + 9 * s ** 4)   # noqa: E731
    d = (tp(t0 + h) - tp(t0 - h)) / (2 * h)
    assert np.allclose(d / np.linalg.norm(d), n0, atol=1e-6)
assert np.allclose(U, T @ np.array([[0.0, 1.0], [-1.0, 0.0]]))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, V, role, lab, title in ((axes[0], Nf, "normal", r"$\mathbf{n}$", "(a)"),
                                (axes[1], U, "accent", r"$\mathbf{u}$", "(b)")):
    ax.plot(curve[:, 0], curve[:, 1], color=C["main"], lw=dgfig.LW["main"])
    for p, tv, v in zip(P, T, V):
        ax.annotate("", xy=p + 0.28 * tv, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3,
                                                                   mutation_scale=8))
        ax.annotate("", xy=p + 0.32 * v, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C[role], lw=1.4,
                                                                  mutation_scale=9))
    ax.plot([0], [0], "o", ms=5, mfc="white", mec=C["main"], zorder=5)
    ax.text(P[-1, 0] + 0.05, P[-1, 1] + 0.35, lab, color=C[role], fontsize=12)
    ax.text(P[-1, 0] + 0.2, P[-1, 1] + 0.02, r"$\mathbf{t}$", color=C["tangent"], fontsize=12)
    ax.text(0.08, -0.35, r"$\kappa = 0$", fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.35, 1.6)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc="left")
fig.tight_layout()
dgfig.save(fig, __file__)
