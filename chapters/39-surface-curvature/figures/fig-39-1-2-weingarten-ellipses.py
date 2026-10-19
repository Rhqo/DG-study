"""Figure 39.1.2: the Weingarten map W = -dN as a linear map of the tangent plane (torus of revolution).

Torus (R, r) = (2, 0.8) with N = x_u x x_v / |...| (inward). In the tangent plane we use the orthonormal basis
e1 = x_u / r (meridian), e2 = x_v / (R + r cos u) (circle of latitude). W e1 = (1/r) e1, W e2 = cos u/(R + r cos u) e2
(Example 39.1.3). (a) u = 0 (outer equator): eigenvalues 1.25, 0.357. (b) u = pi (inner equator): 1.25, -0.833.
Blue: unit vectors X(t) = cos t e1 + sin t e2, eight of them marked 0..7. Vermilion: W X(t). The small arrows show
the direction of travel: (a) same direction (det W > 0), (b) opposite (det W < 0).

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-1-2-weingarten-ellipses.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg39  # noqa: E402

C = dgfig.COLORS
R, r = ddg39.R_TORUS, ddg39.r_TORUS

panels = []
for u in (0.0, np.pi):
    W = ddg39.weingarten(ddg39.torus, u, 0.4)
    g, hh, n, xu, xv = ddg39.forms(ddg39.torus, u, 0.4)
    # orthonormal basis e1 = x_u / |x_u|, e2 = x_v / |x_v|; W in that basis = diag(|x_u|, |x_v|) Wmat diag(...)^-1
    S = np.diag([np.linalg.norm(xu), np.linalg.norm(xv)])
    Wo = S @ W @ np.linalg.inv(S)
    k_exact = np.array([1 / r, np.cos(u) / (R + r * np.cos(u))])
    assert np.allclose(Wo, np.diag(k_exact), atol=1e-6)               # Example 39.1.3
    assert n @ (-ddg39.torus(u, 0.4) + np.array([R * np.cos(0.4), R * np.sin(0.4), 0])) > 0   # inward
    panels.append((u, np.diag(k_exact)))

t = np.linspace(0, 2 * np.pi, 400)
tk = np.arange(8) * np.pi / 4
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
for ax, (u, Wd), title in zip(axes, panels, ("(a)", "(b)")):
    circ = np.stack([np.cos(t), np.sin(t)], 1)
    img = circ @ Wd.T
    ax.axhline(0, color=C["aux"], lw=0.6, zorder=0)
    ax.axvline(0, color=C["aux"], lw=0.6, zorder=0)
    ax.plot(*circ.T, color=C["tangent"], lw=1.6)
    ax.plot(*img.T, color=C["normal"], lw=1.8)
    Pk = np.stack([np.cos(tk), np.sin(tk)], 1)
    Qk = Pk @ Wd.T
    ax.plot(*Pk.T, "o", color=C["tangent"], ms=4)
    ax.plot(*Qk.T, "o", color=C["normal"], ms=4)
    for k in range(8):
        off = 0.16 * Pk[k]
        ax.text(*(Pk[k] + off), f"${k}$", color=C["tangent"], fontsize=10, ha="center", va="center")
        q = Qk[k]
        qn = q / max(np.linalg.norm(q), 1e-9)
        pos = q + np.array([0.0, -0.17]) if k % 4 == 0 else q - 0.17 * qn
        ax.text(*pos, f"${k}$", color=C["normal"], fontsize=10, ha="center", va="center")
    # direction arrows
    for curve, col, idx in ((circ, C["tangent"], 60), (img, C["normal"], 140)):
        ax.annotate("", xy=curve[idx + 6], xytext=curve[idx],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, mutation_scale=13))
    ax.text(1.62, -0.16, r"$e_1$", fontsize=12)
    ax.text(0.08, 1.45, r"$e_2$", fontsize=12)
    k1, k2 = Wd[0, 0], Wd[1, 1]
    ax.text(-1.6, -1.55, rf"$\kappa = {k1:.2f},\ {k2:.3f}$", fontsize=11)
    dgfig.schematic_axes(ax, (-1.75, 1.75), (-1.7, 1.6))
    ax.set_title(title + (r"  $u = 0$" if u == 0 else r"  $u = \pi$"), loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.06)
dgfig.save(fig, __file__)
