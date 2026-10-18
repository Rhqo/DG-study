"""Figure 37.4.3: nonzero edge lengths are not enough.

The triangles gamma_0 = (0, 0), gamma_1 = (1, 0), gamma_2(lambda) = (2, lambda): every edge length stays >= 1, but at
lambda = 0 the curve backtracks (turning angles +-pi are undefined) and the discrete turning number jumps from 1 to -1.
(a) the triangles for lambda = 0.6, 0.25, 0, -0.25, -0.6; (b) the turning angle sum / 2 pi and the minimal edge length
as functions of lambda.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-4-3-nonregular-path.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402

C = dgfig.COLORS


def tri(lam):
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, lam]])


lams = np.linspace(-1, 1, 401)
lams = lams[np.abs(lams) > 1e-9]
ks = np.array([dd.turning_angles(tri(l_), True).sum() / (2 * np.pi) for l_ in lams])
emin = np.array([dd.curve_edge_lengths(tri(l_), True).min() for l_ in lams])
assert np.allclose(ks[lams > 0], 1) and np.allclose(ks[lams < 0], -1) and np.allclose(emin, 1)
th0 = np.array([0.0, 0.0])
E0 = dd.curve_edge_vectors(tri(0.0), True)
T0 = E0 / np.linalg.norm(E0, axis=1, keepdims=True)
assert np.allclose(T0[2], -T0[1]) and np.allclose(T0[0], -T0[2])    # backtracking at gamma_2 and gamma_0

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
for j, lam in enumerate([0.6, 0.25, 0.0, -0.25, -0.6]):
    P = tri(lam) + np.array([0.0, -1.45 * j])
    col = C["normal"] if lam == 0 else (C["tangent"] if lam > 0 else C["accent"])
    for i in range(3):
        a, b = P[i], P[(i + 1) % 3]
        off = np.array([0.0, 0.07]) if (lam == 0 and i == 2) else np.zeros(2)
        ax.annotate("", xy=b + off, xytext=a + off, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3,
                                                                    mutation_scale=9, shrinkA=1, shrinkB=1))
    ax.plot(P[:, 0], P[:, 1], "o", color=C["main"], ms=3, zorder=5)
    lab = r"$\lambda = %g$" % lam
    ax.text(-2.35, P[0, 1] - 0.05, lab, fontsize=10, color=col)
    kk = "undefined" if lam == 0 else (r"$k = 1$" if lam > 0 else r"$k = -1$")
    ax.text(2.25, P[2, 1] - 0.05 if lam != 0 else P[2, 1] - 0.05, kk, fontsize=10, color=col)
ax.set_xlim(-2.4, 3.7)
ax.set_ylim(-6.6, 0.9)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")

bx.plot(lams[lams > 0], ks[lams > 0], color=C["main"], lw=1.6)
bx.plot(lams[lams < 0], ks[lams < 0], color=C["main"], lw=1.6, label=r"$\frac{1}{2\pi}\sum_i\theta_i$")
bx.plot([0, 0], [1, -1], color=C["aux"], lw=0.8, ls=":")
bx.text(0.05, 0.25, r"$\min_i \ell_i = 1$ for every $\lambda$", fontsize=10, color=C["tangent"])
bx.plot([0], [1], "o", mfc="white", mec=C["main"], ms=5, zorder=5)
bx.plot([0], [-1], "o", mfc="white", mec=C["main"], ms=5, zorder=5)
bx.set_xlabel(r"$\lambda$")
bx.set_yticks([-1, 0, 1])
bx.legend(fontsize=10, frameon=False, loc="upper left")
bx.set_title("(b)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
