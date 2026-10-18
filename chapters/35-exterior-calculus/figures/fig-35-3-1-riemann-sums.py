"""Figure 35.3.1: Riemann sums of the line integral of alpha = x dy around the unit circle
(Section 35.3, Proposition 35.3.5 and Example 35.3.6).

The circle gamma(t) = (cos t, sin t), t in [0, 2 pi], counterclockwise, is sampled at p_i = gamma(2 pi i / n).
The Riemann sum S_n = sum_i alpha_{p_i}(p_{i+1} - p_i) = sum_i x_i (y_{i+1} - y_i) equals (n/2) sin(2 pi / n) exactly.
(a) n = 8: the blue arrows are the chord vectors p_{i+1} - p_i; each number is alpha_{p_i}(p_{i+1} - p_i) at the tail
    point p_i (black dot). Grey: the circle. The eight numbers add up to S_8 = 2.828 (pi = 3.142).
(b) The error pi - S_n against n on log-log axes (black dots) with the reference line (2 pi^3 / 3) n^-2 (grey dashed):
    the error decays like n^-2.
Self-checks: S_n = (n/2) sin(2 pi / n) for every n; S_n -> pi; the n^-2 asymptotics; the integral of alpha along the
circle is pi by Gauss-Legendre quadrature (lib/ch35.line_integral_num).
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import ch35  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS


def riemann(n):
    t = 2 * np.pi * np.arange(n + 1) / n
    P = np.stack([np.cos(t), np.sin(t)], 1)
    vals = P[:-1, 0] * (P[1:, 1] - P[:-1, 1])
    return P, vals


ns = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
S = np.array([riemann(n)[1].sum() for n in ns])
assert np.allclose(S, ns / 2 * np.sin(2 * np.pi / ns))
err = np.pi - S
assert np.all(err > 0) and np.all(np.diff(err) < 0)
assert abs(err[-1] * ns[-1] ** 2 / (2 * np.pi ** 3 / 3) - 1) < 1e-3
exact = ch35.line_integral_num(lambda p: np.array([0.0, p[0]]), lambda t: np.array([np.cos(t), np.sin(t)]),
                               lambda t: np.array([-np.sin(t), np.cos(t)]), 0, 2 * np.pi)
assert abs(exact - np.pi) < 1e-12
P8, v8 = riemann(8)
assert abs(v8.sum() - 2 * np.sqrt(2)) < 1e-12

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw=dict(width_ratios=[1.05, 1]))
ts = np.linspace(0, 2 * np.pi, 300)
axA.plot(np.cos(ts), np.sin(ts), color=C["aux"], lw=0.9, zorder=1)
for i in range(8):
    p, q = P8[i], P8[i + 1]
    axA.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle="-|>", mutation_scale=11, color=C["tangent"], lw=1.4,
                                  shrinkA=0, shrinkB=0, zorder=4))
    axA.plot(*p, "o", color=C["main"], ms=3, zorder=5)
    m = 0.5 * (p + q)
    lab = m / np.linalg.norm(m) * 1.26
    txt = f"{v8[i]:+.3f}" if abs(v8[i]) > 5e-4 else "0"
    axA.text(lab[0], lab[1], txt, ha="center", va="center", fontsize=9.5, color=C["tangent"])
axA.text(0, 0.12, r"$\alpha = x\,dy$", ha="center", fontsize=11, color=C["covector"])
axA.text(0, -0.18, r"$S_8 = 2.828$", ha="center", fontsize=11)
dgfig.schematic_axes(axA, xlim=(-1.55, 1.55), ylim=(-1.5, 1.5))
axA.set_title(r"(a) $n = 8$", fontsize=12)

axB.loglog(ns, err, "o", color=C["main"], ms=4, zorder=3, label=r"$\pi - S_n$")
axB.loglog(ns, 2 * np.pi ** 3 / 3 / ns.astype(float) ** 2, "--", color=C["aux"], lw=1.0, zorder=2,
           label=r"$\frac{2\pi^3}{3}\,n^{-2}$")
axB.set_xlabel(r"$n$")
axB.set_ylabel("error")
axB.legend(frameon=False, fontsize=10, loc="upper right")
axB.set_title(r"(b) error of $S_n$", fontsize=12)
for sp_ in ("top", "right"):
    axB.spines[sp_].set_visible(False)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.9, wspace=0.3)
dgfig.save(fig, __file__)
