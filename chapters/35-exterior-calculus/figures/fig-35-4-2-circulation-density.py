"""Figure 35.4.2: the exterior derivative as circulation per unit area (Section 35.4, Proposition 35.4.13).

The 1-form alpha = r dtheta = (-y dx + x dy)/r = star dr on R^2 minus the origin has d alpha = dx ^ dy / r
(Example 35.2.7(c)). At p = (1, 0.5) the coefficient is 1/|p| = 0.894.
(a) The squares Q_eps = p + [0, eps]^2 for eps = 0.8, 0.4, 0.2 with their counterclockwise boundaries (blue arrows on
    the largest one, black outlines); the black dot at the lower left is the origin.
(b) The circulation per area (1/eps^2) * (line integral of alpha around the boundary of Q_eps) against eps
    (black dots), with the limit (d alpha)_p(e1, e2) = 0.894 (grey dashed line). The error is about proportional
    to eps.
Self-checks: for each eps the boundary integral equals the integral of d alpha over Q_eps (Corollary 35.4.10,
Gauss-Legendre); the ratio converges to 1/|p| with error O(eps).
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import ch35  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

C = dgfig.COLORS
p = np.array([1.0, 0.5])
limit = 1 / np.linalg.norm(p)


def alpha(q):
    r = np.hypot(*q)
    return np.array([-q[1], q[0]]) / r


def loop_integral(eps):
    corners = [p, p + [eps, 0], p + [eps, eps], p + [0, eps], p]
    tot = 0.0
    for a, b in zip(corners[:-1], corners[1:]):
        tot += ch35.line_integral_num(alpha, lambda t, a=a, b=b: a + t * (b - a), lambda t, a=a, b=b: b - a, 0, 1, m=30)
    return tot


def area_integral(eps, m=30):
    t, w = ch35.gauss01(m)
    return sum(wi * wj * eps ** 2 / np.hypot(p[0] + eps * ti, p[1] + eps * tj)
               for ti, wi in zip(t, w) for tj, wj in zip(t, w))


eps_list = np.array([0.8, 0.4, 0.2, 0.1, 0.05, 0.025])
ratio = []
for e in eps_list:
    L = loop_integral(e)
    assert abs(L - area_integral(e)) < 1e-12
    ratio.append(L / e ** 2)
ratio = np.array(ratio)
err = np.abs(ratio - limit)
assert np.all(np.diff(err) < 0) and abs(err[-1] / err[-2] - 0.5) < 0.05
assert abs(limit - 0.894) < 5e-4

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
for e in (0.8, 0.4, 0.2):
    axA.add_patch(Rectangle(tuple(p), e, e, facecolor=C["region"], alpha=0.18, edgecolor=C["main"], lw=0.9, zorder=2))
corners = [p, p + [0.8, 0], p + [0.8, 0.8], p + [0, 0.8], p]
for a, b in zip(corners[:-1], corners[1:]):
    axA.add_patch(FancyArrowPatch(tuple(a + 0.5 * (b - a)), tuple(a + 0.85 * (b - a)), arrowstyle="-|>",
                                  mutation_scale=11, color=C["tangent"], lw=1.5, zorder=4))
axA.plot(*p, "o", color=C["main"], ms=3.5, zorder=5)
axA.text(p[0] - 0.05, p[1] - 0.15, r"$p$", fontsize=12, ha="right")
for e, yo in [(0.8, 0.82), (0.4, 0.42), (0.2, 0.22)]:
    axA.text(p[0] + e + 0.03, p[1] + e - 0.02, rf"$\varepsilon = {e}$", fontsize=10, va="top")
axA.plot([0], [0], "o", color=C["main"], ms=2.5)
axA.text(0.05, -0.12, r"$0$", fontsize=10)
dgfig.schematic_axes(axA, xlim=(-0.15, 2.45), ylim=(-0.25, 1.75))
axA.set_title(r"(a) squares $Q_\varepsilon$ at $p$", fontsize=12)

axB.plot(eps_list, ratio, "o-", color=C["main"], ms=4, lw=1.0, zorder=3)
axB.axhline(limit, color=C["aux"], ls="--", lw=1.0, zorder=1)
axB.text(0.45, limit - 0.012, r"$(d\alpha)_p(e_1, e_2) = 1/|p| = 0.894$", fontsize=10, va="top", ha="center",
         color="#555555")
axB.set_xscale("log")
axB.set_xlabel(r"$\varepsilon$")
axB.set_ylabel("circulation / area")
axB.set_title(r"(b) $\varepsilon^{-2}\,\int_{\partial Q_\varepsilon}\alpha$", fontsize=12)
for sp_ in ("top", "right"):
    axB.spines[sp_].set_visible(False)
fig.subplots_adjust(left=0.01, right=0.98, bottom=0.17, top=0.9, wspace=0.32)
dgfig.save(fig, __file__)
print("ratios:", np.round(ratio, 4))
