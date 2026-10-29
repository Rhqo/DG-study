"""Figure 46.2.2: point clouds on the unit circle converge (weak-*) to a length measure, but which one depends on
the weights.

(a) N = 24 nonuniform points t_j = 2 pi u_j + 0.5 sin(2 pi u_j), u_j = (j + 1/2)/N, drawn with area proportional to
the Voronoi (arc) weight, and the support of the bump test function phi (centre (-1, 0), radius 0.6).
(b) mu_N(phi) against N for three choices of weights: Voronoi weights (-> int phi dH^1), equal weights 2 pi / N
(-> int phi theta dH^1 with theta = 1 / (1 + 0.5 cos 2 pi u)), unit weights (grow like N).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-2-2-weak-convergence.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
A = 0.5
CENTER, RADIUS = np.array([-1.0, 0.0]), 0.6


def phi(X):
    return v.bump(X, CENTER, RADIUS)


# the two limits: int phi dH^1 and int phi theta dH^1 = 2 pi int_0^1 phi(x(t(u))) du
lim_vor = v.integrate_1d(lambda t: phi(v.circle(t)), 0, 2 * np.pi, n=64, pieces=64)
lim_eq = 2 * np.pi * v.integrate_1d(lambda u: phi(v.circle(v.warp(u, A))), 0, 1, n=64, pieces=64)

Ns = np.array([6, 12, 24, 48, 96, 192, 384, 768])
vals = {"vor": [], "eq": [], "unit": []}
for N in Ns:
    t, m_vor, m_eq = v.circle_samples(N, a=A)
    f = phi(v.circle(t))
    vals["vor"].append(np.sum(m_vor * f))
    vals["eq"].append(np.sum(m_eq * f))
    vals["unit"].append(np.sum(f))
vals = {k: np.array(x) for k, x in vals.items()}

# self-checks: Voronoi weights converge to the length measure, equal weights to theta H^1, unit weights grow like N
assert abs(vals["vor"][-1] - lim_vor) < 1e-4, (vals["vor"][-1], lim_vor)
assert abs(vals["eq"][-1] - lim_eq) < 1e-4, (vals["eq"][-1], lim_eq)
assert abs(vals["unit"][-1] / Ns[-1] - lim_eq / (2 * np.pi)) < 1e-4
assert lim_eq > 1.5 * lim_vor   # near t = pi the equal-weight cloud carries about twice the length

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.4})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.add_patch(Circle(CENTER, RADIUS, facecolor=C["region"], alpha=0.25, edgecolor=C["region"], lw=0.8, zorder=1))
ax.plot(*v.circle(tt).T, color=C["aux"], lw=0.8, ls=":", zorder=2)
t, m_vor, m_eq = v.circle_samples(24, a=A)
X = v.circle(t)
ax.scatter(X[:, 0], X[:, 1], s=60 * m_vor / m_vor.mean(), color=C["main"], zorder=3)
ax.text(-1.68, 0.72, r"supp $\varphi$", fontsize=11, ha="left", va="bottom")
ax.text(0.0, 0.0, r"$N = 24$", fontsize=11, ha="center", va="center")
ax.set_aspect("equal")
ax.set_xlim(-1.7, 1.25)
ax.set_ylim(-1.3, 1.3)
ax.set_axis_off()
ax.set_title("(a) samples, test function", fontsize=12)

ax = axes[1]
ax.loglog(Ns, vals["vor"], "o-", color=C["main"], lw=1.4, ms=4, label="Voronoi weights")
ax.loglog(Ns, vals["eq"], "s-", color=C["accent"], lw=1.4, ms=4, label=r"equal weights $2\pi/N$")
ax.loglog(Ns, vals["unit"], "^-", color=C["normal"], lw=1.4, ms=4, label="unit weights")
ax.axhline(lim_vor, color=C["aux"], lw=0.8, ls="--")
ax.axhline(lim_eq, color=C["accent"], lw=0.8, ls="--")
ax.text(720, lim_vor * 0.84, r"$\int\varphi\,d\mathcal{H}^1$", fontsize=10, ha="right", va="top")
ax.text(720, lim_eq * 1.12, r"$\int\varphi\,\theta\,d\mathcal{H}^1$", fontsize=10, ha="right", va="bottom")
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$\mu_N(\varphi)$")
ax.set_xticks(Ns[::2])
ax.set_xticklabels([str(n) for n in Ns[::2]])
ax.minorticks_off()
ax.set_ylim(0.3, 300)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_title(r"(b) $\mu_N(\varphi)$ vs. $N$", fontsize=12)

dgfig.save(fig, __file__)
print(f"limits: Voronoi {lim_vor:.4f}, equal {lim_eq:.4f}; N=24: {vals['vor'][2]:.4f} {vals['eq'][2]:.4f} {vals['unit'][2]:.4f}")
