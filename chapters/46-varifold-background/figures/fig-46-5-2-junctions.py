"""Figure 46.5.2: junctions. The first variation at a junction point p is -(sum of the unit directions) delta_p.

(a) three segments from a junction p to the fixed endpoints a1 = (-1, 0), a2 = (1, 0), a3 = (0, 1). Solid: the T junction
p = (0, 0), with the unit directions u_l (blue) and their sum u1 + u2 + u3 = (0, 1) (vermilion), the direction in which
moving p shortens the network fastest. Dashed: the 120-degree junction at p* = (0, 1/sqrt3), where the sum is 0.
(b) total length L(p) for p = (0, y): L = 2 sqrt(1 + y^2) + (1 - y); slope -1 at y = 0, minimum 1 + sqrt3 at y = 1/sqrt3.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-5-2-junctions.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
A = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def units(p):
    d = A - p
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def length(p):
    return float(np.sum(np.linalg.norm(A - p, axis=1)))


p0 = np.zeros(2)
ps = np.array([0.0, 1 / math.sqrt(3)])
U0, Us = units(p0), units(ps)
# self-checks: grad L(p) = -sum u_l; the sum vanishes at p* and the angles there are 120 degrees
h = 1e-6
grad = np.array([(length(p0 + h * e) - length(p0 - h * e)) / (2 * h) for e in np.eye(2)])
assert np.allclose(grad, -U0.sum(axis=0), atol=1e-8) and np.allclose(U0.sum(axis=0), [0, 1])
assert np.allclose(Us.sum(axis=0), 0, atol=1e-12)
for i in range(3):
    for j in range(i + 1, 3):
        assert abs(math.degrees(math.acos(Us[i] @ Us[j])) - 120) < 1e-9
assert abs(length(ps) - (1 + math.sqrt(3))) < 1e-12 and abs(length(p0) - 3) < 1e-12

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})
ax = axes[0]
for a in A:
    ax.plot([p0[0], a[0]], [p0[1], a[1]], color=C["main"], lw=1.6)
    ax.plot([ps[0], a[0]], [ps[1], a[1]], color=C["aux"], lw=1.1, ls="--")
ax.plot(*A.T, "s", color=C["main"], ms=4)
for u in U0:
    ax.annotate("", xy=p0 + 0.45 * u, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.5,
                                                              shrinkA=0, shrinkB=0, mutation_scale=10))
s = U0.sum(axis=0)
ax.annotate("", xy=p0 + 0.45 * s + np.array([0.12, 0]), xytext=p0 + np.array([0.12, 0]),
            arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=11))
ax.text(0.27, 0.2, r"$u_1 + u_2 + u_3$", fontsize=10, color=C["normal"], ha="left", va="center")
ax.plot(*p0, "o", color=C["main"], ms=4)
ax.plot(*ps, "o", color=C["aux"], ms=4)
ax.text(-0.06, -0.08, r"$p$", fontsize=11, ha="right", va="top")
ax.text(-0.08, ps[1] + 0.02, r"$p^*$", fontsize=11, ha="right", va="bottom", color=C["aux"])
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-0.25, 1.12)
ax.set_axis_off()
ax.set_title("(a) T and Y junctions", fontsize=12)

ax = axes[1]
y = np.linspace(-0.3, 1.0, 400)
L = 2 * np.sqrt(1 + y ** 2) + (1 - y)
ax.plot(y, L, color=C["main"], lw=1.5)
ax.plot([-0.25, 0.25], [3 + 0.25, 3 - 0.25], color=C["normal"], lw=1.2, ls="--")
ax.plot(0, 3, "o", color=C["main"], ms=4)
ax.plot(1 / math.sqrt(3), 1 + math.sqrt(3), "o", color=C["aux"], ms=5)
ax.text(0.05, 3.05, r"slope $-1$", fontsize=10, ha="left", va="bottom", color=C["normal"])
ax.text(1 / math.sqrt(3), 1 + math.sqrt(3) - 0.05, r"$1+\sqrt{3}$", fontsize=10, ha="center", va="top")
ax.set_xlabel(r"$y$  ($p = (0, y)$)")
ax.set_ylabel(r"total length $L$")
ax.set_ylim(2.6, 3.45)
ax.set_title(r"(b) $L$ along the stem", fontsize=12)

dgfig.save(fig, __file__)
