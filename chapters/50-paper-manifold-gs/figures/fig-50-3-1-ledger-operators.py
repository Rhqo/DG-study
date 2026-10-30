"""Figure 50.3.1: the three ledger operators of [ManifoldGS26] Section 4 on the examples of Section 50.3.

(a) split, seen from above the plane z = 0: the parent (mass 1, scales 0.3 and 0.15), the two positions drawn by the
3DGS rule (t1 = (0.24, 0.06), t2 = (0.12, -0.09), scales / 1.6) and the conservative split that recentres them to
+-(0.06, 0.075) (Example 50.3.5). (b) merge, seen from the side (x-z plane): two atoms of mass 1 whose normals differ by
30 degrees and the merged atom (mass 2, barycentre, plane = top-2 eigenspace of the mean projection, Example 50.3.7).
(c) prune on a line: two removed atoms of mass 0.05 move their mass to the nearest kept atom (Example 50.3.9).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-3-1-ledger-operators.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

C = dgfig.COLORS

# ---------------------------------------------------------------- the numbers of the three examples
s1, s2 = 0.3, 0.15
raw = np.array([[0.24, 0.06], [0.12, -0.09]])
tbar = raw.mean(0)
kids = raw - tbar
assert np.allclose(tbar, [0.18, -0.015]) and np.allclose(kids, [[0.06, 0.075], [-0.06, -0.075]])
assert abs(2 / 1.6 ** 2 - 0.78125) < 1e-12                       # 3DGS split: ellipse area x 0.78
th = math.radians(30)
n1, n2 = np.array([0.0, 0.0, 1.0]), np.array([math.sin(th), 0.0, math.cos(th)])
Pbar = 0.5 * (np.eye(3) - np.outer(n1, n1)) + 0.5 * (np.eye(3) - np.outer(n2, n2))
w, U = np.linalg.eigh(Pbar)
m = U[:, 0] * np.sign(U[2, 0])
P = np.eye(3) - np.outer(m, m)
defect = 2 * np.linalg.norm(Pbar - P)
assert abs(math.degrees(math.acos(m[2])) - 15) < 1e-9 and abs(defect - 2 * math.sqrt(2) * math.sin(th / 2) ** 2) < 1e-12
kept, removed, qr = np.array([0.0, 1.0]), np.array([0.1, 0.9]), 0.05
dest = kept[np.abs(removed[:, None] - kept[None, :]).argmin(1)]
assert abs(np.sum(qr * (dest - removed))) < 1e-15 and abs(np.sum(qr * np.abs(dest - removed)) - 0.01) < 1e-15

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0]))

# ---------------------------------------------------------------- (a) split
ax.add_patch(Ellipse((0, 0), 2 * s1, 2 * s2, fc=C["surface"], ec=C["aux"], lw=0.8, alpha=0.6))
ax.plot(0, 0, "x", color=C["main"], ms=7, mew=1.6)
for t in raw:
    ax.add_patch(Ellipse(t, 2 * s1 / 1.6, 2 * s2 / 1.6, fc="none", ec=C["aux"], lw=0.9, ls="--"))
    ax.plot(*t, "o", mfc="white", mec=C["aux"], ms=5)
ax.plot(*tbar, "x", color=C["aux"], ms=6, mew=1.3)
for t, k in zip(raw, kids):
    ax.add_patch(Ellipse(k, 2 * s1 / 1.6, 2 * s2 / 1.6, fc="none", ec=C["tangent"], lw=1.3))
    ax.plot(*k, "o", color=C["tangent"], ms=5)
    ax.annotate("", xy=k, xytext=t, arrowprops=dict(arrowstyle="->", color=C["tangent"], lw=0.8, shrinkA=3, shrinkB=3))
ax.text(0.40, -0.17, "3DGS", color=C["aux"], fontsize=8.5, ha="center")
ax.text(-0.37, -0.21, "ledger", color=C["tangent"], fontsize=8.5, ha="center")
ax.set_xlim(-0.48, 0.55)
ax.set_ylim(-0.27, 0.27)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("(a) split", fontsize=11)

# ---------------------------------------------------------------- (b) merge (x-z plane, half length 0.09)
L = 0.09


def seg(axis, c, n, color, lw, ls="-"):
    d = np.array([n[2], -n[0]])                                     # in-plane direction in the x-z plane
    axis.plot([c[0] - L * d[0], c[0] + L * d[0]], [c[1] - L * d[1], c[1] + L * d[1]], color=color, lw=lw, ls=ls)
    axis.annotate("", xy=(c[0] + 0.07 * n[0], c[1] + 0.07 * n[2]), xytext=c,
                  arrowprops=dict(arrowstyle="->", color=C["normal"], lw=1.0, shrinkA=0, shrinkB=0))


seg(bx, (0.0, 0.0), n1, C["main"], 1.6)
seg(bx, (0.2, 0.0), n2, C["main"], 1.6)
seg(bx, (0.1, -0.13), m, C["tangent"], 2.2)
bx.plot([0.0, 0.2], [0, 0], "o", color=C["main"], ms=4)
bx.plot(0.1, -0.13, "o", color=C["tangent"], ms=4)
bx.text(0.1, -0.205, "merged, normal 15°", color=C["tangent"], fontsize=8.5, ha="center")
bx.text(0.0, 0.10, "0°", fontsize=8.5, ha="center")
bx.text(0.2, 0.10, "30°", fontsize=8.5, ha="center")
bx.annotate("", xy=(0.1, -0.10), xytext=(0.1, -0.025), arrowprops=dict(arrowstyle="->", color=C["aux"], lw=0.8))
bx.set_xlim(-0.12, 0.32)
bx.set_ylim(-0.24, 0.15)
bx.set_aspect("equal")
bx.set_xticks([])
bx.set_yticks([])
bx.set_title("(b) merge", fontsize=11)

# ---------------------------------------------------------------- (c) prune
cx.axhline(0, color=C["aux"], lw=0.6)
cx.plot(kept, [0, 0], "o", color=C["main"], ms=9)
cx.plot(removed, [0, 0], "o", mfc="white", mec=C["normal"], ms=6, mew=1.3)
for r, d in zip(removed, dest):
    cx.annotate("", xy=(d, 0.045), xytext=(r, 0.045),
                arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=2))
    cx.text(r, 0.10, f"{r:g}", color=C["normal"], fontsize=8.5, ha="center")
for k in kept:
    cx.text(k, -0.105, f"{k:g}", fontsize=8.5, ha="center")
    cx.text(k, -0.185, "0.5 → 0.55", fontsize=8.5, ha="center")
cx.text(0.5, 0.10, "0.05 each", color=C["normal"], fontsize=8.5, ha="center")
cx.set_xlim(-0.28, 1.28)
cx.set_ylim(-0.24, 0.2)
cx.set_aspect(1.4)
cx.axis("off")
cx.set_title("(c) prune", fontsize=11)

fig.tight_layout()
dgfig.save(fig, __file__)
print(f"split: raw barycentre {tbar}, children {kids.tolist()}; merge: normal 15 deg, defect {defect:.4f}; "
      f"prune: moves {dest - removed}")
