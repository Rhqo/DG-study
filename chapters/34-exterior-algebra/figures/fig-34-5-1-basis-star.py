"""Figure 34.5.1: the basis k-vectors of R^3 and the Hodge star (Section 34.5, Example 34.5.4, Proposition 34.5.2).

At the corner of the unit cube: the basis 1-vectors e1, e2, e3 (arrows along the edges) and the basis 2-vectors
e2^e3, e3^e1, e1^e2 (the three unit squares through the origin). Each square has the same color as the arrow
perpendicular to it, because star e1 = e2^e3, star e2 = e3^e1, star e3 = e1^e2 (and star of each square is that
arrow). Curved arrows give the orientation of each square (e2 -> e3, e3 -> e1, e1 -> e2): the right-hand rule
around the matching arrow. The dashed cube is the 3-vector e1^e2^e3 = star 1.
Self-checks: e_i ^ (paired square) = +e123 for each pair (sign via the permutation), and each square is
perpendicular to its arrow.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import ch34  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
E = np.eye(3)
pairs = [(0, (1, 2), C["tangent"]), (1, (2, 0), C["third"]), (2, (0, 1), C["accent"])]
for i, (j, k), _ in pairs:
    assert ch34.perm_sign((i, j, k)) == 1                      # e_i ^ e_j ^ e_k = +e123
    assert abs(E[i] @ E[j]) < 1e-12 and abs(E[i] @ E[k]) < 1e-12
    assert np.allclose(ch34.hodge_t(E[i], 3), ch34.vec_tensor(E[j], E[k]))

fig = plt.figure(figsize=(4.8, 4.4))
ax = dgfig.axes3d(fig, elev=24, azim=32)
o = np.zeros(3)

# dashed unit cube (the 3-vector)
for a in range(3):
    for s1 in (0, 1):
        for s2 in (0, 1):
            b, c = [m for m in range(3) if m != a]
            p = np.zeros(3)
            p[b], p[c] = s1, s2
            q = p.copy()
            q[a] = 1
            if s1 == 0 and s2 == 0:
                continue
            ax.plot(*np.array([p, q]).T, color=C["aux"], lw=0.7, ls=(0, (3, 2)), zorder=1)

for i, (j, k), col in pairs:
    sq = np.array([o, E[j], E[j] + E[k], E[k]])
    ax.add_collection3d(Poly3DCollection([sq], facecolor=col, alpha=0.22, edgecolor=col, linewidth=0.9, zorder=2))
    cen = 0.5 * (E[j] + E[k])
    ts = np.linspace(0.3, 1.25 * np.pi + 0.3, 40)
    arc = cen + 0.17 * (np.cos(ts)[:, None] * E[j] + np.sin(ts)[:, None] * E[k])
    ax.plot(*arc.T, color=col, lw=1.0, zorder=6)
    dgfig.arrow3d(ax, arc[-3], arc[-1] - arc[-3], role=col, lw=1.0, head=8, zorder=6)
    dgfig.arrow3d(ax, o, 1.35 * E[i], role=col, lw=2.0, zorder=8)

labels = [(1.55 * E[0], r"$e_1$"), (1.5 * E[1], r"$e_2$"), (1.45 * E[2] + np.array([0, 0, 0.02]), r"$e_3$")]
for pos, lab in labels:
    ax.text(*pos, lab, fontsize=13, ha="center", va="center")
sqlab = [(np.array([0.0, 0.75, 0.62]), r"$e_2\wedge e_3$", C["tangent"]),
         (np.array([0.75, 0.0, 0.95]), r"$e_3\wedge e_1$", "#00704F"),
         (np.array([0.8, 0.75, 0.0]), r"$e_1\wedge e_2$", "#9A6200")]
for pos, lab, col in sqlab:
    ax.text(*pos, lab, fontsize=12, color=col, ha="center")
ax.text(0.15, 1.25, 1.42, r"$e_1\wedge e_2\wedge e_3 = \star 1$", fontsize=11, color="#555555", ha="center")

dgfig.equal_aspect(ax, np.array([[0, 0, 0], [1.5, 1.5, 1.5]]), zoom=1.05)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
dgfig.save(fig, __file__)
