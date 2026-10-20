"""Figure 42.6.2: semi-implicit mean curvature flow of the icosphere (Section 42.6, Example 42.6.7).

dgmesh.icosphere(3), steps (M - h L) x' = M x with L and M (mixed Voronoi) of the current mesh, h = 0.01, 0.05, 0.2.
Plotted: r^2 (mean squared distance of the vertices from their centroid; dots) against t = k h, the exact smooth
solution r^2 = 1 - 4t (black) and, for each h, the recurrence r_{k+1} = r_k / (1 + 2h / r_k^2) that one gets from
M^{-1} L x = -(2/r^2) x (thin lines of the same color).
Self-check: for every h the mesh values agree with the recurrence to 2e-3 and lie above 1 - 4t; r^2 stays positive.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-6-2-sphere-radius.py``
"""

import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
V0, F = dm.icosphere(3)
runs = {0.01: col["tangent"], 0.05: col["third"], 0.2: col["accent"]}
data = {}
for h in runs:
    n = int(round(0.6 / h))
    V = V0.copy()
    r2 = [1.0]
    pred = [1.0]
    for k in range(n):
        L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
        V = np.linalg.solve(np.diag(m) - h * L, m[:, None] * V)
        r2.append(np.mean(np.sum((V - V.mean(0)) ** 2, 1)))
        rp = np.sqrt(pred[-1])
        pred.append((rp / (1 + 2 * h / rp ** 2)) ** 2)
        if r2[-1] < 1e-3:
            break
    t = h * np.arange(len(r2))
    r2, pred = np.array(r2), np.array(pred)
    assert np.abs(r2 - pred).max() < 2e-3 and np.all(r2 >= 1 - 4 * t - 1e-12) and np.all(r2 > 0)
    data[h] = (t, r2, pred)
    print(f"h = {h}: r^2 at t = 0.2: {np.interp(0.2, t, r2):.4f} (exact 0.2000), first steps {np.round(r2[:4], 4)}")

fig, ax = plt.subplots(figsize=(4.8, 3.2))
tt = np.linspace(0, 0.25, 50)
ax.plot(tt, 1 - 4 * tt, color=col["main"], lw=1.6, label=r"exact $1 - 4t$")
for h, c in runs.items():
    t, r2, pred = data[h]
    ax.plot(t, pred, "-", color=c, lw=0.8)
    ax.plot(t, r2, "o", color=c, ms=3.2 if h < 0.05 else 4.5, label=rf"$h = {h}$")
ax.axvline(0.25, color=col["aux"], ls=":", lw=0.9)
ax.text(0.255, 0.85, r"$t = \frac{1}{4}$", fontsize=10, color=col["aux"])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$r^2$")
ax.set_xlim(0, 0.62)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.15)
dgfig.save(fig, __file__)
