"""Figure 41.5.3: mean curvature flow smooths a noisy sphere, Gauss curvature flow roughens it.

The noisy sphere of Section 42.6: icosphere(3) with every vertex scaled by 1 + 0.05 N(0, 1) (seed 5); 373 of its 642
vertices have a negative angle defect. A second mesh uses 1 + 0.002 N(0, 1) (same seed; every angle defect positive).
Explicit flows x_i' = (L x)_i / A_i and x_i' = -(K N)_i / A_i with h = 0.1 x (2 / lambda_max of the starting mesh), up
to t = 0.033. (a) roughness (standard deviation / mean of the distance to the centroid) against t, log scale;
(b) histograms of the distances of the 642 vertices from the centroid: the noisy sphere (noise 0.05) and the two
flows after 10 steps (t = 0.0066).

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-5-3-noisy-sphere.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS
T = 0.033
runs = {}
snap = {}
for amp in (0.05, 0.002):
    Vn, F = ddg41.noisy_sphere(3, amp, 5)
    if amp == 0.05:
        assert np.sum(dm.angle_defect(Vn, F) < 0) == 373
    else:
        assert np.all(dm.angle_defect(Vn, F) > 0)
    h = 0.1 * 2 / ddg41.largest_eigenvalue(Vn, F)
    steps = int(np.ceil(T / h))
    for name, vel in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
        rec = []

        def record(k, V):
            rec.append((k * h, ddg41.roughness(V)))
            if amp == 0.05 and k == 10:
                snap[name] = V.copy()
        ddg41.explicit_flow(Vn, F, vel, h, steps, record=record)
        runs[(amp, name)] = np.array(rec)
m5, g5, g2 = runs[(0.05, "mean")], runs[(0.05, "gauss")], runs[(0.002, "gauss")]
assert m5[-1, 1] < 0.011 and g5[-1, 1] > 0.3 and g2[-1, 1] < 0.0005
assert np.all(np.diff(m5[:, 1]) < 0)

fig = plt.figure(figsize=(7.4, 3.0))
ax = fig.add_axes([0.085, 0.17, 0.37, 0.73])
ax.semilogy(m5[:, 0], m5[:, 1], color=C["tangent"], lw=1.6, label="mean, noise 0.05")
ax.semilogy(g5[:, 0], g5[:, 1], color=C["normal"], lw=1.6, label="Gauss, noise 0.05")
ax.semilogy(g2[:, 0], g2[:, 1], color=C["accent"], lw=1.6, ls="--", label="Gauss, noise 0.002")
ax.set_xlabel(r"$t$")
ax.set_ylabel("roughness")
ax.set_xlim(0, T)
ax.set_xticks([0, 0.01, 0.02, 0.03])
ax.set_title("(a) roughness", loc="left", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="center right", bbox_to_anchor=(1.02, 0.36))

V0, F = ddg41.noisy_sphere(3, 0.05, 5)
ax2 = fig.add_axes([0.6, 0.17, 0.38, 0.73])
dist = {key: np.linalg.norm(X - X.mean(0), axis=1) for key, X in (("start", V0), ("mean", snap["mean"]), ("gauss", snap["gauss"]))}
bins = np.linspace(0.3, 1.2, 46)
ax2.hist(dist["start"], bins, histtype="step", color=C["aux"], lw=1.4, label="t = 0")
ax2.hist(dist["mean"], bins, histtype="step", color=C["tangent"], lw=1.6, label="mean, t = 0.0066")
ax2.hist(dist["gauss"], bins, histtype="step", color=C["normal"], lw=1.6, label="Gauss, t = 0.0066")
ax2.set_xlabel("distance from the centroid")
ax2.set_ylabel("vertices")
ax2.set_xlim(0.3, 1.2)
ax2.set_title("(b) noise 0.05", loc="left", fontsize=11)
ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
dgfig.save(fig, __file__)
print("h (0.05)", T, "rough mean", m5[[0, 10, -1], 1], "gauss", g5[[0, 10, -1], 1], "gauss 0.002", g2[[0, -1], 1],
      "t at step 10", g5[10, 0], "ranges", {k: (round(v.min(), 3), round(v.max(), 3)) for k, v in dist.items()})
