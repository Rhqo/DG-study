"""Figure 41.5.1: explicit discrete curvature flows on the unit icosphere(3) against the exact shrinking spheres.

Mean curvature flow x_i' = (L x)_i / A_i (minus the area gradient over the mixed area) and Gauss curvature flow
x_i' = -(K N)_i / A_i (minus the gradient of (1/2) sum l theta over the mixed area), forward Euler with h = 6.5e-4
(about 0.1 of the initial stability limit 2 / lambda_max = 0.0065, Example 41.5.5), up to t = 0.2. r is the mean
distance of the vertices from their centroid.
(a) r^2 against t with the exact 1 - 4t (sphere under mean curvature flow, Example 41.5.5);
(b) r^3 against t with the exact 1 - 3t (Gauss curvature flow, Example 41.5.9). Dots every 10 steps.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-5-1-sphere-flows.py
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
V0, F = dm.icosphere(3)
lam = ddg41.largest_eigenvalue(V0, F)
assert abs(2 / lam - 0.0065) < 1e-4
h, T = 6.5e-4, 0.2
steps = int(round(T / h))
res = {}
for name, vel in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
    rec = []
    ddg41.explicit_flow(V0, F, vel, h, steps, record=lambda k, V: rec.append((k * h, np.linalg.norm(V - V.mean(0), axis=1).mean(),
                                                                              dm.face_areas(V, F).sum())))
    res[name] = np.array(rec)
tm, rm, Am = res["mean"].T
tg, rg, Ag = res["gauss"].T
assert np.all(np.diff(Am) < 0) and np.all(np.diff(Ag) < 0)
em = np.abs(rm ** 2 - (1 - 4 * tm)).max()
eg = np.abs(rg ** 3 - (1 - 3 * tg)).max()
assert em < 0.01 and eg < 0.005

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
tt = np.linspace(0, 0.2, 100)
for ax, t, y, ex, lab, col, title in ((axes[0], tm, rm ** 2, 1 - 4 * tt, r"$r^2$", C["tangent"], "(a) mean curvature flow"),
                                      (axes[1], tg, rg ** 3, 1 - 3 * tt, r"$r^3$", C["third"], "(b) Gauss curvature flow")):
    ax.plot(tt, ex, color=C["main"], lw=1.4, label="exact sphere")
    ax.plot(t[::10], y[::10], "o", color=col, ms=3.5, label="icosphere(3)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(lab)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax.set_xlim(0, 0.205)
axes[0].text(0.01, 0.25, r"$1-4t$", fontsize=11)
axes[1].text(0.01, 0.45, r"$1-3t$", fontsize=11)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.9, wspace=0.3)
dgfig.save(fig, __file__)
print("lambda_max", lam, "h_c", 2 / lam, "steps", steps, "max |r^2 - (1 - 4t)|", em, "max |r^3 - (1 - 3t)|", eg,
      "t = 0.2:", rm[-1] ** 2, rg[-1] ** 3)
