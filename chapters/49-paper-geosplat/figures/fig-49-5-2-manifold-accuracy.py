"""Figure 49.5.2: the manifold estimator of [GeoSplat25, App. E] on the running surfaces (Section 49.5).

Points uniform with respect to area, about 4000 per unit area; random-walk generator with t = 0.005; the estimated
normal is oriented like the true one; true and estimated values are matched in increasing (signed) order
(ordering by absolute value, as the paper does, would swap the two values at the saddle where |tau_1| = |tau_2|).
  unit sphere (50000 points; true (-1, -1) with the outward normal), 8 random points;
  cylinder of radius 1, |z| <= 1 (50000 points; true (-1, 0)), 8 random points with |z| < 0.5;
  saddle z = (x^2 - y^2)/2 (40000 points over [-1.2, 1.2]^2; true values from the shape operator with the upward
  normal), the origin and 7 random points with x^2 + y^2 < 0.25;
  torus T_{2,0.8} (250000 points; true -1.25 and -cos u/(2 + 0.8 cos u)), 8 points along a meridian.
Plotted: estimated against true principal curvature, both tau_1 and tau_2 of every point.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-5-2-manifold-accuracy.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
rng = np.random.default_rng(1)
t = 0.005
pairs = {}


def run(name, X, idx, nref, true):
    out = []
    for i, nr, tr in zip(idx, nref, true):
        est = np.sort(geo49.manifold_estimate(X, X[i], t, n_ref=nr)["tau"])
        out.append((np.sort(np.array(tr, float)), est))
    pairs[name] = out


X = geo49.sphere_points(50000, seed=2)
idx = rng.choice(50000, 8, replace=False)
run("sphere", X, idx, X[idx], [(-1.0, -1.0)] * 8)
X = geo49.cylinder_points(50000, seed=3)
idx = rng.choice(np.where(np.abs(X[:, 2]) < 0.5)[0], 8, replace=False)
run("cylinder", X, idx, np.c_[X[idx, 0], X[idx, 1], 0 * idx], [(-1.0, 0.0)] * 8)
X = 2 * geo49.saddle_points(40000, seed=4)
idx = [0] + list(rng.choice(np.where(np.hypot(X[:, 0], X[:, 1]) < 0.5)[0], 7, replace=False))
tr, nr = [], []
for i in idx:
    g = np.array([X[i, 0], -X[i, 1]])
    w = np.sqrt(1 + g @ g)
    tr.append(np.linalg.eigvals(np.linalg.solve(np.eye(2) + np.outer(g, g), np.diag([1.0, -1.0]) / w)).real)
    nr.append(np.array([-g[0], -g[1], 1.0]) / w)
run("saddle", X, idx, nr, tr)
T = geo49.torus_points(250000, seed=5)
us = np.linspace(0, np.pi, 8)
Qp = np.stack([2 + 0.8 * np.cos(us), 0 * us, 0.8 * np.sin(us)], axis=1)
run("torus", np.vstack([Qp, T]), range(8), (Qp - np.array([2.0, 0, 0])) / 0.8,
    [(-1.25, -np.cos(u) / (2 + 0.8 * np.cos(u))) for u in us])

allerr = np.concatenate([np.abs(e - tr_) for v in pairs.values() for tr_, e in v])
print("mean |error|", allerr.mean(), "max", allerr.max())
for k, v in pairs.items():
    print(k, np.round(np.mean([e for _, e in v], 0), 3), np.round(np.mean([np.abs(e - tr_) for tr_, e in v]), 3))
assert allerr.mean() < 0.12 and allerr.max() < 0.35

fig, ax = plt.subplots(figsize=(4.6, 4.2))
marks = {"sphere": (C["main"], "o"), "cylinder": (C["tangent"], "s"), "saddle": (C["normal"], "^"), "torus": (C["third"], "D")}
for name, v in pairs.items():
    col, mk = marks[name]
    tv = np.concatenate([tr_ for tr_, _ in v])
    ev = np.concatenate([e for _, e in v])
    ax.plot(tv, ev, mk, color=col, ms=5, alpha=0.85, label=name, mfc="none" if name in ("cylinder", "torus") else col)
lim = [-1.65, 1.2]
ax.plot(lim, lim, color=C["aux"], lw=0.9, ls="--")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("true principal curvature")
ax.set_ylabel("estimated principal curvature")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
