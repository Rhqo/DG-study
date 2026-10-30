"""Figure 49.6.3: which Gaussians pass the flatness test MAC < xi of the upsampling (6) on a synthetic room
(Section 49.6, Example 49.6.2).

Scene (lib/geo49.py, room_points): the six faces of the box [0,4] x [0,3] x [0,2.5] (metres) and a ball of radius 0.4
on the floor, 1500 points per square metre, each moved by isotropic Gaussian noise sigma. At sampled points we run the
manifold estimator of App. E (random-walk generator, t = 0.004 m^2, so sqrt t = 6.3 cm and about 19 points within the
kernel area pi t) and take MAC = (|tau_1| + |tau_2|)/2 (Definition 49.3.4).
Groups: "flat" = face points farther than 6 sqrt t from every edge and from the ball (their kernel sees one plane);
"edge" = face points within 5 cm of an edge (two faces meet at 90 degrees); "ball" = ball points with z > 0.45
(true MAC 1/0.4 = 2.5).
(a) fraction of points with MAC < xi, against xi (log scale): flat with sigma = 1, 3, 10 mm, edge and ball with
    sigma = 3 mm.  Without noise the flat MAC is exactly 0 (to rounding), so that curve would be 1 everywhere.
(b) top view of the floor, sigma = 3 mm: 2000 random floor points, filled if MAC < 0.3 per metre, hollow otherwise.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-6-3-mac-threshold.py
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
T, NQ, XI_B = 0.004, 150, 0.3
rng = np.random.default_rng(6)
bc = np.array(geo49.ROOM_BALL_C)
far = 6 * np.sqrt(T)


def macs(X, idx):
    return np.array([geo49.mac(geo49.manifold_estimate(X, X[i], T)["tau"]) for i in idx])


curves = {}
for sig in (0.001, 0.003, 0.01):
    X, X0, lab = geo49.room_points(1500, sig, seed=0)
    de = geo49.room_edge_distance(X0)
    flat = np.where((lab == 0) & (de > far) & (np.linalg.norm(X0 - bc, axis=1) > geo49.ROOM_BALL_R + far))[0]
    curves[f"flat, sigma = {sig * 1000:g} mm"] = macs(X, rng.choice(flat, NQ, replace=False))
    if sig == 0.003:
        edge = np.where((lab == 0) & (de < 0.05))[0]
        ball = np.where((lab == 1) & (X0[:, 2] > 0.45))[0]
        curves["edge, sigma = 3 mm"] = macs(X, rng.choice(edge, NQ, replace=False))
        curves["ball, sigma = 3 mm"] = macs(X, rng.choice(ball, NQ, replace=False))
        floor = np.where((lab == 0) & (X0[:, 2] == 0.0))[0]
        fl = rng.choice(floor, 2000, replace=False)
        floor_mac = macs(X, fl)
        floor_xy, floor_de = X0[fl, :2], de[fl]
for k, m in curves.items():
    print(k, "median", round(float(np.median(m)), 4), "pass 0.001:", np.mean(m < 1e-3), "pass 0.3:", round(float(np.mean(m < XI_B)), 3))
near_ball = np.linalg.norm(floor_xy - bc[:2], axis=1) < 0.25
print("floor: pass 0.3", round(float(np.mean(floor_mac < XI_B)), 3), "within 5 cm of a wall", round(float(np.mean(floor_mac[floor_de < 0.05] < XI_B)), 3),
      "within 0.25 of the ball contact", round(float(np.mean(floor_mac[near_ball] < XI_B)), 3), "far", round(float(np.mean(floor_mac[floor_de > far] < XI_B)), 3))
med = {k: np.median(v) for k, v in curves.items()}
assert all(np.mean(m < 1e-3) == 0 for m in curves.values())                       # xi_min = 0.001 selects nothing
assert med["flat, sigma = 1 mm"] < med["flat, sigma = 3 mm"] < med["flat, sigma = 10 mm"] < 0.5
assert 2.0 < med["flat, sigma = 10 mm"] / med["flat, sigma = 3 mm"] < 5.0                # roughly proportional to sigma
assert np.mean(curves["flat, sigma = 3 mm"] < XI_B) > 0.8 and np.mean(curves["ball, sigma = 3 mm"] < XI_B) == 0
assert np.mean(curves["edge, sigma = 3 mm"] < XI_B) < 0.1
assert 1.6 < med["ball, sigma = 3 mm"] < 2.6 and med["edge, sigma = 3 mm"] > 1.0

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
xs = np.logspace(-4, 1.2, 400)
sty = {"flat, sigma = 1 mm": (C["tangent"], ":"), "flat, sigma = 3 mm": (C["tangent"], "-"),
       "flat, sigma = 10 mm": (C["tangent"], "--"), "edge, sigma = 3 mm": (C["normal"], "-."),
       "ball, sigma = 3 mm": (C["main"], "-")}
for k in sty:
    m = curves[k]
    col, ls = sty[k]
    ax.plot(xs, [np.mean(m < x) for x in xs], color=col, ls=ls, lw=1.6, label=k.replace("sigma", r"$\sigma$"))
ax.axvline(1e-3, color=C["aux"], lw=1.0)
ax.text(1.2e-3, 0.3, r"$\xi_{\min} = 0.001$" + "\n(paper)", fontsize=8, color=C["aux"])
ax.axvline(XI_B, color=C["accent"], lw=1.0, ls="--")
ax.text(XI_B * 0.8, 1.04, r"$\xi = 0.3$", fontsize=8, color=C["accent"], ha="right")
ax.axvline(2.5, color=C["main"], lw=0.6, ls=":")
ax.text(3.0, 0.08, "true ball MAC 2.5", rotation=90, fontsize=7, va="bottom")
ax.set_xscale("log")
ax.set_xlim(1e-4, 15)
ax.set_ylim(-0.02, 1.1)
ax.set_xlabel(r"threshold $\xi$ (1/m)")
ax.set_ylabel(r"fraction with MAC $< \xi$")
ax.set_title("(a) who passes the flatness test", fontsize=10)
ax.legend(loc="upper left", fontsize=7.2, frameon=False, bbox_to_anchor=(0.0, 0.93))
ax = axes[1]
ok = floor_mac < XI_B
ax.scatter(floor_xy[ok, 0], floor_xy[ok, 1], s=5, color=C["tangent"], lw=0, label=r"MAC $< 0.3$ (upsampled)")
ax.scatter(floor_xy[~ok, 0], floor_xy[~ok, 1], s=7, facecolors="none", edgecolors=C["normal"], lw=0.6, label=r"MAC $\geq 0.3$")
ax.add_patch(plt.Circle(bc[:2], geo49.ROOM_BALL_R, fill=False, ls="--", color=C["main"], lw=0.9))
ax.text(bc[0], bc[1] + 0.5, "ball (seen from above)", ha="center", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
ax.set_aspect("equal")
ax.set_xlim(-0.05, 4.05)
ax.set_ylim(-0.05, 3.05)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(r"(b) floor, $\sigma$ = 3 mm, $\xi$ = 0.3", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False, markerscale=2)
fig.tight_layout()
dgfig.save(fig, __file__)
