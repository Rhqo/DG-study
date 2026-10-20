"""Figure 44.5.1: two geodesics of the torus of revolution (Section 44.5, Example 44.5.9).

Torus x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u), (R, r) = (2, 0.8) (GUIDELINES 7). Geodesics from
x(0, 0) on the outer equator, unit speed, angle psi to the parallel: (a) psi = 45 deg, (b) psi = 75 deg, t in [0, 40].
Solved with ch44.surface_geodesic (dgsym.christoffel + dgnum.rk4, 8001 steps). View from above (elev 90): the upper
half sin u >= 0 is visible (solid orange), the lower half is hidden (thin dashed). Gray circles: outer equator rho = 2.8,
inner equator rho = 1.2, the top circle rho = 2; blue dashed (a): the turning parallels rho = c = 2.8 cos 45 deg.
Interactive version: both geodesics on the torus.
Self-checks: the Clairaut constant rho^2 dv/dt is constant to 1e-10; (a) never reaches rho < c - 1e-6; (b) crosses
the inner equator.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-5-1-torus-geodesics.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig
import dgsym

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
to = dgsym.EXAMPLES["torus"]
X = to["expr"].subs(dict(zip(to["params"], (2, 0.8))))
ts = np.linspace(0, 40, 8001)
runs = {}
for psi in (np.pi / 4, 5 * np.pi / 12):
    C, W, P = ch44.surface_geodesic(X, to["coords"], {}, np.array([0.0, 0.0]),
                                    np.array([np.sin(psi) / 0.8, np.cos(psi) / 2.8]), ts)
    rho = 2 + 0.8 * np.cos(C[:, 0])
    cl = rho ** 2 * W[:, 1]
    assert np.ptp(cl) < 1e-10
    runs[psi] = (C, P, rho, cl[0])
c45 = runs[np.pi / 4][3]
assert runs[np.pi / 4][2].min() > c45 - 1e-6 and runs[5 * np.pi / 12][2].min() < 1.2 + 1e-6
print(f"Clairaut constants: {c45:.4f}, {runs[5 * np.pi / 12][3]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
tt = np.linspace(0, 2 * np.pi, 400)
for ax, psi, lab in zip(axes, (np.pi / 4, 5 * np.pi / 12), ("(a)", "(b)")):
    C, P, rho, c = runs[psi]
    for R_, ls in ((2.8, "-"), (1.2, "-"), (2.0, ":")):
        ax.plot(R_ * np.cos(tt), R_ * np.sin(tt), color=col["aux"], lw=0.7, ls=ls)
    vis = np.sin(C[:, 0]) >= 0
    ax.plot(np.where(vis, P[:, 0], np.nan), np.where(vis, P[:, 1], np.nan), color=col["accent"], lw=1.0)
    ax.plot(np.where(~vis, P[:, 0], np.nan), np.where(~vis, P[:, 1], np.nan), color=col["accent"], lw=0.4, ls="--")
    if psi < np.pi / 3:
        ax.plot(c * np.cos(tt), c * np.sin(tt), color=col["tangent"], lw=0.9, ls="--")
    ax.plot(2.8, 0, "o", color=col["main"], ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_axis_off()
    ax.set_title(f"{lab} $\\psi_0 = {np.degrees(psi):.0f}^\\circ$, $\\rho\\cos\\psi = {c:.3f}$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
dgfig.save(fig, __file__)

import plotly.graph_objects as go  # noqa: E402
Vt, Ft = dm.torus_mesh(2, 0.8, 32, 80)
traces = [dm.plotly_mesh(Vt, Ft, color="#D0D0D0", name="torus", opacity=0.85)]
for psi, c_ in ((np.pi / 4, col["accent"]), (5 * np.pi / 12, col["tangent"])):
    P = runs[psi][1]
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=c_, width=4),
                               name=f"psi0 = {np.degrees(psi):.0f} deg"))
dm.plotly_save(traces, __file__, "torus (2, 0.8): geodesics from the outer equator at 45 and 75 degrees", "fig-44-5-1")
