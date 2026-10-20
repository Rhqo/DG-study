"""Figure 44.5.2: Clairaut's relation and the accuracy of RK4 (Section 44.5, Example 44.5.9, Implementation box).

Same two torus geodesics as Figure 44.5.1 ((R, r) = (2, 0.8), psi_0 = 45 and 75 degrees).
(a) u(t) (mod 2 pi, shown as cos u, i.e. (rho - R)/r) for t in [0, 40]: psi_0 = 45 deg (orange) oscillates between
    cos u = 1 and cos u = (c - R)/r = -0.0251 (blue dashed); psi_0 = 75 deg (green) reaches cos u = -1 (inner equator).
(b) the drift max_t |c(t) - c(0)| of the Clairaut quantity c = rho^2 dv/dt over t in [0, 20] against the RK4 step h,
    for psi_0 = 45 deg; gray dashed: slope 4.
Self-checks: the fitted slope in (b) is between 3.5 and 4.5.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-5-2-conservation.py
"""

import pathlib
import sys

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
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.3, 1.0]})
for psi, c_ in ((np.pi / 4, col["accent"]), (5 * np.pi / 12, col["third"])):
    C, W, P = ch44.surface_geodesic(X, to["coords"], {}, np.array([0.0, 0.0]),
                                    np.array([np.sin(psi) / 0.8, np.cos(psi) / 2.8]), ts)
    axa.plot(ts, np.cos(C[:, 0]), color=c_, lw=1.2, label=rf"$\psi_0 = {np.degrees(psi):.0f}^\circ$")
cturn = (2.8 * np.cos(np.pi / 4) - 2) / 0.8
axa.axhline(cturn, color=col["tangent"], lw=0.9, ls="--")
axa.text(30.5, cturn + 0.07, r"$(c - R)/r$", fontsize=9.5, color=col["tangent"])
axa.set_xlabel(r"$t$ (arc length)")
axa.set_ylabel(r"$\cos u = (\rho - R)/r$")
axa.set_ylim(-1.15, 1.15)
axa.legend(frameon=False, loc="lower left", fontsize=9.5)
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("(a) latitude along the geodesic", fontsize=11)
hs, drift = [], []
for n in (100, 200, 400, 800, 1600):
    tq = np.linspace(0, 20, n + 1)
    C, W, P = ch44.surface_geodesic(X, to["coords"], {}, np.array([0.0, 0.0]),
                                    np.array([np.sin(np.pi / 4) / 0.8, np.cos(np.pi / 4) / 2.8]), tq)
    cl = (2 + 0.8 * np.cos(C[:, 0])) ** 2 * W[:, 1]
    hs.append(20 / n)
    drift.append(np.abs(cl - cl[0]).max())
slope = np.polyfit(np.log(hs), np.log(drift), 1)[0]
assert 3.5 < slope < 4.5, slope
print("RK4 drift:", [f"{d_:.1e}" for d_ in drift], f"slope {slope:.2f}")
axb.loglog(hs, drift, "o-", color=col["accent"], lw=1.4, ms=4)
hh = np.array([hs[0], hs[-1]])
axb.loglog(hh, drift[0] * (hh / hs[0]) ** 4, color=col["aux"], lw=0.9, ls="--")
axb.text(hs[2], drift[2] * 3, "slope 4", fontsize=9.5, color=col["aux"])
axb.set_xlabel(r"RK4 step $h$")
axb.set_ylabel(r"$\max_t|c(t) - c(0)|$")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title(f"(b) Clairaut drift, slope {slope:.2f}", fontsize=11)
fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18, wspace=0.35)
dgfig.save(fig, __file__)
