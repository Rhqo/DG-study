"""Figure 32.4.3: ROUND for the four discrete flows on the unevenly sampled unit circle (12 vertices at the angles
ddg32.UNEVEN_DEG). Matched time steps h = dt (D) and h = dt / mean(lbar) (A, B, C), dt = 1e-3, up to t = 0.3.
(a) the polygons at t = 0 and t = 0.3 for B and D; (b) the roundness defect against time (log scale).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-4-3-roundness.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402

C = dgfig.COLORS
S = ddg32.KIND_STYLE
U = ddg32.uneven_circle()
lb = dd.curve_dual_lengths(U, True)
dt, T, every = 1e-3, 0.3, 5
steps = int(round(T / dt))
tt = dt * every * np.arange(steps // every + 1)
runs = {}
for key, k, normal in (("A", "A", None), ("B", "B", None), ("C", "C", None), ("D", "D", "circumcenter"),
                       ("Dbis", "D", "bisector")):
    h = dt if k == "D" else dt / lb.mean()
    tr = ddg32.flow(U, k, h, steps, normal=normal or "circumcenter", every=every)
    me = min(dd.curve_edge_lengths(X, True).min() for X in tr) / dd.curve_edge_lengths(U, True).min()
    assert me > 0.2, (key, me)
    runs[key] = tr
rd = {key: np.array([ddg32.roundness_defect(X) for X in tr]) for key, tr in runs.items()}
assert np.all(rd["D"] < 1e-13)                                       # Proposition 32.4.10
assert all(rd[k][-1] > 0.05 for k in "ABC") and 1e-4 < rd["Dbis"][-1] < 1e-2
cD, rD = ddg32.fit_circle(runs["D"][-1])
assert np.linalg.norm(cD) < 1e-12 and np.allclose(np.linalg.norm(runs["D"][-1], axis=1), rD)
r_m = 1.0
for _ in range(steps):
    r_m = r_m - dt / r_m                                             # r_{m+1} = r_m - h / r_m
assert abs(rD - r_m) < 1e-12

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
s = np.linspace(0, 2 * np.pi, 400)


def poly(a, X, **kw):
    Xc = np.vstack([X, X[:1]])
    a.plot(Xc[:, 0], Xc[:, 1], **kw)
    a.plot(X[:, 0], X[:, 1], "o", ms=3.2, color=kw.get("color"), zorder=5)


ax.plot(np.cos(s), np.sin(s), color=C["aux"], lw=dgfig.LW["aux"], ls="--")
poly(ax, U, color=C["main"], lw=1.4)
ax.plot(rD * np.cos(s), rD * np.sin(s), color=S["D"]["color"], lw=dgfig.LW["aux"], ls="--")
poly(ax, runs["D"][-1], color=S["D"]["color"], lw=1.4)
poly(ax, runs["B"][-1], color=S["B"]["color"], lw=1.4, ls="-")
ax.text(-0.15, 1.08, r"$t = 0$", fontsize=10.5)
jb = int(np.argmin(runs["B"][-1][:, 0]))                       # leftmost vertex of the B polygon
ax.annotate("B", xy=runs["B"][-1][jb], xytext=(-1.28, 0.62), fontsize=11, color=S["B"]["color"],
            fontweight="bold", arrowprops=dict(arrowstyle="-", color=S["B"]["color"], lw=0.7))
ax.text(-0.42, -0.25, "D", fontsize=11, color=S["D"]["color"], fontweight="bold")
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.25)
ax.set_ylim(-1.2, 1.25)
ax.set_axis_off()
ax.set_title(r"(a)  $t = 0$ and $t = 0.3$", fontsize=12)

floor = 1e-17
for key, lab in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("Dbis", r"D with $\mathbf{n}_i$")):
    k = key[0]
    st = dict(color=S[k]["color"], ls=S[k]["ls"], lw=1.6)
    if key == "Dbis":
        st.update(ls="-", lw=1.0, alpha=0.8)
    bx.semilogy(tt[1:], np.maximum(rd[key][1:], floor), label=lab, **st)
bx.set_xlabel(r"time $t$")
bx.set_ylabel("roundness defect")
bx.set_ylim(1e-17, 1)
bx.legend(frameon=False, fontsize=9, loc="center right", ncol=1)
bx.set_title("(b)", fontsize=12)
for sp_ in ("top", "right"):
    bx.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
for key in rd:
    print(key, "defect at t=0.3: %.2e" % rd[key][-1], " fitted r: %.4f" % ddg32.fit_circle(runs[key][-1])[1])
print("smooth r(0.3) =", np.sqrt(1 - 0.6), " D-flow r =", rD)
