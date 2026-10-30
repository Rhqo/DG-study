"""Figure 47.6.1: approximating a piecewise smooth curve by cell data ([BLM17] Lemma 6.1, Def 6.2, Thm 6.4).

The standard double bubble in the plane with outer radii 0.6 and 1 (three circular arcs meeting at 120 degrees at
two junction points; total length L = 8.2549) is cut by the square grid of side a (lines x = 0.123 a + k a,
y = 0.071 a + k a). Each cell K gets m_K = length in K and a line P_K minimizing int_K ||P - T|| ds (operator norm),
computed from 400000 equally spaced samples. Cells whose centre is within 2 delta (delta = sqrt2 a) of a junction
are called singular (the set K^sing in the proof of Thm 6.4), the others regular.
(a) a = 0.1: regular cells (light blue), singular cells (orange), the lines P_K (blue segments), junctions (black).
(b) the three terms of the bound (6.1) of Lemma 6.1 with U = R^2 against a: position term delta ||V|| (black),
plane term over regular cells (blue) and over singular cells (orange), and their sum (grey dashed). All decay with
slope 1, as (6.8) predicts for beta = 1.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-6-1-cell-approximation.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import var47 as v  # noqa: E402

C = dgfig.COLORS
db = v.double_bubble_2d(0.6, 1.0)
arcs = db["arcs"]
L = sum(r * (b - a) for c, r, a, b in arcs)
NS = 400000
X, T, _, _ = v.sample_arcs(arcs, NS)
w = np.full(NS, L / NS)
J = [db["p"], db["p"] * np.array([1.0, -1.0])]


def terms(a):
    org = (0.123 * a, 0.071 * a)
    cells = v.cell_planes(X, T, w, a, org)
    delta = math.sqrt(2) * a
    reg = sing = 0.0
    kinds = {}
    for (i, j), (m, al, mis) in cells.items():
        c = np.array([org[0] + (i + 0.5) * a, org[1] + (j + 0.5) * a])
        s = min(np.linalg.norm(c - q) for q in J) < 2 * delta
        kinds[(i, j)] = s
        if s:
            sing += mis
        else:
            reg += mis
    return dict(pos=delta * L, reg=reg, sing=sing, cells=cells, kinds=kinds, org=org)


As = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
res = [terms(a) for a in As]
pos = np.array([r["pos"] for r in res])
reg = np.array([r["reg"] for r in res])
sing = np.array([r["sing"] for r in res])
tot = pos + reg + sing
assert abs(L - 8.254888) < 1e-5
for arr in (pos, reg, tot):
    sl = np.polyfit(np.log(As[1:]), np.log(arr[1:]), 1)[0]
    assert 0.85 < sl < 1.15, sl
assert np.all(sing < 2 * As)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.28})
ax = axes[0]
r0 = res[1]
a0 = As[1]
for (i, j), (m, al, mis) in r0["cells"].items():
    x0, y0 = r0["org"][0] + i * a0, r0["org"][1] + j * a0
    col = C["accent"] if r0["kinds"][(i, j)] else C["region"]
    ax.add_patch(Rectangle((x0, y0), a0, a0, facecolor=col, alpha=0.45, edgecolor="white", lw=0.5))
    c = np.array([x0 + a0 / 2, y0 + a0 / 2])
    u = np.array([math.cos(al), math.sin(al)])
    ax.plot(*np.array([c - 0.4 * a0 * u, c + 0.4 * a0 * u]).T, color=C["tangent"], lw=1.0)
ax.plot(X[::200, 0], X[::200, 1], ",", color=C["main"])
for c, rr, t0, t1 in arcs:
    tt = np.linspace(t0, t1, 300)
    ax.plot(c[0] + rr * np.cos(tt), c[1] + rr * np.sin(tt), color=C["main"], lw=0.6)
for q in J:
    ax.plot(*q, "o", color=C["main"], ms=3.5)
ax.set_aspect("equal")
ax.set_xlim(-0.72, 1.95)
ax.set_ylim(-1.1, 1.1)
ax.set_axis_off()
ax.set_title(r"(a) cells, $a = 0.1$", fontsize=11)

ax = axes[1]
ax.loglog(As, pos, "o-", color=C["main"], ms=3.5, lw=1.2, label=r"$\delta\,\|V\|$")
ax.loglog(As, reg, "s-", color=C["tangent"], ms=3.5, lw=1.2, label="planes, regular")
ax.loglog(As, sing, "^-", color=C["accent"], ms=3.5, lw=1.2, label="planes, singular")
ax.loglog(As, tot, "--", color=C["aux"], lw=1.2, label="sum")
ax.set_xlabel(r"cell side $a$")
ax.set_title(r"(b) terms of (6.1), slope $1$", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
dgfig.save(fig, __file__)
for a, r in (zip(As, res) if __name__ == "__main__" else ()):
    print(f"a = {a}: cells {len(r['cells'])}, singular {sum(r['kinds'].values())}, pos {r['pos']:.4f}, "
          f"reg {r['reg']:.4f}, sing {r['sing']:.4f}")
