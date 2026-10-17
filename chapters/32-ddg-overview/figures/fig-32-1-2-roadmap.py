"""Figure 32.1.2: the chapters of Track 2 and their dependencies (schematic).

The dependency list is GUIDELINES 9 ("Track 2 안의 의존 관계"). An arrow a -> b means "b uses a".
Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-1-2-roadmap.py
"""
import dgfig
dgfig.setup()
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

C = dgfig.COLORS

TITLE = {
    32: "Plane-curve\ncurvature", 33: "Combinatorial\nsurfaces", 34: "Exterior\nalgebra",
    35: "Exterior\ncalculus", 36: "Discrete exterior\ncalculus", 37: "Curves", 38: "Surfaces",
    39: "Curvature of\nsurfaces", 40: "Discrete\ncurvature I", 41: "Discrete\ncurvature II",
    42: "Laplace-Beltrami", 43: "Conformal\ngeometry", 44: "Geodesics", 45: "Vector fields",
}
DEPS = {35: [34], 36: [33, 35], 37: [32, 36], 38: [35, 36, 37], 39: [38], 40: [33, 39], 41: [40],
        42: [36, 41], 43: [42], 44: [39, 42], 45: [36, 42, 44]}
# two branches (PLAN 13): B = curves/surfaces/curvature (left), A = forms/DEC (right)
_X = {"B": 0.0, "M": 2.75, "A": 5.5}
_DY = 1.5
POS = {32: (_X["B"], 10), 33: (_X["M"], 10), 34: (_X["A"], 10), 35: (_X["A"], 10 - _DY), 36: (_X["A"], 10 - 2 * _DY),
       37: (_X["B"], 10 - 3 * _DY), 38: (_X["B"], 10 - 4 * _DY), 39: (_X["B"], 10 - 5 * _DY),
       40: (_X["B"], 10 - 6 * _DY), 41: (_X["B"], 10 - 7 * _DY), 42: (_X["A"], 10 - 7 * _DY),
       43: (_X["A"], 10 - 8.4 * _DY), 44: (_X["M"], 10 - 8.4 * _DY), 45: (_X["M"], 10 - 9.6 * _DY)}

# self-check: every dependency points to an earlier row or the same row (the layout is a valid order)
for b, al in DEPS.items():
    for a in al:
        assert POS[a][1] >= POS[b][1], (a, b)
        assert a < b
assert set(POS) == set(TITLE) == set(range(32, 46))

W, H = 2.3, 1.12
fig, ax = plt.subplots(figsize=(6.2, 9.2))


def top(k, dx=0.0):
    return (POS[k][0] + dx, POS[k][1] + H / 2)


def bottom(k, dx=0.0):
    return (POS[k][0] + dx, POS[k][1] - H / 2)


def left(k, dy=0.0):
    return (POS[k][0] - W / 2, POS[k][1] + dy)


def right(k, dy=0.0):
    return (POS[k][0] + W / 2, POS[k][1] + dy)


# explicit routes (polylines from box a to box b). Long edges run down the empty middle column, through the gaps
# between the columns, or around the right edge, so that no edge passes behind a box it does not connect.
GAP = (_X["M"] + _X["A"]) / 2                     # x in the gap between the middle and right columns
FAR = _X["A"] + W / 2 + 0.3                       # x to the right of the right column
ROUTES = {
    (34, 35): [bottom(34), top(35)],
    (35, 36): [bottom(35), top(36)],
    (33, 36): [bottom(33, 0.35), (_X["M"] + 0.35, POS[36][1] + 0.2), left(36, 0.2)],
    (33, 40): [bottom(33, -0.35), (_X["M"] - 0.35, POS[40][1]), right(40)],
    (32, 37): [bottom(32), top(37)],
    (36, 37): [left(36, -0.2), right(37, 0.2)],
    (35, 38): [left(35, -0.2), right(38, 0.25)],
    (36, 38): [left(36, -0.4), right(38, -0.15)],
    (37, 38): [bottom(37), top(38)],
    (38, 39): [bottom(38), top(39)],
    (39, 40): [bottom(39), top(40)],
    (40, 41): [bottom(40), top(41)],
    (39, 44): [right(39), (_X["M"] - 0.7, POS[39][1]), top(44, -0.7)],
    (41, 42): [right(41), left(42)],
    (36, 42): [bottom(36), top(42)],
    (42, 43): [bottom(42), top(43)],
    (42, 44): [bottom(42, -0.85), top(44, 0.6)],
    (42, 45): [left(42, -0.3), (GAP, POS[42][1] - 0.3), (GAP, POS[45][1] + 0.2), right(45, 0.2)],
    (44, 45): [bottom(44), top(45)],
    (36, 45): [right(36), (FAR, POS[36][1]), (FAR, POS[45][1] - 0.2), right(45, -0.2)],
}
assert set(ROUTES) == {(a, b) for b, al in DEPS.items() for a in al}


def inside(pt, k, margin=0.03):
    x, y = pt
    return abs(x - POS[k][0]) < W / 2 - margin and abs(y - POS[k][1]) < H / 2 - margin


# self-check: no route enters a box other than its two ends (no edge hidden behind a box)
for (a, b), pts in ROUTES.items():
    for p, q in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, 401):
            pt = (p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]))
            hit = [k for k in POS if k not in (a, b) and inside(pt, k)]
            assert not hit, ((a, b), hit)

for (a, b), pts in ROUTES.items():
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    ax.add_patch(FancyArrowPatch(path=MPath(pts, codes), arrowstyle="-|>", mutation_scale=9, color=C["aux"],
                                 lw=0.9, shrinkA=1, shrinkB=1, zorder=1, joinstyle="miter"))

for k, (x, y) in POS.items():
    here = k == 32
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C["region"] if here else "white", alpha=1.0, ec=C["main"],
                                lw=1.6 if here else 0.9, zorder=2))
    ax.text(x, y + 0.3, str(k), ha="center", va="center", fontsize=12, fontweight="bold", zorder=3)
    ax.text(x, y - 0.15, TITLE[k], ha="center", va="center", fontsize=9.2, linespacing=1.05, zorder=3)

ax.text(_X["B"], 10.68, "curves, surfaces,\ncurvature", ha="center", va="bottom", fontsize=10, color=C["tangent"])
ax.text(_X["A"], 10.68, "forms and DEC", ha="center", va="bottom", fontsize=10, color=C["third"])
ax.annotate("you are\nhere", xy=(-0.6, 10 - H / 2), xytext=(-0.6, 8.3), fontsize=10, ha="center",
            linespacing=1.0, arrowprops=dict(arrowstyle="->", color=C["main"], lw=0.8))
ax.set_xlim(-1.3, FAR + 0.15)
ax.set_ylim(10 - 9.6 * _DY - 0.65, 11.45)
ax.set_axis_off()
dgfig.save(fig, __file__)
