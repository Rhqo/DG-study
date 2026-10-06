"""Figure 0.1.3: Tour의 각 절과 깊은 장의 대응 (0.1절, schematic).

왼쪽 열: Tour의 절 0.1–0.8. 가운데 열: 각 절의 Go deeper가 가리키는 깊은 장.
오른쪽: 깊은 장이 속한 Part (GUIDELINES.md §9의 장 제목과 폴더).
자기검사: 대응표가 Chapter 1–26을 빠짐없이 한 번씩 덮는다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-1-3-roadmap.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C = dgfig.COLORS

# (Tour 절, 깊은 장 상자의 글, 덮는 장 번호, Part 이름)
ROWS = [
    ("0.1  What Is DG?", "Ch. 1–3  Preliminaries", range(1, 4), "Part 0"),
    ("0.2  Curves", "Ch. 4–5  Curves", range(4, 6), "Part I"),
    ("0.3  Surfaces", "Ch. 6–7  Surfaces, First Form", range(6, 8), "Part I"),
    ("0.4  Curvature", "Ch. 8  Gauss Map", range(8, 9), "Part I"),
    ("0.5  Intrinsic Geometry", "Ch. 9–10  Geodesics, Gauss–Bonnet", range(9, 11), "Part I"),
    ("0.6  Manifolds", "Ch. 11–15, 20  Manifolds", list(range(11, 16)) + [20], "Part II"),
    ("0.7  Forms", "Ch. 16–19  Forms, Stokes", range(16, 20), "Part II"),
    ("0.8  Riemannian", "Ch. 21–26  Riemannian", range(21, 27), "Part III"),
]
covered = sorted(c for _, _, cs, _ in ROWS for c in cs)
assert covered == list(range(1, 27)), covered

fig = plt.figure(figsize=(6.6, 4.3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11)
ax.set_ylim(-0.4, len(ROWS) + 0.2)
ax.set_axis_off()

H = 0.72
xL, wL = 0.15, 3.1
xR, wR = 4.25, 4.75
for i, (tour, deep, _, part) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    ax.add_patch(FancyBboxPatch((xL, y), wL, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=C["region"], ec=C["tangent"], alpha=1.0, lw=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((xL, y), wL, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec="none", alpha=0.55, lw=0, zorder=2.5))
    ax.text(xL + 0.15, y + H / 2, tour, fontsize=10.5, va="center", ha="left", zorder=3)
    ax.add_patch(FancyBboxPatch((xR, y), wR, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#EFEFEF", ec=C["aux"], lw=0.8, zorder=2))
    ax.text(xR + 0.15, y + H / 2, deep, fontsize=10.5, va="center", ha="left", zorder=3)
    ax.add_patch(FancyArrowPatch((xL + wL + 0.08, y + H / 2), (xR - 0.08, y + H / 2),
                                 arrowstyle="-|>", mutation_scale=11, color=C["main"], lw=1.0, zorder=4))

# Part 괄호
groups = {}
for i, (_, _, _, part) in enumerate(ROWS):
    groups.setdefault(part, []).append(len(ROWS) - 1 - i)
xb = xR + wR + 0.18
for part, ys in groups.items():
    y0, y1 = min(ys) + 0.02, max(ys) + H - 0.02
    ax.plot([xb, xb + 0.12, xb + 0.12, xb], [y0, y0, y1, y1], color=C["aux"], lw=1.0)
    ax.text(xb + 0.24, (y0 + y1) / 2, part, fontsize=10.5, va="center", ha="left")

ax.text(xL + wL / 2, len(ROWS) - 0.05, "Tour", fontsize=11.5, ha="center", va="bottom", weight="bold")
ax.text(xR + wR / 2, len(ROWS) - 0.05, "Go deeper", fontsize=11.5, ha="center", va="bottom", weight="bold")

dgfig.save(fig, __file__)
