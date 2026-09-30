"""그림 4.5.1: 가까운 세 점을 지나는 원은 접촉원으로 수렴한다 (4.5절, 명제 4.5.7).

곡선 γ(x) = (x, x²/2), 점 x0 = 0.5. 부호곡률 κ_s = (1 + x0²)^{−3/2} (예 4.4.6(b)),
곡률중심 c = (−x0³, 1 + 3x0²/2) = (−0.125, 1.375), 곡률반지름 (1 + x0²)^{3/2} ≈ 1.3975 (예 4.5.2(b)).
h = 1.0, 0.55, 0.25에 대해 세 점 γ(x0 − h), γ(x0), γ(x0 + h)를 지나는 원(회색 점선, 짧아질수록 진하게)과
접촉원(주황)을 그린다. 곡률중심 c는 청록 점이다. 원의 중심(작은 점)이 곡률중심 c로 다가간다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-5-1-three-point-circles.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW
X0 = 0.5
HS = (1.0, 0.55, 0.25)


def gamma(x):
    return np.stack([x, x ** 2 / 2], axis=-1)


def circumcircle(p1, p2, p3):
    """세 점을 지나는 원: 2<p2 − p1, c> = |p2|² − |p1|², 2<p3 − p1, c> = |p3|² − |p1|²."""
    A = 2 * np.array([p2 - p1, p3 - p1])
    b = np.array([p2 @ p2 - p1 @ p1, p3 @ p3 - p1 @ p1])
    c = np.linalg.solve(A, b)
    return c, np.linalg.norm(p1 - c)


kap = (1 + X0 ** 2) ** -1.5
center = np.array([-X0 ** 3, 1 + 1.5 * X0 ** 2])
R_osc = 1 / kap
# 자기검사: 곡률중심 = γ + n_s/κ_s, 세 점 원의 중심이 수렴
t_vec = np.array([1.0, X0]) / np.hypot(1, X0)
n_vec = np.array([-t_vec[1], t_vec[0]])
assert np.allclose(gamma(np.array(X0)) + n_vec / kap, center)
errs = []
for h in (1.0, 0.55, 0.25, 0.1, 0.03, 0.01):
    c, rr = circumcircle(gamma(np.array(X0 - h)), gamma(np.array(X0)), gamma(np.array(X0 + h)))
    errs.append(np.linalg.norm(c - center))
    assert abs(rr - np.linalg.norm(gamma(np.array(X0)) - c)) < 1e-12
assert all(e1 > e2 for e1, e2 in zip(errs, errs[1:])) and errs[-1] < 1e-3
assert abs(R_osc - 1.25 ** 1.5) < 1e-12

fig, ax = plt.subplots(figsize=(4.8, 4.4))
dgfig.schematic_axes(ax, (-1.75, 2.1), (-0.45, 3.35))
ax.plot([-1.75, 2.1], [0, 0], color=C["aux"], lw=0.5, zorder=0)
ax.plot([0, 0], [-0.45, 3.35], color=C["aux"], lw=0.5, zorder=0)
xs = np.linspace(-1.75, 2.1, 400)
P = gamma(xs)
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=4)
ang = np.linspace(0, 2 * np.pi, 400)
greys = ("#BBBBBB", "#999999", "#666666")
for h, col in zip(HS, greys):
    pts = [gamma(np.array(X0 + d)) for d in (-h, 0.0, h)]
    c, rr = circumcircle(*pts)
    ax.plot(c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang), color=col, lw=1.0, ls=(0, (4, 2.5)), zorder=2)
    ax.plot(*c, "o", color=col, ms=3.5, zorder=3)
    for q in (pts[0], pts[2]):
        ax.plot(*q, "o", color=col, mec=C["main"], mew=0.6, ms=4.5, zorder=6)
ax.plot(center[0] + R_osc * np.cos(ang), center[1] + R_osc * np.sin(ang), color=C["accent"], lw=1.5, zorder=3)
ax.plot(*center, "o", color=C["third"], ms=5, zorder=5)
p0 = gamma(np.array(X0))
ax.plot(*p0, "o", color=C["main"], ms=5, zorder=7)
ax.plot(*np.stack([p0, center]).T, color=C["normal"], lw=1.0, zorder=3)
ax.text(p0[0] + 0.1, p0[1] - 0.16, r"$\gamma(x_0)$", fontsize=11)
ax.text(center[0] - 0.12, center[1] + 0.12, r"$c$", fontsize=12, color=C["third"], ha="right")
ax.text(1.88, 1.3, r"$\gamma$", fontsize=12)
dgfig.save(fig, __file__)
