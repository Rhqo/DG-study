"""그림 11.5.1: 닫힌 원판 B̄² 의 경계 차트 ψ_2^+ (11.5절, 예 11.5.13).

V_2^+ = {x ∈ B̄² : x^2 > 0}(위쪽 반원판),  ψ_2^+(x) = (x^1, √(1 - (x^1)^2) - x^2) ∈ ℍ².
치역은 {(y, t) : |y| < 1, 0 ≤ t < √(1 - y^2)} (ℍ² 안의 반원판 모양, ℍ²에서 열린집합).
경계 원의 위쪽 호(x^2 = √(1 - (x^1)^2))는 ∂ℍ² = {t = 0} 위로 간다.
격자: y = x^1 = -0.75, …, 0.75 (세로선)와 t = 0, 0.15, 0.3, 0.45, 0.6, 0.75 (경계 호를 아래로 민 곡선).
점: 내부점 p = (0.3, 0.45) ↦ (0.3, √0.91 - 0.45), 경계점 q = (-0.6, 0.8) ↦ (-0.6, 0).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-5-1-boundary-chart.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

C = dgfig.COLORS
LW = dgfig.LW


def psi(x):
    x = np.asarray(x, float)
    return np.stack([x[..., 0], np.sqrt(1 - x[..., 0] ** 2) - x[..., 1]], axis=-1)


def psi_inv(w):
    w = np.asarray(w, float)
    return np.stack([w[..., 0], np.sqrt(1 - w[..., 0] ** 2) - w[..., 1]], axis=-1)


p = np.array([0.3, 0.45])
q = np.array([-0.6, 0.8])

# 자기검사 (§12.1)
rng = np.random.default_rng(5)
pts = rng.uniform(-1, 1, size=(4000, 2))
pts = pts[(np.hypot(*pts.T) <= 1) & (pts[:, 1] > 0)]
img = psi(pts)
assert np.all(img[:, 1] >= -1e-12)                                   # ℍ² 안
assert np.all(img[:, 1] < np.sqrt(1 - img[:, 0] ** 2) + 1e-12)       # t < √(1 - y²)
assert np.allclose(psi_inv(img), pts)                                # 역사상
arc = np.linspace(-0.99, 0.99, 50)
assert np.allclose(psi(np.stack([arc, np.sqrt(1 - arc ** 2)], 1))[:, 1], 0)   # 경계 호 ↦ ∂ℍ²
assert np.isclose(psi(q)[1], 0) and psi(p)[1] > 0

fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax in (aL, aR):
    dgfig.schematic_axes(ax, (-1.3, 1.3), (-1.2, 1.25))

# 왼쪽: 닫힌 원판
ax = aL
ax.add_patch(Circle((0, 0), 1, facecolor=C["surface"], alpha=0.45, lw=0, zorder=0))
ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=C["main"], lw=LW["main"], zorder=4))
tt = np.linspace(0, np.pi, 200)
upper = np.stack([np.cos(tt), np.sin(tt)], axis=1)
ax.add_patch(Polygon(np.vstack([upper, [[-1, 0]]]), closed=True, facecolor=C["region"], alpha=0.35, lw=0, zorder=1))
ax.plot([-1, 1], [0, 0], color=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3)
ax.plot(*upper.T, color=C["accent"], lw=2.6, zorder=5)
for y0 in np.linspace(-0.75, 0.75, 7):
    top = np.sqrt(1 - y0 ** 2)
    ax.plot([y0, y0], [0.0, top], color=C["tangent"], lw=0.45, zorder=2)
for t0 in (0.15, 0.3, 0.45, 0.6, 0.75):
    ys = np.linspace(-1, 1, 400)
    xs2 = np.sqrt(1 - ys ** 2) - t0
    m = xs2 > 0
    ax.plot(ys[m], xs2[m], color=C["tangent"], lw=0.45, zorder=2)
for e in (-1, 1):
    ax.plot(e, 0, "o", ms=5, mfc="white", mec=C["main"], zorder=6)       # (±1, 0) ∉ V_2^+
ax.plot(*p, "o", ms=4.5, color=C["main"], zorder=7)
ax.plot(*q, "o", ms=4.5, color=C["accent"], zorder=7)
ax.text(p[0] + 0.06, p[1] + 0.05, r"$p$", fontsize=12, zorder=8)
ax.text(q[0] - 0.16, q[1] + 0.07, r"$q$", fontsize=12, color=C["accent"], zorder=8)
ax.text(0.55, -0.62, r"$\overline{B}{}^2$", fontsize=12)
ax.text(-0.25, 0.25, r"$V_2^+$", fontsize=11, color=C["tangent"])

# 오른쪽: ℍ²
ax = aR
ax.fill_between([-1.25, 1.25], 0, 1.2, color=C["surface"], alpha=0.25, lw=0, zorder=0)
ax.plot([-1.25, 1.25], [0, 0], color=C["accent"], lw=1.0, zorder=1)
reg = np.vstack([upper, [[-1, 0]]])
ax.add_patch(Polygon(reg, closed=True, facecolor=C["region"], alpha=0.35, lw=0, zorder=1))
ax.plot(*upper.T, color=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3)
ax.plot([-1, 1], [0, 0], color=C["accent"], lw=2.6, zorder=5)
for e in (-1, 1):
    ax.plot(e, 0, "o", ms=5, mfc="white", mec=C["main"], zorder=6)
for y0 in np.linspace(-0.75, 0.75, 7):
    ax.plot([y0, y0], [0, np.sqrt(1 - y0 ** 2)], color=C["tangent"], lw=0.45, zorder=2)
for t0 in (0.15, 0.3, 0.45, 0.6, 0.75):
    y1 = np.sqrt(1 - t0 ** 2)
    ax.plot([-y1, y1], [t0, t0], color=C["tangent"], lw=0.45, zorder=2)
pp, qq = psi(p), psi(q)
ax.plot(*pp, "o", ms=4.5, color=C["main"], zorder=7)
ax.plot(*qq, "o", ms=4.5, color=C["accent"], zorder=7)
ax.text(pp[0] + 0.06, pp[1] + 0.05, r"$\psi_2^+(p)$", fontsize=10, zorder=8)
ax.text(qq[0] - 0.3, qq[1] - 0.22, r"$\psi_2^+(q)$", fontsize=10, color=C["accent"], zorder=8)
ax.text(0.72, 0.95, r"$\mathbb{H}^2$", fontsize=12)
ax.text(0.55, -0.22, r"$\partial\mathbb{H}^2$", fontsize=11, color=C["accent"])

dgfig.map_arrow(fig, aL, aR, r"$\psi_2^+$", xy_from=(0.8, 0.8), xy_to=(0.22, 0.82), rad=-0.3, label_offset=(0.0, 0.035))
fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.97, bottom=0.02)
dgfig.save(fig, __file__)
print("psi(p) =", psi(p))
