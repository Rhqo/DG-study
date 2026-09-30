"""그림 12.1.1: 매끄러운 사상의 좌표표현 F̂ = ψ∘F∘φ^{-1} (12.1절, 정의 12.1.5). (개념도)

위 왼쪽: 다양체 M(얼룩)과 차트의 정의역 U(원판), 위 오른쪽: 다양체 N과 차트의 정의역 V(원판).
F는 U를 V 안으로 보낸다(F(U) ⊆ V, 주황 영역 = F(U)).
아래: φ(U) ⊆ R^m, ψ(V) ⊆ R^n과 좌표표현 F̂ = ψ∘F∘φ^{-1}.
φ, ψ, F는 명시적인 단사 매끄러운 평면 사상(회전·확대 + 작은 이차 섭동)으로 정했다. 모양은 개념도다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-1-1-coordinate-representation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

C = dgfig.COLORS
LW = dgfig.LW

cM, cN = np.array([-1.45, 1.55]), np.array([1.45, 1.55])
cU, rU = np.array([-1.35, 1.55]), 0.55
cV, rV = np.array([1.45, 1.52]), 0.62


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def F(z):
    """U → V: 축소·회전 + 작은 이차 섭동 (F(U)가 V 안에 들어가도록)."""
    w = (np.asarray(z, float) - cU) @ rot(0.5).T * 0.72
    w = w + 0.18 * np.stack([w[..., 1] ** 2, 0.5 * w[..., 0] * w[..., 1]], axis=-1)
    return w + cV + np.array([0.02, -0.03])


def phi(z):
    w = (np.asarray(z, float) - cU) @ rot(-0.2).T * 1.1
    w = w + 0.1 * np.stack([-w[..., 0] * w[..., 1], w[..., 0] ** 2], axis=-1)
    return w + np.array([-1.45, -0.75])


def psi(z):
    w = (np.asarray(z, float) - cV) @ rot(0.3).T * 1.0
    w = w + 0.12 * np.stack([w[..., 1] ** 2, -w[..., 0] * w[..., 1]], axis=-1)
    return w + np.array([1.45, -0.75])


def circle(c, r, n=240):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + r * np.stack([np.cos(t), np.sin(t)], axis=1)


# 자기검사 (§12.1): F(U) ⊆ V (표본점), φ, ψ, F는 표본에서 단사이고 야코비 행렬식이 0이 아니다.
rng = np.random.default_rng(12)
t = rng.uniform(0, 2 * np.pi, 400)
sU = cU + rU * np.sqrt(rng.uniform(0, 1, 400))[:, None] * np.stack([np.cos(t), np.sin(t)], 1)
assert np.all(np.linalg.norm(F(sU) - cV, axis=1) < rV), "F(U)가 V 밖으로 나간다"
for f, s in ((F, sU), (phi, sU), (psi, cV + (F(sU) - cV))):
    img = f(s)
    din = np.linalg.norm(s[:, None] - s[None], axis=-1)
    dout = np.linalg.norm(img[:, None] - img[None], axis=-1)
    assert np.all((din < 1e-12) == (dout < 1e-12))
    h = 1e-6
    for z in s[:60]:
        J = np.column_stack([(f(z + [h, 0]) - f(z - [h, 0])) / (2 * h), (f(z + [0, h]) - f(z - [0, h])) / (2 * h)])
        assert abs(np.linalg.det(J)) > 0.2

fig, ax = plt.subplots(figsize=(6.0, 4.6))
dgfig.schematic_axes(ax, (-2.75, 2.75), (-1.75, 2.85))

# 위: M과 N
dgfig.blob(ax, center=tuple(cM), radius=0.95, seed=7, amp=0.1, fill="surface", fill_alpha=0.35)
dgfig.blob(ax, center=tuple(cN), radius=0.98, seed=2, amp=0.1, fill="surface", fill_alpha=0.35)
for c, r in ((cU, rU), (cV, rV)):
    P = circle(c, r)
    ax.add_patch(Polygon(P, closed=True, facecolor=C["region"], alpha=0.22, lw=0, zorder=2))
    ax.add_patch(Polygon(P, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3))
FU = F(circle(cU, rU))
ax.add_patch(Polygon(FU, closed=True, facecolor=C["accent"], alpha=0.45, lw=0, zorder=3))
ax.text(cU[0] - 0.12, cU[1] + 0.12, r"$U$", fontsize=13, ha="center", va="center")
ax.text(cV[0] + 0.52, cV[1] + 0.50, r"$V$", fontsize=13, ha="center")
ax.text(-2.45, 2.62, r"$M$", fontsize=14, va="top")
ax.text(2.25, 2.62, r"$N$", fontsize=14, va="top")
p = cU + np.array([0.18, -0.12])
ax.plot(*p, "o", color=C["main"], ms=3.5, zorder=6)
ax.text(p[0] + 0.07, p[1] - 0.13, r"$p$", fontsize=11, zorder=6)
Fp = F(p)
ax.plot(*Fp, "o", color=C["main"], ms=3.5, zorder=6)
ax.text(Fp[0] + 0.07, Fp[1] + 0.05, r"$F(p)$", fontsize=10, zorder=6)

# 아래: 두 좌표 영역
for f, c, r, lab, pos in ((phi, cU, rU, r"$\varphi(U)\subseteq\mathbb{R}^m$", (-2.05, -1.62)),
                          (psi, cV, rV, r"$\psi(V)\subseteq\mathbb{R}^n$", (0.95, -1.62))):
    P = f(circle(c, r))
    ax.add_patch(Polygon(P, closed=True, facecolor=C["region"], alpha=0.3, lw=0, zorder=2))
    ax.add_patch(Polygon(P, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3))
    ax.text(*pos, lab, fontsize=11)
ax.add_patch(Polygon(psi(FU), closed=True, facecolor=C["accent"], alpha=0.45, lw=0, zorder=3))
for q, lab, off in ((phi(p), r"$\varphi(p)$", (0.07, -0.16)), (psi(Fp), r"$\psi(F(p))$", (0.07, 0.06))):
    ax.plot(*q, "o", color=C["main"], ms=3.5, zorder=6)
    ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=10, zorder=6)


def arrow(p0, p1, rad, label, lab_xy, fs=13):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, connectionstyle=f"arc3,rad={rad}",
                        color=C["main"], lw=LW["vector"], shrinkA=2, shrinkB=2, zorder=10)
    ax.add_patch(a)
    ax.text(*lab_xy, label, fontsize=fs, ha="center", va="center")


arrow(cU + np.array([0.35, 0.45]), cV + np.array([-0.45, 0.45]), -0.35, r"$F$", (0.0, 2.55))
arrow(cU + np.array([-0.35, -0.5]), phi(cU) + np.array([-0.2, 0.62]), 0.25, r"$\varphi$", (-2.2, 0.35))
arrow(cV + np.array([0.4, -0.52]), psi(cV) + np.array([0.3, 0.68]), -0.25, r"$\psi$", (2.25, 0.35))
arrow(phi(cU) + np.array([0.65, -0.05]), psi(cV) + np.array([-0.7, -0.05]), 0.3,
      r"$\hat F=\psi\circ F\circ\varphi^{-1}$", (0.0, -1.2), fs=12)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
