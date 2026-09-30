"""그림 11.2.1: 두 차트의 좌표변환 ψ∘φ^{-1} (11.2절, 정의 11.2.1). (개념도)

위: 위상다양체 M(얼룩)과 겹치는 두 열린집합 U(원판, 중심 cU, 반지름 0.62)와 V(원판, 중심 cV, 반지름 0.58).
아래 왼쪽: φ(U) ⊆ R^n, 아래 오른쪽: ψ(V) ⊆ R^n. 진하게 칠한 곳이 φ(U ∩ V), ψ(U ∩ V)이다.
φ, ψ는 명시적인 단사 매끄러운 사상(회전·확대 + 작은 이차 섭동)으로 정했다. 모양은 개념도다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-2-1-transition.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

C = dgfig.COLORS
LW = dgfig.LW

cU, rU = np.array([-0.42, 1.55]), 0.62
cV, rV = np.array([0.40, 1.62]), 0.58


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def phi(z):
    w = (np.asarray(z, float) - cU) @ rot(0.25).T * 1.05
    w = w + 0.12 * np.stack([w[..., 1] ** 2, -w[..., 0] * w[..., 1]], axis=-1)
    return w + np.array([-1.35, -0.55])


def psi(z):
    w = (np.asarray(z, float) - cV) @ rot(-0.35).T * 1.1
    w = w + 0.14 * np.stack([-w[..., 0] * w[..., 1], w[..., 0] ** 2], axis=-1)
    return w + np.array([1.35, -0.55])


def circle(c, r, n=240):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return c + r * np.stack([np.cos(t), np.sin(t)], axis=1)


def lens(n=200):
    """U ∩ V의 경계: V 안에 있는 U의 호 + U 안에 있는 V의 호."""
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    a = cU + rU * np.stack([np.cos(t), np.sin(t)], axis=1)
    b = cV + rV * np.stack([np.cos(t), np.sin(t)], axis=1)
    a_in = a[np.linalg.norm(a - cV, axis=1) <= rV]
    b_in = b[np.linalg.norm(b - cU, axis=1) <= rU]
    pts = np.concatenate([a_in, b_in])
    ctr = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
    return pts[np.argsort(ang)]


# 자기검사 (§12.1): 렌즈의 점은 U와 V 모두에 있다. φ, ψ는 U, V 위에서 단사(표본)이고 야코비 행렬식이 0이 아니다.
L = lens()
assert np.all(np.linalg.norm(L - cU, axis=1) <= rU + 1e-9) and np.all(np.linalg.norm(L - cV, axis=1) <= rV + 1e-9)
rng = np.random.default_rng(3)
for f, c, r in ((phi, cU, rU), (psi, cV, rV)):
    s = c + r * np.sqrt(rng.uniform(0, 1, 300))[:, None] * np.stack([np.cos(t := rng.uniform(0, 2 * np.pi, 300)), np.sin(t)], 1)
    img = f(s)
    din = np.linalg.norm(s[:, None] - s[None], axis=-1)
    dout = np.linalg.norm(img[:, None] - img[None], axis=-1)
    assert np.all((din < 1e-12) == (dout < 1e-12))
    h = 1e-6
    for z in s[:50]:
        J = np.column_stack([(f(z + [h, 0]) - f(z - [h, 0])) / (2 * h), (f(z + [0, h]) - f(z - [0, h])) / (2 * h)])
        assert abs(np.linalg.det(J)) > 0.3

fig, ax = plt.subplots(figsize=(6.0, 4.6))
dgfig.schematic_axes(ax, (-2.55, 2.55), (-1.55, 3.05))

# M
dgfig.blob(ax, center=(0.0, 1.6), radius=1.25, seed=4, amp=0.1, fill="surface", fill_alpha=0.35)
for c, r in ((cU, rU), (cV, rV)):
    P = circle(c, r)
    ax.add_patch(Polygon(P, closed=True, facecolor=C["region"], alpha=0.22, lw=0, zorder=2))
    ax.add_patch(Polygon(P, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3))
ax.add_patch(Polygon(L, closed=True, facecolor=C["accent"], alpha=0.45, lw=0, zorder=3))
ax.text(cU[0] - 0.42, cU[1] + 0.32, r"$U$", fontsize=13)
ax.text(cV[0] + 0.34, cV[1] + 0.34, r"$V$", fontsize=13)
ax.text(-1.75, 2.75, r"$M$", fontsize=14, va="top")

# 아래 두 치역
for f, c, r, lab, pos in ((phi, cU, rU, r"$\varphi(U)$", (-2.45, -1.35)), (psi, cV, rV, r"$\psi(V)$", (1.55, -1.35))):
    P = f(circle(c, r))
    ax.add_patch(Polygon(P, closed=True, facecolor=C["region"], alpha=0.3, lw=0, zorder=2))
    ax.add_patch(Polygon(P, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=3))
    ax.add_patch(Polygon(f(L), closed=True, facecolor=C["accent"], alpha=0.45, lw=0, zorder=3))
    ax.text(*pos, lab, fontsize=12)
qa, qb = phi(L).mean(axis=0), psi(L).mean(axis=0)
ax.text(qa[0] + 0.05, qa[1] + 0.62, r"$\varphi(U\cap V)$", fontsize=10, ha="center")
ax.text(qb[0] - 0.05, qb[1] + 0.62, r"$\psi(U\cap V)$", fontsize=10, ha="center")


def arrow(p0, p1, rad, label, lab_xy, fs=13):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, connectionstyle=f"arc3,rad={rad}",
                        color=C["main"], lw=LW["vector"], shrinkA=2, shrinkB=2, zorder=10)
    ax.add_patch(a)
    ax.text(*lab_xy, label, fontsize=fs, ha="center", va="center")


arrow((-0.95, 1.05), (-1.45, 0.2), 0.25, r"$\varphi$", (-1.55, 0.72))
arrow((0.95, 1.1), (1.45, 0.2), -0.25, r"$\psi$", (1.55, 0.72))
pa = phi(L).mean(axis=0) + np.array([0.12, -0.15])
pb = psi(L).mean(axis=0) + np.array([-0.12, -0.15])
arrow(pa, pb, 0.35, r"$\psi\circ\varphi^{-1}$", (0.0, -1.05))

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
