"""그림 4.1.1: 매개곡선과 속도벡터 (4.1절, 예 4.1.3, 비예 4.1.9, 예 4.1.10).

(a) 원 γ(t) = (cos t, sin t) (기준 예제, r = 1)와 β(t) = γ(2t): 자취는 같고 속도는 두 배.
    같은 세 점에서 γ'(t)(파랑)와 β'(t/2) = 2γ'(t)(주황)를 그린다.
(b) 뾰족점 곡선 γ(t) = (t², t³): 매끄러운 사상이지만 γ'(0) = 0이고 자취 y² = x³는 원점에서 뾰족하다.
(c) 스스로 만나는 곡선 γ(t) = (t² − 1, t³ − t): γ(1) = γ(−1) = (0, 0)에서 속도가 두 개다.

벡터는 모두 실제 길이의 1/2배(a)와 1/4배(b, c)로 그린다(캡션에 적음).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-1-1-velocity.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

# 기준 예제의 원 (dgsym.EXAMPLES) — r = 1, 평면곡선이므로 앞의 두 성분만 쓴다
ex = dgsym.EXAMPLES["circle"]
(t,) = ex["coords"]
(r,) = ex["params"]
circ = sp.lambdify(t, ex["expr"].subs(r, 1)[:2], "numpy")
dcirc = sp.lambdify(t, ex["expr"].diff(t).subs(r, 1)[:2], "numpy")


def cusp(t):
    return np.array([t ** 2, t ** 3])


def dcusp(t):
    return np.array([2 * t, 3 * t ** 2])


def node(t):
    return np.array([t ** 2 - 1, t ** 3 - t])


def dnode(t):
    return np.array([2 * t, 3 * t ** 2 - 1])


# 자기검사 -------------------------------------------------------------------
for tt in np.linspace(-3, 3, 13):
    p = np.array(circ(tt), float)
    assert abs(np.hypot(*p) - 1) < 1e-12                      # 자취는 단위원
    assert abs(np.dot(p, np.array(dcirc(tt), float))) < 1e-12  # <γ, γ'> = 0
    x, y = cusp(tt)
    assert abs(y ** 2 - x ** 3) < 1e-9                        # y² = x³
    x, y = node(tt)
    assert abs(y ** 2 - x ** 2 * (x + 1)) < 1e-9              # y² = x²(x + 1)
    assert np.linalg.norm(dnode(tt)) > 0.5                    # 정칙 (|γ'| ≥ 1은 아니지만 0이 아님)
assert np.allclose(dcusp(0.0), 0)
assert np.allclose(node(1.0), 0) and np.allclose(node(-1.0), 0)
assert np.allclose(dnode(1.0), [2, 2]) and np.allclose(dnode(-1.0), [-2, 2])


def arrow(ax, base, vec, color, lw=None, z=5, scale=1.0):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + scale * np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=10, shrinkA=0, shrinkB=0), zorder=z)


def frame(ax, xlim, ylim, tag):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)


fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))

# (a) 원과 두 배 빠른 원 ----------------------------------------------------------
ax = axs[0]
XL, YL = (-1.85, 1.85), (-1.85, 1.85)
frame(ax, XL, YL, "(a)")
ts = np.linspace(0, 2 * np.pi, 400)
P = np.array(circ(ts)).T
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=3)
for tt in (np.pi / 6, 5 * np.pi / 6, 3 * np.pi / 2):
    p = np.array(circ(tt), float)
    v = np.array(dcirc(tt), float)
    arrow(ax, p, 2 * v, C["accent"], scale=0.5, z=5)     # β'(t/2) = 2γ'(t)
    arrow(ax, p, v, C["tangent"], scale=0.5, z=6)        # γ'(t)
    ax.plot(*p, "o", color=C["main"], ms=3, zorder=7)
ax.text(0.70, 0.93, r"$\gamma'$", color=C["tangent"], fontsize=12, ha="left", va="center")
ax.text(0.47, 1.43, r"$\beta'$", color=C["accent"], fontsize=12, ha="left", va="center")
ax.text(-0.78, -0.95, r"$\gamma,\ \beta$", color=C["main"], fontsize=11, ha="right")

ax.set_xlim(*XL); ax.set_ylim(*YL)

# (b) 뾰족점 --------------------------------------------------------------------
ax = axs[1]
XL, YL = (-0.95, 1.5), (-1.5, 1.5)
frame(ax, XL, YL, "(b)")
ts = np.linspace(-1.12, 1.12, 400)
P = cusp(ts).T
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=3)
for tt in (-1.0, -0.62, -0.38, 0.38, 0.62, 1.0):
    p = cusp(tt)
    arrow(ax, p, dcusp(tt), C["tangent"], scale=0.25, z=6)
    ax.plot(*p, "o", color=C["main"], ms=2.5, zorder=7)
ax.plot(0, 0, "o", color=C["normal"], ms=4, zorder=8)
ax.text(-0.1, 0.16, r"$\gamma'(0)=0$", color=C["normal"], fontsize=10, ha="right")
ax.text(0.62, 1.2, r"$t>0$", color=C["main"], fontsize=10)
ax.text(0.62, -1.3, r"$t<0$", color=C["main"], fontsize=10)
ax.set_xlim(*XL); ax.set_ylim(*YL)

# (c) 스스로 만나는 곡선 -----------------------------------------------------------
ax = axs[2]
XL, YL = (-1.35, 1.35), (-1.2, 1.2)
frame(ax, XL, YL, "(c)")
ts = np.linspace(-1.52, 1.52, 400)
P = node(ts).T
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=3)
arrow(ax, (0, 0), dnode(1.0), C["tangent"], scale=0.25, z=6)
arrow(ax, (0, 0), dnode(-1.0), C["tangent"], scale=0.25, z=6)
ax.plot(0, 0, "o", color=C["main"], ms=3.5, zorder=7)
ax.text(0.5, 0.52, r"$\gamma'(1)$", color=C["tangent"], fontsize=10, ha="left")
ax.text(-0.5, 0.52, r"$\gamma'(-1)$", color=C["tangent"], fontsize=10, ha="right")
ax.text(-1.0, -0.12, r"$t=0$", color=C["main"], fontsize=10, ha="right", va="top")
ax.plot(*node(0.0), "o", color=C["main"], ms=2.5, zorder=7)
ax.set_xlim(*XL); ax.set_ylim(*YL)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
