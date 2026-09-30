"""그림 4.4.1: 부호곡률 — 단위접벡터가 도는 방향과 빠르기 (4.4절, 정의 4.4.2, 예 4.4.6(d), 정리 4.4.9).

(a) 평면곡선 γ(x) = (x, sin x), −π ≤ x ≤ π와 다섯 점 x = −2.2, −1.1, 0, 1.1, 2.2에서의
    단위접벡터 t(파랑)와 부호 단위법벡터 n_s = Jt(주홍). κ_s = −sin x/(1 + cos²x)^{3/2}:
    x < 0에서 κ_s > 0(왼쪽으로 돈다), x > 0에서 κ_s < 0(오른쪽으로 돈다), x = 0은 변곡점.
(b) 같은 다섯 점의 t를 원점으로 옮긴 것(단위원 위의 점). 접선각 θ = arctan(cos x)는 −π/4에서 π/4로 늘었다가(κ_s > 0)
    다시 −π/4로 준다(κ_s < 0). 안쪽 호(올라감)와 바깥쪽 호(내려감)로 두 구간을 나누어 그린다.

벡터는 (a)에서 실제 길이의 0.6배, (b)에서 실제 길이로 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-4-1-signed-curvature.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

C = dgfig.COLORS
LW = dgfig.LW
J = np.array([[0.0, -1.0], [1.0, 0.0]])          # +90° 회전


def gamma(x):
    return np.stack([x, np.sin(x)], axis=-1)


def d1(x):
    return np.stack([np.ones_like(x), np.cos(x)], axis=-1)


def d2(x):
    return np.stack([np.zeros_like(x), -np.sin(x)], axis=-1)


def unit_t(x):
    v = d1(x)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def kappa_s(x):
    """명제 4.4.5: κ_s = <γ'', Jγ'>/|γ'|³."""
    v, a = d1(x), d2(x)
    return np.einsum("...i,...i->...", a, v @ J.T) / np.linalg.norm(v, axis=-1) ** 3


XS = np.array([-2.2, -1.1, 0.0, 1.1, 2.2])

# 자기검사: 공식과 정의(dt/ds = κ_s n_s, 수치미분)의 일치, 부호
xx = np.linspace(-np.pi, np.pi, 2001)
assert np.allclose(kappa_s(xx), -np.sin(xx) / (1 + np.cos(xx) ** 2) ** 1.5)
h = 1e-5
for x0 in XS:
    dt_dx = (unit_t(np.array(x0 + h)) - unit_t(np.array(x0 - h))) / (2 * h)
    ds_dx = np.linalg.norm(d1(np.array(x0)))
    n_s = J @ unit_t(np.array(x0))
    assert np.allclose(dt_dx / ds_dx, kappa_s(np.array(x0)) * n_s, atol=1e-8)
assert np.all(kappa_s(xx[(xx < -1e-9) & (xx > -np.pi + 1e-9)]) > 0)
assert np.all(kappa_s(xx[(xx > 1e-9) & (xx < np.pi - 1e-9)]) < 0)
theta = np.arctan(np.cos(xx))                      # 접선각
dtheta_ds = np.gradient(theta, xx) / np.linalg.norm(d1(xx), axis=1)
assert np.allclose(dtheta_ds[5:-5], kappa_s(xx)[5:-5], atol=1e-4)   # 정리 4.4.9: κ_s = dθ/ds


def arrow(ax, base, vec, color, lw=None, scale=1.0, z=5):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + scale * np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=10, shrinkA=0, shrinkB=0), zorder=z)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[2.1, 1]))

# (a) --------------------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-3.4, 3.4), (-1.75, 1.75))
ax.plot([-3.4, 3.4], [0, 0], color=C["aux"], lw=0.5, zorder=0)
P = gamma(np.linspace(-np.pi, np.pi, 500))
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=3)
for x0 in XS:
    p = gamma(np.array(x0))
    tv = unit_t(np.array(x0))
    arrow(ax, p, tv, C["tangent"], scale=0.6, z=6)
    arrow(ax, p, J @ tv, C["normal"], scale=0.6, z=6)
    ax.plot(*p, "o", color=C["main"], ms=3.5, zorder=7)
p = gamma(np.array(-1.1)); tv = unit_t(np.array(-1.1))
ax.text(*(p + 0.6 * tv + [0.05, -0.2]), r"$\mathbf{t}$", color=C["tangent"], fontsize=12)
ax.text(*(p + 0.6 * (J @ tv) + [-0.32, 0.02]), r"$\mathbf{n}_s$", color=C["normal"], fontsize=12)
ax.text(-2.1, -1.55, r"$\kappa_s>0$", fontsize=11, ha="center")
ax.text(2.1, 1.4, r"$\kappa_s<0$", fontsize=11, ha="center")
ax.text(0.12, -0.38, r"$\kappa_s=0$", fontsize=10, ha="left", color=C["aux"])
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) 접선의 지표 (단위원 위) ------------------------------------------------------
ax = axs[1]
dgfig.schematic_axes(ax, (-0.45, 1.55), (-1.25, 1.25))
th = np.linspace(-np.pi / 2 - 0.3, np.pi / 2 + 0.3, 200)
ax.plot(np.cos(th), np.sin(th), color=C["aux"], lw=0.6, ls=(0, (3, 2.5)), zorder=1)
ax.plot([0, 1.45], [0, 0], color=C["aux"], lw=0.5, zorder=0)
for rad, sgn, lab in ((0.8, 1, r"$\kappa_s>0$"), (1.22, -1, r"$\kappa_s<0$")):
    a = np.linspace(-np.pi / 4, np.pi / 4, 60)
    ax.plot(rad * np.cos(a), rad * np.sin(a), color=C["accent"], lw=1.4, zorder=2)
    a0, a1 = (-0.15, 0.15) if sgn > 0 else (0.15, -0.15)
    ax.add_patch(FancyArrowPatch((rad * np.cos(a0), rad * np.sin(a0)), (rad * np.cos(a1), rad * np.sin(a1)),
                                 connectionstyle=f"arc3,rad={0.08 * sgn}", arrowstyle="-|>", mutation_scale=10,
                                 color=C["accent"], lw=1.4, zorder=3))
ax.text(0.47, 0.0, r"$\kappa_s>0$", fontsize=9.5, ha="center", va="center", color=C["accent"], rotation=90)
ax.text(1.43, 0.0, r"$\kappa_s<0$", fontsize=9.5, ha="center", va="center", color=C["accent"], rotation=90)
for x0 in XS:
    arrow(ax, (0, 0), unit_t(np.array(x0)), C["tangent"], lw=1.1, z=5)
ax.plot(0, 0, "o", color=C["main"], ms=3, zorder=6)
ax.text(0.28, 0.1, r"$\theta$", fontsize=11, zorder=6)
ax.add_patch(FancyArrowPatch((0.24, 0.0), (0.24 * np.cos(np.pi / 4), 0.24 * np.sin(np.pi / 4)),
                             connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=7,
                             color=C["main"], lw=0.8, zorder=6))
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
