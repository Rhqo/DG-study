"""그림 7.1.1: 같은 크기의 좌표 칸이 구면 위에서는 크기가 다르다 (7.1절 동기, 예 7.1.2).

x(θ, φ) = (r sin θ cos φ, r sin θ sin φ, r cos θ) (dgsym.EXAMPLES["sphere"]), r = 1.
왼쪽: U = (0, π) × (0, 2π)의 좌표격자(간격 π/12)와 크기가 같은 두 칸
      A = [5π/12, 7π/12] × [φ_A, φ_A + π/6] (적도 근처, φ_A = π/4), B = [π/6, π/3] × [φ_B, φ_B + π/6] (북극 쪽, φ_B = π/4).
오른쪽: 구면과 위도원·경선 격자, 두 칸의 상 x(A), x(B)(하늘색),
        두 칸의 왼쪽 아래 꼭짓점 p_A = x(5π/12, φ_A), p_B = x(π/6, φ_B)에서 x_θ, x_φ (파랑, 0.5배).
        |x_θ| = r이고 |x_φ| = r sin θ이므로 북극 근처에서 x_φ가 짧다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-1-1-sphere-grid.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Xt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")
E, F, G = (sp.lambdify((th, ph), c, "numpy") for c in dgsym.first_ff(Xe, th, ph, ex["positive"]))


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


D = np.pi / 6                      # 칸의 크기
SC = 0.5                           # 벡터 배율
cells = {"A": (5 * np.pi / 12, np.pi / 4), "B": (np.pi / 6, np.pi / 4)}   # (θ₀, φ₀): 왼쪽 아래 꼭짓점

# 자기검사: |x_θ| = 1 = √E, |x_φ| = sin θ = √G, x_θ ⊥ x_φ
for t0, p0 in cells.values():
    a, b = np.array(Xt(t0, p0), float), np.array(Xp(t0, p0), float)
    assert np.isclose(a @ a, E(t0, p0)) and np.isclose(b @ b, G(t0, p0))
    assert np.isclose(a @ b, 0.0) and np.isclose(np.linalg.norm(b), np.sin(t0))

fig = plt.figure(figsize=(6.6, 3.6))

# 왼쪽: U --------------------------------------------------------------------------------
axU = fig.add_axes([0.03, 0.08, 0.22, 0.84])
axU.fill([0, np.pi, np.pi, 0], [0, 0, 2 * np.pi, 2 * np.pi], color=C["region"], alpha=0.18, lw=0)
for k in range(1, 12):
    axU.plot([k * np.pi / 12] * 2, [0, 2 * np.pi], color="#9A9A9A", lw=0.35)
for k in range(1, 24):
    axU.plot([0, np.pi], [k * np.pi / 12] * 2, color="#9A9A9A", lw=0.35)
axU.plot([0, np.pi, np.pi, 0, 0], [0, 0, 2 * np.pi, 2 * np.pi, 0], color=C["main"], lw=0.8)
for name, (t0, p0) in cells.items():
    axU.fill([t0, t0 + D, t0 + D, t0], [p0, p0, p0 + D, p0 + D], color=C["region"], alpha=0.75, lw=0)
    axU.plot([t0, t0 + D, t0 + D, t0, t0], [p0, p0, p0 + D, p0 + D, p0], color=C["tangent"], lw=1.0)
    axU.text(t0 + D / 2, p0 + D / 2, f"${name}$", fontsize=11, ha="center", va="center")
axU.text(np.pi / 2, -0.35, r"$\theta$", fontsize=12, ha="center", va="top")
axU.text(-0.3, np.pi, r"$\varphi$", fontsize=12, ha="right", va="center")
axU.text(np.pi, -0.1, r"$\pi$", fontsize=10, ha="center", va="top")
axU.text(0, -0.1, r"$0$", fontsize=10, ha="center", va="top")
axU.text(-0.08, 2 * np.pi, r"$2\pi$", fontsize=10, ha="right", va="center")
axU.text(np.pi / 2, 2 * np.pi + 0.25, r"$U$", fontsize=13, ha="center")
dgfig.schematic_axes(axU, (-0.8, np.pi + 0.2), (-0.8, 2 * np.pi + 0.7))

# 오른쪽: 구면 ------------------------------------------------------------------------------
ax = fig.add_axes([0.27, -0.02, 0.73, 1.04], projection="3d", computed_zorder=False)
ax.view_init(elev=24, azim=58)
ax.set_axis_off()
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = xmap(Tg, Pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.22, grid=False, zorder=1)


def draw(curve, role="#8A8A8A", lw=0.4, zorder=3):
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=dgfig.visible_mask(ax, curve, curve), hidden=None,
                  zorder=zorder)


s = np.linspace(0, 2 * np.pi, 400)
for k in range(1, 12):                      # 위도원 θ = kπ/12
    draw(xmap(k * np.pi / 12 * np.ones_like(s), s))
s2 = np.linspace(0, np.pi, 200)
for k in range(24):                         # 경선 φ = kπ/12
    draw(xmap(s2, k * np.pi / 12 * np.ones_like(s2)))

view = dgfig.view_vector(ax)
for name, (t0, p0) in cells.items():
    tt = np.linspace(t0, t0 + D, 20)
    pp = np.linspace(p0, p0 + D, 20)
    edge = np.concatenate([xmap(tt, p0 * np.ones_like(tt)), xmap((t0 + D) * np.ones_like(pp), pp),
                           xmap(tt[::-1], (p0 + D) * np.ones_like(tt)), xmap(t0 * np.ones_like(pp), pp[::-1])])
    assert np.all(edge @ view > 0)          # 두 칸은 보이는 쪽에 있다
    poly = Poly3DCollection([edge * 1.002], facecolor=C["region"], alpha=0.7, edgecolor=C["tangent"],
                            linewidth=1.0, zorder=6)
    ax.add_collection3d(poly)
    p = np.array(X(t0, p0), float)
    dgfig.point3d(ax, p, size=10, zorder=12)
    dgfig.arrow3d(ax, p, SC * np.array(Xt(t0, p0), float), role="tangent", zorder=14, head=9)
    dgfig.arrow3d(ax, p, SC * np.array(Xp(t0, p0), float), role="tangent", zorder=14, head=9)
    lab = {"A": (r"$\mathbf{x}_\theta$", r"$\mathbf{x}_\varphi$", r"$p_A$"),
           "B": (r"$\mathbf{x}_\theta$", r"$\mathbf{x}_\varphi$", r"$p_B$")}[name]
    tip_t = p + SC * np.array(Xt(t0, p0), float)
    tip_p = p + SC * np.array(Xp(t0, p0), float)
    ax.text(*(tip_t + np.array([0.14, -0.09, -0.02])), lab[0], color=C["tangent"], fontsize=11, zorder=15)
    ax.text(*(tip_p + np.array([0.0, 0.06, 0.06])), lab[1], color=C["tangent"], fontsize=11, zorder=15)
    ax.text(*(p + np.array([0.0, -0.13, 0.03])), lab[2], fontsize=11, zorder=15)
    ax.text(*(1.04 * np.array(X(t0 + D / 2, p0 + D / 2), float) + np.array([0.02, 0.0, 0.0])),
            f"$\\mathbf{{x}}({name})$", fontsize=10, zorder=15, ha="center", va="center")

dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.35)
dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(1.0, 0.62), xy_to=(0.28, 0.66), rad=-0.3)

dgfig.save(fig, __file__)
