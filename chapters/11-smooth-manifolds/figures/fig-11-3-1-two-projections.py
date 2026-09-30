"""그림 11.3.1: S^1의 두 입체사영 σ(북극 N에서)와 σ̃(남극 S에서) (11.3절, 명제 11.3.1).

x = (3/5, 4/5): σ(x) = 3, σ̃(x) = 1/3.
z = (-4/5, -3/5): σ(z) = -1/2, σ̃(z) = -2.
두 경우 모두 σ̃ = σ/|σ|^2, 즉 σ·σ̃ = 1 (n = 1에서 u ↦ 1/u).
공식은 §7: σ(x) = x^1/(1 - x^2), σ̃(x) = x^1/(1 + x^2).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-3-1-two-projections.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def sigma(p):
    return p[0] / (1 - p[1])


def sigma_t(p):
    return p[0] / (1 + p[1])


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


N, S = np.array([0.0, 1.0]), np.array([0.0, -1.0])
pts = {"x": np.array([3 / 5, 4 / 5]), "z": np.array([-4 / 5, -3 / 5])}

# 자기검사 (§12.1)
for p in pts.values():
    assert abs(np.hypot(*p) - 1) < 1e-12
    s, st = sigma(p), sigma_t(p)
    assert abs(s * st - 1) < 1e-12
    # σ(p)는 N, p를 잇는 직선과 가로축의 교점, σ̃(p)는 S, p를 잇는 직선과 가로축의 교점
    assert abs(cross2(p - N, np.array([s, 0]) - N)) < 1e-12
    assert abs(cross2(p - S, np.array([st, 0]) - S)) < 1e-12
assert np.isclose(sigma(pts["x"]), 3) and np.isclose(sigma_t(pts["x"]), 1 / 3)
assert np.isclose(sigma(pts["z"]), -0.5) and np.isclose(sigma_t(pts["z"]), -2)

fig, ax = plt.subplots(figsize=(6.0, 3.4))
dgfig.schematic_axes(ax, (-2.6, 3.35), (-1.3, 1.35))
ax.annotate("", xy=(3.3, 0), xytext=(-2.55, 0), arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(3.3, -0.12, r"$\mathbb{R}$", fontsize=11, ha="right", va="top")
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=C["main"], lw=LW["main"], zorder=3)

for name, p in pts.items():
    s, st = sigma(p), sigma_t(p)
    # N → p → σ(p) (파랑), S → p → σ̃(p) (주황)
    # 직선은 극에서 p와 교점 중 더 먼 쪽까지 그린다 (세 점이 한 직선 위에 있다)
    far_n = p if np.linalg.norm(p - N) > np.hypot(s - N[0], N[1]) else np.array([s, 0.0])
    far_s = p if np.linalg.norm(p - S) > np.hypot(st - S[0], S[1]) else np.array([st, 0.0])
    ax.plot([N[0], far_n[0]], [N[1], far_n[1]], color=C["tangent"], lw=1.0, zorder=2)
    ax.plot([S[0], far_s[0]], [S[1], far_s[1]], color=C["accent"], lw=1.0, zorder=2)
    ax.plot(s, 0, "o", ms=5, color=C["tangent"], zorder=5)
    ax.plot(st, 0, "s", ms=5, color=C["accent"], zorder=5)
    ax.plot(*p, "o", ms=5, color=C["main"], zorder=6)

ax.plot(*N, "o", ms=5, mfc="white", mec=C["main"], zorder=6)
ax.plot(*S, "o", ms=5, mfc="white", mec=C["main"], zorder=6)
ax.text(0.08, 1.08, r"$N$", fontsize=12)
ax.text(0.08, -1.2, r"$S$", fontsize=12)
ax.text(0.62, 0.85, r"$x$", fontsize=12)
ax.text(-0.98, -0.68, r"$z$", fontsize=12)
ax.text(3.0, 0.1, r"$\sigma(x)=3$", fontsize=10, color=C["tangent"], ha="center", va="bottom")
ax.text(0.36, -0.1, r"$\tilde\sigma(x)=\frac{1}{3}$", fontsize=10, color=C["accent"], ha="left", va="top")
ax.text(-0.5, 0.1, r"$\sigma(z)=-\frac{1}{2}$", fontsize=10, color=C["tangent"], ha="center", va="bottom")
ax.text(-2.0, -0.12, r"$\tilde\sigma(z)=-2$", fontsize=10, color=C["accent"], ha="center", va="top")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
