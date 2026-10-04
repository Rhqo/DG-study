"""그림 8.2.2: 뫼니에 정리 — 구면 위에서 같은 접선을 갖는 위도원과 대원 (8.2절, 정리 8.2.5, 예 8.2.6).

단위구면 x(θ, φ) (dgsym.EXAMPLES["sphere"]), r = 1, 바깥쪽 N = x/r. p = x(θ₀, 0), θ₀ = π/3.
위도원 γ₁ (θ = θ₀, 주황)과 p에서 같은 접벡터 t = e₂를 갖는 대원 γ₂(s) = cos s · p + sin s · e₂ (검정).
p에서 N (주홍 실선, 0.5배), 위도원의 곡률벡터 κ₁n₁ (주황 점선, 실제 길이 1/sin θ₀),
대원의 곡률벡터 κ₂n₂ = −N (주황 점선, 실제 길이 1), t (파랑, 0.5배).
회색 점선은 κ₁n₁의 끝에서 N 방향 직선으로 내린 수선이다: 발은 p − N = p + κ₂n₂.
자기검사: 두 곡선의 p에서의 속도가 같다, ⟨κ₁n₁, N⟩ = ⟨κ₂n₂, N⟩ = −1/r, κ₁ = 1/(r sin θ₀).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-2-2-meusnier.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Xf = sp.lambdify((th, ph), list(ex["expr"].subs(rs, 1)), "numpy")
th0 = np.pi / 3
p = np.array(Xf(th0, 0.0), float)
N = p.copy()                                      # 바깥쪽 N = x/r, r = 1
e2 = np.array([0.0, 1.0, 0.0])

# 위도원 γ₁(s) = x(θ₀, s/sin θ₀) (단위속력), 대원 γ₂(s) = cos s p + sin s e₂ (단위속력)
def lat(s):
    return np.stack(Xf(th0 * np.ones_like(s), s / np.sin(th0)), axis=-1)


def great(s):
    return np.outer(np.cos(s), p) + np.outer(np.sin(s), e2)


h = 1e-5
v1 = (lat(np.array([h])) - lat(np.array([-h])))[0] / (2 * h)
v2 = (great(np.array([h])) - great(np.array([-h])))[0] / (2 * h)
a1 = (lat(np.array([h])) - 2 * lat(np.array([0.0])) + lat(np.array([-h])))[0] / h ** 2
a2 = (great(np.array([h])) - 2 * great(np.array([0.0])) + great(np.array([-h])))[0] / h ** 2
assert np.allclose(v1, e2, atol=1e-8) and np.allclose(v2, e2, atol=1e-8)
assert abs(a1 @ N + 1) < 1e-4 and abs(a2 @ N + 1) < 1e-4                    # 법성분 −1/r
assert abs(np.linalg.norm(a1) - 1 / np.sin(th0)) < 1e-4                      # κ₁ = 1/(r sin θ₀)
k1n1 = np.array([-np.cos(0.0), -np.sin(0.0), 0.0]) / np.sin(th0)             # 위도원의 κn: 축을 향함
assert np.allclose(a1, k1n1, atol=1e-4)
k2n2 = -N

fig = plt.figure(figsize=(5.0, 4.4))
ax = dgfig.axes3d(fig, elev=30, azim=-60)
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = np.stack(Xf(Tg, Pg), axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.16, grid=False, zorder=1)
view = dgfig.view_vector(ax)
s = np.linspace(0, 2 * np.pi, 400)
L1 = lat(s * np.sin(th0))
G2 = great(s)
dgfig.curve3d(ax, L1, role="accent", lw=1.8, visible=L1 @ view > 0, hidden="dashed", zorder=5)
dgfig.curve3d(ax, G2, role="main", lw=1.2, visible=G2 @ view > 0, hidden="dashed", zorder=5)
# 위도원의 중심과 축
cen = np.array([0, 0, np.cos(th0)])
ax.plot([0, 0], [0, 0], [-1.15, 1.2], color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.plot(*np.stack([p, cen]).T, color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=4)
# 벡터
dgfig.arrow3d(ax, p, 0.5 * N, role="normal", label=r"$\mathbf{N}$", zorder=14, lw=1.6, label_offset=(0.02, 0, 0.05))
from matplotlib.patches import FancyArrowPatch  # noqa: F401  (arrow3d 내부)
for vec, lab, off in ((k1n1, r"$\kappa_1\mathbf{n}_1$", (0.0, 0.05, 0.07)), (k2n2, r"$\kappa_2\mathbf{n}_2$", (0.0, -0.02, -0.1))):
    arr = dgfig.arrow3d(ax, p, vec, role="accent", label=lab, zorder=13, lw=1.5, label_offset=off, fontsize=11)
    arr.set_linestyle((0, (4, 2.5)))
dgfig.arrow3d(ax, p, 0.5 * e2, role="tangent", label=r"$\mathbf{t}$", zorder=14, label_offset=(0, 0.04, 0.04))
tip = p + k1n1
foot = p - N
ax.plot(*np.stack([tip, foot]).T, color=C["aux"], lw=0.8, ls=(0, (1.5, 1.5)), zorder=12)
dgfig.point3d(ax, p, size=14, zorder=16)
ax.text(*(p + np.array([0.05, 0.08, -0.12])), r"$p$", fontsize=12, zorder=16)
ax.text(*(lat(np.array([2.2 * np.sin(th0)]))[0] + np.array([0, 0, 0.06])), r"$\gamma_1$", color=C["accent"], fontsize=12, zorder=16)
ax.text(*(great(np.array([2.2]))[0] + np.array([0.0, 0.05, 0.05])), r"$\gamma_2$", color=C["main"], fontsize=12, zorder=16)
dgfig.equal_aspect(ax, np.array([[-1.05, -1.05, -1.05], [1.3, 1.05, 1.3]]), zoom=1.3)

dgfig.save(fig, __file__)
