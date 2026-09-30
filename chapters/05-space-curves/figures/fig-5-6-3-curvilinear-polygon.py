"""그림 5.6.3: 곡선다각형과 외각 (5.6절, 정의 5.6.9, 정리 5.6.11).

꼭짓점 P0 = (0, -1), P1 = (cos 30°, sin 30°), P2 = (cos 150°, sin 150°)를 반시계 방향으로 잇는 세 조각
γ_i(u) = P_i + u(P_{i+1} - P_i) + 0.12 sin(πu) ν_i (0 ≤ u ≤ 1, ν_i는 현의 왼쪽 단위법선, 곧 안쪽)으로 만든
오목한 곡선 삼각형. 각 꼭짓점에서 들어오는 단위접벡터 t⁻ (회색 점선 연장)과 나가는 단위접벡터 t⁺ (파랑),
그리고 t⁻에서 t⁺까지의 외각 ε_i (주황 호)를 그린다. 가장 낮은 점은 꼭짓점 P0이다.
자기검사: Σ∫κ_s|γ'| + Σε_i = 2π (정리 5.6.11), 외각은 (-π, π) 안.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-6-3-curvilinear-polygon.py``
"""

import dgfig

dgfig.setup()

import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
t = sp.symbols("t", real=True)
P = [np.array([math.cos(a), math.sin(a)]) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
H = 0.12


def edge(p, q):
    d = q - p
    nu = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return sp.Matrix([p[0] + t * d[0] + H * sp.sin(sp.pi * t) * nu[0], p[1] + t * d[1] + H * sp.sin(sp.pi * t) * nu[1]])


def ang(u, v):
    return 2 * math.atan((u[0] * v[1] - u[1] * v[0]) / (1 + u @ v))


edges = [edge(P[i], P[(i + 1) % 3]) for i in range(3)]
total = 0.0
tan_in, tan_out = [], []
for e in edges:
    ks = dgsym.signed_curvature(e, t)
    spd = sp.sqrt(sp.diff(e[0], t) ** 2 + sp.diff(e[1], t) ** 2)
    f = sp.lambdify(t, ks * spd, "numpy")
    xs = np.linspace(0, 1, 20001)
    total += np.trapezoid(np.broadcast_to(f(xs), xs.shape), xs)
    d = sp.lambdify(t, list(sp.diff(e, t)), "numpy")
    a0, a1 = np.array(d(0.0), float), np.array(d(1.0), float)
    tan_out.append(a0 / np.linalg.norm(a0))          # 조각의 시작 = 꼭짓점 P_i에서 나가는 방향
    tan_in.append(a1 / np.linalg.norm(a1))           # 조각의 끝 = 꼭짓점 P_{i+1}로 들어오는 방향
eps = [ang(tan_in[i - 1], tan_out[i]) for i in range(3)]      # 꼭짓점 P_i: 들어오는 것은 조각 i-1의 끝
assert all(abs(x) < math.pi - 1e-3 for x in eps)
assert abs(total + sum(eps) - 2 * math.pi) < 1e-7
pts = np.concatenate([np.array(sp.lambdify(t, list(e), "numpy")(np.linspace(0, 1, 400))).T for e in edges])
assert abs(pts[:, 1].min() + 1) < 1e-9                        # 가장 낮은 점 = P0

fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.fill(*pts.T, color=C["region"], alpha=0.18, lw=0)
ax.plot(*np.vstack([pts, pts[:1]]).T, color=C["main"], lw=dgfig.LW["main"], zorder=3)
for i in range(3):
    p = P[i]
    Tm, Tp = tan_in[i - 1], tan_out[i]
    ax.plot(*np.stack([p, p + 0.42 * Tm]).T, color=C["aux"], lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.annotate("", xy=p + 0.42 * Tp, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4,
                                                               mutation_scale=11, shrinkA=0, shrinkB=0), zorder=5)
    a0 = math.atan2(Tm[1], Tm[0])
    arc = np.linspace(a0, a0 + eps[i], 60)
    rr = 0.24
    ax.plot(p[0] + rr * np.cos(arc), p[1] + rr * np.sin(arc), color=C["accent"], lw=1.4, zorder=4)
    mid = a0 + eps[i] / 2
    ax.text(p[0] + 0.36 * math.cos(mid), p[1] + 0.36 * math.sin(mid), r"$\varepsilon_{%d}$" % i, fontsize=12,
            color=C["accent"], ha="center", va="center")
    ax.plot(*p, "o", color=C["main"], ms=4, zorder=6)
off = {0: (0.0, -0.13, "center", "top"), 1: (0.08, 0.0, "left", "center"), 2: (-0.08, 0.0, "right", "center")}
for i in range(3):
    dx, dy, ha, va = off[i]
    ax.text(P[i][0] + dx, P[i][1] + dy, r"$P_{%d}$" % i, fontsize=12, ha=ha, va=va)
ax.text(*(P[0] + 0.42 * tan_out[0] + np.array([0.04, -0.04])), r"$\mathbf{t}^+$", fontsize=11, color=C["tangent"], va="top")
ax.text(*(P[0] + 0.42 * tan_in[2] + np.array([-0.04, -0.04])), r"$\mathbf{t}^-$", fontsize=11, color=C["aux"], ha="right", va="top")
ax.axhline(-1, color=C["aux"], lw=0.6, ls=(0, (4, 3)), zorder=1)
dgfig.schematic_axes(ax, xlim=(-1.45, 1.45), ylim=(-1.45, 1.05))
dgfig.save(fig, __file__)
