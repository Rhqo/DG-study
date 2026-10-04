"""그림 5.6.4: 가장 낮은 점이 꼭짓점일 때 양의 방향과 회전지수의 부호 (5.6절, 명제 5.6.12의 증명).

그림 5.6.3의 오목한 곡선 삼각형(꼭짓점 P0 = (0, -1), P1 = (cos 30°, sin 30°), P2 = (cos 150°, sin 150°),
각 변은 현 + 0.12 sin(πu)·(안쪽 단위법선))을 시계 방향 P0 → P2 → P1 → P0으로 돈다. 회전지수는 -1이다.
(a) 곡선 전체, 둘러싼 영역 Ω(하늘색), 가장 낮은 점을 지나는 수평선 아래의 반평면(회색), (b)에서 확대할 곳(회색 사각형).
(b) p = P0 근처: 나가는 방향 t⁺ = e(β)(파랑), 들어오는 조각을 거꾸로 따라가는 방향 -t⁻ = e(α)(회색), α와 β를 재는 호(주황),
나가는 조각 위의 점 γ(t1)(u = U1)에서의 t1(파랑), 왼쪽 법선 J t1(주홍)과 그 방향의 반직선(주홍 점선).
자기검사: Σ∫κ_s|γ'| + Σε = -2π, 부호 있는 넓이 < 0 (그린 정리), 가장 낮은 점은 꼭짓점 P0, 0 ≤ α < β ≤ π,
β - α = π + ε0 (식 (5.6.7)), 6단계의 조건 β ≥ π/2 + 2δ, 반직선이 곡선과 만나지 않고 y < y0로 들어감,
γ(t1) ± η J t1의 감는 수 (왼쪽은 0 = Ω 밖, 오른쪽은 ±1 = Ω 안).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-6-4-orientation-vertex.py``
"""

import dgfig

dgfig.setup()

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
t = sp.symbols("t", real=True)
P = [np.array([math.cos(a), math.sin(a)]) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
H = 0.12
U1 = 0.07                                     # 시험할 점: 나가는 조각(P0 → P2)의 u = 0.07 (그림 5.6.3의 변 P2 → P0에서 u = 0.93)
ZOOM = (-0.62, 0.2, -1.1, -0.71)              # (b)의 범위 x0, x1, y0, y1


def J(v):
    return np.array([-v[1], v[0]])


def ang(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return 2 * np.arctan((u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) / (1 + (u * v).sum(-1)))


def ev(fn, x):
    """lambdify한 벡터 함수를 배열 x에서 계산한다 (상수 성분도 x의 모양으로 늘인다)."""
    return np.array([np.broadcast_to(c, np.shape(x)) for c in fn(x)], float)


def edge_ccw(p, q):
    """그림 5.6.3의 조각: p에서 q로, 현의 왼쪽(안쪽) 법선 방향으로 0.12 sin(πu)만큼 휜다."""
    d = q - p
    nu = J(d) / np.linalg.norm(d)
    return sp.Matrix([p[0] + t * d[0] + H * sp.sin(sp.pi * t) * nu[0], p[1] + t * d[1] + H * sp.sin(sp.pi * t) * nu[1]])


# 시계 방향: 그림 5.6.3의 조각을 거꾸로 따라간다 (P0 → P2, P2 → P1, P1 → P0)
ccw = [edge_ccw(P[i], P[(i + 1) % 3]) for i in range(3)]
pieces = [ccw[2].subs(t, 1 - t), ccw[1].subs(t, 1 - t), ccw[0].subs(t, 1 - t)]
G = [sp.lambdify(t, list(e), "numpy") for e in pieces]
DG = [sp.lambdify(t, list(sp.diff(e, t)), "numpy") for e in pieces]
us = np.linspace(0, 1, 20001)
curve = [ev(g, us).T for g in G]
poly = np.concatenate([c[:-1] for c in curve] + [curve[0][:1]])     # 검사용 닫힌 꺾은선

# 자기검사 1: 회전지수 -1, 부호 있는 넓이 < 0, 가장 낮은 점 = P0 ---------------------------------------------
total, area = 0.0, 0.0
t_in, t_out = [], []
for e, g, dg in zip(pieces, G, DG):
    ks = dgsym.signed_curvature(e, t)
    spd = sp.sqrt(sp.diff(e[0], t) ** 2 + sp.diff(e[1], t) ** 2)
    f = sp.lambdify(t, ks * spd, "numpy")
    total += np.trapezoid(np.broadcast_to(f(us), us.shape), us)
    xy, dxy = ev(g, us), ev(dg, us)
    area += 0.5 * np.trapezoid(xy[0] * dxy[1] - xy[1] * dxy[0], us)
    a0, a1 = ev(dg, np.array([0.0, 1.0])).T
    t_out.append(a0 / np.linalg.norm(a0))
    t_in.append(a1 / np.linalg.norm(a1))
eps = [float(ang(t_in[i - 1], t_out[i])) for i in range(3)]
assert all(abs(x) < math.pi - 1e-3 for x in eps)
assert abs(total + sum(eps) + 2 * math.pi) < 1e-7                      # 회전지수 -1
assert area < 0                                                        # 둘러싼 영역은 오른쪽 (그린 정리)
p = P[0]
y0 = p[1]
assert abs(poly[:, 1].min() - y0) < 1e-12 and np.allclose(curve[0][0], p)

# 자기검사 2: 식 (5.6.7)과 δ, 6단계의 조건 --------------------------------------------------------------
tp, tm = t_out[0], t_in[2]
beta = math.atan2(tp[1], tp[0]) % (2 * math.pi)
alpha = math.atan2(-tm[1], -tm[0]) % (2 * math.pi)
assert 0 <= alpha < beta <= math.pi and abs(beta - alpha - (math.pi + eps[0])) < 1e-12
delta = min(math.pi / 8, (beta - alpha) / 4)
assert beta >= math.pi / 2 + 2 * delta - 1e-12                         # 6단계 (좌우 대칭이라 7단계도 경계에서 성립)

# 자기검사 3: 왼쪽 법선 방향의 반직선 ---------------------------------------------------------------------
x1 = ev(G[0], np.array([U1]))[:, 0]
d1 = ev(DG[0], np.array([U1]))[:, 0]
T1 = d1 / np.linalg.norm(d1)
nu = J(T1)
assert nu[1] < 0                                                       # 아래를 향한다
lam_H = (y0 - x1[1]) / nu[1]                                           # 수평선 y = y0를 지나는 곳
lam_end = (y0 - 1.0 - x1[1]) / nu[1]


def crosses(a, b, pts):
    """선분 ab가 꺾은선 pts와 만나는가 (방향 판정)."""
    q0, q1 = pts[:-1], pts[1:]

    def orient(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])

    o1, o2 = orient(a[None], b[None], q0), orient(a[None], b[None], q1)
    o3, o4 = orient(q0, q1, a[None]), orient(q0, q1, b[None])
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


assert not crosses(x1 + 1e-4 * nu, x1 + lam_end * nu, poly)            # 반직선은 곡선을 만나지 않는다
assert x1[1] + lam_end * nu[1] < y0 - 0.5                               # 그리고 y < y0로 들어간다


def winding(q):
    v = poly - q[None]
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return float(np.sum(ang(v[:-1], v[1:])) / (2 * math.pi))


assert abs(winding(x1 + 1e-3 * nu)) < 1e-6                             # γ(t1) + η J t1은 Ω 밖 (양의 방향이 아니다)
assert abs(abs(winding(x1 - 1e-3 * nu)) - 1) < 1e-6                     # 오른쪽은 Ω 안
assert ZOOM[0] < x1[0] + lam_H * nu[0] and x1[1] + lam_H * nu[1] > ZOOM[2]   # (b)에서 수평선을 지나는 곳이 보인다

# 그리기 ------------------------------------------------------------------------------------------------
uc = np.linspace(0, 1, 301)                                            # 그리기용 (검사는 위의 촘촘한 꺾은선으로)
curve_c = [ev(g, uc).T for g in G]
poly_c = np.concatenate([c[:-1] for c in curve_c] + [curve_c[0][:1]])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw=dict(width_ratios=[2.4 / 2.2, 0.82 / 0.39]))


def base(ax, xl, yl):
    ax.fill(*poly_c.T, color=C["region"], alpha=0.22, lw=0, zorder=0)
    ax.fill_between([xl[0] - 1, xl[1] + 1], [yl[0] - 1] * 2, [y0] * 2, color=C["surface"], alpha=0.4, lw=0, zorder=0)
    ax.axhline(y0, color=C["aux"], lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.plot(*poly_c.T, color=C["main"], lw=dgfig.LW["main"], zorder=3)


def arrow(ax, base_, vec, color, lw=1.4, ms=11):
    ax.annotate("", xy=base_ + vec, xytext=base_, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                                                  mutation_scale=ms, shrinkA=0, shrinkB=0), zorder=6)


def orient_marks(ax, where):
    for c, u0 in zip(curve, where):                                    # 진행 방향 표시
        i = int(u0 * (len(c) - 1))
        ax.annotate("", xy=c[i + 60], xytext=c[i - 60], arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=0.1,
                                                                        mutation_scale=11, shrinkA=0, shrinkB=0), zorder=4)


# (a) 전체
XA, YA = (-1.2, 1.2), (-1.42, 0.78)
base(axa, XA, YA)
orient_marks(axa, (0.5, 0.5, 0.5))
axa.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False, ec=C["aux"], lw=0.7, zorder=2))
for i in range(3):
    axa.plot(*P[i], "o", color=C["main"], ms=3.5, zorder=7)
axa.text(p[0] + 0.07, p[1] - 0.05, r"$P_0$", fontsize=12, ha="left", va="top")
axa.text(P[1][0] - 0.1, P[1][1] + 0.06, r"$P_1$", fontsize=12, ha="right", va="bottom")
axa.text(P[2][0] + 0.1, P[2][1] + 0.06, r"$P_2$", fontsize=12, ha="left", va="bottom")
axa.text(0.0, 0.1, r"$\Omega$", fontsize=13, ha="center", va="center")
axa.text(1.15, y0 - 0.07, r"$y < y_0$", fontsize=11, ha="right", va="top", color=C["aux"])
axa.text(-1.18, 0.76, "(a)", fontsize=11, ha="left", va="top")
dgfig.schematic_axes(axa, xlim=XA, ylim=YA)

# (b) p 근처
XB, YB = (ZOOM[0], ZOOM[1]), (ZOOM[2], ZOOM[3])
base(axb, XB, YB)
orient_marks(axb, (0.1, 0.5, 0.9))
L0, L1 = 0.055, 0.06
arrow(axb, p, L0 * tp, C["tangent"], ms=9)
arrow(axb, p, L0 * (-tm), C["aux"], ms=9)
axb.text(*(p + 0.6 * L0 * tp + np.array([-0.01, 0.0])), r"$\mathbf{t}^+$", fontsize=12, color=C["tangent"], ha="right", va="center")
axb.text(*(p + 0.6 * L0 * (-tm) + np.array([0.02, 0.0])), r"$-\mathbf{t}^-$", fontsize=12, color=C["aux"], ha="left", va="center")
for rr, a_, lab, lr, la in ((0.022, alpha, r"$\alpha$", 0.034, 0.3), (0.075, beta, r"$\beta$", 0.09, 2.2)):
    arc = np.linspace(0, a_, 60)
    axb.plot(p[0] + rr * np.cos(arc), p[1] + rr * np.sin(arc), color=C["accent"], lw=1.1, zorder=5)
    axb.text(p[0] + lr * math.cos(la), p[1] + lr * math.sin(la), lab, fontsize=11, color=C["accent"], ha="center", va="center")
axb.plot(*np.stack([x1 + L1 * nu, x1 + lam_end * nu]).T, color=C["normal"], lw=1.0, ls=(0, (2.5, 2)), zorder=5)
arrow(axb, x1, L1 * T1, C["tangent"], ms=9)
arrow(axb, x1, L1 * nu, C["normal"], ms=9)
axb.plot(*x1, "o", color=C["main"], ms=3.5, zorder=7)
axb.plot(*p, "o", color=C["main"], ms=3.5, zorder=7)
axb.text(x1[0] - 0.012, x1[1] + 0.004, r"$\gamma(t_1)$", fontsize=12, ha="right", va="bottom")
axb.text(*(x1 + L1 * T1 + np.array([-0.008, 0.004])), r"$\mathbf{t}_1$", fontsize=12, color=C["tangent"], ha="right", va="bottom")
axb.text(*(x1 + L1 * nu + np.array([0.0, -0.012])), r"$J\mathbf{t}_1$", fontsize=12, color=C["normal"], ha="center", va="top")
axb.text(p[0] + 0.008, p[1] - 0.012, r"$p$", fontsize=12, ha="left", va="top")
axb.text(0.0, -0.75, r"$\Omega$", fontsize=13, ha="center", va="center")
axb.text(ZOOM[1] - 0.01, y0 - 0.02, r"$y < y_0$", fontsize=11, ha="right", va="top", color=C["aux"])
axb.text(ZOOM[0] + 0.005, ZOOM[3] - 0.005, "(b)", fontsize=11, ha="left", va="top")
dgfig.schematic_axes(axb, xlim=XB, ylim=YB)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)
dgfig.save(fig, __file__)

print(f"alpha = {math.degrees(alpha):.2f} deg, beta = {math.degrees(beta):.2f} deg, eps0 = {math.degrees(eps[0]):.2f} deg, "
      f"delta = {math.degrees(delta):.2f} deg, area = {area:.4f}, x1 = {x1}, |x1 - p| = {np.linalg.norm(x1 - p):.4f}")
