"""그림 5.6.2: 접선회전정리의 증명에 쓰는 할선 사상 (5.6절, 정리 5.6.7).

(a) 삼각형 T = {(s1, s2) : 0 ≤ s1 ≤ s2 ≤ L}과 꼭짓점 A = (0, 0), B = (0, L), C = (L, L). (개념도)
    들어 올린 각 θ의 값 θ(A) = 0, θ(B) = π, θ(C) = 2π를 적는다(값은 (b)의 곡선에서 수치로 확인).
(b) 콩 모양 곡선 γ(t) = ρ(t)(cos t, sin t), ρ = 1 + cos(2t)/4 + 3 sin(3t)/25 (단순 닫힌 곡선, 극좌표 ρ > 0).
    매개변수는 가장 낮은 점 p = γ(0)에서 시작하도록 옮겼다. p에서의 접선(회색 점선)은 곡선 전체의 아래에 있고,
    단위접벡터 t(0) = e1 (파랑). 할선 방향 ψ(s1, s2) = (γ(s2) - γ(s1))/|γ(s2) - γ(s1)| (주황)의 한 예를 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-6-2-secant-map.py``
"""

import dgfig

dgfig.setup()

import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

C = dgfig.COLORS
t = sp.symbols("t", real=True)
rho = 1 + sp.cos(2 * t) / 4 + sp.Rational(3, 25) * sp.sin(3 * t)
bean = rho * sp.Matrix([sp.cos(t), sp.sin(t)])
gf = sp.lambdify(t, list(bean), "numpy")
df = sp.lambdify(t, list(sp.diff(bean, t)), "numpy")
yd = sp.lambdify(t, sp.diff(bean[1], t), "numpy")
ydd = sp.lambdify(t, sp.diff(bean[1], t, 2), "numpy")
tt = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
t0 = tt[np.argmin(np.array(gf(tt))[1])]
for _ in range(30):
    t0 -= yd(t0) / ydd(t0)
G = lambda s: np.array(gf(np.asarray(s, float) + t0), float)
DG = lambda s: np.array(df(np.asarray(s, float) + t0), float)
L = 2 * np.pi


def ang(u, v):
    det = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = np.sum(u * v, axis=-1)
    return 2 * np.arctan(det / (1 + dot))


def psi(s1, s2):
    if abs(s2 - s1) < 1e-12:
        d = DG(s1)
    elif abs(s1) < 1e-12 and abs(s2 - L) < 1e-12:
        d = -DG(0.0)
    else:
        d = G(s2) - G(s1)
    return d / np.linalg.norm(d)


def change(points):
    vals = np.array([psi(*q) for q in points])
    st = ang(vals[:-1], vals[1:])
    assert np.max(np.abs(st)) < 0.5
    return float(np.sum(st))


# 자기검사: 극좌표 곡선은 ρ > 0이면 단순, 가장 낮은 점에서 t = e1, 변 AB·BC에서 π, 대각선에서 2π
assert np.min(np.array(sp.lambdify(t, rho, "numpy")(tt))) > 0
v0 = DG(0.0) / np.linalg.norm(DG(0.0))
assert abs(v0[1]) < 1e-12 and v0[0] > 0
n = 3000
dAB = change([(0.0, x) for x in np.linspace(0, L, n)])
dBC = change([(x, L) for x in np.linspace(0, L, n)])
dAC = change([(x, x) for x in np.linspace(0, L, n)])
assert abs(dAB - math.pi) < 1e-6 and abs(dBC - math.pi) < 1e-6 and abs(dAC - 2 * math.pi) < 1e-6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
# (a) 삼각형
A, B, Cc = np.array([0, 0.0]), np.array([0, L]), np.array([L, L])
ax1.fill(*np.stack([A, B, Cc]).T, color=C["region"], alpha=0.25, lw=0)
for p, q, col in ((A, B, C["main"]), (B, Cc, C["main"]), (A, Cc, C["tangent"])):
    ax1.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, mutation_scale=11,
                                                    shrinkA=0, shrinkB=0))
ax1.text(*(A + [-0.25, -0.25]), r"$A$", fontsize=12, ha="right", va="top")
ax1.text(*(B + [-0.25, 0.1]), r"$B$", fontsize=12, ha="right", va="bottom")
ax1.text(*(Cc + [0.2, 0.1]), r"$C$", fontsize=12, ha="left", va="bottom")
ax1.text(*(A + [0.3, -0.35]), r"$\theta=0$", fontsize=11, ha="left", va="top")
ax1.text(*(B + [0.25, 0.25]), r"$\theta=\pi$", fontsize=11, ha="left", va="bottom")
ax1.text(*(Cc + [-0.1, -0.5]), r"$\theta=2\pi$", fontsize=11, ha="right", va="top")
ax1.text(L * 0.62, L * 0.44, r"$\mathbf{t}(s)$", fontsize=12, color=C["tangent"], ha="left", va="top")
q = np.array([1.3, 4.3])
ax1.plot(*q, "o", color=C["accent"], ms=4, zorder=5)
ax1.text(*(q + [0.2, 0.0]), r"$(s_1, s_2)$", fontsize=11, color=C["accent"], va="center")
ax1.set_xlabel(r"$s_1$", fontsize=12)
ax1.set_ylabel(r"$s_2$", fontsize=12, rotation=0, labelpad=10)
ax1.set_aspect("equal")
ax1.set_xlim(-1.2, L + 1.3)
ax1.set_ylim(-1.2, L + 1.0)
for sp_ in ("top", "right"):
    ax1.spines[sp_].set_visible(False)
ax1.set_xticks([0, L], [r"$0$", r"$L$"])
ax1.set_yticks([0, L], [r"$0$", r"$L$"])
ax1.text(-0.2, 1.02, "(a)", transform=ax1.transAxes, fontsize=12, va="bottom", ha="left")

# (b) 곡선
ss = np.linspace(0, L, 800)
P = G(ss)
ax2.plot(*P, color=C["main"], lw=dgfig.LW["main"], zorder=3)
p0 = G(0.0)
ax2.axhline(p0[1], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=1)
ax2.annotate("", xy=p0 + 0.45 * v0, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.5,
                                                             mutation_scale=11, shrinkA=0, shrinkB=0), zorder=6)
ax2.plot(*p0, "o", color=C["main"], ms=4, zorder=7)
ax2.text(*(p0 + [0.0, -0.08]), r"$p=\gamma(0)$", fontsize=11, ha="center", va="top")
ax2.text(*(p0 + 0.45 * v0 + [0.05, 0.1]), r"$\mathbf{t}(0)$", fontsize=11, color=C["tangent"], va="bottom")
s1, s2 = 1.3, 4.3
P1, P2 = G(s1), G(s2)
ax2.annotate("", xy=P2, xytext=P1, arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.5, mutation_scale=11,
                                                  shrinkA=2, shrinkB=2), zorder=6)
for Pk, lab, off, ha in ((P1, r"$\gamma(s_1)$", (0.06, 0.0), "left"), (P2, r"$\gamma(s_2)$", (-0.06, 0.02), "right")):
    ax2.plot(*Pk, "o", color=C["main"], ms=3.5, zorder=7)
    ax2.text(*(Pk + np.array(off)), lab, fontsize=11, ha=ha, va="center")
mid = (P1 + P2) / 2
ax2.text(*(mid + [0.05, 0.12]), r"$\psi(s_1,s_2)$", fontsize=11, color=C["accent"], ha="left")
dgfig.schematic_axes(ax2)
ax2.set_xlim(P[0].min() - 0.45, P[0].max() + 0.35)
ax2.set_ylim(p0[1] - 0.35, P[1].max() + 0.2)
ax2.text(0.0, 1.0, "(b)", transform=ax2.transAxes, fontsize=12, va="top", ha="left")
fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.12, wspace=0.12)
dgfig.save(fig, __file__)
