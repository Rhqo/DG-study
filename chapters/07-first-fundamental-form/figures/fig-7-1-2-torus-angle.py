"""그림 7.1.2: 매개변수 평면의 각과 곡면 위의 각은 다르다 (7.1절, 예 7.1.12).

x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u) (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8.
곡선 α(t) = x(t, t)(주황)는 U의 대각선 u = v의 상이다.
두 점 t₁ = π/4(바깥쪽), t₂ = 5π/6(안쪽)에서 α'(t) = x_u + x_v(파랑)와 u-좌표곡선의 속도 x_u(청록)를 그리고,
그 사이의 각 ϑ를 호로 표시한다. cos ϑ = √E / √(E + G) = r / √(r² + (R + r cos t)²).
왼쪽: U = (0, 2π)²와 대각선. U에서는 두 점 모두 각이 π/4이다.
두 벡터는 방향만 보이도록 길이 0.6으로 맞추어 그렸다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-1-2-torus-angle.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
Xe = ex["expr"].subs({Rs: RV, rs: rV})
X = sp.lambdify((u, v), list(Xe), "numpy")
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")
Es, Fs, Gs = dgsym.first_ff(ex["expr"], u, v, ex["positive"])
Ef, Gf = (sp.lambdify(u, c.subs({Rs: RV, rs: rV}), "numpy") for c in (Es, Gs))
assert Fs == 0


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


def angle_formula(t):
    return np.arccos(np.sqrt(Ef(t)) / np.sqrt(Ef(t) + Gf(t)))


T1, T2 = np.pi / 4, 5 * np.pi / 6
angles = {}
for t0 in (T1, T2):
    a = np.array(Xu(t0, t0), float)
    w = a + np.array(Xv(t0, t0), float)
    th = np.arccos(a @ w / np.linalg.norm(a) / np.linalg.norm(w))
    # 자기검사: 벡터로 잰 각 = (7.1.x)의 공식, 그리고 U에서의 각 π/4보다 크다
    assert np.isclose(th, angle_formula(t0))
    assert th > np.pi / 4
    angles[t0] = th
assert angles[T1] > angles[T2]          # 바깥쪽일수록 G가 커서 각이 커진다

fig = plt.figure(figsize=(6.8, 3.3))

# 왼쪽: U -------------------------------------------------------------------------------
axU = fig.add_axes([0.02, 0.12, 0.25, 0.76])
T = 2 * np.pi
axU.fill([0, T, T, 0], [0, 0, T, T], color=C["region"], alpha=0.18, lw=0)
for k in range(1, 12):
    axU.plot([k * T / 12] * 2, [0, T], color="#9A9A9A", lw=0.35)
    axU.plot([0, T], [k * T / 12] * 2, color="#9A9A9A", lw=0.35)
axU.plot([0, T, T, 0, 0], [0, 0, T, T, 0], color=C["main"], lw=0.8)
axU.plot([0, T], [0, T], color=C["accent"], lw=1.8)
for t0, lab in ((T1, r"$t_1$"), (T2, r"$t_2$")):
    axU.plot([t0], [t0], "o", color=C["main"], ms=3.5, zorder=5)
    axU.annotate("", xy=(t0 + 1.1, t0), xytext=(t0, t0),
                 arrowprops=dict(arrowstyle="-|>", color=C["third"], lw=1.3, mutation_scale=9))
    axU.annotate("", xy=(t0 + 0.78, t0 + 0.78), xytext=(t0, t0),
                 arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3, mutation_scale=9))
    s = np.linspace(0, np.pi / 4, 20)
    axU.plot(t0 + 0.55 * np.cos(s), t0 + 0.55 * np.sin(s), color=C["main"], lw=0.8)
    axU.text(t0 + 0.8, t0 + 0.33, r"$\frac{\pi}{4}$", fontsize=10, ha="left", va="center")
    axU.text(t0 - 0.2, t0 + 0.15, lab, fontsize=11, ha="right", va="bottom")
axU.text(T / 2, T + 0.3, r"$U$", fontsize=13, ha="center")
axU.text(T / 2, -0.35, r"$u$", fontsize=12, ha="center", va="top")
axU.text(-0.35, T / 2, r"$v$", fontsize=12, ha="right", va="center")
dgfig.schematic_axes(axU, (-0.9, T + 0.4), (-0.9, T + 0.8))

# 오른쪽: 원환면 ------------------------------------------------------------------------
ax = fig.add_axes([0.28, -0.02, 0.72, 1.04], projection="3d", computed_zorder=False)
ax.view_init(elev=48, azim=10)
ax.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(0, T, 49), np.linspace(0, T, 97))
Sg = xmap(Ug, Vg)
dgfig.surface(ax, Sg[..., 0], Sg[..., 1], Sg[..., 2], alpha=0.25, grid=False, zorder=1)


def normal_out(P):
    rr = np.hypot(P[:, 0], P[:, 1])
    c = np.stack([RV * P[:, 0] / rr, RV * P[:, 1] / rr, 0 * rr], axis=1)
    return P - c


def draw(curve, role="#8A8A8A", lw=0.4, zorder=3, hidden=None):
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=dgfig.visible_mask(ax, curve, normal_out(curve)),
                  hidden=hidden, zorder=zorder)


s = np.linspace(0, T, 300)
for k in range(12):
    draw(xmap(k * T / 12 * np.ones_like(s), s))
    draw(xmap(s, k * T / 12 * np.ones_like(s)))
draw(xmap(s, s), role="accent", lw=1.4, zorder=6, hidden="dashed")

SC = 0.6
for t0, lab in ((T1, r"$\vartheta_1$"), (T2, r"$\vartheta_2$")):
    p = np.array(X(t0, t0), float)
    a = np.array(Xu(t0, t0), float)
    w = a + np.array(Xv(t0, t0), float)
    dgfig.point3d(ax, p, size=12, zorder=12)
    dgfig.arrow3d(ax, p, SC * a / np.linalg.norm(a), role="third", zorder=14, head=9)
    dgfig.arrow3d(ax, p, SC * w / np.linalg.norm(w), role="tangent", zorder=14, head=9)
    e1 = a / np.linalg.norm(a)
    e2 = w - (w @ e1) * e1
    e2 /= np.linalg.norm(e2)
    ss = np.linspace(0, angles[t0], 30)
    arc = p + 0.3 * (np.outer(np.cos(ss), e1) + np.outer(np.sin(ss), e2))
    ax.plot(*arc.T, color=C["main"], lw=0.9, zorder=13)
    mid = p + 0.44 * (np.cos(angles[t0] / 2) * e1 + np.sin(angles[t0] / 2) * e2)
    ax.text(*mid, lab, fontsize=11, zorder=15, ha="center", va="center")
    ax.text(*(p + (SC + 0.12) * e1), r"$\mathbf{x}_u$", color=C["third"], fontsize=11, zorder=15,
            ha="center", va="center")
    ax.text(*(p + (SC + 0.12) * w / np.linalg.norm(w)), r"$\alpha'$", color=C["tangent"], fontsize=11,
            zorder=15, ha="center", va="center")

dgfig.equal_aspect(ax, np.array([[-2.8, -2.8, -0.8], [2.8, 2.8, 0.8]]), zoom=1.35)
dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.97, 0.8), xy_to=(0.16, 0.74), rad=-0.3)

dgfig.save(fig, __file__)
