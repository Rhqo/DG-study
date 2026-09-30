"""그림 7.4.2: 메르카토르 사상과 항정선 (7.4절, 예 7.4.10, 연습 7.4.5).

구면 x(θ, φ) (dgsym.EXAMPLES["sphere"]), r = 1. 메르카토르 사상 ψ(x(θ, φ)) = (φ, log cot(θ/2), 0).
왼쪽: 구면과 위도원 θ = kπ/12, 경선 φ = kπ/6, 그리고 경선과 늘 각 β = 5π/12를 이루는 곡선(항정선, 주황)을
      적도 (θ, φ) = (π/2, 0.3)에서 북쪽으로 t ≤ 3까지(θ ≥ 0.099).
오른쪽: 메르카토르 평면 (s, t) = (φ, log cot(θ/2))의 띠 0 < s < 2π, |t| ≤ 3. 경선은 같은 간격의 세로선,
       위도원은 가로선(극으로 갈수록 간격이 넓어진다), 항정선은 기울기가 cot β인 선분들(s = 2π에서 s = 0으로 이어진다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-4-2-mercator.py``
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
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Es, Fs, Gs = dgsym.first_ff(Xe, th, ph, ex["positive"])
Ym = sp.Matrix([ph, sp.log(sp.cot(th / 2)), 0])
Em, Fm, Gm = dgsym.first_ff(Ym, th, ph, ex["positive"])
# 자기검사: 메르카토르 매개화는 구면의 계수에 1/sin²θ를 곱한 것 (등각)
assert all(sp.simplify(a - b / sp.sin(th) ** 2) == 0 for a, b in zip((Em, Fm, Gm), (Es, Fs, Gs)))


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


BETA = 5 * np.pi / 12
T0, S0, TMAX = 0.0, 0.3, 3.0
tt = np.linspace(T0, TMAX, 3000)
ss = S0 + np.tan(BETA) * (tt - T0)            # 직선: ds/dt = tan β (세로선과 각 β)
theta = 2 * np.arctan(np.exp(-tt))           # t = log cot(θ/2)
lox = xmap(theta, ss)
# 자기검사: 경선 방향 x_θ와 항정선의 각은 β
dth = np.gradient(theta, tt)
dph = np.gradient(ss, tt)
v = np.stack([np.cos(theta) * np.cos(ss) * dth - np.sin(theta) * np.sin(ss) * dph,
              np.cos(theta) * np.sin(ss) * dth + np.sin(theta) * np.cos(ss) * dph, -np.sin(theta) * dth], axis=1)
xt = np.stack([np.cos(theta) * np.cos(ss), np.cos(theta) * np.sin(ss), -np.sin(theta)], axis=1)
cosang = np.abs(np.sum(v * xt, 1)) / np.linalg.norm(v, axis=1)
assert np.allclose(cosang[5:-5], np.cos(BETA), atol=1e-3)

fig = plt.figure(figsize=(6.6, 3.4))

# 왼쪽: 구면 ------------------------------------------------------------------------------
ax = fig.add_axes([0.0, -0.02, 0.46, 1.04], projection="3d", computed_zorder=False)
ax.view_init(elev=28, azim=40)
ax.set_axis_off()
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = xmap(Tg, Pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.2, grid=False, zorder=1)
view = dgfig.view_vector(ax)
s = np.linspace(0, 2 * np.pi, 300)
for k in range(1, 12):
    c = xmap(k * np.pi / 12 * np.ones_like(s), s)
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.45 if k != 6 else 0.9, visible=c @ view > 0, hidden=None, zorder=3)
s2 = np.linspace(0, np.pi, 200)
for k in range(12):
    c = xmap(s2, k * np.pi / 6 * np.ones_like(s2))
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.45, visible=c @ view > 0, hidden=None, zorder=3)
dgfig.curve3d(ax, lox, role="accent", lw=1.6, visible=lox @ view > 0, hidden="dashed", zorder=6)
dgfig.point3d(ax, lox[0], size=12, zorder=12)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.3)

# 오른쪽: 메르카토르 평면 --------------------------------------------------------------------
axM = fig.add_axes([0.52, 0.08, 0.46, 0.84])
T = 2 * np.pi
axM.fill([0, T, T, 0], [-TMAX, -TMAX, TMAX, TMAX], color=C["region"], alpha=0.15, lw=0)
for k in range(1, 12):
    t_k = np.log(1 / np.tan(k * np.pi / 24))
    if abs(t_k) <= TMAX:
        axM.plot([0, T], [t_k, t_k], color="#9A9A9A", lw=0.45 if k != 6 else 0.9)
for k in range(13):
    axM.plot([k * np.pi / 6] * 2, [-TMAX, TMAX], color="#9A9A9A", lw=0.45)
pieces = np.floor(ss / T)
for m in np.unique(pieces):
    msk = pieces == m
    axM.plot(ss[msk] - m * T, tt[msk], color=C["accent"], lw=1.6)
axM.plot([S0], [T0], "o", color=C["main"], ms=3.5)
axM.text(T + 0.15, 0, r"$t = 0$", fontsize=10, va="center")
axM.text(T / 2, -TMAX - 0.35, r"$s = \varphi$", fontsize=11, ha="center", va="top")
axM.text(-0.2, TMAX * 0.7, r"$t$", fontsize=11, ha="right")
dgfig.schematic_axes(axM, (-0.4, T + 1.0), (-TMAX - 0.9, TMAX + 0.3))
axM.set_aspect("equal")
dgfig.map_arrow(fig, ax, axM, r"$\psi$", xy_from=(0.83, 0.8), xy_to=(0.02, 0.9), rad=-0.3)

dgfig.save(fig, __file__)
