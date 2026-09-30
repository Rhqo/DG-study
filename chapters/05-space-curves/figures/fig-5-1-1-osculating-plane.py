"""그림 5.1.1: 나선의 한 점에서의 단위접벡터, 주법선벡터, 접촉평면, 접촉원 (5.1절).

기준 예제의 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]), a = 1, b = 0.3.
점 γ(t0), t0 = π/2 에서
- 단위접벡터 t (파랑), 주법선벡터 n (주홍): 매개화에서 계산한 실제 벡터(실제 길이).
- 접촉평면 (하늘색): γ(t0)를 지나고 t, n이 생성하는 평면의 일부.
- 접촉원 (주황): 중심 c = γ + n/κ, 반지름 1/κ = (a² + b²)/a = 1.09.
원기둥 x² + y² = a²는 옅은 회색으로 그렸다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-1-1-osculating-plane.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS
ex = dgsym.EXAMPLES["helix"]
(t,) = ex["coords"]
a, b = ex["params"]
A, B = 1.0, 0.3
T0 = np.pi / 2

gam = sp.lambdify(t, list(ex["expr"].subs({a: A, b: B})), "numpy")
kap_sym, tau_sym = dgsym.curvature_torsion(ex["expr"], t)
Tv, Nv, Bv = dgsym.frenet_frame(ex["expr"], t)
kap = float(kap_sym.subs({a: A, b: B}))
Tf = sp.lambdify(t, list(Tv.subs({a: A, b: B})), "numpy")
Nf = sp.lambdify(t, list(Nv.subs({a: A, b: B})), "numpy")


def G(tt):
    tt = np.asarray(tt, float)
    return np.stack([np.broadcast_to(c, tt.shape) for c in gam(tt)], axis=-1)


p = G(T0)
tv = np.array(Tf(T0), float)
nv = np.array(Nf(T0), float)
rad = 1.0 / kap
c = p + rad * nv

# 자기검사 --------------------------------------------------------------------
assert abs(kap - A / (A ** 2 + B ** 2)) < 1e-12                      # κ = a/(a²+b²) (§7)
assert abs(np.linalg.norm(tv) - 1) < 1e-12 and abs(np.linalg.norm(nv) - 1) < 1e-12
assert abs(tv @ nv) < 1e-12                                           # <t, n> = 0
h = 1e-4                                                              # γ'' ≈ κ n (호의 길이로 미분)
spd = np.hypot(A, B)
g2 = (G(T0 + h) - 2 * p + G(T0 - h)) / h ** 2 / spd ** 2
assert np.allclose(g2, kap * nv, atol=1e-6)
assert np.allclose(c, [-(B ** 2 / A) * np.cos(T0), -(B ** 2 / A) * np.sin(T0), B * T0])   # 중심은 축 반대편, 반지름 b²/a

fig = plt.figure(figsize=(6, 4.8))
ax = dgfig.axes3d(fig, elev=24, azim=-32)
V = dgfig.view_vector(ax)

# 원기둥 (옅게)
uu, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(-0.45, 2.0, 15))
dgfig.surface(ax, A * np.cos(uu), A * np.sin(uu), zz, alpha=0.10, grid=False, zorder=1, shade=False)

# 나선: 원기둥의 앞쪽(보이는 쪽)은 실선, 뒤쪽은 옅은 점선
ts = np.linspace(-1.5, 6.6, 700)
H = G(ts)
nrm = np.stack([np.cos(ts), np.sin(ts), 0 * ts], axis=1)
dgfig.curve3d(ax, H, visible=nrm @ V > 0, zorder=4)

# 접촉평면: γ(t0) + x t + y n, -1.35 ≤ x ≤ 1.35, -0.35 ≤ y ≤ 2.45
xs, ys = (-1.2, 1.2), (-0.3, 2 * rad + 0.2)
corners = [p + xs[0] * tv + ys[0] * nv, p + xs[1] * tv + ys[0] * nv,
           p + xs[1] * tv + ys[1] * nv, p + xs[0] * tv + ys[1] * nv]
ax.add_collection3d(Poly3DCollection([corners], facecolor=C["region"], alpha=0.22,
                                     edgecolor=C["tangent"], linewidth=0.8, zorder=2))

# 접촉원
ang = np.linspace(0, 2 * np.pi, 400)
circ = c + rad * (np.cos(ang)[:, None] * (-nv) + np.sin(ang)[:, None] * tv)
assert np.allclose(np.linalg.norm(circ - c, axis=1), rad)
ax.plot(*circ.T, color=C["accent"], lw=1.6, zorder=6)

# 반지름 선분과 중심
ax.plot(*np.stack([p, c]).T, color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=5)
dgfig.point3d(ax, c, role="main", size=14, zorder=9)
ax.text(*(c + np.array([0.0, 0.0, -0.18])), r"$c$", fontsize=12, zorder=12, ha="center", va="top")

# 벡터와 점
dgfig.point3d(ax, p, size=18, zorder=10)
dgfig.arrow3d(ax, p, tv, role="tangent", zorder=11)
ax.text(*(p + 0.75 * tv + np.array([0.12, 0.0, 0.05])), r"$\mathbf{t}$", color=C["tangent"], fontsize=12, zorder=12, ha="left", va="bottom")
dgfig.arrow3d(ax, p, nv, role="normal", zorder=11)
ax.text(*(p + 0.5 * nv + np.array([0.0, 0.0, 0.09])), r"$\mathbf{n}$", color=C["normal"], fontsize=12, zorder=12, ha="center", va="bottom")
ax.text(*(p + np.array([0.0, 0.25, -0.28])), r"$\gamma(s_0)$", fontsize=12, zorder=12, ha="left")
ax.text(*(p + 0.8 * rad * nv + np.array([0.0, 0.0, -0.1])), r"$1/\kappa$", fontsize=11, color=C["aux"],
        zorder=12, ha="center", va="top")

allpts = np.concatenate([H, circ, np.array(corners)])
dgfig.equal_aspect(ax, allpts, zoom=1.22)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)
