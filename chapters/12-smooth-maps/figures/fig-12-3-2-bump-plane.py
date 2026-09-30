"""그림 12.3.2: ℝ²의 범프 함수 H(x) = h(1 + (|x − c| − r₁)/(r₂ − r₁)) (12.3절, 보조정리 12.3.3).

c = 0, r₁ = 1, r₂ = 2. 곡면 z = H(x)를 −2.6 ≤ x¹, x² ≤ 2.6에서 그린다.
H는 닫힌 원판 |x| ≤ 1(주황 원)에서 1이고, |x| ≥ 2(회색 원)에서 0이며, 받침은 닫힌 원판 |x| ≤ 2다.
평면 z = 0 위에 두 원을 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-3-2-bump-plane.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def f(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.exp(-1.0 / t[pos])
    return out


def h(t):
    a, b = f(2 - np.asarray(t, float)), f(np.asarray(t, float) - 1)
    return a / (a + b)


r1, r2 = 1.0, 2.0


def H(x1, x2):
    rho = np.hypot(x1, x2)
    return h(1 + (rho - r1) / (r2 - r1))


# 자기검사: H = 1 (|x| ≤ 1), H = 0 (|x| ≥ 2), 0 < H < 1 사이, 반지름에만 의존
rng = np.random.default_rng(3)
P = rng.uniform(-3, 3, (4000, 2))
rho = np.hypot(P[:, 0], P[:, 1])
V = H(P[:, 0], P[:, 1])
assert np.all(V[rho <= r1] == 1) and np.all(V[rho >= r2] == 0)
m = (rho > r1 + 0.05) & (rho < r2 - 0.05)   # 경계 근처는 부동소수 언더플로
assert np.all((V[m] > 0) & (V[m] < 1))
ang = rng.uniform(0, 2 * np.pi, 4000)
assert np.allclose(H(rho * np.cos(ang), rho * np.sin(ang)), V)

fig = plt.figure(figsize=(5.0, 3.3))
ax = dgfig.axes3d(fig, elev=24, azim=-58)
ax.set_position([-0.06, 0.0, 1.12, 1.08])
g = np.linspace(-2.6, 2.6, 131)
X1, X2 = np.meshgrid(g, g)
Z = H(X1, X2)
dgfig.surface(ax, X1, X2, Z, alpha=0.5, grid=True, grid_every=10)
t = np.linspace(0, 2 * np.pi, 300)
ax.plot(r1 * np.cos(t), r1 * np.sin(t), 0 * t, color=C["accent"], lw=1.8, zorder=6)
ax.plot(r2 * np.cos(t), r2 * np.sin(t), 0 * t, color=C["aux"], lw=1.2, zorder=6)
ax.plot(r1 * np.cos(t), r1 * np.sin(t), 1 + 0 * t, color=C["accent"], lw=1.0, ls=(0, (3, 2)), zorder=6)
ax.text(-0.3, 0.2, 1.12, r"$H=1$", fontsize=11, zorder=7)
ax.set_box_aspect((5.2, 5.2, 2.2), zoom=1.2)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.set_zlim(0, 1.1)
plt.rcParams["savefig.bbox"] = "standard"   # 3D 축의 tight bbox는 곡면을 잘라 낸다
dgfig.save(fig, __file__)
