"""그림 12.3.3: S² 위의 범프 함수 ψ(x) = k(2x³), k(s) = 1 − h(1 + s) (12.3절, 예 12.3.9).

곡면의 색은 ψ의 값이다(연회색 0 → 파랑 1). ψ는 북극 쪽 닫힌 극관 K = {x³ ≥ 1/2}(주황 원 위쪽)에서 1이고,
아래 반구 {x³ ≤ 0}(적도 = 회색 원 아래)에서 0이다. 받침은 닫힌 위 반구 {x³ ≥ 0}이다.
시점 elev = 18°, azim = −60°. 가려진 원 부분은 점선이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-3-3-bump-sphere.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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


def k(s):
    return 1 - h(1 + np.asarray(s, float))


def psi(x3):
    return k(2 * np.asarray(x3, float))


# 자기검사: k = 0 (s ≤ 0), k = 1 (s ≥ 1), 0 ≤ k ≤ 1; ψ = 1 on x³ ≥ 1/2, ψ = 0 on x³ ≤ 0, ψ > 0 on x³ > 0.05
s = np.linspace(-2, 3, 5001)
assert np.all(k(s[s <= 0]) == 0) and np.all(k(s[s >= 1]) == 1) and np.all((k(s) >= 0) & (k(s) <= 1))
z = np.linspace(-1, 1, 4001)
assert np.all(psi(z[z >= 0.5]) == 1) and np.all(psi(z[z <= 0]) == 0) and np.all(psi(z[z > 0.05]) > 0)

cmap = LinearSegmentedColormap.from_list("bump", ["#EDEDED", C["tangent"]])
fig = plt.figure(figsize=(4.2, 4.0))
ax = dgfig.axes3d(fig, elev=18, azim=-60)
th, ph = np.meshgrid(np.linspace(0, np.pi, 91), np.linspace(0, 2 * np.pi, 181), indexing="ij")
X, Y, Z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
ax.plot_surface(X, Y, Z, facecolors=cmap(psi(Z)), rstride=1, cstride=1, linewidth=0, antialiased=True,
                shade=False, alpha=0.8, zorder=1, rasterized=True)
# 윤곽선: 시선에 수직인 대원
vv = dgfig.view_vector(ax)
e1 = np.cross(vv, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(vv, e1)
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(*(np.outer(np.cos(tt), e1) + np.outer(np.sin(tt), e2)).T, color=C["main"], lw=0.9, zorder=4)
t = np.linspace(0, 2 * np.pi, 400)
for zc, role, lw in ((0.5, "accent", 1.8), (0.0, "aux", 1.2)):
    rc = np.sqrt(1 - zc ** 2)
    pts = np.stack([rc * np.cos(t), rc * np.sin(t), zc + 0 * t], axis=1)
    vis = dgfig.visible_mask(ax, pts, pts)
    dgfig.curve3d(ax, pts, role=role, lw=lw, visible=vis, zorder=5)
dgfig.point3d(ax, (0, 0, 1), label=r"$N$", label_offset=(0.05, 0.05, 0.08), size=12)
dgfig.equal_aspect(ax, X, Y, Z, zoom=1.35)
dgfig.save(fig, __file__)
