"""Figure 0.3.3: first fundamental form이 UV 평면에 그리는 ellipse (0.3절, Example 0.3.6, Exercise 0.3.4).

각 격자점 q에서 {w ∈ R² : I_q(w) = ε²} = {(a, b) : E a² + 2F ab + G b² = ε²}를 그린다.
surface 위에서 길이가 ε인 tangent vector들이 UV 평면에서 차지하는 모양이다.
왼쪽: sphere S²(1)의 standard parametrization (E, F, G) = (1, 0, sin²θ). 가로 φ, 세로 θ (위쪽이 북극 θ = 0),
      equirectangular 지도와 같은 배치. ε = 0.13. ellipse의 반축은 θ 방향 ε, φ 방향 ε/sin θ.
오른쪽: torus of revolution (R, r) = (2, 0.8)의 (E, F, G) = (r², 0, (R + r cos u)²). 가로 v, 세로 u. ε = 0.13.
       반축은 u 방향 ε/r, v 방향 ε/(R + r cos u).
자기검사: dgsym.first_ff의 E, F, G와 §7의 값이 같고, 그린 ellipse 위의 점이 I_q(w) = ε²를 만족한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-3-3-metric-ellipses.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Ellipse

import dgsym

C = dgfig.COLORS
EPS = 0.13

sph = dgsym.EXAMPLES["sphere"]
th, ph = sph["coords"]
(rs,) = sph["params"]
Es = dgsym.first_ff(sph["expr"], th, ph, sph["positive"])
assert all(sp.simplify(a - sph["expected"][k]) == 0 for a, k in zip(Es, "EFG"))
fS = sp.lambdify(th, [e.subs(rs, 1) for e in Es], "numpy")

tor = dgsym.EXAMPLES["torus"]
u, v = tor["coords"]
Rs, rr = tor["params"]
Et = dgsym.first_ff(tor["expr"], u, v, tor["positive"])
assert all(sp.simplify(a - tor["expected"][k]) == 0 for a, k in zip(Et, "EFG"))
fT = sp.lambdify(u, [e.subs({Rs: 2, rr: sp.Rational(4, 5)}) for e in Et], "numpy")


def semi_axes(E, F, G):
    """I(w) = ε²인 ellipse: Gram matrix의 eigenvalue λ에 대해 반축 ε/√λ (eigenvector 방향)."""
    g = np.array([[E, F], [F, G]], float)
    lam, vec = np.linalg.eigh(g)
    return EPS / np.sqrt(lam), vec


def draw(ax, x0, y0, E, F, G, horiz_is_second):
    """UV 평면의 점 (x0, y0)에 ellipse. 첫 좌표가 세로축(θ 또는 u), 둘째 좌표가 가로축이다."""
    ax_len, vec = semi_axes(E, F, G)
    # 좌표 (a, b) = (첫째, 둘째) → 그림 (가로, 세로) = (b, a)
    d = vec[:, 0]
    ang = np.degrees(np.arctan2(d[0], d[1]))
    e = Ellipse((x0, y0), 2 * ax_len[0], 2 * ax_len[1], angle=ang, fc=C["region"], ec=C["tangent"],
                alpha=0.9, lw=0.8)
    ax.add_patch(e)
    # 자기검사: ellipse 위의 점 w가 I(w) = ε²
    tt = np.linspace(0, 2 * np.pi, 7)
    for t_ in tt:
        w = ax_len[0] * np.cos(t_) * vec[:, 0] + ax_len[1] * np.sin(t_) * vec[:, 1]
        assert np.isclose(w @ np.array([[E, F], [F, G]]) @ w, EPS ** 2)


fig = plt.figure(figsize=(6.6, 3.2))
axL = fig.add_axes([0.07, 0.15, 0.42, 0.75])
axR = fig.add_axes([0.57, 0.15, 0.42, 0.75])

for k in range(1, 8):
    t0 = k * np.pi / 8
    for j in range(8):
        p0 = (j + 0.5) * 2 * np.pi / 8
        E, F, G = fS(t0)
        draw(axL, p0, t0, E, F, G, True)
axL.set_xlim(0, 2 * np.pi)
axL.set_ylim(np.pi, 0)
axL.set_xticks([0, np.pi, 2 * np.pi])
axL.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axL.set_yticks([0, np.pi / 2, np.pi])
axL.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
axL.set_xlabel(r"$\varphi$", labelpad=1)
axL.set_ylabel(r"$\theta$", rotation=0, labelpad=8)
axL.set_aspect("equal")

for k in range(8):
    u0 = (k + 0.5) * 2 * np.pi / 8
    for j in range(8):
        v0 = (j + 0.5) * 2 * np.pi / 8
        E, F, G = fT(u0)
        draw(axR, v0, u0, E, F, G, True)
axR.set_xlim(0, 2 * np.pi)
axR.set_ylim(0, 2 * np.pi)
axR.set_xticks([0, np.pi, 2 * np.pi])
axR.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axR.set_yticks([0, np.pi, 2 * np.pi])
axR.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axR.set_xlabel(r"$v$", labelpad=1)
axR.set_ylabel(r"$u$", rotation=0, labelpad=8)
axR.set_aspect("equal")
for a in (axL, axR):
    for s_ in a.spines.values():
        s_.set_linewidth(0.6)

dgfig.save(fig, __file__)
