"""그림 5.3.2: 구면 위 원들의 프레네 틀과 구면의 단위법벡터 (5.3절, 예 5.3.7).

구면 S²(r) (dgsym.EXAMPLES["sphere"]), r = 1, 바깥쪽 단위법벡터 N = x/r (§7).
점 p = x(θ0, φ0), θ0 = π/3, φ0 = -0.35.
(a) 위도원 θ = θ0: t = (-sin φ, cos φ, 0), n = -(cos φ, sin φ, 0) (z축을 향함), b = e3.  N은 n과 각 π/2 - θ0 = 30°.
(b) p를 지나는 경선을 포함하는 대원: t = ∂x/∂θ/|∂x/∂θ|, n = -N (구의 중심을 향함), b는 상수(대원의 평면의 법벡터).
프레네 틀은 dgsym.frenet_frame으로 계산하고 실제 길이의 0.5배, N은 주황 점선 화살표로 0.5배로 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-3-2-sphere-frenet.py``
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
(r,) = ex["params"]
X1 = ex["expr"].subs(r, 1)
Xf = sp.lambdify((th, ph), list(X1), "numpy")
TH0, PH0 = np.pi / 3, -0.35
SC = 0.5

lat = X1.subs(th, sp.Float(TH0))                    # φ의 곡선
mer = X1.subs(ph, sp.Float(PH0))                    # θ의 곡선 (대원의 일부)
Tl, Nl, Bl = (sp.lambdify(ph, list(v), "numpy") for v in dgsym.frenet_frame(lat, ph))
Tm, Nm, Bm = (sp.lambdify(th, list(v), "numpy") for v in dgsym.frenet_frame(mer, th, (sp.sin(th),)))
p = np.array(Xf(TH0, PH0), float)
Nsph = p / 1.0
fl = [np.array(f(PH0), float) for f in (Tl, Nl, Bl)]
fm = [np.array(f(TH0), float) for f in (Tm, Nm, Bm)]

# 자기검사 --------------------------------------------------------------------
assert np.allclose(fl[2], [0, 0, 1])                                   # 위도원 b = e3
assert np.allclose(fl[1], [-np.cos(PH0), -np.sin(PH0), 0])              # n은 z축 쪽
assert abs(np.degrees(np.arccos(fl[1] @ (-Nsph))) - 30) < 1e-9          # n과 -N 사이 30°
assert np.allclose(Nsph, -np.sin(TH0) * fl[1] + np.cos(TH0) * fl[2])   # N = -sinθ0 n + cosθ0 b
assert np.allclose(fm[1], -Nsph)                                        # 대원: n = -N
assert np.allclose(fm[2], [np.sin(PH0), -np.cos(PH0), 0]) or np.allclose(fm[2], [-np.sin(PH0), np.cos(PH0), 0])
for fr in (fl, fm):
    M = np.stack(fr, axis=1)
    assert np.allclose(M.T @ M, np.eye(3)) and abs(np.linalg.det(M) - 1) < 1e-12


def base(ax):
    V = dgfig.view_vector(ax)
    uu, vv = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
    S = np.stack(Xf(uu, vv), axis=-1)
    dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.12, grid=False, zorder=1, shade=False)
    w1 = np.cross(V, [0, 0, 1.0]); w1 /= np.linalg.norm(w1)
    w2 = np.cross(V, w1)
    ang = np.linspace(0, 2 * np.pi, 400)
    rim = np.cos(ang)[:, None] * w1 + np.sin(ang)[:, None] * w2
    ax.plot(*rim.T, color=C["aux"], lw=0.6, zorder=1.5)
    return V, ang


def frame(ax, fr, labels):
    for v, role, lab, off in zip(fr, ("tangent", "normal", "third"), labels,
                                 ((0.05, 0.0, 0.03), (0.0, 0.0, -0.1), (0.04, 0.0, 0.05))):
        dgfig.arrow3d(ax, p, v, role=role, scale=SC, zorder=11, head=10)
        ax.text(*(p + SC * v + np.array(off)), lab, color=C[role], fontsize=12, zorder=12)
    dgfig.arrow3d(ax, p, Nsph, role="accent", scale=SC, zorder=10, head=10, ls=(0, (3, 2)))
    ax.text(*(p + SC * Nsph + np.array([0.04, 0.0, 0.06])), r"$\mathbf{N}$", color=C["accent"], fontsize=12, zorder=12)
    dgfig.point3d(ax, p, size=16, zorder=12)


fig = plt.figure(figsize=(6.4, 3.6))
ax1 = dgfig.axes3d(fig, pos=121, elev=20, azim=28)
V, ang = base(ax1)
LAT = np.stack(Xf(TH0 + 0 * ang, ang), axis=-1)
dgfig.curve3d(ax1, LAT, visible=LAT @ V > 0, zorder=5)
cz = np.array([0, 0, np.cos(TH0)])
ax1.plot(*np.stack([p, cz]).T, color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=4)
dgfig.point3d(ax1, cz, size=10, zorder=6)
frame(ax1, fl, (r"$\mathbf{t}$", r"$\mathbf{n}$", r"$\mathbf{b}$"))
ax1.text2D(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=11, va="top")
dgfig.equal_aspect(ax1, np.array([[-1, -1, -1], [1, 1, 1.1]]), zoom=1.3)

ax2 = dgfig.axes3d(fig, pos=122, elev=48, azim=-55)
V, ang = base(ax2)
e_rho = np.array([np.cos(PH0), np.sin(PH0), 0.0])
GC = np.cos(ang)[:, None] * np.array([0, 0, 1.0]) + np.sin(ang)[:, None] * e_rho
dgfig.curve3d(ax2, GC, role="main", visible=GC @ V > 0, zorder=5)
ax2.plot(*np.stack([p, np.zeros(3)]).T, color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=4)
dgfig.point3d(ax2, np.zeros(3), size=10, zorder=6)
frame(ax2, fm, (r"$\mathbf{t}$", r"$\mathbf{n}$", r"$\mathbf{b}$"))
ax2.text2D(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=11, va="top")
dgfig.equal_aspect(ax2, np.array([[-1, -1, -1], [1, 1, 1.1]]), zoom=1.3)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0)
dgfig.save(fig, __file__)
