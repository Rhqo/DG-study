"""그림 5.2.1: 나선을 따라 도는 접촉평면과 종법선벡터 (5.2절, 예 5.2.6).

기준 예제의 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]).
(a) a = 1, b = 0.6 (오른손 나선, τ = b/(a²+b²) ≈ 0.441 > 0)
(b) a = 1, b = -0.6 (왼손 나선, (a)를 xy평면에 대해 반사한 것, τ ≈ -0.441 < 0)
보는 쪽의 세 점 t = t_i에서 단위접벡터 t (파랑), 종법선벡터 b (청록)를 dgsym.frenet_frame으로 계산해 실제 길이의 0.6배로 그리고,
접촉평면의 일부(하늘색 평행사변형, t와 n 방향으로 ±0.45)를 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-2-1-binormal.py``
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
kap_s, tau_s = dgsym.curvature_torsion(ex["expr"], t)
Tv, Nv, Bv = dgsym.frenet_frame(ex["expr"], t)
SC = 0.6
TS = (-2.25, -1.05, 0.15)


def panel(ax, B, label, tag):
    sub = {a: 1.0, b: B}
    gam = sp.lambdify(t, list(ex["expr"].subs(sub)), "numpy")
    Tf, Nf, Bf = (sp.lambdify(t, list(v.subs(sub)), "numpy") for v in (Tv, Nv, Bv))
    tau = float(tau_s.subs(sub))
    ts = np.linspace(-3.9, 2.4, 600)
    uu, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(-3.9 * abs(B) - 0.2, 2.4 * abs(B) + 0.2, 9))
    dgfig.surface(ax, np.cos(uu), np.sin(uu), zz, alpha=0.08, grid=False, zorder=1, shade=False)
    H = np.stack([np.broadcast_to(c, ts.shape) for c in gam(ts)], axis=1)
    V = dgfig.view_vector(ax)
    nrm = np.stack([np.cos(ts), np.sin(ts), 0 * ts], axis=1)
    dgfig.curve3d(ax, H, visible=nrm @ V > 0, zorder=4)
    pts = [H]
    for i, ti in enumerate(TS):
        p = np.array(gam(ti), float)
        tv, nv, bv = (np.array(f(ti), float) for f in (Tf, Nf, Bf))
        # 자기검사: 정규직교, 양의 방향, b' = -τ n (수치 미분)
        M = np.stack([tv, nv, bv], axis=1)
        assert np.allclose(M.T @ M, np.eye(3), atol=1e-12) and abs(np.linalg.det(M) - 1) < 1e-12
        hh = 1e-5
        spd = np.hypot(1.0, B)
        db = (np.array(Bf(ti + hh), float) - np.array(Bf(ti - hh), float)) / (2 * hh) / spd
        assert np.allclose(db, -tau * nv, atol=1e-7)
        w = 0.45
        corners = [p - w * tv - w * nv, p + w * tv - w * nv, p + w * tv + w * nv, p - w * tv + w * nv]
        ax.add_collection3d(Poly3DCollection([corners], facecolor=C["region"], alpha=0.28,
                                             edgecolor=C["tangent"], linewidth=0.6, zorder=2))
        dgfig.point3d(ax, p, size=12, zorder=10)
        dgfig.arrow3d(ax, p, tv, role="tangent", scale=SC, zorder=11, head=9)
        dgfig.arrow3d(ax, p, bv, role="third", scale=SC, zorder=11, head=9)
        if i == 1:
            ax.text(*(p + SC * tv + np.array([0.08, 0.0, 0.02])), r"$\mathbf{t}$", color=C["tangent"], fontsize=12, zorder=12)
            ax.text(*(p + SC * bv + np.array([0.06, 0.0, 0.06])), r"$\mathbf{b}$", color=C["third"], fontsize=12, zorder=12)
        pts += [np.array(corners), (p + SC * bv)[None], (p + SC * tv)[None]]
    ax.text2D(0.02, 0.98, tag, transform=ax.transAxes, fontsize=11, va="top")
    ax.text2D(0.5, 0.02, label, transform=ax.transAxes, fontsize=12, ha="center", va="bottom")
    dgfig.equal_aspect(ax, np.concatenate(pts), zoom=1.25)
    return tau


fig = plt.figure(figsize=(6.4, 3.9))
ax1 = dgfig.axes3d(fig, pos=121, elev=24, azim=-60)
tau1 = panel(ax1, 0.6, r"$b=0.6,\ \tau\approx 0.441$", "(a)")
ax2 = dgfig.axes3d(fig, pos=122, elev=24, azim=-60)
tau2 = panel(ax2, -0.6, r"$b=-0.6,\ \tau\approx -0.441$", "(b)")
assert abs(tau1 - 0.6 / 1.36) < 1e-12 and abs(tau2 + 0.6 / 1.36) < 1e-12
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0)
dgfig.save(fig, __file__)
