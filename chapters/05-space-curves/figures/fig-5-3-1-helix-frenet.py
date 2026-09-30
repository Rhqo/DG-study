"""그림 5.3.1: 나선을 따라 움직이는 프레네 틀 (5.3절, 예 5.3.3).

기준 예제의 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]), a = 1, b = 0.3, -2.6 ≤ t ≤ 7.4.
보는 쪽의 t = -1.7, -0.2, 4.6, 6.1에서 프레네 틀 (t 파랑, n 주홍, b 청록)을 dgsym.frenet_frame으로 계산해 실제 길이의 0.55배로 그린다.
틀은 z축 둘레를 일정한 각속도 1/c (c = √(a²+b²))로 돌며 올라간다 (다르부 벡터 ω = (0, 0, 1/c)).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-3-1-helix-frenet.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
ex = dgsym.EXAMPLES["helix"]
(t,) = ex["coords"]
a, b = ex["params"]
A, B = 1.0, 0.3
sub = {a: A, b: B}
gam = sp.lambdify(t, list(ex["expr"].subs(sub)), "numpy")
Tv, Nv, Bv = dgsym.frenet_frame(ex["expr"], t)
Tf, Nf, Bf = (sp.lambdify(t, list(v.subs(sub)), "numpy") for v in (Tv, Nv, Bv))
kap_s, tau_s = dgsym.curvature_torsion(ex["expr"], t)
kap, tau = float(kap_s.subs(sub)), float(tau_s.subs(sub))
cc = np.hypot(A, B)
SC = 0.55
TS = (-1.7, -0.2, 4.6, 6.1)


def G(tt):
    tt = np.asarray(tt, float)
    return np.stack([np.broadcast_to(c, tt.shape) for c in gam(tt)], axis=-1)


# 자기검사: 프레네-세레 공식 (호의 길이로 미분 = (1/c) d/dt), 다르부 벡터
omega = np.array([0.0, 0.0, 1 / cc])
hh = 1e-5
for ti in TS:
    fr = [np.array(f(ti), float) for f in (Tf, Nf, Bf)]
    dfr = [(np.array(f(ti + hh), float) - np.array(f(ti - hh), float)) / (2 * hh) / cc for f in (Tf, Nf, Bf)]
    tv, nv, bv = fr
    assert np.allclose(dfr[0], kap * nv, atol=1e-8)
    assert np.allclose(dfr[1], -kap * tv + tau * bv, atol=1e-8)
    assert np.allclose(dfr[2], -tau * nv, atol=1e-8)
    for v, dv in zip(fr, dfr):
        assert np.allclose(dv, np.cross(omega, v), atol=1e-8)
    assert abs(np.linalg.det(np.stack(fr, axis=1)) - 1) < 1e-12

fig = plt.figure(figsize=(6, 4.6))
ax = dgfig.axes3d(fig, elev=20, azim=-40)
V = dgfig.view_vector(ax)
uu, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(-2.6 * B - 0.1, 7.4 * B + 0.1, 11))
dgfig.surface(ax, A * np.cos(uu), A * np.sin(uu), zz, alpha=0.08, grid=False, zorder=1, shade=False)
ts = np.linspace(-2.6, 7.4, 900)
H = G(ts)
nrm = np.stack([np.cos(ts), np.sin(ts), 0 * ts], axis=1)
dgfig.curve3d(ax, H, visible=nrm @ V > 0, zorder=4)
ax.plot([0, 0], [0, 0], [-2.6 * B - 0.2, 7.4 * B + 0.35], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(0.04, 0, 7.4 * B + 0.42, r"$z$", color=C["aux"], fontsize=11)

tips = []
for i, ti in enumerate(TS):
    p = G(ti)
    tv, nv, bv = (np.array(f(ti), float) for f in (Tf, Nf, Bf))
    assert p @ V > 0 or np.array([np.cos(ti), np.sin(ti), 0]) @ V > 0     # 보이는 쪽
    dgfig.point3d(ax, p, size=14, zorder=10)
    dgfig.arrow3d(ax, p, tv, role="tangent", scale=SC, zorder=11, head=10)
    dgfig.arrow3d(ax, p, nv, role="normal", scale=SC, zorder=11, head=10)
    dgfig.arrow3d(ax, p, bv, role="third", scale=SC, zorder=11, head=10)
    tips += [p + SC * tv, p + SC * nv, p + SC * bv]
    if i == 1:
        ax.text(*(p + SC * tv + np.array([0.02, 0.06, 0.03])), r"$\mathbf{t}$", color=C["tangent"], fontsize=12, zorder=12)
        ax.text(*(p + SC * nv + np.array([0.0, 0.0, -0.12])), r"$\mathbf{n}$", color=C["normal"], fontsize=12, zorder=12, ha="center", va="top")
        ax.text(*(p + SC * bv + np.array([0.04, 0.0, 0.05])), r"$\mathbf{b}$", color=C["third"], fontsize=12, zorder=12)

dgfig.equal_aspect(ax, np.concatenate([H, np.array(tips)]), zoom=1.3)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)
