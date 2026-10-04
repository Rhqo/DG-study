"""그림 8.2.1: 안장면의 원점에서의 법단면 (8.2절, 정의 8.2.9, 예 8.2.11).

안장면 x(u, v) = (u, v, u² − v²) (dgsym.EXAMPLES["saddle"]), 위쪽 N, 원점 p = 0, N(p) = e₃.
방향 w(β) = (cos β, sin β, 0)의 법단면은 평면 Span{w(β), e₃}과 곡면의 교선
γ_β(s) = (s cos β, s sin β, s² cos 2β)이다 (β = 0, π/6, π/4, π/2, 주황).
β = π/6의 법평면 조각(하늘색)과 w(π/6)(파랑), N(p)(주홍)을 그렸다.
자기검사: γ_β가 곡면 위에 있다, γ_β는 법평면 안에 있다, γ_β''(0)의 e₃ 성분 = 2 cos 2β = II_p(w(β)).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-2-1-normal-sections.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["saddle"]
u, v = ex["coords"]
Xs = ex["expr"]
X = sp.lambdify((u, v), list(Xs), "numpy")
Ns = dgsym.unit_normal(Xs, u, v)
assert Ns.subs({u: 0, v: 0}) == sp.Matrix([0, 0, 1])
Ws = dgsym.shape_operator(Xs, u, v).subs({u: 0, v: 0})
assert Ws == sp.diag(2, -2)


def section(beta, s):
    return np.stack([s * np.cos(beta), s * np.sin(beta), s ** 2 * np.cos(2 * beta)], axis=-1)


betas = [0.0, np.pi / 6, np.pi / 4, np.pi / 2]
s = np.linspace(-0.85, 0.85, 201)
for b in betas:
    pts = section(b, s)
    assert np.allclose(pts[:, 2], pts[:, 0] ** 2 - pts[:, 1] ** 2)          # 곡면 위
    nrm = np.array([-np.sin(b), np.cos(b), 0.0])                           # 법평면의 법선
    assert np.allclose(pts @ nrm, 0)                                       # 법평면 안
    wv = np.array([np.cos(b), np.sin(b)])
    II = wv @ np.array(Ws, float) @ wv
    assert abs(2 * np.cos(2 * b) - II) < 1e-12                             # γ''(0)·e₃ = II(w)

fig = plt.figure(figsize=(6.0, 4.5))
ax = dgfig.axes3d(fig, elev=20, azim=-52)
rr, tt = np.meshgrid(np.linspace(0, 0.9, 28), np.linspace(0, 2 * np.pi, 97))
U, V = rr * np.cos(tt), rr * np.sin(tt)
S = np.stack(X(U, V), axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)
edge = np.stack(X(0.9 * np.cos(tt[:, 0]), 0.9 * np.sin(tt[:, 0])), axis=-1)
ax.plot(*edge.T, color=C["aux"], lw=0.6, zorder=2)

# β = π/6의 법평면 조각
b0 = np.pi / 6
w0 = np.array([np.cos(b0), np.sin(b0), 0.0])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
L, Hh = 0.92, 0.72
corners = [(-L) * w0 + (-0.3) * np.array([0, 0, 1]), L * w0 + (-0.3) * np.array([0, 0, 1]),
           L * w0 + Hh * np.array([0, 0, 1]), (-L) * w0 + Hh * np.array([0, 0, 1])]
ax.add_collection3d(Poly3DCollection([corners], facecolor=C["region"], alpha=0.12, edgecolor=C["region"],
                                     linewidth=0.6, zorder=3))

labels = {0.0: r"$\beta=0$", np.pi / 6: r"$\beta=\pi/6$", np.pi / 4: r"$\beta=\pi/4$", np.pi / 2: r"$\beta=\pi/2$"}
offs = {0.0: (0.04, 0.0, 0.02), np.pi / 6: (0.06, -0.02, 0.0), np.pi / 4: (0.06, 0.0, -0.03), np.pi / 2: (0.02, 0.06, -0.1)}
for b in betas:
    pts = section(b, s)
    dgfig.curve3d(ax, pts, role="accent", lw=1.8, zorder=6)
    end = pts[-1]
    ax.text(*(end + np.array(offs[b])), labels[b], fontsize=10, color=C["main"], zorder=15)
dgfig.arrow3d(ax, [0, 0, 0], 0.55 * w0, role="tangent", label=r"$w$", zorder=12, label_offset=(0.03, 0.03, -0.05))
dgfig.arrow3d(ax, [0, 0, 0], [0, 0, 0.6], role="normal", label=r"$\mathbf{N}(p)$", zorder=12, label_offset=(0.0, 0.0, 0.06))
dgfig.point3d(ax, [0, 0, 0], size=12, zorder=13)
ax.text(0.05, -0.12, -0.1, r"$p$", fontsize=11, zorder=14)
dgfig.equal_aspect(ax, S.reshape(-1, 3), np.array([[0, 0, 0.75]]), zoom=1.15)

dgfig.save(fig, __file__)
