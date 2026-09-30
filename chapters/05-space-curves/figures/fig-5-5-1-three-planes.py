"""그림 5.5.1: 한 점에서의 프레네 틀과 세 평면 (5.5절, 정의 5.5.1, 따름정리 5.5.5).

기준 예제의 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]), a = 1, b = 0.6, 점 γ(t0), t0 = 0.
- 접촉평면 (t, n이 생성, 하늘색), 전직평면 (t, b가 생성, 회색), 법평면 (n, b가 생성, 연한 주황).
- 곡선은 전직평면의 n 쪽에 있고, τ > 0이므로 s가 늘 때 접촉평면을 -b 쪽에서 +b 쪽으로 지난다.
프레네 틀(t 파랑, n 주홍, b 청록)은 실제 길이의 0.7배. 평면 조각은 각 방향으로 ±0.8.
곡선은 -1.15 ≤ t ≤ 1.15만 그리고, 접촉평면의 +b 쪽은 실선, -b 쪽은 점선으로 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-5-1-three-planes.py``
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
A, B = 1.0, 0.6
sub = {a: A, b: B}
gam = sp.lambdify(t, list(ex["expr"].subs(sub)), "numpy")
Tf, Nf, Bf = (sp.lambdify(t, list(v.subs(sub)), "numpy") for v in dgsym.frenet_frame(ex["expr"], t))
_, tau_s = dgsym.curvature_torsion(ex["expr"], t)
tau = float(tau_s.subs(sub))
T0 = 0.0


def G(tt):
    tt = np.asarray(tt, float)
    return np.stack([np.broadcast_to(c, tt.shape) for c in gam(tt)], axis=-1)


p = G(T0)
tv, nv, bv = (np.array(f(T0), float) for f in (Tf, Nf, Bf))
ts = np.linspace(-1.15, 1.15, 400)
H = G(ts)
X, Y, Z = ((H - p) @ v for v in (tv, nv, bv))

# 자기검사: 곡선은 전직평면의 n 쪽 (y > 0), τ > 0이면 z는 t와 같은 부호 (따름정리 5.5.5)
assert tau > 0
mask = np.abs(ts) > 1e-9
assert np.all(Y[mask] > 0)
assert np.all(np.sign(Z[mask]) == np.sign(ts[mask]))
M = np.stack([tv, nv, bv], axis=1)
assert np.allclose(M.T @ M, np.eye(3)) and abs(np.linalg.det(M) - 1) < 1e-12

fig = plt.figure(figsize=(5.8, 4.8))
Vd = 0.9 * tv + 0.75 * nv + 0.65 * bv                          # 세 평면이 모두 비스듬히 보이는 방향
Vd /= np.linalg.norm(Vd)
ax = dgfig.axes3d(fig, elev=np.degrees(np.arcsin(Vd[2])), azim=np.degrees(np.arctan2(Vd[1], Vd[0])))
w = 0.8


def patch(e1, e2, face, alpha, edge):
    corners = [p - w * e1 - w * e2, p + w * e1 - w * e2, p + w * e1 + w * e2, p - w * e1 + w * e2]
    ax.add_collection3d(Poly3DCollection([corners], facecolor=face, alpha=alpha, edgecolor=edge,
                                         linewidth=0.7, zorder=2))
    return np.array(corners)


pts = [patch(tv, nv, C["region"], 0.28, C["tangent"]),
       patch(tv, bv, C["surface"], 0.45, C["aux"]),
       patch(nv, bv, C["accent"], 0.14, C["accent"])]
# 곡선: 접촉평면(z = 0) 위쪽(+b 쪽)은 실선, 아래쪽은 옅게
above = Z >= 0
dgfig.curve3d(ax, H, visible=above, zorder=6)
dgfig.point3d(ax, p, size=16, zorder=10)
for v, role, lab, off in ((tv, "tangent", r"$\mathbf{t}$", (0.05, 0.05, 0.0)), (nv, "normal", r"$\mathbf{n}$", (0.0, 0.08, -0.05)),
                          (bv, "third", r"$\mathbf{b}$", (0.03, 0.0, 0.06))):
    dgfig.arrow3d(ax, p, v, role=role, scale=0.7, zorder=11, head=10)
    ax.text(*(p + 0.7 * v + np.array(off)), lab, color=C[role], fontsize=12, zorder=12)
ax.text(*(p - 0.28 * nv - 0.12 * bv), r"$\gamma(s_0)$", fontsize=11, zorder=12, ha="center", va="top")
dgfig.equal_aspect(ax, np.concatenate(pts + [H]), zoom=1.3)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)
