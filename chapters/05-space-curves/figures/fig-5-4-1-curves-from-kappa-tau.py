"""그림 5.4.1: 곡률과 비틀림률로부터 곡선 만들기 (5.4절, 예 5.4.10).

곡선론의 기본정리(정리 5.4.8)의 존재 증명대로 프레네-세레 연립방정식
    γ' = T,  T' = κN,  N' = -κT + τB,  B' = -τN
을 s = 0에서 γ(0) = 0, (T, N, B)(0) = (e1, e2, e3)로 놓고 dgnum.rk4로 푼다.
(a) κ ≡ 1, τ ≡ 0.4, |s| ≤ 2π/√1.16 (두 바퀴, 길이 4πc, c = 1/√1.16): 나선 (따름정리 5.4.9: a = κ/(κ²+τ²) ≈ 0.862, b = τ/(κ²+τ²) ≈ 0.345).
(b) κ ≡ 1, τ(s) = s/4, -11 ≤ s ≤ 11: |τ|가 커질수록 가늘게 감긴다.
출발점(주황 점)과 출발 틀 (e1 파랑, e2 주홍, e3 청록, 실제 길이의 0.6배)을 표시한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-4-1-curves-from-kappa-tau.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

import dgnum

C = dgfig.COLORS


def build(kap, tau, s0, s1, n):
    ts = np.linspace(s0, s1, n)
    i0 = int(np.argmin(np.abs(ts)))
    assert abs(ts[i0]) < 1e-12                                   # s = 0이 격자점

    def f(s, y):
        T, N, B = y[3:6], y[6:9], y[9:12]
        return np.concatenate([T, kap(s) * N, -kap(s) * T + tau(s) * B, -tau(s) * N])

    y0 = np.concatenate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    fwd = dgnum.rk4(f, y0, ts[i0:])
    bwd = dgnum.rk4(f, y0, ts[i0::-1])[::-1]
    return ts, np.concatenate([bwd[:-1], fwd])


def check(ts, Y, kap, tau):
    """틀이 정규직교·양의 방향으로 남는지, 수치 곡선에서 다시 잰 κ, τ가 주어진 함수와 같은지."""
    F = Y[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)          # 열이 T, N, B
    G = np.einsum("kij,kil->kjl", F, F)
    assert np.max(np.abs(G - np.eye(3))) < 1e-8
    assert np.max(np.abs(np.linalg.det(F) - 1)) < 1e-8
    h = ts[1] - ts[0]
    g = Y[:, :3]
    d1 = np.gradient(g, h, axis=0, edge_order=2)
    d2 = np.gradient(d1, h, axis=0, edge_order=2)
    d3 = np.gradient(d2, h, axis=0, edge_order=2)
    sl = slice(5, -5)
    cr = np.cross(d1, d2)
    k_num = np.linalg.norm(cr, axis=1) / np.linalg.norm(d1, axis=1) ** 3
    t_num = np.einsum("ij,ij->i", cr, d3) / np.einsum("ij,ij->i", cr, cr)
    assert np.max(np.abs(k_num[sl] - kap(ts[sl]))) < 1e-3
    assert np.max(np.abs(t_num[sl] - tau(ts[sl]))) < 2e-3


fig = plt.figure(figsize=(6.4, 3.8))

# (a) 상수 κ, τ: 나선
K0, T0 = 1.0, 0.4
kapA = lambda s: K0 + 0 * np.asarray(s, float)
tauA = lambda s: T0 + 0 * np.asarray(s, float)
L = 4 * np.pi * np.hypot(K0, T0) / (K0 ** 2 + T0 ** 2)          # 나선 두 바퀴의 길이 4πc, c = 1/√(κ²+τ²)
tsA, YA = build(kapA, tauA, -L / 2, L / 2, 2401)
check(tsA, YA, kapA, tauA)
aa, bb = K0 / (K0 ** 2 + T0 ** 2), T0 / (K0 ** 2 + T0 ** 2)
axis_pts = YA[:, :3] + aa * YA[:, 6:9]                           # γ + a n은 축 위에 있다
u = (T0 * np.array([1, 0, 0]) + K0 * np.array([0, 0, 1])) / np.hypot(K0, T0)
rel = axis_pts - axis_pts[0]
assert np.max(np.linalg.norm(np.cross(rel, u), axis=1)) < 1e-7    # 축은 다르부 벡터 방향의 직선
dist = np.linalg.norm(np.cross(YA[:, :3] - axis_pts[0], u), axis=1)
assert np.max(np.abs(dist - aa)) < 1e-7                          # 축까지의 거리 = a

axA = dgfig.axes3d(fig, pos=121, elev=18, azim=-70)
axA.plot(*YA[:, :3].T, color=C["main"], lw=dgfig.LW["main"], zorder=4)
ax_line = np.stack([axis_pts[0] - 2.5 * u, axis_pts[0] + 2.5 * u])
axA.plot(*ax_line.T, color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=2)

# (b) τ(s) = s/4
kapB = lambda s: 1.0 + 0 * np.asarray(s, float)
tauB = lambda s: np.asarray(s, float) / 4
tsB, YB = build(kapB, tauB, -11, 11, 4401)
check(tsB, YB, kapB, tauB)
axB = dgfig.axes3d(fig, pos=122, elev=12, azim=60)
axB.plot(*YB[:, :3].T, color=C["main"], lw=1.4, zorder=4)

for ax, Y, tag, lab in ((axA, YA, "(a)", r"$\kappa=1,\ \tau=0.4$"), (axB, YB, "(b)", r"$\kappa=1,\ \tau=s/4$")):
    i0 = len(Y) // 2
    p = Y[i0, :3]
    dgfig.point3d(ax, p, role="accent", size=22, zorder=10)
    for j, role in enumerate(("tangent", "normal", "third")):
        dgfig.arrow3d(ax, p, Y[i0, 3 + 3 * j:6 + 3 * j], role=role, scale=0.6, zorder=11, head=9)
    ax.text2D(0.02, 0.98, tag, transform=ax.transAxes, fontsize=11, va="top")
    ax.text2D(0.5, -0.02, lab, transform=ax.transAxes, fontsize=12, ha="center", va="top")
    dgfig.equal_aspect(ax, Y[:, :3], zoom=1.2)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0)
dgfig.save(fig, __file__)
