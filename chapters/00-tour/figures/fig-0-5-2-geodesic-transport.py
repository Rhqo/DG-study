"""Figure 0.5.2: (a) great circle과 circle of latitude, (b) circle of latitude를 따른 parallel transport (0.5절).

unit sphere (dgsym.EXAMPLES["sphere"], r = 1, 바깥쪽 N).
(a) 여위도 θ₀ = π/4(북위 45°)의 두 점 p = x(π/4, −π/3), q = x(π/4, π/3).
    circle of latitude의 호(검정, 길이 sin θ₀ · 2π/3 ≈ 1.481)와 great circle의 호(주황, 길이 arccos(1/4) ≈ 1.318).
(b) 여위도 θ₀ = π/3인 circle of latitude를 φ가 증가하는 쪽으로 한 바퀴 돌며 V₀ = e_θ(남쪽, 파랑)를
    parallel transport한다(dgnum.parallel_transport, Christoffel symbols는 dgsym에서). 30°마다 V(φ)를 주황으로 그리고,
    한 바퀴 뒤의 V(2π)를 굵은 주황으로 그린다. 바깥에서 보아 반시계 방향으로 2π(1 − cos θ₀) = π만큼 돌아,
    처음과 반대 방향(북쪽)을 가리킨다.
자기검사: (a)의 두 길이, (b)에서 V는 길이를 유지하고 회전각 = 2π(1 − cos θ₀) (mod 2π).
great circle 호의 가속도는 sphere에 수직이고, circle of latitude의 가속도는 수직이 아니다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-5-2-geodesic-transport.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgnum
import dgsym

C = dgfig.COLORS
ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Xt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")


def P(t, p):
    return np.array(X(t, p), float)


# (a) ----------------------------------------------------------------------------------------
t0 = np.pi / 4
p, q = P(t0, -np.pi / 3), P(t0, np.pi / 3)
s = np.linspace(-np.pi / 3, np.pi / 3, 300)
lat = np.stack([P(t0, x) for x in s])
L_lat = np.sum(np.linalg.norm(np.diff(lat, axis=0), axis=1))
d = np.arccos(p @ q)
w = np.linspace(0, 1, 300)
gc = np.stack([(np.sin((1 - a) * d) * p + np.sin(a * d) * q) / np.sin(d) for a in w])
L_gc = np.sum(np.linalg.norm(np.diff(gc, axis=0), axis=1))
assert np.isclose(L_lat, np.sin(t0) * 2 * np.pi / 3, rtol=1e-4) and np.isclose(L_gc, np.arccos(0.25), rtol=1e-4)
assert np.allclose(np.linalg.norm(gc, axis=1), 1)
acc_gc = np.gradient(np.gradient(gc, axis=0), axis=0)[5:-5]
tang = acc_gc - np.einsum("ij,ij->i", acc_gc, gc[5:-5])[:, None] * gc[5:-5]
assert np.max(np.linalg.norm(tang, axis=1)) < 1e-8                 # great circle: 가속도 ⟂ sphere
acc_lat = np.gradient(np.gradient(lat, axis=0), axis=0)[5:-5]
tang_l = acc_lat - np.einsum("ij,ij->i", acc_lat, lat[5:-5])[:, None] * lat[5:-5]
assert np.min(np.linalg.norm(tang_l, axis=1)) > 1e-6                # circle of latitude: 접성분 있음

# (b) ----------------------------------------------------------------------------------------
t1 = np.pi / 3
g = dgsym.induced_metric(ex["expr"], (th, ph), ex["positive"])
Gam = dgnum.numeric_christoffel(g, (th, ph), {rs: 1}, ex["positive"])
ts = np.linspace(0, 2 * np.pi, 3601)
V = dgnum.parallel_transport(Gam, lambda t: np.array([t1, t]), lambda t: np.array([0.0, 1.0]),
                             np.array([1.0, 0.0]), ts)


def to3(k):
    return V[k, 0] * np.array(Xt(t1, ts[k])) + V[k, 1] * np.array(Xp(t1, ts[k]))


v0, v1 = to3(0), to3(len(ts) - 1)
assert np.allclose([np.linalg.norm(to3(k)) for k in range(0, len(ts), 300)], 1, atol=1e-8)
N0 = P(t1, 0.0)
ang = np.arctan2(np.cross(v0, v1) @ N0, v0 @ v1)
assert np.isclose(np.mod(ang, 2 * np.pi), np.mod(2 * np.pi * (1 - np.cos(t1)), 2 * np.pi), atol=1e-6)

fig = plt.figure(figsize=(6.6, 3.3))
T, PP = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing="ij")
S = np.stack([np.broadcast_to(c, T.shape) for c in X(T, PP)], axis=-1)

ax = fig.add_axes([0.0, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=22, azim=0)
ax.set_proj_type("ortho")
ax.set_axis_off()
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)
view = dgfig.view_vector(ax)
for k in range(1, 6):
    cc = np.stack([P(k * np.pi / 6, x) for x in np.linspace(-np.pi, np.pi, 240)])
    dgfig.curve3d(ax, cc, role="#A8A8A8", lw=0.4, visible=cc @ view > 0, hidden=None, zorder=2)
for k in range(12):
    cc = np.stack([P(x, k * np.pi / 6) for x in np.linspace(0, np.pi, 120)])
    dgfig.curve3d(ax, cc, role="#A8A8A8", lw=0.4, visible=cc @ view > 0, hidden=None, zorder=2)
dgfig.curve3d(ax, lat, role="main", lw=1.8, zorder=5)
dgfig.curve3d(ax, gc, role="accent", lw=2.0, zorder=6)
dgfig.point3d(ax, p, label=r"$p$", label_offset=(0, -0.16, -0.05), zorder=11)
dgfig.point3d(ax, q, label=r"$q$", label_offset=(0, 0.06, -0.05), zorder=11)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
ax.text2D(0.03, 0.93, r"$(\mathrm{a})$", transform=ax.transAxes, fontsize=11)

ax = fig.add_axes([0.5, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=58, azim=-30)
ax.set_proj_type("ortho")
ax.set_axis_off()
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)
view = dgfig.view_vector(ax)
circ = np.stack([P(t1, x) for x in np.linspace(0, 2 * np.pi, 360)])
dgfig.curve3d(ax, circ, role="main", lw=1.6, visible=circ @ view > 0, hidden="dashed", zorder=5)
for j in range(1, 12):
    k = j * 300
    base = P(t1, ts[k])
    if base @ view > 0.05:
        dgfig.arrow3d(ax, base, to3(k), role="accent", scale=0.32, zorder=8, head=8, lw=1.3)
base0 = P(t1, 0.0)
dgfig.arrow3d(ax, base0, v0, role="tangent", scale=0.36, label=r"$V_0$", label_offset=(0.04, 0.05, -0.02), zorder=10)
dgfig.arrow3d(ax, base0, v1, role="accent", scale=0.36, lw=2.0, zorder=10)
ax.text(*(base0 + 0.36 * v1 + np.array([0.0, 0.08, 0.06])), r"$V(2\pi)$", color=C["accent"], fontsize=11, zorder=12)
dgfig.point3d(ax, base0, size=14, zorder=11)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
ax.text2D(0.03, 0.93, r"$(\mathrm{b})$", transform=ax.transAxes, fontsize=11)

dgfig.save(fig, __file__)
