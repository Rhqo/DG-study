"""Figure 0.3.2: tangent plane과 unit normal vector (0.3절, Definition 0.3.3, Example 0.3.4).

sphere S²(r), r = 1 (dgsym.EXAMPLES["sphere"]), standard parametrization x(θ, φ).
점 p = x(0.55, −0.52)에서 x_θ, x_φ (파랑, 실제 길이), N = (x_θ × x_φ)/|x_θ × x_φ| = x/r (주홍, 바깥쪽),
tangent plane T_pS (하늘색 평행사변형, 반너비 0.55).
곡선 α(t) = x(0.55 + 0.6t, −0.52 + 0.9t) (주황)와 그 속도 α'(0) = 0.6 x_θ + 0.9 x_φ (주황 화살표, T_pS 안).
자기검사: N은 단위벡터, x_θ와 x_φ에 수직, N = p/r. α'(0)은 T_pS 안에 있다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-3-2-sphere-tangent-plane.py``
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
(rs,) = ex["params"]
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Xt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")
Nsym = dgsym.unit_normal(ex["expr"], th, ph, ex["positive"])
assert sp.simplify(Nsym - ex["expected"]["normal"]) == sp.zeros(3, 1)
Nf = sp.lambdify((th, ph), list(Nsym), "numpy")

q = (0.55, -0.52)
p = np.array(X(*q), float)
xt, xp, N = (np.array(f(*q), float) for f in (Xt, Xp, Nf))
assert np.isclose(N @ N, 1) and abs(N @ xt) < 1e-12 and abs(N @ xp) < 1e-12 and np.allclose(N, p)
a0, b0 = 0.6, 0.9
w = a0 * xt + b0 * xp
assert abs(w @ N) < 1e-12

fig = plt.figure(figsize=(4.8, 4.2))
ax = dgfig.axes3d(fig, elev=22, azim=-55)
ax.set_proj_type("ortho")
T, P = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing="ij")
S = np.stack([np.broadcast_to(c, T.shape) for c in X(T, P)], axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.35, grid=False, zorder=1)
view = dgfig.view_vector(ax)
for k in range(1, 12):                       # circle of latitude
    tt = np.linspace(-np.pi, np.pi, 300)
    L = np.stack([np.broadcast_to(c, tt.shape) for c in X(k * np.pi / 12 * np.ones_like(tt), tt)], axis=-1)
    dgfig.curve3d(ax, L, role="#9A9A9A", lw=0.35, visible=L @ view > 0, hidden=None, zorder=2)
for k in range(12):                          # meridian
    tt = np.linspace(0, np.pi, 200)
    L = np.stack([np.broadcast_to(c, tt.shape) for c in X(tt, k * np.pi / 6 * np.ones_like(tt))], axis=-1)
    dgfig.curve3d(ax, L, role="#9A9A9A", lw=0.35, visible=L @ view > 0, hidden=None, zorder=2)
tt = np.linspace(-0.35, 0.35, 100)
al = np.stack([np.broadcast_to(c, tt.shape) for c in X(q[0] + a0 * tt, q[1] + b0 * tt)], axis=-1)
dgfig.curve3d(ax, al, role="accent", lw=1.8, zorder=6)
dgfig.tangent_plane(ax, p, xt, xp, size=0.55, zorder=5)
dgfig.arrow3d(ax, p, xt, role="tangent", scale=0.5, label=r"$\mathbf{x}_\theta$", label_offset=(0, 0, -0.08), zorder=10)
dgfig.arrow3d(ax, p, xp, role="tangent", scale=0.5, label=r"$\mathbf{x}_\varphi$", label_offset=(0, 0.05, 0.05), zorder=10)
dgfig.arrow3d(ax, p, w, role="accent", scale=0.5, label=r"$\alpha'(0)$", label_offset=(0.0, 0.12, 0.06), zorder=10)
dgfig.arrow3d(ax, p, N, role="normal", scale=0.6, label=r"$\mathbf{N}$", label_offset=(0, 0, 0.06), zorder=10)
dgfig.point3d(ax, p, label=r"$p$", label_offset=(0.03, -0.12, -0.12), zorder=11)
ax.text(*(p - 0.5 * xt / np.linalg.norm(xt) - 0.75 * xp / np.linalg.norm(xp)), r"$T_pS$", color=C["tangent"],
        fontsize=12, zorder=12)
dgfig.equal_aspect(ax, S.reshape(-1, 3), (p + 0.65 * N)[None], zoom=1.25)

dgfig.save(fig, __file__)
