"""Figure 0.2.3: helix와 Frenet frame (t, n, b) (0.2절, Frenet–Serret formulas (0.2.5)).

helix γ(t) = (a cos t, a sin t, bt), a = 1, b = 0.3 (dgsym.EXAMPLES["helix"]), 0 ≤ t ≤ 4π.
세 점 t = π/3, 5π/3, 7π/3에서 t (파랑), n (주홍), b (청록), 길이 0.7배.
자기검사: frame은 orthonormal이고 b = t × n, Frenet–Serret 식 b' = −τ n (arc length 기준)이 성립한다.
κ = a/(a² + b²), τ = b/(a² + b²) (§7).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-2-3-helix-frenet.py``
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
vals = {a: 1, b: sp.Rational(3, 10)}
G = ex["expr"].subs(vals)
T, Nn, B = dgsym.frenet_frame(G, t)
kap, tau = dgsym.curvature_torsion(G, t)
assert sp.simplify(kap - ex["expected"]["kappa"].subs(vals)) == 0
assert sp.simplify(tau - ex["expected"]["tau"].subs(vals)) == 0
speed = sp.sqrt(G.diff(t).dot(G.diff(t)))
assert sp.simplify(B.diff(t) / speed + tau * Nn) == sp.zeros(3, 1)          # b' = −τ n
assert sp.simplify(T.cross(Nn) - B) == sp.zeros(3, 1)
fG, fT, fN, fB = (sp.lambdify(t, list(M), "numpy") for M in (G, T, Nn, B))

tt = np.linspace(0, 4 * np.pi, 800)
P = np.stack([np.broadcast_to(c, tt.shape) for c in fG(tt)], axis=1)

fig = plt.figure(figsize=(4.6, 4.4))
ax = dgfig.axes3d(fig, elev=15, azim=0)
ax.set_proj_type("ortho")
view = dgfig.view_vector(ax)
radial = np.stack([P[:, 0], P[:, 1], 0 * P[:, 2]], axis=1)
dgfig.curve3d(ax, P, role="main", visible=radial @ view > 0, hidden="dashed", zorder=4)
# 원기둥의 윤곽(보조선)
for z0 in (0.0, 4 * np.pi * 0.3):
    c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(c), np.sin(c), z0 + 0 * c, color=C["aux"], lw=0.5, ls=(0, (3, 3)), zorder=1)
for k, t0 in enumerate((np.pi / 3, 5 * np.pi / 3, 7 * np.pi / 3)):
    p = np.array(fG(t0), float)
    vt, vn, vb = (np.array(f(t0), float) for f in (fT, fN, fB))
    assert np.allclose([vt @ vt, vn @ vn, vb @ vb, vt @ vn, vt @ vb, vn @ vb], [1, 1, 1, 0, 0, 0])
    lab = k == 2                     # t = 7π/3: t, n, b가 잘 갈라져 보이는 frame에 라벨
    dgfig.arrow3d(ax, p, vt, role="tangent", scale=0.7, label=r"$\mathbf{t}$" if lab else None,
                  label_offset=(0, 0, 0.08), zorder=10)
    dgfig.arrow3d(ax, p, vn, role="normal", scale=0.7, label=r"$\mathbf{n}$" if lab else None,
                  label_offset=(0, 0, 0.1), zorder=10)
    dgfig.arrow3d(ax, p, vb, role="third", scale=0.7, label=r"$\mathbf{b}$" if lab else None,
                  label_offset=(0.05, 0, 0.06), zorder=10)
    dgfig.point3d(ax, p, size=14, zorder=11)
dgfig.equal_aspect(ax, P, np.array([[1.3, 1.3, -0.2], [-1.3, -1.3, 4.0]]), zoom=1.15)

dgfig.save(fig, __file__)
