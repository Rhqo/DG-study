"""Figure 0.2.2: osculating circle의 반지름 = 1/|κ_s| (0.2절, Example 0.2.6, Exercise 0.2.1).

ellipse γ(t) = (2 cos t, sin t) (반시계 방향). κ_s(t) = 2/(4 sin² t + cos² t)^{3/2}.
t = 0에서 κ_s = 2 (반지름 1/2), t = π/2에서 κ_s = 1/4 (반지름 4), t = π/4에서도 그린다.
center of curvature c = γ + n_s/κ_s (n_s = J t, 주황 점).
자기검사: dgsym.signed_curvature의 식과 위 식이 같다. osculating circle은 γ와 2차까지 접한다
(γ(t0 ± h)에서 원까지의 거리가 O(h³)).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-2-2-osculating.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ts = sp.symbols("t", real=True)
ell = sp.Matrix([2 * sp.cos(ts), sp.sin(ts), 0])
ks_sym = dgsym.signed_curvature(ell, ts)
assert sp.simplify(ks_sym - 2 / (4 * sp.sin(ts) ** 2 + sp.cos(ts) ** 2) ** sp.Rational(3, 2)) == 0
ks = sp.lambdify(ts, ks_sym, "numpy")
assert np.isclose(ks(0.0), 2.0) and np.isclose(ks(np.pi / 2), 0.25)


def gam(t):
    return np.array([2 * np.cos(t), np.sin(t)])


def tan(t):
    v = np.array([-2 * np.sin(t), np.cos(t)])
    return v / np.linalg.norm(v)


def center(t):
    tv = tan(t)
    ns = np.array([-tv[1], tv[0]])
    return gam(t) + ns / ks(t)


for t0 in (0.0, np.pi / 4, np.pi / 2):          # 2차 접촉: 거리 = O(h³)
    c0, R0 = center(t0), 1 / ks(t0)
    d = [abs(np.linalg.norm(gam(t0 + h) - c0) - R0) for h in (1e-2, 5e-3)]
    assert d[1] < d[0] / 6, d                     # h를 반으로 → 1/8 정도

t = np.linspace(0, 2 * np.pi, 600)
E = np.stack([gam(x) for x in t])

fig = plt.figure(figsize=(6.0, 3.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(*E.T, color=C["main"], lw=dgfig.LW["main"], zorder=4)
c = np.linspace(0, 2 * np.pi, 800)
for t0, lab, lpos in ((0.0, r"$\frac{1}{2}$", (1.62, 0.12)), (np.pi / 2, r"$4$", (-2.55, 0.62)),
                      (np.pi / 4, r"$\approx 1.98$", (0.62, -0.62))):
    c0, R0 = center(t0), 1 / ks(t0)
    ax.plot(c0[0] + R0 * np.cos(c), c0[1] + R0 * np.sin(c), color=C["tangent"], lw=1.0, zorder=3)
    p = gam(t0)
    ax.plot(*p, "o", color=C["main"], ms=4, zorder=6)
    if np.all(np.abs(c0) < [2.9, 1.7]):
        ax.plot(*c0, "o", color=C["accent"], ms=4.5, zorder=6)
        ax.plot(*np.stack([p, c0]).T, color=C["accent"], lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.text(*lpos, lab, fontsize=12, color=C["tangent"], ha="center", va="center")
assert np.isclose(1 / ks(np.pi / 4), 1.976, atol=1e-3)
ax.annotate("", xy=gam(0.55) + 0.02, xytext=gam(0.45), arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.2,
                                                                      mutation_scale=12))
ax.text(-0.15, -0.72, r"$\gamma$", fontsize=13)
dgfig.schematic_axes(ax, (-2.9, 2.9), (-1.7, 1.45))

dgfig.save(fig, __file__)
