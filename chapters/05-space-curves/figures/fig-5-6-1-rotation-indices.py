"""그림 5.6.1: 닫힌 곡선과 회전지수 (5.6절, 예 5.6.5).

(a) 타원 γ(t) = (2cos t, sin t): 단순, rot = 1.
(b) 리마송 γ(t) = (1 + 2cos t)(cos t, sin t): 단순하지 않음, rot = 2 (04장 연습 4.4.8).
(c) 8자 곡선 γ(t) = (sin t, sin t cos t): 단순하지 않음, rot = 0.
각 곡선 위 아홉 점에서 단위접벡터(파랑, 곡선의 가로 폭의 0.13배 길이)를 그리고, 회전지수는 dgsym.signed_curvature로 계산한
(1/2π)∫κ_s|γ'|dt를 반올림해 표시한다(자기검사: 정수와 1e-8 이내).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-6-1-rotation-indices.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
t = sp.symbols("t", real=True)
curves = [
    (sp.Matrix([2 * sp.cos(t), sp.sin(t)]), "(a)", 1),
    ((1 + 2 * sp.cos(t)) * sp.Matrix([sp.cos(t), sp.sin(t)]), "(b)", 2),
    (sp.Matrix([sp.sin(t), sp.sin(t) * sp.cos(t)]), "(c)", 0),
]
SC = 0.35
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, (g, tag, want) in zip(axs, curves):
    ks = dgsym.signed_curvature(g, t)
    spd = sp.sqrt(sp.diff(g[0], t) ** 2 + sp.diff(g[1], t) ** 2)
    f = sp.lambdify(t, ks * spd, "numpy")
    ts = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
    val = np.mean(np.broadcast_to(f(ts), ts.shape))              # (1/2π)∫ = 평균
    assert abs(val - want) < 1e-8, (tag, val)
    gf = sp.lambdify(t, list(g), "numpy")
    df = sp.lambdify(t, list(sp.diff(g, t)), "numpy")
    tt = np.linspace(0, 2 * np.pi, 800)
    P = np.array(gf(tt))
    SC = 0.13 * (P[0].max() - P[0].min())
    ax.plot(*P, color=C["main"], lw=dgfig.LW["main"], zorder=3)
    for s0 in np.linspace(0, 2 * np.pi, 9, endpoint=False) + 0.15:
        p = np.array(gf(s0), float)
        v = np.array(df(s0), float)
        v /= np.linalg.norm(v)
        ax.annotate("", xy=p + SC * v, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3, mutation_scale=11, shrinkA=0, shrinkB=0),
                    zorder=5)
    p0 = np.array(gf(0.0), float)
    ax.plot(*p0, "o", color=C["accent"], ms=4.5, zorder=6)
    dgfig.schematic_axes(ax)
    lo, hi = P.min(axis=1) - 0.45, P.max(axis=1) + 0.45
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1] - 0.35, hi[1])
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, fontsize=12, va="top")
    ax.text(0.5, 0.0, r"$\mathrm{rot}(\gamma)=%d$" % want, transform=ax.transAxes, fontsize=13, ha="center", va="bottom")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.1)
dgfig.save(fig, __file__)
