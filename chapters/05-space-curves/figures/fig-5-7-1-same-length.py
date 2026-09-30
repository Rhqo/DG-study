"""그림 5.7.1: 길이가 같은 닫힌 곡선들이 둘러싸는 넓이 (5.7절, 정리 5.7.7).

세 곡선을 모두 길이 L = 2π가 되도록 닮음변환(확대·축소)한 뒤 같은 축척으로 그린다.
(a) 단위원: A = π, 4πA/L² = 1.
(b) 타원 (2cos t, sin t)를 길이 2π로 줄인 것.
(c) 콩 모양 곡선 ρ(t)(cos t, sin t), ρ = 1 + cos(2t)/4 + 3 sin(3t)/25 를 길이 2π로 줄인 것.
넓이 A = ½∫(xy' - yx')dt와 길이 L은 수치 적분(주기함수의 사다리꼴 공식)으로 계산하고, 비 4πA/L²를 적는다.
자기검사: (a)의 비는 1, (b)·(c)는 1보다 작다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-7-1-same-length.py``
"""

import dgfig

dgfig.setup()

import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

C = dgfig.COLORS
t = sp.symbols("t", real=True)
rho = 1 + sp.cos(2 * t) / 4 + sp.Rational(3, 25) * sp.sin(3 * t)
curves = [(sp.Matrix([sp.cos(t), sp.sin(t)]), "(a)"),
          (sp.Matrix([2 * sp.cos(t), sp.sin(t)]), "(b)"),
          (rho * sp.Matrix([sp.cos(t), sp.sin(t)]), "(c)")]
ts = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
ratios = []
for ax, (g, tag) in zip(axs, curves):
    f = sp.lambdify(t, list(g), "numpy")
    d = sp.lambdify(t, list(sp.diff(g, t)), "numpy")
    x, y = (np.broadcast_to(c, ts.shape) for c in f(ts))
    dx, dy = (np.broadcast_to(c, ts.shape) for c in d(ts))
    L = np.mean(np.hypot(dx, dy)) * 2 * np.pi
    A = np.mean(x * dy - y * dx) / 2 * 2 * np.pi
    lam = 2 * np.pi / L                                          # 길이를 2π로
    X, Y = lam * x, lam * y
    X, Y = X - X.mean(), Y - Y.mean()
    A2, L2 = lam ** 2 * A, lam * L
    ratio = 4 * math.pi * A2 / L2 ** 2
    ratios.append(ratio)
    Xp, Yp = X[::25], Y[::25]                                   # 그릴 때는 800점이면 충분하다
    ax.fill(Xp, Yp, color=C["region"], alpha=0.25, lw=0)
    ax.plot(np.append(Xp, Xp[0]), np.append(Yp, Yp[0]), color=C["main"], lw=dgfig.LW["main"])
    dgfig.schematic_axes(ax, xlim=(-1.9, 1.9), ylim=(-1.45, 1.25))
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, fontsize=12, va="top")
    ax.text(0.5, 0.0, r"$A=%.3f,\ \ 4\pi A/L^2=%.3f$" % (A2, ratio), transform=ax.transAxes, fontsize=11,
            ha="center", va="bottom")
assert abs(ratios[0] - 1) < 1e-9 and ratios[1] < 1 and ratios[2] < 1
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)
dgfig.save(fig, __file__)
