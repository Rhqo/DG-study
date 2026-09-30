"""그림 5.5.2: 세 평면에 비친 곡선과 국소 표준형 (5.5절, 정리 5.5.2, 예 5.5.4).

기준 예제의 나선 (dgsym.EXAMPLES["helix"]), a = 1, b = 0.6의 호의 길이 매개화에서 s0 = 0, -2 ≤ s ≤ 2.
프레네 좌표 x = <γ - γ(0), t0>, y = <·, n0>, z = <·, b0>로 쓴 곡선(검정)과 국소 표준형의 3차 근사
x ≈ s - κ²s³/6, y ≈ κs²/2 + κ's³/6 (κ' = 0), z ≈ κτs³/6 (주황 점선)을
(a) 접촉평면 (x, y), (b) 전직평면 (x, z), (c) 법평면 (y, z)에 비춘다.
κ = a/(a²+b²) ≈ 0.735, τ = b/(a²+b²) ≈ 0.441.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-5-2-projections.py``
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
A, B = 1.0, 0.6
sub = {a: A, b: B}
kap = float(ex["expected"]["kappa"].subs(sub))
tau = float(ex["expected"]["tau"].subs(sub))
cc = np.hypot(A, B)
gam = sp.lambdify(t, list(ex["expr"].subs(sub)), "numpy")
Tf, Nf, Bf = (sp.lambdify(t, list(v.subs(sub)), "numpy") for v in dgsym.frenet_frame(ex["expr"], t))
t0v, n0v, b0v = (np.array(f(0.0), float) for f in (Tf, Nf, Bf))
ss = np.linspace(-2, 2, 801)
P = np.stack([np.broadcast_to(c, ss.shape) for c in gam(ss / cc)], axis=1) - np.array(gam(0.0), float)
X, Y, Z = P @ t0v, P @ n0v, P @ b0v
Xa, Ya, Za = ss - kap ** 2 * ss ** 3 / 6, kap * ss ** 2 / 2, kap * tau * ss ** 3 / 6

# 자기검사: 나머지는 s⁴ 차수 (|s| ≤ 0.5에서 |오차|/s⁴ 유계), 법평면의 뾰족점 z² ≈ (2τ²/(9κ)) y³
small = (np.abs(ss) <= 0.5) & (np.abs(ss) > 1e-3)
err = np.sqrt((X - Xa) ** 2 + (Y - Ya) ** 2 + (Z - Za) ** 2)
assert np.max(err[small] / ss[small] ** 4) < 0.1
yy = Ya[small]
assert np.allclose(Za[small] ** 2, 2 * tau ** 2 / (9 * kap) * yy ** 3, rtol=1e-12)
assert np.all(Y[np.abs(ss) > 1e-9] > 0)

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
panels = ((X, Y, Xa, Ya, r"$\mathbf{t}$", r"$\mathbf{n}$", "(a)"),
          (X, Z, Xa, Za, r"$\mathbf{t}$", r"$\mathbf{b}$", "(b)"),
          (Y, Z, Ya, Za, r"$\mathbf{n}$", r"$\mathbf{b}$", "(c)"))
colors = {r"$\mathbf{t}$": C["tangent"], r"$\mathbf{n}$": C["normal"], r"$\mathbf{b}$": C["third"]}
for ax, (U, W, Ua, Wa, lu, lw, tag) in zip(axs, panels):
    ax.plot(U, W, color=C["main"], lw=dgfig.LW["main"], zorder=3)
    ax.plot(Ua, Wa, color=C["accent"], lw=1.3, ls=(0, (4, 2.5)), zorder=4)
    lim_u = max(np.max(np.abs(U)), 0.4) * 1.12
    lim_w = max(np.max(np.abs(W)), 0.4) * 1.12
    lo_u = -0.25 * lim_u if np.min(U) > -1e-12 else -lim_u      # y ≥ 0인 축은 음의 쪽을 짧게
    lo_w = -0.25 * lim_w if np.min(W) > -1e-12 else -lim_w
    ax.annotate("", xy=(lim_u, 0), xytext=(lo_u, 0),
                arrowprops=dict(arrowstyle="-|>", color=colors[lu], lw=0.9, mutation_scale=10))
    ax.annotate("", xy=(0, lim_w), xytext=(0, lo_w),
                arrowprops=dict(arrowstyle="-|>", color=colors[lw], lw=0.9, mutation_scale=10))
    ax.text(lim_u, -0.06 * lim_w, lu, color=colors[lu], fontsize=13, ha="right", va="top")
    ax.text(0.06 * lim_u, lim_w, lw, color=colors[lw], fontsize=13, ha="left", va="top")
    ax.plot(0, 0, "o", color=C["main"], ms=3.5, zorder=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(lo_u * 1.02, lim_u * 1.05)
    ax.set_ylim(lo_w * 1.05, lim_w * 1.05)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, fontsize=12, va="top", ha="left")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.12)
dgfig.save(fig, __file__)
