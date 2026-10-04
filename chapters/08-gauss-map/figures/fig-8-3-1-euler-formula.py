"""그림 8.3.1: 법곡률을 방향의 함수로 본 그래프 — 원환면의 세 점 (8.3절, 정리 8.3.5, 예 8.3.6).

원환면 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8, N = N_x (안쪽).
점 x(u, v)에서 e₁ = x_u/r, e₂ = x_v/(R + r cos u), w(β) = cos β e₁ + sin β e₂일 때
κ_n(β) = II(w(β)) (dgsym.shape_operator와 제1기본형식으로 계산). u = 0, π/2, π.
최댓값(β = 0, π)과 최솟값(β = π/2)을 점으로 표시한다.
자기검사: 직접 계산한 κ_n(β)가 오일러 공식 κ₁cos²β + κ₂sin²β와 같다, 최대·최소가 고윳값과 같다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-3-1-euler-formula.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
vals = {Rs: 2.0, rs: 0.8}
W = dgsym.shape_operator(ex["expr"], u, v, ex["positive"]).subs(vals)
E, F, G = (c.subs(vals) for c in dgsym.first_ff(ex["expr"], u, v, ex["positive"]))
Wf = sp.lambdify(u, W, "numpy")
Ef, Gf = sp.lambdify(u, E, "numpy"), sp.lambdify(u, G, "numpy")
assert sp.simplify(F) == 0

beta = np.linspace(0, np.pi, 401)          # β = π/2가 격자점
fig, ax = plt.subplots(figsize=(5.6, 3.4))
styles = [(0.0, C["main"], "-", r"$u=0$"), (np.pi / 2, C["tangent"], "--", r"$u=\pi/2$"), (np.pi, C["accent"], "-.", r"$u=\pi$")]
for u0, col, ls, lab in styles:
    Wm = np.array(Wf(u0), float)
    I = np.diag([float(Ef(u0)), float(Gf(u0))])
    a = np.cos(beta) / np.sqrt(I[0, 0])            # w = a x_u + b x_v
    b = np.sin(beta) / np.sqrt(I[1, 1])
    coef = np.stack([a, b])
    kn = np.einsum("in,ij,jn->n", coef, I @ Wm, coef)   # II(w) = ⟨Ww, w⟩ = cᵀ I W c
    k1, k2 = 1 / 0.8, np.cos(u0) / (2.0 + 0.8 * np.cos(u0))
    assert np.allclose(kn, k1 * np.cos(beta) ** 2 + k2 * np.sin(beta) ** 2)
    ev = np.sort(np.linalg.eigvals(Wm).real)
    assert np.allclose([kn.max(), kn.min()], [ev[1], ev[0]])
    ax.plot(beta, kn, color=col, ls=ls, lw=1.8, label=lab)
    ax.plot([0, np.pi], [k1, k1], "o", color=col, ms=4.5)
    ax.plot([np.pi / 2], [k2], "o", color=col, ms=4.5)
ax.axhline(0, color=C["aux"], lw=0.8, ls=":")
ax.set_xlim(0, np.pi)
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
ax.set_xticklabels([r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\kappa_n(\beta)$")
ax.text(0.14, 1.25 + 0.06, r"$\kappa_1 = 1/r$", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", frameon=False)
ax.set_ylim(-1.0, 1.45)

dgfig.save(fig, __file__)
