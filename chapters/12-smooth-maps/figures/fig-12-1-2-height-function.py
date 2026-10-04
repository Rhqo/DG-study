"""그림 12.1.2: S² 위의 높이함수 h(x) = x³을 두 입체사영 차트로 읽은 좌표표현 (12.1절, 예 12.1.3(c)).

(a) σ 차트: ĥ(u) = h∘σ^{-1}(u) = (|u|² − 1)/(|u|² + 1)의 등위선.
(b) σ̃ 차트: h∘σ̃^{-1}(v) = (1 − |v|²)/(1 + |v|²)의 등위선.
등위값 c = −0.8, −0.6, …, 0.8. 등위선 h = c는 위도원이고, σ 차트에서 반지름 √((1 + c)/(1 − c)),
σ̃ 차트에서 반지름 √((1 − c)/(1 + c))인 원이다(두 반지름은 서로 역수, 좌표변환 u ↦ u/|u|²).
적도 h = 0은 두 차트 모두에서 단위원(주황)이다. 점 0은 (a)에서 남극 −N, (b)에서 북극 N이다.
식은 dgsym.stereographic(2)에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-1-2-height-function.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from dgsym import stereographic

C = dgfig.COLORS
LW = dgfig.LW

st = stereographic(2)
U = st["chart_coords"]
h_north = sp.simplify(st["sigma_inv"][2])        # h∘σ^{-1}
h_south = sp.simplify(st["sigma_south_inv"][2])  # h∘σ̃^{-1}
fN = sp.lambdify(U, h_north, "numpy")
fS = sp.lambdify(U, h_south, "numpy")

# 자기검사 (§12.1): 식이 본문과 같고, 좌표변환과 맞으며, 등위선 반지름이 본문의 값이다.
u1, u2 = U
r2 = u1 ** 2 + u2 ** 2
assert sp.simplify(h_north - (r2 - 1) / (r2 + 1)) == 0
assert sp.simplify(h_south - (1 - r2) / (1 + r2)) == 0
tr = st["transition"]                            # σ̃∘σ^{-1}(u) = u/|u|²
assert sp.simplify(h_south.subs({u1: tr[0], u2: tr[1]}, simultaneous=True) - h_north) == 0
levels = np.round(np.arange(-0.8, 0.81, 0.2), 10)
for c in levels:
    rN, rS = np.sqrt((1 + c) / (1 - c)), np.sqrt((1 - c) / (1 + c))
    assert abs(fN(rN, 0.0) - c) < 1e-12 and abs(fS(rS, 0.0) - c) < 1e-12 and abs(rN * rS - 1) < 1e-12

L = 3.2
g = np.linspace(-L, L, 321)
X, Y = np.meshgrid(g, g)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
for ax, f, lab, pole, title in ((axes[0], fN, r"$\hat h = h\circ\sigma^{-1}$", r"$\sigma(-N)=0$", "(a)"),
                                (axes[1], fS, r"$h\circ\tilde\sigma^{-1}$", r"$\tilde\sigma(N)=0$", "(b)")):
    Z = f(X, Y)
    ax.set_aspect("equal")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.imshow(Z, extent=(-L, L, -L, L), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.35,
              interpolation="bilinear")
    cs = ax.contour(X, Y, Z, levels=[c for c in levels if abs(c) > 1e-9], colors=C["covector"], linewidths=0.9,
                    linestyles="solid")
    ax.clabel(cs, fmt="%.1f", fontsize=7.5, inline_spacing=2)
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=C["accent"], lw=1.8)
    ax.plot([0], [0], "o", color=C["main"], ms=3.5)
    ax.annotate(pole, xy=(0, 0), xytext=(0.9, -2.3), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=C["main"], lw=0.6, shrinkA=1, shrinkB=2))
    ax.axhline(0, color=C["aux"], lw=0.5)
    ax.axvline(0, color=C["aux"], lw=0.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.tick_params(labelsize=8)
    ax.set_title(title + "  " + lab, fontsize=11)
axes[0].set_xlabel(r"$u^1$", fontsize=10)
axes[0].set_ylabel(r"$u^2$", fontsize=10)
axes[1].set_xlabel(r"$v^1$", fontsize=10)
axes[1].set_ylabel(r"$v^2$", fontsize=10)
fig.tight_layout(w_pad=1.5)
dgfig.save(fig, __file__)
