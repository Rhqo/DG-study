"""그림 12.4.1: ℝ의 열린덮개 U_k = (k − 1, k + 1) (k ∈ ℤ)에 종속된 단위분할 ψ_k (12.4절, 예 12.4.6(b)).

b(s) = h(1/2 + 2|s|)는 보조정리 12.3.4의 범프(c = 0, r₁ = 1/4, r₂ = 3/4)이고 받침은 [−3/4, 3/4]이다.
g_k(t) = b(t − k), G = Σ_k g_k, ψ_k = g_k/G. 위: ψ_k (k = −2, …, 2, 색을 번갈아)와 합 Σψ_k = 1(검정 점선).
아래: 덮개의 원소 U_k(가는 선)와 받침 supp ψ_k = [k − 3/4, k + 3/4](굵은 선). −2.6 ≤ t ≤ 2.6.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-4-1-partition-line.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def f(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = np.exp(-1.0 / t[pos])
    return out


def h(t):
    a, b = f(2 - np.asarray(t, float)), f(np.asarray(t, float) - 1)
    return a / (a + b)


def bump(s):
    return h(0.5 + 2 * np.abs(np.asarray(s, float)))


ks = np.arange(-6, 7)
t = np.linspace(-2.6, 2.6, 2601)
g = np.array([bump(t - k) for k in ks])
G = g.sum(axis=0)
psi = g / G

# 자기검사 (§12.1): 받침 ⊆ [k − 3/4, k + 3/4] ⊂ U_k, G > 0, Σψ = 1, 각 점에서 0이 아닌 항은 많아야 2개
for k, gk in zip(ks, g):
    assert np.all(gk[np.abs(t - k) >= 0.75] == 0)
assert np.all(G > 0) and np.allclose(psi.sum(axis=0), 1.0)
assert np.all((g > 0).sum(axis=0) <= 2)
assert np.allclose(bump(np.array([0.0, 0.25, -0.25])), 1.0)

fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.2, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.25], hspace=0.08))
cols = [C["tangent"], C["accent"], C["third"], C["covector"]]
show = [k for k in ks if -2 <= k <= 2]
for k in ks:
    if -3 <= k <= 3:
        i = list(ks).index(k)
        ax.plot(t, psi[i], color=cols[k % 4], lw=1.6 if k in show else 1.0, alpha=1.0 if k in show else 0.45)
ax.plot(t, psi.sum(axis=0), color=C["main"], lw=1.0, ls=(0, (4, 3)))
for k in show:
    ax.text(k, 1.07, rf"$\psi_{{{k}}}$", color=cols[k % 4], fontsize=10, ha="center")
ax.set_ylim(-0.05, 1.2)
ax.set_yticks([0, 0.5, 1])
ax.tick_params(labelsize=8.5)
ax.text(2.55, 0.92, r"$\sum_k\psi_k=1$", fontsize=9.5, ha="right", va="top")
for k in show:
    y = -0.35 * (k % 2) - 0.2
    bx.plot([k - 1, k + 1], [y, y], color=cols[k % 4], lw=0.8)
    bx.plot([k - 0.75, k + 0.75], [y, y], color=cols[k % 4], lw=3.0, solid_capstyle="butt")
    for e in (k - 1, k + 1):
        bx.plot([e], [y], "o", mfc="white", mec=cols[k % 4], ms=3.5)
bx.set_ylim(-0.75, 0.05)
bx.set_yticks([])
for side in ("left", "right", "top"):
    bx.spines[side].set_visible(False)
bx.set_xticks([-2, -1, 0, 1, 2])
bx.tick_params(labelsize=8.5)
bx.set_xlabel(r"$t$", fontsize=10)
ax.set_xlim(-2.6, 2.6)
dgfig.save(fig, __file__)
