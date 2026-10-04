"""그림 12.5.2: ℝ의 매끄러운 소진 함수 f = Σ_k (|k| + 1) ψ_k (12.5절, 예 12.5.8(d)).

ψ_k는 예 12.4.6(b)의 단위분할(U_k = (k − 1, k + 1))이다. f(t)는 −4.6 ≤ t ≤ 4.6에서 검정으로,
비교를 위해 |t| + 1을 회색 점선으로 그린다. 수평선 c = 3(주황)과 부분수준집합 f^{-1}((−∞, 3])(가로축 위의 굵은 주황 선분)을 표시한다.
f는 |t| ≤ 1/4 근처에서 1이고, 정수 k 근처 [k − 1/4, k + 1/4]에서 |k| + 1인 평평한 계단을 매끄럽게 이은 모양이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-5-2-exhaustion-function.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def f0(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = np.exp(-1.0 / t[m])
    return out


def h(t):
    p, q = f0(2 - np.asarray(t, float)), f0(np.asarray(t, float) - 1)
    return p / (p + q)


b = lambda s: h(0.5 + 2 * np.abs(np.asarray(s, float)))
ks = np.arange(-8, 9)
t = np.linspace(-4.6, 4.6, 4601)
g = np.array([b(t - q) for q in ks])
psi = g / g.sum(axis=0)
fexh = (np.abs(ks)[:, None] + 1) * psi
fexh = fexh.sum(axis=0)

# 자기검사: f ≥ 1, |t| ≥ m + 1이면 f(t) ≥ m + 2 (따라서 {f ≤ m + 1} ⊆ [−m − 1, m + 1]), 정수 근처 평평함
assert np.all(fexh >= 1 - 1e-12)
for m0 in range(0, 3):
    assert np.all(fexh[np.abs(t) >= m0 + 1] >= m0 + 2 - 1e-12)
for q in range(-3, 4):
    m = np.abs(t - q) <= 0.25
    assert np.allclose(fexh[m], abs(q) + 1)
c = 3.0
sub = fexh <= c
assert np.all(np.abs(t[sub]) <= 2.3) and np.all(sub[np.abs(t) <= 2.25])   # {f ≤ 3} = [−9/4, 9/4] (경계는 언더플로 여유)

fig, ax = plt.subplots(figsize=(6.0, 2.9))
ax.plot(t, np.abs(t) + 1, color=C["aux"], lw=1.0, ls=(0, (4, 3)))
ax.plot(t, fexh, color=C["main"], lw=LW["main"])
ax.axhline(c, color=C["accent"], lw=1.0)
lo, hi = t[sub].min(), t[sub].max()
ax.plot([lo, hi], [0, 0], color=C["accent"], lw=4, solid_capstyle="butt")
ax.text(4.5, c + 0.12, r"$c=3$", color=C["accent"], fontsize=10, ha="right")
ax.text(0, 0.25, r"$f^{-1}((-\infty,3])$", color=C["accent"], fontsize=10, ha="center")
ax.text(3.9, 3.55, r"$|t|+1$", color=C["aux"], fontsize=10)
ax.text(-0.9, 2.25, r"$f$", fontsize=11)
ax.set_xlim(-4.6, 4.6)
ax.set_ylim(-0.2, 5.8)
ax.set_xticks(range(-4, 5))
ax.set_yticks(range(0, 6))
ax.tick_params(labelsize=8.5)
ax.set_xlabel(r"$t$", fontsize=10)
fig.tight_layout()
dgfig.save(fig, __file__)
