"""그림 12.3.1: 매끄러운 계단 함수의 재료 (12.3절, 보조정리 12.3.1).

(a) 예 2.2.20의 함수 f(t) = e^{-1/t} (t > 0), f(t) = 0 (t ≤ 0), −1 ≤ t ≤ 5. 점선은 점근선 y = 1.
(b) h(t) = f(2 − t)/(f(2 − t) + f(t − 1)), −0.5 ≤ t ≤ 3.5 (검정). 분자 f(2 − t)(파랑 점선)와 f(t − 1)(청록 점선)도 그린다.
h는 t ≤ 1에서 1, t ≥ 2에서 0이고 그 사이에서 매끄럽게 줄어든다. h(3/2) = 1/2.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-3-1-smooth-step.py``
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


# 자기검사 (§12.1): 분모 > 0, h = 1 (t ≤ 1), h = 0 (t ≥ 2), 0 < h < 1 (1 < t < 2), h(3 − t) = 1 − h(t)
tt = np.linspace(-3, 6, 9001)
assert np.all(f(2 - tt) + f(tt - 1) > 0)
H = h(tt)
assert np.all(H[tt <= 1] == 1) and np.all(H[tt >= 2] == 0)
mid = (tt > 1.05) & (tt < 1.95)   # 경계 근처는 e^{-1/ε}가 부동소수에서 0으로 내려간다
assert np.all((H[mid] > 0) & (H[mid] < 1))
assert np.allclose(h(3 - tt), 1 - H)
assert abs(h(np.array([1.5]))[0] - 0.5) < 1e-15

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
ax = axes[0]
t = np.linspace(-1, 5, 1201)
ax.axhline(1, color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
ax.plot(t, f(t), color=C["main"], lw=LW["main"])
ax.set_xlim(-1, 5)
ax.set_ylim(-0.12, 1.12)
ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
ax.set_yticks([0, 0.5, 1])
ax.set_title(r"(a)  $f(t)=e^{-1/t}\ (t>0)$", fontsize=11)
ax.set_xlabel(r"$t$", fontsize=10)

ax = axes[1]
t = np.linspace(-0.5, 3.5, 1201)
ax.plot(t, f(2 - t), color=C["tangent"], lw=1.1, ls=(0, (4, 2.5)), label=r"$f(2-t)$")
ax.plot(t, f(t - 1), color=C["third"], lw=1.1, ls=(0, (4, 2.5)), label=r"$f(t-1)$")
ax.plot(t, h(t), color=C["main"], lw=LW["main"], label=r"$h(t)$")
ax.axvline(1, color=C["aux"], lw=0.6)
ax.axvline(2, color=C["aux"], lw=0.6)
ax.plot([1.5], [0.5], "o", color=C["accent"], ms=4, zorder=5)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.12, 1.12)
ax.set_xticks([0, 1, 1.5, 2, 3])
ax.set_xticklabels(["0", "1", "1.5", "2", "3"])
ax.set_yticks([0, 0.5, 1])
ax.set_title(r"(b)  $h(t)$", fontsize=11)
ax.set_xlabel(r"$t$", fontsize=10)
ax.legend(loc="upper right", fontsize=8.5, frameon=False, handlelength=1.8)
for a in axes:
    a.tick_params(labelsize=8.5)
fig.tight_layout(w_pad=1.2)
dgfig.save(fig, __file__)
