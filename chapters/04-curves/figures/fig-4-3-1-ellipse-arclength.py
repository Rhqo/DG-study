"""그림 4.3.1: 같은 타원, 두 가지 매개화 (4.3절, 예 4.3.12).

타원 γ(t) = (2 cos t, sin t) (연습 4.1.2에서 A = 2, B = 1).
(a) t = 2πk/16 (k = 0, …, 15)인 16개의 점과 속도 γ'(t). 속력은 1(긴 축의 끝)에서 2(짧은 축의 끝)까지 변한다.
(b) 호의 길이 매개화 β = γ∘h (정리 4.3.9)에서 s = kL/16인 16개의 점과 속도 β'(s) (길이 1).
    L = ∫_0^{2π} √(4 sin²t + cos²t) dt ≈ 9.6884. h = s⁻¹은 수치로 구한다(초등함수로 쓸 수 없다).
두 그림 모두 벡터를 실제 길이의 0.4배로 그린다(같은 배율).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-3-1-ellipse-arclength.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW
A, B = 2.0, 1.0
SCALE = 0.4
N = 16


def gamma(t):
    return np.stack([A * np.cos(t), B * np.sin(t)], axis=-1)


def dgamma(t):
    return np.stack([-A * np.sin(t), B * np.cos(t)], axis=-1)


# 호의 길이 s(t) = ∫_0^t |γ'|: 조밀한 격자에서 사다리꼴 적분 (심프슨으로 교차검사)
tt = np.linspace(0, 2 * np.pi, 400001)
spd = np.linalg.norm(dgamma(tt), axis=1)
s_of_t = np.concatenate([[0.0], np.cumsum((spd[1:] + spd[:-1]) / 2 * np.diff(tt))])
L = s_of_t[-1]
simpson = (tt[1] - tt[0]) / 3 * (spd[0] + spd[-1] + 4 * spd[1:-1:2].sum() + 2 * spd[2:-1:2].sum())
assert abs(L - simpson) < 1e-8
assert abs(L - 9.68844822054768) < 1e-8           # 완전타원적분: L = 4A·E(m), m = 1 − B²/A² = 3/4 (mpmath로 확인)


def h(s):
    """h = s⁻¹: 호의 길이 s에 대응하는 매개변수 t (단조 보간)."""
    return np.interp(s, s_of_t, tt)


tk_a = 2 * np.pi * np.arange(N) / N
tk_b = h(L * np.arange(N) / N)

# 자기검사: (b)의 점들은 이웃한 점 사이의 호의 길이가 모두 L/16, β'(s) = γ'(h)/|γ'(h)|는 단위벡터
assert np.allclose(np.diff(np.interp(tk_b, tt, s_of_t)), L / N, atol=1e-6)
vb = dgamma(tk_b) / np.linalg.norm(dgamma(tk_b), axis=1)[:, None]
assert np.allclose(np.linalg.norm(vb, axis=1), 1)
sp_a = np.linalg.norm(dgamma(tk_a), axis=1)
assert abs(sp_a.min() - 1) < 1e-12 and abs(sp_a.max() - 2) < 1e-12   # 연습 4.1.2: 최소 B, 최대 A


def arrow(ax, base, vec, color):
    ax.annotate("", xy=base + SCALE * vec, xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=9,
                                shrinkA=0, shrinkB=0), zorder=5)


fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
P = gamma(np.linspace(0, 2 * np.pi, 400))
for ax, tks, vecs, tag in ((axs[0], tk_a, dgamma(tk_a), "(a)"), (axs[1], tk_b, vb, "(b)")):
    dgfig.schematic_axes(ax, (-2.75, 2.75), (-1.5, 1.5))
    ax.plot(P[:, 0], P[:, 1], color=C["aux"], lw=0.9, zorder=3)
    for tk, v in zip(tks, vecs):
        p = gamma(tk)
        arrow(ax, p, v, C["tangent"])
        ax.plot(*p, "o", color=C["main"], ms=3, zorder=6)
    ax.plot(*gamma(0.0), "o", color=C["accent"], ms=5, zorder=7)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)
axs[0].text(0.0, 0.0, r"$\gamma(t)$", ha="center", va="center", fontsize=12)
axs[1].text(0.0, 0.0, r"$\beta(s)$", ha="center", va="center", fontsize=12)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
