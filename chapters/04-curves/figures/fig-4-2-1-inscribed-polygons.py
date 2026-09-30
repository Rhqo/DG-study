"""그림 4.2.1: 내접 꺾은선의 길이는 호의 길이로 수렴한다 (4.2절, 정리 4.2.11).

곡선 γ(t) = (t, t²), [t1, t2] = [−1, 1]. 호의 길이 L = √5 + asinh(2)/2 ≈ 2.9579.
(a) 등간격 분할 m = 2(점선), m = 4(실선)의 내접 꺾은선.
(b) m = 1, …, 12에 대한 꺾은선 길이 ℓ(P_m)과 L(가로선). ℓ(P_m) ≤ L이고 m이 커지면 L로 다가간다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-2-1-inscribed-polygons.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

T1, T2 = -1.0, 1.0
L = np.sqrt(5) + np.arcsinh(2) / 2          # ∫_{-1}^{1} √(1 + 4t²) dt
M2 = 2.0                                    # max |γ''| = |(0, 2)|


def gamma(t):
    return np.stack([t, t ** 2], axis=-1)


def poly_len(m):
    u = np.linspace(T1, T2, m + 1)
    return float(np.sum(np.linalg.norm(np.diff(gamma(u), axis=0), axis=1)))


# 자기검사: 수치적분으로 L 확인, ℓ(P) ≤ L, 정리 4.2.11의 오차 상한
tt = np.linspace(T1, T2, 200001)
speed = np.sqrt(1 + 4 * tt ** 2)
L_num = float(np.sum((speed[1:] + speed[:-1]) / 2 * np.diff(tt)))
assert abs(L_num - L) < 1e-8
ms = np.arange(1, 13)
ells = np.array([poly_len(m) for m in ms])
assert np.all(ells <= L + 1e-12)
assert np.all(np.diff(ells) > 0)                              # 등간격 분할에서는 증가 (이 예에서)
assert np.all(L - ells <= 2 * M2 * (T2 - T1) * (T2 - T1) / ms + 1e-12)   # 0 ≤ L − ℓ ≤ 2M(t2−t1)|P|

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) 곡선과 내접 꺾은선 --------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-1.25, 1.25), (-0.18, 1.3))
P = gamma(np.linspace(T1, T2, 400))
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=3)
for m, ls, col, ms_ in ((2, (0, (4, 3)), C["accent"], 4.5), (4, "-", C["tangent"], 3.5)):
    V = gamma(np.linspace(T1, T2, m + 1))
    ax.plot(V[:, 0], V[:, 1], color=col, lw=1.2, ls=ls, zorder=4)
    ax.plot(V[:, 0], V[:, 1], "o", color=col, ms=ms_, zorder=5)
ax.text(-1.0, 1.08, r"$\gamma(t_1)$", ha="center", va="bottom", fontsize=11)
ax.text(1.0, 1.08, r"$\gamma(t_2)$", ha="center", va="bottom", fontsize=11)
ax.text(0.43, 0.02, r"$m=2$", color=C["accent"], fontsize=10.5)
ax.text(-0.95, 0.42, r"$m=4$", color=C["tangent"], fontsize=10.5)
ax.text(0.0, 1.02, "(a)", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)

# (b) 꺾은선 길이 ------------------------------------------------------------------
ax = axs[1]
ax.axhline(L, color=C["main"], lw=1.0, zorder=1)
ax.plot(ms, ells, "o", color=C["tangent"], ms=4.5, zorder=3)
ax.plot([2, 4], [poly_len(2), poly_len(4)], "o", color="none", mec=C["accent"], ms=8, mew=1.2, zorder=4)
ax.text(12.3, L + 0.02, r"$L$", ha="right", va="bottom", fontsize=12)
ax.set_xlim(0.3, 12.7)
ax.set_ylim(1.95, 3.05)
ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$\ell(P_m)$")
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
ax.text(-0.3, 1.02, "(b)", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)

fig.subplots_adjust(wspace=0.28, left=0.01, right=0.99, top=0.9, bottom=0.16)
dgfig.save(fig, __file__)
