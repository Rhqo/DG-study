"""그림 4.4.2: 부호곡률이 평면곡선을 결정한다 (4.4절, 정리 4.4.10, 예 4.4.12).

위: 주어진 함수 k(s). 아래: θ' = k, γ' = (cos θ, sin θ), γ(s0) = (0, 0), θ(s0) = 0으로 풀어 얻은 단위속력 곡선.
(a) k(s) = 1, s ∈ [0, 2π]        → 반지름 1인 원
(b) k(s) = s, s ∈ [−6, 6]         → 클로소이드(오일러 나선). s → ±∞에서 두 점 ±(√π/2, √π/2)로 감겨 들어간다.
(c) k(s) = 2 cos s, s ∈ [0, 4π]   → 사행 곡선
s0는 각 구간에서 (a) 0, (b) 0, (c) 0이다. ODE는 dgnum.rk4로 푼다(scipy 사용 불가, 부록 E).
출발점과 출발 방향은 주황 점과 파란 화살표(길이 1의 0.5배)로 표시한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-4-2-curvature-determines.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

import dgnum

C = dgfig.COLORS
LW = dgfig.LW


def solve(k, s0, s1, n=6001):
    """정리 4.4.10의 해: s0에서 s1까지 (s1 < s0이면 거꾸로) 적분. 반환값 (s, θ, x, y)."""
    ss = np.linspace(s0, s1, n)
    Y = dgnum.rk4(lambda s, y: np.array([k(s), np.cos(y[0]), np.sin(y[0])]), [0.0, 0.0, 0.0], ss)
    return ss, Y[:, 0], Y[:, 1], Y[:, 2]


def solve_both(k, lo, hi, s0=0.0):
    b = solve(k, s0, hi)
    if lo >= s0:
        return b
    a = solve(k, s0, lo)
    return tuple(np.concatenate([x[::-1], y[1:]]) for x, y in zip(a, b))


CASES = [
    ("(a)", lambda s: np.ones_like(np.asarray(s, float)), 0.0, 2 * np.pi, r"$\kappa_s=1$"),
    ("(b)", lambda s: np.asarray(s, float), -6.0, 6.0, r"$\kappa_s=s$"),
    ("(c)", lambda s: 2 * np.cos(s), 0.0, 4 * np.pi, r"$\kappa_s=2\cos s$"),
]

sols = [solve_both(k, lo, hi) for _, k, lo, hi, _ in CASES]

# 자기검사 ---------------------------------------------------------------------
s, th, x, y = sols[0]
assert np.allclose(np.hypot(x, y - 1), 1, atol=1e-9)                # 중심 (0, 1), 반지름 1인 원
assert abs(x[-1]) < 1e-9 and abs(y[-1]) < 1e-9                       # 한 바퀴 돌아 제자리
s, th, x, y = sols[1]
assert np.allclose(th, s ** 2 / 2, atol=1e-8)                        # θ(s) = s²/2 (연습 4.4.7)
i0 = np.argmin(np.abs(s))
assert np.allclose(x[::-1] + x, 0, atol=1e-8) and np.allclose(y[::-1] + y, 0, atol=1e-8)   # γ(−s) = −γ(s)
lim = np.sqrt(np.pi) / 2                                              # ∫_0^∞ cos(u²/2) du = ∫_0^∞ sin(u²/2) du = √π/2
assert np.hypot(x[-1] - lim, y[-1] - lim) < 1 / 6 + 0.02              # 꼬리는 대략 1/s 크기
for (s, th, x, y), (_, k, _, _, _) in zip(sols, CASES):
    sp_ = np.hypot(np.gradient(x, s), np.gradient(y, s))
    assert np.allclose(sp_[3:-3], 1, atol=1e-5)                      # 단위속력
    # 부호곡률을 곡선에서 다시 계산 (명제 4.4.5) → 주어진 k와 일치
    x1, y1 = np.gradient(x, s), np.gradient(y, s)
    x2, y2 = np.gradient(x1, s), np.gradient(y1, s)
    kap = (x1 * y2 - x2 * y1) / (x1 ** 2 + y1 ** 2) ** 1.5
    assert np.allclose(kap[5:-5], k(s)[5:-5], atol=2e-4)

fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 3.2], hspace=0.12, wspace=0.12,
                      left=0.04, right=0.99, top=0.93, bottom=0.02)
LIMS = [((-1.3, 1.3), (-0.3, 2.3)), ((-1.5, 1.5), (-1.5, 1.5)), ((-0.35, 3.2), (-0.3, 2.8))]
for j, ((tag, k, lo, hi, lab), (s, th, x, y), (xl, yl)) in enumerate(zip(CASES, sols, LIMS)):
    axk = fig.add_subplot(gs[0, j])
    axk.plot(s, k(s), color=C["main"], lw=1.2)
    axk.axhline(0, color=C["aux"], lw=0.5)
    axk.set_xlim(s[0], s[-1])
    axk.tick_params(labelsize=8, length=2)
    for sp_ in ("top", "right"):
        axk.spines[sp_].set_visible(False)
    axk.set_title(tag, loc="left", fontsize=10, pad=3)
    axk.set_title(lab, loc="right", fontsize=10, pad=3)
    if j == 0:
        axk.set_ylim(-0.4, 1.6)
    axk.set_xlabel(r"$s$", fontsize=9, labelpad=0)

    ax = fig.add_subplot(gs[1, j])
    dgfig.schematic_axes(ax, xl, yl)
    ax.plot(x, y, color=C["main"], lw=1.5, zorder=3)
    i0 = int(np.argmin(np.abs(s)))
    ax.plot(x[i0], y[i0], "o", color=C["accent"], ms=5, zorder=6)
    ax.annotate("", xy=(x[i0] + 0.5, y[i0]), xytext=(x[i0], y[i0]),
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.5, mutation_scale=10,
                                shrinkA=0, shrinkB=0), zorder=7)
dgfig.save(fig, __file__)
