"""12.4절 컴팩트 소진과 단위분할: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

정리 12.4.12의 증명을 ℝ과 ℝ² ∖ {0}의 구체적인 덮개에 대해 수치로 따라가 본다(독립 검산).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/verify/v-12-4-partitions-of-unity.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary


def f(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = np.exp(-1.0 / t[m])
    return out


def h(t):
    a, b = f(2 - np.asarray(t, float)), f(np.asarray(t, float) - 1)
    return a / (a + b)


def k(s):
    return 1 - h(1 + np.asarray(s, float))


def H(x, c, r1, r2):
    """보조정리 12.3.4의 범프 (x: (..., n) 배열)."""
    rho = np.linalg.norm(np.atleast_2d(x) - c, axis=-1)
    return h(1 + (rho - r1) / (r2 - r1))


# ---------------------------------------------------------------- 비예 12.4.4: 받침이 0에 쌓이는 합
ks = np.arange(1, 200)
c_ = 1.0 / ks
s_ = 1.0 / (4 * ks ** 2)
check("비예 12.4.4: |c_k - c_{k+1}| > s_k + s_{k+1} (받침 서로소)", np.all(np.abs(c_[:-1] - c_[1:]) > s_[:-1] + s_[1:]))
Fvals = np.array([sum(H(np.array([[ck]]), np.array([cj]), sj / 2, sj)[0] for cj, sj in zip(c_[:60], s_[:60])) for ck in c_[:60]])
close("비예 12.4.4: F(1/k) = 1 (k ≤ 60)", Fvals, np.ones(60), 1e-12)
check("비예 12.4.4: F(0) = 0", all(H(np.array([[0.0]]), np.array([cj]), sj / 2, sj)[0] == 0 for cj, sj in zip(c_, s_)))

# ---------------------------------------------------------------- 예 12.4.6(b), 연습 12.4.4: ℝ의 구간 덮개
b = lambda s: h(0.5 + 2 * np.abs(np.asarray(s, float)))
t = np.linspace(-5.3, 5.3, 21201)
kk = np.arange(-9, 10)
g = np.array([b(t - q) for q in kk])
G = g.sum(axis=0)
psi = g / G
check("예 12.4.6(b): b = h(1 + (|s| - 1/4)/(1/2)) = h(1/2 + 2|s|)", np.allclose(h(1 + (np.abs(t) - 0.25) / 0.5), b(t)))
check("예 12.4.6(b): supp g_k ⊆ [k - 3/4, k + 3/4] ⊆ U_k", all(np.all(gq[np.abs(t - q) >= 0.75] == 0) for q, gq in zip(kk, g)))
check("예 12.4.6(b): G > 0 (가장 가까운 정수 k에서 g_k > 0)", np.all(G > 0))
close("예 12.4.6(b): Σψ_k = 1", psi.sum(axis=0), np.ones_like(t), 1e-12)
check("예 12.4.6(b): 각 점 근처에서 0이 아닌 g_k는 많아야 3개 ((p-1/4, p+1/4)와 만나는 받침)",
      np.all(np.array([np.sum(np.abs(q - p) < 0.75 + 0.25) for p in np.linspace(-4, 4, 801) for q in [kk]]) <= 3))
mask = np.abs(t) <= 4
Gshift = np.array([b(t + 1 - q) for q in kk]).sum(axis=0)
close("연습 12.4.4: G(t + 1) = G(t)", Gshift[mask], G[mask], 1e-12)
i0 = list(kk).index(0)
one = np.isclose(psi[i0], 1.0, atol=0, rtol=0)
# |t|가 1/4보다 조금 크면 g_{±1}(t) ~ e^{-1/ε}가 부동소수에서 0으로 내려가므로 경계 근처 0.05는 뺀다
check("연습 12.4.4: ψ_0 = 1 ⇔ |t| ≤ 1/4 (수치, 경계 근처 0.05 제외)",
      np.all(one[np.abs(t) <= 0.25]) and not np.any(one[(np.abs(t) > 0.30) & (np.abs(t) < 2)]))

# ---------------------------------------------------------------- 예 12.4.6(c): S^n의 두 영역
rho = lambda x: k(x + 0.5)
z = np.linspace(-1, 1, 20001)
psi2 = rho(z)
psi1 = 1 - psi2
check("예 12.4.6(c): ρ(t) = 0 (t ≤ -1/2), 1 (t ≥ 1/2)", np.all(rho(z[z <= -0.5]) == 0) and np.all(rho(z[z >= 0.5]) == 1))
check("예 12.4.6(c): ψ₂ ≠ 0 ⇒ x^{n+1} > -1/2 (supp ψ₂ ∌ S)", np.all(z[psi2 != 0] > -0.5))
check("예 12.4.6(c): ψ₁ ≠ 0 ⇒ x^{n+1} < 1/2 (supp ψ₁ ∌ N)", np.all(z[psi1 != 0] < 0.5))
close("예 12.4.6(c): ψ₁ + ψ₂ = 1", psi1 + psi2, np.ones_like(z), 0)

# ---------------------------------------------------------------- 예 12.4.9(b), 예 12.4.10, 연습 12.4.2: 컴팩트 소진
ok = all(1 / (j + 2) < 1 / (j + 1) and j < j + 1 for j in range(1, 1000))
check("예 12.4.10: 반지름 비교 V_j ⊆ K_j ⊆ V_{j+1} (j < 1000)", ok)
rng = np.random.default_rng(124)
r = np.exp(rng.uniform(-8, 5, 5000))
jj = np.maximum(np.maximum(np.ceil(r), np.ceil(1 / r) - 1), 1)
check("예 12.4.10: 모든 x ≠ 0은 어떤 K_j에 속함 (표본)", np.all((1 / (jj + 1) <= r) & (r <= jj)))
x01 = rng.uniform(1e-6, 1 - 1e-6, 5000)
jx = np.maximum(np.ceil(1 / np.minimum(x01, 1 - x01)) - 2, 1)
check("연습 12.4.2: (0,1)의 모든 점은 K_j = [1/(j+2), 1-1/(j+2)]에 속함 (표본)",
      np.all((1 / (jx + 2) <= x01) & (x01 <= 1 - 1 / (jx + 2))))
check("연습 12.4.2: K_j ⊆ V_{j+1} (1/(j+3) < 1/(j+2))", all(1 / (j + 3) < 1 / (j + 2) for j in range(1, 1000)))

# ---------------------------------------------------------------- 명제 12.4.11의 구성 (ℝ, W_i = (i-2, i) 류의 덮개)
# 폐포가 컴팩트한 열린집합들 W_i = (-i, i)의 덮개에서 K_j, V_j를 증명대로 만든다.
W = [(-(i + 1) / 2.0, (i + 1) / 2.0) for i in range(1, 60)]
m = [1]
Ks = []
for j in range(1, 20):
    lo = min(W[q][0] for q in range(m[-1])); hi = max(W[q][1] for q in range(m[-1]))
    Ks.append((lo, hi))                                  # K_j = W̄_1 ∪ … ∪ W̄_{m_j}
    need = next(q for q in range(1, len(W)) if max(W[i][1] for i in range(q)) > hi and min(W[i][0] for i in range(q)) < lo)
    m.append(max(need, m[-1] + 1))
check("명제 12.4.11: m_j는 순증가, m_j ≥ j", all(m[i] < m[i + 1] for i in range(len(m) - 1)) and all(m[j - 1] >= j for j in range(1, len(m) + 1)))
check("명제 12.4.11: K_j ⊆ V_{j+1} (구간 비교)",
      all(Ks[j][1] < max(W[i][1] for i in range(m[j + 1])) and Ks[j][0] > min(W[i][0] for i in range(m[j + 1])) for j in range(len(Ks) - 1)))

# ---------------------------------------------------------------- 정리 12.4.12의 구성을 ℝ에서 수치로 따라가기
# 덮개 U_α = (α - 0.7, α + 0.7), α ∈ 0.5ℤ,  소진 K_j = [-j, j], V_j = (-j, j)
t = np.linspace(-6.0, 6.0, 12001)
alphas = np.arange(-16, 17) / 2.0
funcs = []                                               # (중심 p, 반지름 ε/2, α(p), 단계 j)
for j in range(1, 7):
    Klo, Khi = -j, j
    Vprev = j - 1
    A = [p for p in np.arange(-j, j + 1e-9, 0.05) if abs(p) >= Vprev]      # A_j = K_j ∖ V_{j-1}의 표본점
    for p in A:
        al = alphas[np.argmin(np.abs(alphas - p))]
        # U_α ∩ O_j 안에 들어가는 반지름 ε: O_j = V_{j+1} ∖ K_{j-2} = (-(j+1), j+1) ∖ [-(j-2), j-2]
        eps = min(0.7 - abs(p - al), (j + 1) - abs(p), (abs(p) - max(j - 2, 0)) if j >= 3 else np.inf)
        funcs.append((p, eps / 2, al, j))
gs = np.array([H(t[:, None], np.array([p]), r / 2, r) for (p, r, al, j) in funcs])
check("정리 12.4.12 2단계: 각 g_i의 받침 ⊆ U_α(i) ∩ O_j",
      all(np.all(np.abs(t[gi > 0] - al) < 0.7) and np.all(np.abs(t[gi > 0]) < j + 1) and (j < 3 or np.all(np.abs(t[gi > 0]) > j - 2))
          for gi, (p, r, al, j) in zip(gs, funcs)))
Gt = gs.sum(axis=0)
inside = np.abs(t) <= 5.9
check("정리 12.4.12 4단계: G ≥ 1 (|t| ≤ 5.9)", np.all(Gt[inside] >= 1 - 1e-12))
cnt = np.array([np.sum([(abs(x - p) < r) for (p, r, al, j) in funcs]) for x in np.linspace(-5, 5, 201)])
check("정리 12.4.12 3단계: 각 점에서 0이 아닌 g_i는 유한개 (|t| ≤ 5)", np.all(cnt < len(funcs)))
psis = {}
for gi, (p, r, al, j) in zip(gs, funcs):
    psis.setdefault(al, np.zeros_like(t))
    psis[al] += gi / np.where(Gt > 0, Gt, 1)
total = sum(psis.values())
close("정리 12.4.12 5단계: Σ_α ψ_α = 1 (|t| ≤ 5.9)", total[inside], np.ones(inside.sum()), 1e-12)
check("정리 12.4.12 5단계: supp ψ_α ⊆ U_α", all(np.all(np.abs(t[(v > 0) & inside] - al) < 0.7) for al, v in psis.items()))

# ---------------------------------------------------------------- 연습 12.4.3(c): 볼록결합도 단위분할
# 예 12.4.6(b)의 ψ_k와, 반 칸 옮긴 덮개에서 만든 ψ'_k(받침 ⊆ U_k가 되도록 폭을 줄인 범프)의 볼록결합
b2 = lambda s_: h(1 + (np.abs(np.asarray(s_, float)) - 0.3) / 0.35)          # 받침 [-0.65, 0.65]
t0 = np.linspace(-5.3, 5.3, 21201)
g1 = np.array([b(t0 - q) for q in kk]); psi1_ = g1 / g1.sum(axis=0)
g2 = np.array([b2(t0 - q) for q in kk]); psi2_ = g2 / g2.sum(axis=0)
mix = 0.3 * psi1_ + 0.7 * psi2_
close("연습 12.4.3(c): 두 단위분할의 볼록결합의 합 = 1", mix.sum(axis=0), np.ones_like(t0), 1e-12)
check("연습 12.4.3(c): 볼록결합의 받침 ⊆ U_k", all(np.all(mq[np.abs(t0 - q) >= 0.75] == 0) for q, mq in zip(kk, mix)))

summary()
