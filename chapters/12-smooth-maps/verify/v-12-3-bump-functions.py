"""12.3절 범프 함수: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

f(t) = e^{-1/t} (t > 0), 0 (t ≤ 0)은 예 2.2.20의 함수다. 기호 계산은 t > 0 쪽에서, 경계 동작은 극한으로 확인한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/verify/v-12-3-bump-functions.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

t, s = sp.symbols("t s", real=True)
tp = sp.symbols("t", positive=True)


def f_sym(x):
    return sp.Piecewise((sp.exp(-1 / x), x > 0), (0, True))


def f_num(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = np.exp(-1.0 / x[m])
    return out


def h_num(x):
    a, b = f_num(2 - np.asarray(x, float)), f_num(np.asarray(x, float) - 1)
    return a / (a + b)


def lam_num(x):
    return 1 - h_num(1 + np.asarray(x, float))


# ---------------------------------------------------------------- 보조정리 12.3.1
ts = np.linspace(-5, 8, 13001)
check("보조정리 12.3.1: 분모 f(2-t) + f(t-1) > 0 (수치, -5 ≤ t ≤ 8)", np.all(f_num(2 - ts) + f_num(ts - 1) > 0))
check("보조정리 12.3.1: t ≤ 1이면 h = 1", np.all(h_num(ts[ts <= 1]) == 1))
check("보조정리 12.3.1: t ≥ 2이면 h = 0", np.all(h_num(ts[ts >= 2]) == 0))
mid = (ts > 1.05) & (ts < 1.95)
check("보조정리 12.3.1: 1 < t < 2이면 0 < h < 1 (1.05 ≤ t ≤ 1.95, 수치)", np.all((h_num(ts[mid]) > 0) & (h_num(ts[mid]) < 1)))
# 1 < t < 2에서의 기호식: h = e^{-1/(2-t)}/(e^{-1/(2-t)} + e^{-1/(t-1)}) 은 양수이고 1보다 작다
w1, w2 = sp.symbols("w1 w2", positive=True)   # w1 = 2 - t > 0, w2 = t - 1 > 0
a_, b_ = sp.exp(-1 / w1), sp.exp(-1 / w2)
check("보조정리 12.3.1: 1 < t < 2에서 두 항이 양수 (기호)", a_.is_positive and b_.is_positive)
check("보조정리 12.3.1: 1 < t < 2에서 h < 1 (분자 < 분모, 기호)", sp.simplify(a_ + b_ - a_).is_positive)
# 경계에서 매끄럽게 이어짐: t → 1+ 에서 1 - h의 모든 계 도함수 → 0 (예: 3계까지)
u = sp.symbols("u", positive=True)          # t = 1 + u
one_minus_h = sp.exp(-1 / u) / (sp.exp(-1 / (1 - u)) + sp.exp(-1 / u))
for kk in range(4):
    d = sp.diff(one_minus_h, u, kk)
    lim = sp.limit(d, u, 0, "+")
    check(f"보조정리 12.3.1: t → 1+에서 (1-h)^({kk}) → 0", lim == 0)

# λ(s) = 1 - h(1 + s)
ss = np.linspace(-3, 4, 7001)
check("본문: λ(s) = 0 (s ≤ 0), λ(s) = 1 (s ≥ 1), 0 ≤ λ ≤ 1",
      np.all(lam_num(ss[ss <= 0]) == 0) and np.all(lam_num(ss[ss >= 1]) == 1) and np.all((lam_num(ss) >= 0) & (lam_num(ss) <= 1)))
check("본문: s > 0.05이면 λ(s) > 0 (수치)", np.all(lam_num(ss[ss > 0.05]) > 0))

# ---------------------------------------------------------------- 연습 12.3.1: h(3 - t) = 1 - h(t), h(3/2) = 1/2
close("연습 12.3.1: h(3-t) = 1 - h(t) (수치)", h_num(3 - ts), 1 - h_num(ts), 1e-14)
close("연습 12.3.1: h(3/2) = 1/2", h_num(np.array([1.5]))[0], 0.5, 1e-15)

# ---------------------------------------------------------------- 연습 12.3.4: q_1, q_2와 점화식
fp = sp.exp(-1 / tp)
q = [sp.Integer(1)]
for kk in range(4):
    q.append(sp.expand(s ** 2 * (q[-1] - sp.diff(q[-1], s))))
sym_equal("연습 12.3.4: q_1(s) = s²", q[1], s ** 2)
sym_equal("연습 12.3.4: q_2(s) = s⁴ - 2s³", q[2], s ** 4 - 2 * s ** 3)
for kk in range(1, 5):
    sym_equal(f"예 2.2.20의 점화식: f^({kk})(t) = q_{kk}(1/t) e^(-1/t) (t > 0)", sp.diff(fp, tp, kk), q[kk].subs(s, 1 / tp) * fp)

# ---------------------------------------------------------------- 보조정리 12.3.4: H(x) = h(1 + (|x-c| - r1)/(r2 - r1))
rng = np.random.default_rng(123)
for n in (1, 2, 3):
    c = rng.normal(size=n)
    r1, r2 = 0.7, 1.9
    P = c + rng.uniform(-3, 3, (6000, n))
    rho = np.linalg.norm(P - c, axis=1)
    H = h_num(1 + (rho - r1) / (r2 - r1))
    check(f"보조정리 12.3.4: |x-c| ≤ r1이면 H = 1 (n = {n})", np.all(H[rho <= r1] == 1))
    check(f"보조정리 12.3.4: |x-c| ≥ r2이면 H = 0 (n = {n})", np.all(H[rho >= r2] == 0))
    m = (rho > r1 + 0.05) & (rho < r2 - 0.05)
    check(f"보조정리 12.3.4: r1 < |x-c| < r2이면 0 < H < 1 (n = {n}, 수치)", np.all((H[m] > 0) & (H[m] < 1)))
# 괄호 안의 수: |x-c| = r1이면 1, r2이면 2
r1s, r2s, d = sp.symbols("r1 r2 d", positive=True)
sarg = 1 + (d - r1s) / (r2s - r1s)
check("보조정리 12.3.4: s(r1) = 1, s(r2) = 2", sp.simplify(sarg.subs(d, r1s) - 1) == 0 and sp.simplify(sarg.subs(d, r2s) - 2) == 0)

# ---------------------------------------------------------------- 예 12.3.9: S² 위의 ψ(x) = λ(2x³)
z = np.linspace(-1, 1, 20001)
psi = lam_num(2 * z)
check("예 12.3.9: x³ ≥ 1/2이면 ψ = 1", np.all(psi[z >= 0.5] == 1))
check("예 12.3.9: x³ ≤ 0이면 ψ = 0", np.all(psi[z <= 0] == 0))
check("예 12.3.9: x³ > 0.03이면 ψ > 0 (수치)", np.all(psi[z > 0.03] > 0))
check("예 12.3.9: ψ ≠ 0이면 x³ > 0 (따라서 받침 ⊆ {x³ ≥ 0} ⊆ U = {x³ > -1/2})", np.all(z[psi != 0] > 0))

# ---------------------------------------------------------------- 예 12.3.11: 서로소인 닫힌공
eps = 1.0
cs = [eps * 2.0 ** (-j) for j in range(1, 30)]
ss_ = [eps * 2.0 ** (-j - 2) for j in range(1, 30)]
ok = True
for i in range(len(cs)):
    ok &= cs[i] + ss_[i] < eps
    for j in range(i + 1, len(cs)):
        ok &= abs(cs[i] - cs[j]) > ss_[i] + ss_[j]
        ok &= abs(cs[i] - cs[j]) >= eps * 2.0 ** (-(i + 1) - 1) - 1e-15
check("예 12.3.11: B̄_{s_j}(c_j)들은 B_ε(x_0) 안에서 서로소 (j ≤ 29)", ok)

# ---------------------------------------------------------------- 연습 12.3.2(a): supp f(t)f(1-t) = [0, 1]
g = f_num(ts) * f_num(1 - ts)
nz = ts[g > 0]
check("연습 12.3.2(a): {f(t)f(1-t) ≠ 0} ⊆ (0, 1)", nz.min() > 0 and nz.max() < 1)
check("연습 12.3.2(a): 0 < t < 1에서 f(t)f(1-t) > 0 (0.02 ≤ t ≤ 0.98)", np.all(g[(ts > 0.02) & (ts < 0.98)] > 0))

summary()
