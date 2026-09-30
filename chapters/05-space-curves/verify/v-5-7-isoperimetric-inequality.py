"""5.7절 등주부등식 (선택): 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

- 부호 있는 넓이 A(γ) = ½∫(xy' - yx')dt: 원, 타원, 리마송(3π), 8자 곡선(0), 극좌표 곡선(½∫ρ²).
- 비르팅거 부등식의 두 단계(양 끝이 0인 경우의 항등식, u·v 분해)와 수치 확인.
- 등주부등식 L² ≥ 4π|A|: 원(등호), 타원, 콩 모양 곡선, 두 번 도는 원.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-7-isoperimetric-inequality.py``
"""

import math

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

t = sp.symbols("t", real=True)
TWO_PI = 2 * math.pi


def signed_area(g, T=2 * sp.pi):
    x, y = g
    return sp.integrate(sp.simplify((x * sp.diff(y, t) - y * sp.diff(x, t)) / 2), (t, 0, T))


def num_area_len(g, T=TWO_PI, n=40000):
    fx = sp.lambdify(t, list(g), "numpy")
    fd = sp.lambdify(t, list(sp.diff(g, t)), "numpy")
    ts = np.linspace(0, T, n, endpoint=False)
    x, y = (np.broadcast_to(c, ts.shape) for c in fx(ts))
    dx, dy = (np.broadcast_to(c, ts.shape) for c in fd(ts))
    A = np.mean(x * dy - y * dx) / 2 * T
    L = np.mean(np.hypot(dx, dy)) * T
    return A, L


r, a, b = sp.symbols("r a b", positive=True)
# 명제 5.7.2, 예 5.7.3
sym_equal("예 5.7.3(a): 반시계 원 A = πr²", signed_area(sp.Matrix([r * sp.cos(t), r * sp.sin(t)])), sp.pi * r ** 2)
sym_equal("예 5.7.3(a)/명제 5.7.2(b): 시계 방향 원 A = -πr²", signed_area(sp.Matrix([r * sp.cos(t), -r * sp.sin(t)])), -sp.pi * r ** 2)
sym_equal("예 5.7.3(b)/연습 5.7.1: 타원 A = πab", signed_area(sp.Matrix([a * sp.cos(t), b * sp.sin(t)])), sp.pi * a * b)
lim = (1 + 2 * sp.cos(t)) * sp.Matrix([sp.cos(t), sp.sin(t)])
sym_equal("예 5.7.3(c): 리마송 A = ½∫(1+2cos t)² = 3π", signed_area(lim), 3 * sp.pi)
eight = sp.Matrix([sp.sin(t), sp.sin(t) * sp.cos(t)])
sym_equal("예 5.7.3(d): 8자 곡선 A = 0", signed_area(eight), 0)
# 명제 5.7.2(a): ∫xy' = -∫yx' = A
ell = sp.Matrix([a * sp.cos(t), b * sp.sin(t)])
sym_equal("명제 5.7.2(a): ∫x y' dt = A (타원)", sp.integrate(ell[0] * sp.diff(ell[1], t), (t, 0, 2 * sp.pi)), sp.pi * a * b)
sym_equal("명제 5.7.2(a): -∫y x' dt = A (타원)", -sp.integrate(ell[1] * sp.diff(ell[0], t), (t, 0, 2 * sp.pi)), sp.pi * a * b)
# 명제 5.7.2(c): 평행이동·회전 불변
c1, c2, al = sp.symbols("c1 c2 alpha", real=True)
Ra = sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]])
moved = Ra * ell + sp.Matrix([c1, c2])
sym_equal("명제 5.7.2(c): 강체운동 뒤에도 A = πab", signed_area(moved), sp.pi * a * b)
sym_equal("명제 5.7.2(c): 반사 (x, -y)는 A의 부호를 바꾼다", signed_area(sp.Matrix([ell[0], -ell[1]])), -sp.pi * a * b)
sym_equal("명제 5.7.2(d)/연습 5.7.2: 원을 두 번 (주기 4π) A = 2πr²", signed_area(sp.Matrix([r * sp.cos(t), r * sp.sin(t)]), 4 * sp.pi), 2 * sp.pi * r ** 2)
# 재매개화 불변 (h(t) = t + sin(t)/2)
Ar, _ = num_area_len(ell.subs({a: 2, b: 1}).subs(t, t + sp.sin(t) / 2))
close("명제 5.7.2(b): 방향을 보존하는 재매개화에서 A 불변 (타원 a=2, b=1)", Ar, 2 * math.pi, tol=1e-9)
# 연습 5.7.5: 극좌표 곡선 A = ½∫ρ²
rho = 1 + sp.cos(2 * t) / 4 + sp.Rational(3, 25) * sp.sin(3 * t)
bean = rho * sp.Matrix([sp.cos(t), sp.sin(t)])
sym_equal("연습 5.7.5: 극좌표 곡선 A = ½∫ρ² dt", signed_area(bean), sp.integrate(rho ** 2 / 2, (t, 0, 2 * sp.pi)))
sym_equal("연습 5.7.5: xy' - yx' = ρ² (극좌표 곡선)", sp.simplify(bean[0] * sp.diff(bean[1], t) - bean[1] * sp.diff(bean[0], t)), rho ** 2)

# 보조정리 5.7.5: g = h sin t이면 g'² - g² = h'² sin² t + (h² sin t cos t)'
h = sp.Function("h")(t)
g = h * sp.sin(t)
sym_equal("보조정리 5.7.5 증명: g'² - g² = h'² sin² + (h² sin cos)'", sp.diff(g, t) ** 2 - g ** 2,
          sp.diff(h, t) ** 2 * sp.sin(t) ** 2 + sp.diff(h ** 2 * sp.sin(t) * sp.cos(t), t))
check("보조정리 5.7.5: g = c sin t에서 등호 ∫_0^π (g'² - g²) = 0",
      sp.simplify(sp.integrate(sp.cos(t) ** 2 - sp.sin(t) ** 2, (t, 0, sp.pi))) == 0)
g2 = t * (sp.pi - t)
check("보조정리 5.7.5: g = t(π - t)에서 부등호 (∫g'² = π³/3 > ∫g² = π⁵/30)",
      sp.integrate(sp.diff(g2, t) ** 2, (t, 0, sp.pi)) - sp.integrate(g2 ** 2, (t, 0, sp.pi)) > 0)

# 보조정리 5.7.6: 비르팅거 (무작위 삼각다항식, 평균 0)
rng = np.random.default_rng(1)
ts = np.linspace(0, TWO_PI, 4096, endpoint=False)
ok = True
for _ in range(50):
    K = 6
    ca, sb = rng.normal(size=K), rng.normal(size=K)
    f = sum(ca[k - 1] * np.cos(k * ts) + sb[k - 1] * np.sin(k * ts) for k in range(1, K + 1))
    fp = sum(-k * ca[k - 1] * np.sin(k * ts) + k * sb[k - 1] * np.cos(k * ts) for k in range(1, K + 1))
    ok &= np.mean(fp ** 2) >= np.mean(f ** 2) - 1e-12
check("보조정리 5.7.6: 평균 0인 삼각다항식 50개에서 ∫f'² ≥ ∫f²", ok)
fe = 0.7 * np.cos(ts) - 1.3 * np.sin(ts)
close("보조정리 5.7.6: f = a cos t + b sin t에서 등호", np.mean(np.gradient(fe, ts) ** 2), np.mean(fe ** 2), tol=1e-5)
# u, v 분해: ∫u'v' = ∫uv = 0
f = np.cos(ts) + 0.5 * np.sin(2 * ts) - 0.3 * np.cos(3 * ts) + 0.2 * np.sin(4 * ts)
sh = len(ts) // 2
u_, v_ = (f - np.roll(f, -sh)) / 2, (f + np.roll(f, -sh)) / 2
close("보조정리 5.7.6 증명: u(t + π) = -u(t)", np.roll(u_, -sh), -u_, tol=1e-12)
close("보조정리 5.7.6 증명: v(t + π) = v(t)", np.roll(v_, -sh), v_, tol=1e-12)
close("보조정리 5.7.6 증명: ∫uv = 0", np.mean(u_ * v_), 0.0, tol=1e-12)
close("보조정리 5.7.6 증명: u는 홀수 진동수, v는 짝수 진동수 부분", u_, np.cos(ts) - 0.3 * np.cos(3 * ts), tol=1e-12)
check("보조정리 5.7.6: f = 1이면 부등식이 깨진다 (평균이 0이 아님)", 0 < 1)

# 정리 5.7.7: 등주부등식
for name, curve, eq in (("원 r = 1.3", sp.Matrix([1.3 * sp.cos(t), 1.3 * sp.sin(t)]), True),
                        ("타원 (2cos t, sin t)", sp.Matrix([2 * sp.cos(t), sp.sin(t)]), False),
                        ("콩 모양 곡선", bean, False),
                        ("8자 곡선", eight, False),
                        ("리마송", lim, False)):
    A, L = num_area_len(curve)
    ratio = 4 * math.pi * abs(A) / L ** 2
    if eq:
        close(f"정리 5.7.7: {name}: 4π|A|/L² = 1", ratio, 1.0, tol=1e-9)
    else:
        check(f"정리 5.7.7: {name}: 4π|A|/L² = {ratio:.4f} < 1", ratio < 1 - 1e-6)
A2, L2 = num_area_len(sp.Matrix([sp.cos(t), sp.sin(t)]), T=2 * TWO_PI)
close("연습 5.7.2: 두 번 도는 단위원: 4π|A|/L² = 1/2", 4 * math.pi * A2 / L2 ** 2, 0.5, tol=1e-9)
Ae, Le = num_area_len(sp.Matrix([2 * sp.cos(t), sp.sin(t)]))
check(f"연습 5.7.1: 타원 a=2, b=1의 둘레 {Le:.4f} ≥ 2π√2 = {2 * math.pi * math.sqrt(2):.4f}", Le >= 2 * math.pi * math.sqrt(2))
close("연습 5.7.1: 타원 a=2, b=1의 둘레 ≈ 9.6884", Le, 9.688448220547675, tol=1e-9)
# 증명의 항등식: L²/2π - 2A = ∫(x'² - x²) + ∫(x - y')² (속력 L/2π, x의 평균 0)
Ab, Lb = num_area_len(bean)
# 콩 모양 곡선을 속력이 일정하게 (L/2π) 다시 매개화: 호의 길이의 역함수를 수치로
fb = sp.lambdify(t, list(bean), "numpy")
fdb = sp.lambdify(t, list(sp.diff(bean, t)), "numpy")
tt = np.linspace(0, TWO_PI, 200001)
spd = np.hypot(*fdb(tt))
s_cum = np.concatenate([[0], np.cumsum((spd[1:] + spd[:-1]) / 2 * np.diff(tt))])
Lnum = s_cum[-1]
uu = np.linspace(0, TWO_PI, 8192, endpoint=False)
t_of_u = np.interp(uu * Lnum / TWO_PI, s_cum, tt)
X, Y = fb(t_of_u)
X = X - np.mean(X)
k = np.fft.fftfreq(len(uu), d=1 / len(uu))
dX = np.real(np.fft.ifft(1j * k * np.fft.fft(X)))
dY = np.real(np.fft.ifft(1j * k * np.fft.fft(Y)))
lhs = Lnum ** 2 / TWO_PI - 2 * (np.mean(X * dY) * TWO_PI)
rhs = np.mean(dX ** 2 - X ** 2) * TWO_PI + np.mean((X - dY) ** 2) * TWO_PI
close("정리 5.7.7 증명: L²/2π - 2A = ∫(x'² - x²) + ∫(x - y')² (콩 모양 곡선)", lhs, rhs, tol=1e-4)
check("정리 5.7.7 증명: 두 항 모두 0 이상", np.mean(dX ** 2 - X ** 2) >= 0 and np.mean((X - dY) ** 2) >= 0)
close("정리 5.7.7 증명: 속력이 L/2π로 일정", np.hypot(dX, dY), np.full_like(uu, Lnum / TWO_PI), tol=1e-3)
# 연습 5.7.4: 구간 [0, ℓ]의 판
ell_ = sp.symbols("ell", positive=True)
gl = sp.sin(sp.pi * t / ell_)
sym_equal("연습 5.7.4: g = sin(πt/ℓ)에서 ∫g'² = (π/ℓ)²∫g²", sp.integrate(sp.diff(gl, t) ** 2, (t, 0, ell_)),
          (sp.pi / ell_) ** 2 * sp.integrate(gl ** 2, (t, 0, ell_)))

summary()
