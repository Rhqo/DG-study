"""5.6절 회전지수와 접선회전정리: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

- 각 함수 ∠(u, v) = 2 arctan(det(u, v)/(1 + <u, v>))의 성질 (보조정리 5.6.6).
- 회전지수 (1/2π)∫κ_s|γ'|dt의 값: 원, 타원, 리마송, 8자 곡선, ρ = 1 + λ cos t 가족 (예 5.6.5, 연습문제).
- 접선회전정리의 증명 구조를 수치로 재현: 할선 사상 ψ를 삼각형의 경계에서 들어 올려 변 AB, BC에서 각각 π,
  대각선에서 2π·(회전지수)가 됨 (정리 5.6.7). 부호곡률은 dgsym.signed_curvature(n_s = J t 규약)로 계산한다.
- 곡선다각형 (정리 5.6.11): 곧은 삼각형, 오목한 곡선 삼각형(가장 낮은 점이 꼭짓점), 뢸로 삼각형에서 Σ∫κ_s + Σε = ±2π.
- 양의 방향이면 +1 (명제 5.6.12): 오목한 곡선 삼각형(h = 0.25, 0.12)을 평면에서 5°씩 돌려도 가장 낮은 점은 꼭짓점이고,
  영역이 왼쪽인 방향(그린 정리의 넓이 부호 > 0, 매끄러운 점의 왼쪽 점의 감는 수 1)은 +2π, 반대 방향은 -2π.
  반대 방향 곡선에서 증명의 4-8단계(α, β, δ, 가르는 벡터 m, 왼쪽 법선 반직선이 곡선을 만나지 않고 y < y0로 들어감)를
  수치로 따라가고, 증명의 삼각함수 계산을 sympy로 확인한다. 연습 5.6.8(원의 방향, 뒤집으면 부호가 바뀜).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-6-rotation-index.py``
"""

import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

t = sp.symbols("t", real=True)
TWO_PI = 2 * math.pi


def ang(u, v):
    """∠(u, v) = 2 arctan(det(u, v)/(1 + <u, v>)), <u, v> > -1 (식 (5.6.2))."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    det = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
    return 2 * np.arctan(det / (1 + dot))


def rot(phi):
    return np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])


def rotation_index(expr, T, n=20000):
    """(1/2π)∫_0^T κ_s|γ'| dt, 주기함수의 사다리꼴 적분 (스펙트럼 정확도)."""
    ks = dgsym.signed_curvature(expr, t)
    sp_ = sp.sqrt(sp.diff(expr[0], t) ** 2 + sp.diff(expr[1], t) ** 2)
    f = sp.lambdify(t, ks * sp_, "numpy")
    ts = np.linspace(0, T, n, endpoint=False)
    return np.mean(np.broadcast_to(f(ts), ts.shape)) * T / TWO_PI


# 보조정리 5.6.6(a): ∠의 성질 ------------------------------------------------------------------
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    a, phi = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi + 1e-3, math.pi - 1e-3)
    u = np.array([math.cos(a), math.sin(a)])
    v = rot(phi) @ u
    worst = max(worst, abs(ang(u, v) - phi))
close("보조정리 5.6.6(a): v = R_φ u (|φ| < π)이면 ∠(u, v) = φ (무작위 200개)", worst, 0.0, tol=1e-12)
e1, e2 = np.array([1.0, 0]), np.array([0, 1.0])
close("정리 5.6.7 증명: ∠(e2, -e1) = π/2, ∠(e2, e1) = -π/2", (ang(e2, -e1), ang(e2, e1)), (math.pi / 2, -math.pi / 2), tol=1e-15)
close("정리 5.6.7 증명: ∠(-e2, e1) = π/2, ∠(-e2, -e1) = -π/2", (ang(-e2, e1), ang(-e2, -e1)), (math.pi / 2, -math.pi / 2), tol=1e-15)
check("보조정리 5.6.6 증명: |u - v| < 1이면 <u, v> > 1/2", all(
    (lambda u, v: (np.linalg.norm(u - v) >= 1) or (u @ v > 0.5))(*(np.array([math.cos(x), math.sin(x)]) for x in rng.uniform(-4, 4, 2)))
    for _ in range(500)))
# 반각 공식 sin φ/(1 + cos φ) = tan(φ/2)
ph = sp.symbols("phi", real=True)
sym_equal("보조정리 5.6.6(a) 증명: sin φ/(1 + cos φ) = tan(φ/2)", sp.sin(ph) / (1 + sp.cos(ph)), sp.tan(ph / 2), {ph: (-3, 3)})

# 예 5.6.5: 회전지수 ------------------------------------------------------------------------
r = 1.3
close("예 5.6.5(a): 반시계 원의 회전지수 = 1", rotation_index(sp.Matrix([r * sp.cos(t), r * sp.sin(t)]), TWO_PI), 1.0, tol=1e-10)
close("예 5.6.5(a): 시계 방향 원의 회전지수 = -1", rotation_index(sp.Matrix([r * sp.cos(t), -r * sp.sin(t)]), TWO_PI), -1.0, tol=1e-10)
close("예 5.6.5(a)/연습 5.6.1: 주기 4π로 본 원 = 2", rotation_index(sp.Matrix([r * sp.cos(t), r * sp.sin(t)]), 2 * TWO_PI), 2.0, tol=1e-10)
close("연습 5.6.1: (cos 3t, sin 3t), 주기 2π = 3", rotation_index(sp.Matrix([sp.cos(3 * t), sp.sin(3 * t)]), TWO_PI), 3.0, tol=1e-10)
close("예 5.6.5(b): 타원 (2cos t, sin t) = 1", rotation_index(sp.Matrix([2 * sp.cos(t), sp.sin(t)]), TWO_PI), 1.0, tol=1e-10)
close("연습 5.6.1: 타원 (2cos t, -sin t) = -1", rotation_index(sp.Matrix([2 * sp.cos(t), -sp.sin(t)]), TWO_PI), -1.0, tol=1e-10)
lim = (1 + 2 * sp.cos(t)) * sp.Matrix([sp.cos(t), sp.sin(t)])
close("예 5.6.5(c): 리마송 ρ = 1 + 2cos t = 2 (연습 4.4.8)", rotation_index(lim, TWO_PI), 2.0, tol=1e-9)
eight = sp.Matrix([sp.sin(t), sp.sin(t) * sp.cos(t)])
close("예 5.6.5(d)/연습 5.6.2: 8자 곡선 (sin t, sin t cos t) = 0", rotation_index(eight, TWO_PI), 0.0, tol=1e-10)
sp8 = sp.diff(eight, t)
check("예 5.6.2(d): 8자 곡선은 정칙 (γ' = (cos t, cos 2t) ≠ 0)",
      sp.simplify(sp8 - sp.Matrix([sp.cos(t), sp.cos(2 * t)])) == sp.zeros(2, 1))
check("연습 5.6.2: γ(t + π) = ρ(γ(t)), ρ(x, y) = (-x, y)",
      sp.simplify(eight.subs(t, t + sp.pi) - sp.Matrix([-eight[0], eight[1]])) == sp.zeros(2, 1))
liss = sp.Matrix([sp.cos(t), sp.sin(2 * t)])
close("연습 5.6.2: (cos t, sin 2t)의 회전지수 = 0", rotation_index(liss, TWO_PI), 0.0, tol=1e-10)
check("연습 5.6.2: (cos t, sin 2t)는 γ(t + π) = (-x, y)", sp.simplify(liss.subs(t, t + sp.pi) - sp.Matrix([-liss[0], liss[1]])) == sp.zeros(2, 1))
check("연습 5.6.2: (cos t, sin 2t)는 γ(π/2) = γ(3π/2) (단순하지 않다)", sp.simplify(liss.subs(t, sp.pi / 2) - liss.subs(t, 3 * sp.pi / 2)) == sp.zeros(2, 1))
# 예 5.6.5(b): ∫_0^{2π} dt/(A² sin² t + B² cos² t) = 2π/(AB)
Aa, Bb = 2.0, 1.0
xs_ = np.linspace(0, TWO_PI, 20000, endpoint=False)
close("예 5.6.5(b): ∫ dt/(A²sin²t + B²cos²t) = 2π/(AB) (A = 2, B = 1)", np.mean(1 / (Aa ** 2 * np.sin(xs_) ** 2 + Bb ** 2 * np.cos(xs_) ** 2)) * TWO_PI, TWO_PI / (Aa * Bb), tol=1e-10)
# 명제 5.6.4: 출발점과 재매개화
bean = (1 + sp.Rational(1, 4) * sp.cos(2 * t) + sp.Rational(3, 25) * sp.sin(3 * t)) * sp.Matrix([sp.cos(t), sp.sin(t)])
ks_b = dgsym.signed_curvature(bean, t)
v_b = sp.sqrt(sp.diff(bean[0], t) ** 2 + sp.diff(bean[1], t) ** 2)
fb = sp.lambdify(t, ks_b * v_b, "numpy")
tt = np.linspace(0, TWO_PI, 20000, endpoint=False)
close("명제 5.6.4(b): 출발점을 옮겨도 같다 (∫_c^{c+T}, c = 1.234)", np.mean(fb(tt + 1.234)), np.mean(fb(tt)), tol=1e-12)
hre = t + sp.sin(t) / 2                                # h' = 1 + cos t/2 > 0, h(t + 2π) = h(t) + 2π
ell = sp.Matrix([2 * sp.cos(t), sp.sin(t)])
close("명제 5.6.4(c): 방향을 보존하는 재매개화 h(t) = t + sin(t)/2에서 같다 (타원)",
      rotation_index(ell.subs(t, hre), TWO_PI), rotation_index(ell, TWO_PI), tol=1e-9)
close("명제 5.6.4(c): 방향을 뒤집으면 부호가 바뀐다", rotation_index(bean.subs(t, -t), TWO_PI), -rotation_index(bean, TWO_PI), tol=1e-10)
close("명제 5.6.4(d): 반사 (x, -y)는 부호를 바꾼다", rotation_index(sp.Matrix([bean[0], -bean[1]]), TWO_PI), -1.0, tol=1e-9)
close("정리 5.6.7: 콩 모양 단순 닫힌 곡선의 회전지수 = 1", rotation_index(bean, TWO_PI), 1.0, tol=1e-9)

# 정리 5.6.7 증명의 수치 재현: 할선 사상의 들어올림 ----------------------------------------------------
# 콩 모양 곡선을 호의 길이로 바꾸는 대신 매개변수 t로 삼각형을 만든다 (명제 5.6.4에서 같은 결과).
gxy = sp.lambdify(t, list(bean), "numpy")
dgxy = sp.lambdify(t, list(sp.diff(bean, t)), "numpy")
ys = np.array(gxy(tt))[1]
t_low = tt[np.argmin(ys)]
# 가장 낮은 점을 정확히: y'(t) = 0의 근을 뉴턴법으로
yd = sp.lambdify(t, sp.diff(bean[1], t), "numpy")
ydd = sp.lambdify(t, sp.diff(bean[1], t, 2), "numpy")
for _ in range(30):
    t_low -= yd(t_low) / ydd(t_low)
G = lambda s: np.array(gxy(np.asarray(s) + t_low))
DG = lambda s: np.array(dgxy(np.asarray(s) + t_low))
v0 = DG(0.0)
check("정리 5.6.7: 가장 낮은 점에서 γ'은 +e1 방향 (반시계 곡선)", abs(v0[1]) < 1e-12 and v0[0] > 0)
check("정리 5.6.7: 모든 점의 y ≥ 가장 낮은 점의 y", np.min(np.array(gxy(tt))[1]) >= G(0.0)[1] - 1e-12)
Lp = TWO_PI


def psi(s1, s2):
    if abs(s2 - s1) < 1e-12:
        d = DG(s1)
    elif abs(s1) < 1e-12 and abs(s2 - Lp) < 1e-12:
        d = -DG(0.0)
    else:
        d = G(s2) - G(s1)
    return d / np.linalg.norm(d)


def lift_along(points):
    vals = np.array([psi(*p) for p in points])
    steps = ang(vals[:-1], vals[1:])
    assert np.max(np.abs(steps)) < 0.5                    # 촘촘하게 잡았는지
    return np.sum(steps)


m = 4000
AB = [(0.0, x) for x in np.linspace(0, Lp, m)]
BC = [(x, Lp) for x in np.linspace(0, Lp, m)]
AC = [(x, x) for x in np.linspace(0, Lp, m)]
dAB, dBC, dAC = lift_along(AB), lift_along(BC), lift_along(AC)
close("정리 5.6.7 증명 4단계: 변 AB에서 각의 변화 = π", dAB, math.pi, tol=1e-6)
close("정리 5.6.7 증명 4단계: 변 BC에서 각의 변화 = π", dBC, math.pi, tol=1e-6)
close("정리 5.6.7 증명 3단계: 대각선에서 각의 변화 = 2π·(회전지수) = 2π", dAC, TWO_PI, tol=1e-6)
yAB = np.array([psi(*p) for p in AB[1:-1]])[:, 1]
yBC = np.array([psi(*p) for p in BC[1:-1]])[:, 1]
check("정리 5.6.7 증명: 변 AB에서 ψ의 y성분 ≥ 0", np.min(yAB) >= -1e-12)
check("정리 5.6.7 증명: 변 BC에서 ψ의 y성분 ≤ 0", np.max(yBC) <= 1e-12)
# 삼각형 내부의 격자에서 보조정리 5.6.6의 사슬 공식으로 들어 올린 θ가 cos/sin을 맞추는지 (몇 점)
for (x, y) in ((1.0, 5.0), (2.5, 3.0), (0.3, 6.0), (4.0, 4.3)):
    N = 400
    pts = [(x * j / N, 0.0 + (y - 0.0) * j / N) for j in range(N + 1)]   # A = (0, 0)에서 선분
    th = lift_along(pts)
    close(f"보조정리 5.6.6: 사슬 공식의 θ는 ψ의 각 ((s1, s2) = ({x}, {y}))",
          np.array([math.cos(th), math.sin(th)]), psi(x, y), tol=1e-9)

# 비예 5.6.8: 단순하지 않으면 F = 0인 점이 있다 (8자 곡선: γ(0) = γ(π))
check("비예 5.6.8: 8자 곡선 γ(0) = γ(π) (단순하지 않다)", sp.simplify(eight.subs(t, 0) - eight.subs(t, sp.pi)) == sp.zeros(2, 1))
check("비예 5.6.8: 리마송 γ(2π/3) = γ(4π/3) = 0", sp.simplify(lim.subs(t, 2 * sp.pi / 3)) == sp.zeros(2, 1)
      and sp.simplify(lim.subs(t, 4 * sp.pi / 3)) == sp.zeros(2, 1))

# 곡선다각형 (정리 5.6.11) --------------------------------------------------------------------
def polygon_total(pieces):
    """pieces: [(expr, a, b)], 반시계 순서로 이어지는 조각들. Σ∫κ_s|γ'| + Σε를 돌려준다."""
    total = 0.0
    tangents = []
    for expr, a0, b0 in pieces:
        ks = dgsym.signed_curvature(expr, t)
        spd = sp.sqrt(sp.diff(expr[0], t) ** 2 + sp.diff(expr[1], t) ** 2)
        f = sp.lambdify(t, ks * spd, "numpy")
        xs = np.linspace(a0, b0, 20001)
        ys_ = np.broadcast_to(f(xs), xs.shape)
        total += np.trapezoid(ys_, xs)
        d = sp.lambdify(t, list(sp.diff(expr, t)), "numpy")
        ta, tb = np.array(d(a0), float), np.array(d(b0), float)
        tangents.append((ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)))
    eps = []
    for i in range(len(pieces)):
        Tm = tangents[i - 1][1]
        Tp = tangents[i][0]
        eps.append(float(ang(Tm, Tp)))
    return total + sum(eps), eps, tangents


P = [np.array([math.cos(a), math.sin(a)]) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)]
u = sp.symbols("u", real=True)


def edge(p, q, hgt):
    """p에서 q로 가는 조각: 현 + hgt·sin(πu)·(현의 왼쪽 법선), u ∈ [0, 1]."""
    d = q - p
    nu = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    return sp.Matrix([p[0] + t * d[0] + hgt * sp.sin(sp.pi * t) * nu[0],
                      p[1] + t * d[1] + hgt * sp.sin(sp.pi * t) * nu[1]])


straight = [(edge(P[i], P[(i + 1) % 3], 0), 0, 1) for i in range(3)]
tot_s, eps_s, _ = polygon_total(straight)
close("예 5.6.10: 곧은 정삼각형(반시계): Σε = 2π", tot_s, TWO_PI, tol=1e-9)
close("예 5.6.10: 외각은 각각 2π/3", eps_s, [TWO_PI / 3] * 3, tol=1e-12)
concave = [(edge(P[i], P[(i + 1) % 3], 0.25), 0, 1) for i in range(3)]   # 왼쪽 법선 = 안쪽 → 오목
tot_c, eps_c, tan_c = polygon_total(concave)
close("정리 5.6.11: 오목한 곡선 삼각형(반시계): Σ∫κ_s + Σε = 2π", tot_c, TWO_PI, tol=1e-7)
check("정리 5.6.11: 오목한 곡선 삼각형의 외각은 (-π, π) 안에 있다 (뾰족점 없음)", all(abs(e) < math.pi - 1e-3 for e in eps_c))
# 가장 낮은 점이 꼭짓점 P0 = (0, -1)인지
allpts = np.concatenate([np.array(sp.lambdify(t, list(e), "numpy")(np.linspace(0, 1, 2001))).T for e, _, _ in concave])
check("정리 5.6.11 증명: 오목한 곡선 삼각형의 가장 낮은 점은 꼭짓점 (0, -1)", abs(np.min(allpts[:, 1]) + 1) < 1e-9)
Tp, Tm = tan_c[0][0], tan_c[2][1]
beta = math.atan2(Tp[1], Tp[0]) % TWO_PI
betap = math.atan2(Tm[1], Tm[0]) % TWO_PI
check("정리 5.6.11 증명: t⁺는 닫힌 위쪽 반원, t⁻는 닫힌 아래쪽 반원", 0 <= beta <= math.pi and math.pi <= betap <= TWO_PI)
check("정리 5.6.11 증명: β' - β > π이면 회전지수 +1", betap - beta > math.pi)
close("정리 5.6.11 증명: 2π·rot = 2(ε0 + β' - β - π)", 2 * (eps_c[0] + betap - beta - math.pi), TWO_PI, tol=1e-9)
rev = [(e.subs(t, 1 - t), 0, 1) for e, _, _ in reversed(concave)]
tot_r, eps_r, tan_r = polygon_total(rev)
close("정리 5.6.11: 방향을 뒤집으면 -2π", tot_r, -TWO_PI, tol=1e-7)
Tp_r, Tm_r = tan_r[0][0], tan_r[-1][1]
beta_r = math.atan2(Tp_r[1], Tp_r[0]) % TWO_PI
betap_r = math.atan2(Tm_r[1], Tm_r[0]) % TWO_PI
check("정리 5.6.11 증명: 뒤집은 곡선은 β' - β < π", betap_r - beta_r < math.pi)
# e1이 t⁻에서 t⁺로 가는 짧은 호에 있는지 (정리 5.6.11(b))
def contains_e1(Tm_, Tp_, eps):
    a0 = math.atan2(Tm_[1], Tm_[0])
    return any(abs(math.sin((a0 + lam * eps) / 2)) < 2e-3 for lam in np.linspace(0, 1, 20001))
check("정리 5.6.11(b): 반시계 곡선은 t⁻에서 t⁺로 가는 짧은 호가 e1을 지난다", contains_e1(Tm, Tp, eps_c[0]))
check("정리 5.6.11(b): 뒤집은 곡선의 짧은 호는 e1을 지나지 않는다", not contains_e1(Tm_r, Tp_r, eps_r[0]))
def contains_m_e1(Tm_, Tp_, eps):
    a0 = math.atan2(Tm_[1], Tm_[0])
    return any(abs(math.cos((a0 + lam * eps) / 2)) < 2e-3 for lam in np.linspace(0, 1, 20001))
check("정리 5.6.11(b): 뒤집은 곡선의 짧은 호는 -e1을 실제로 지난다", contains_m_e1(Tm_r, Tp_r, eps_r[0]))
check("정리 5.6.11(b): 반시계 곡선의 짧은 호는 -e1을 지나지 않는다", not contains_m_e1(Tm, Tp, eps_c[0]))
# 오목 꼭짓점(ε < 0)이 있는 곧은 다각형(L자 육각형): Σε = 2π
_Lh = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
_eds = [np.subtract(_Lh[(i + 1) % 6], _Lh[i]) for i in range(6)]
_epsL = [math.atan2(_eds[i - 1][0] * _eds[i][1] - _eds[i - 1][1] * _eds[i][0], _eds[i - 1][0] * _eds[i][0] + _eds[i - 1][1] * _eds[i][1]) for i in range(6)]
close("정리 5.6.11: L자 육각형(오목 꼭짓점 하나): Σε = 2π", sum(_epsL), TWO_PI, tol=1e-12)
check("정리 5.6.11: L자 육각형의 오목 꼭짓점 (1, 1)에서 ε = -π/2", abs(_epsL[3] + math.pi / 2) < 1e-12)

# 연습 5.6.4: 볼록 n각형의 내각의 합 (n - 2)π
for n in (3, 4, 5, 7):
    Q = [np.array([math.cos(TWO_PI * j / n), math.sin(TWO_PI * j / n)]) for j in range(n)]
    tot, eps, _ = polygon_total([(edge(Q[j], Q[(j + 1) % n], 0), 0, 1) for j in range(n)])
    close(f"연습 5.6.4: 정{n}각형 내각의 합 = (n-2)π", sum(math.pi - e for e in eps), (n - 2) * math.pi, tol=1e-9)

# 연습 5.6.5: ρ = 1 + λ cos t 가족 ------------------------------------------------------------------
lam = sp.symbols("lambda", positive=True)
rho = 1 + lam * sp.cos(t)
num = rho ** 2 + 2 * sp.diff(rho, t) ** 2 - rho * sp.diff(rho, t, 2)
den = rho ** 2 + sp.diff(rho, t) ** 2
sym_equal("연습 5.6.5: 분자 = 1 + 2λ² + 3λ cos t", num, 1 + 2 * lam ** 2 + 3 * lam * sp.cos(t), {lam: (0.1, 3)})
sym_equal("연습 5.6.5: 분모 = 1 + λ² + 2λ cos t", den, 1 + lam ** 2 + 2 * lam * sp.cos(t), {lam: (0.1, 3)})
sym_equal("연습 5.6.5: 비 = 3/2 + ((λ²-1)/2)/(1 + λ² + 2λ cos t)", num / den,
          sp.Rational(3, 2) + ((lam ** 2 - 1) / 2) / (1 + lam ** 2 + 2 * lam * sp.cos(t)), {lam: (0.1, 3)})
for lv, want in ((0.5, 1.0), (0.9, 1.0), (1.5, 2.0), (2.0, 2.0)):
    close(f"연습 5.6.5: λ = {lv}이면 회전지수 {int(want)}", rotation_index(((1 + lv * sp.cos(t)) * sp.Matrix([sp.cos(t), sp.sin(t)])), TWO_PI), want, tol=1e-6)

# 연습 5.6.6: 뢸로 삼각형 (너비 w = √3, 꼭짓점은 단위원 위의 정삼각형) --------------------------------------------
w = math.sqrt(3)
reu = []
for i in range(3):
    c = P[(i + 2) % 3]                         # 호 P_i → P_{i+1}의 중심은 맞은편 꼭짓점
    a0 = math.atan2(*(P[i] - c)[::-1])
    a1 = math.atan2(*(P[(i + 1) % 3] - c)[::-1])
    while a1 < a0:
        a1 += TWO_PI
    reu.append((sp.Matrix([c[0] + w * sp.cos(t), c[1] + w * sp.sin(t)]), a0, a1))
tot_reu, eps_reu, _ = polygon_total(reu)
close("연습 5.6.6: 뢸로 삼각형: 각 호의 중심각 π/3", [b - a for _, a, b in reu], [math.pi / 3] * 3, tol=1e-12)
close("연습 5.6.6: 뢸로 삼각형: 외각 π/3씩", eps_reu, [math.pi / 3] * 3, tol=1e-9)
close("연습 5.6.6: 뢸로 삼각형: Σ∫κ_s + Σε = 2π", tot_reu, TWO_PI, tol=1e-8)

# 명제 5.6.12: 양의 방향이면 회전지수 +1 --------------------------------------------------------------
# 곡선은 조각마다 u ∈ [0, 1]로 매개화한다. 양의 방향(정리 2.6.13)은 두 방법으로 판정한다.
#  (i) 부호 있는 넓이 ½∮(x dy - y dx)의 부호 (그린 정리: 양의 방향이면 넓이 > 0),
#  (ii) 매끄러운 점 γ(u)에서 γ + η Jγ'이 둘러싼 영역 안에 있는지 (감는 수 ±1이면 안, 0이면 밖).
def _ev(fn, x):
    return np.array([np.broadcast_to(c, np.shape(x)) for c in fn(x)], float)


def _J(v):
    v = np.asarray(v, float)
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


class PWCurve:
    """조각마다 매끄러운 닫힌 곡선의 수치 표현. 조각 i는 u ∈ [0, 1]에서 R·γ_i(u)."""

    def __init__(self, pieces_sym, R=None, n=8001, funcs=None):
        self.R = np.eye(2) if R is None else np.asarray(R, float)
        if funcs is None:
            funcs = ([sp.lambdify(t, list(e), "numpy") for e, _, _ in pieces_sym],
                     [sp.lambdify(t, list(sp.diff(e, t)), "numpy") for e, _, _ in pieces_sym])
        self.f, self.df = funcs
        self.us = np.linspace(0, 1, n)
        self.pts = [self.pos(i, self.us) for i in range(len(self.f))]
        self.poly = np.concatenate([q[:-1] for q in self.pts] + [self.pts[0][:1]])

    def rotated(self, R, n):
        return PWCurve(None, R, n, (self.f, self.df))

    def pos(self, i, u):
        return (self.R @ _ev(self.f[i], np.atleast_1d(np.asarray(u, float)))).T

    def der(self, i, u):
        return (self.R @ _ev(self.df[i], np.atleast_1d(np.asarray(u, float)))).T

    def area(self):
        tot = 0.0
        for i in range(len(self.f)):
            q, dq = self.pts[i], self.der(i, self.us)
            tot += 0.5 * np.trapezoid(q[:, 0] * dq[:, 1] - q[:, 1] * dq[:, 0], self.us)
        return tot

    def winding(self, q):
        v = self.poly - np.asarray(q, float)[None]
        v = v / np.linalg.norm(v, axis=1)[:, None]
        return float(np.sum(ang(v[:-1], v[1:])) / TWO_PI)

    def left_point(self, i, u0, dist=2e-3):
        """γ + η Jγ' (η|γ'| = dist): 양의 방향이면 Ω 안이어야 하는 점."""
        d = self.der(i, u0)[0]
        return self.pos(i, u0)[0] + dist * _J(d / np.linalg.norm(d))


def crosses(a, b, pts):
    """선분 ab가 꺾은선 pts와 만나는가 (방향 판정)."""
    q0, q1 = pts[:-1], pts[1:]

    def orient(x, y, z):
        return (y[..., 0] - x[..., 0]) * (z[..., 1] - x[..., 1]) - (y[..., 1] - x[..., 1]) * (z[..., 0] - x[..., 0])

    o1, o2 = orient(a[None], b[None], q0), orient(a[None], b[None], q1)
    o3, o4 = orient(q0, q1, a[None]), orient(q0, q1, b[None])
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def e_(phi):
    return np.array([math.cos(phi), math.sin(phi)])


def lowest_vertex(cv):
    """가장 낮은 점이 꼭짓점인지 확인하고 그 꼭짓점의 조각 번호를 돌려준다 (없으면 None)."""
    vy = [cv.pts[i][0, 1] for i in range(len(cv.pts))]
    j = int(np.argmin(vy))
    k = len(cv.us) // 100
    inner = min(np.min(q[k:-k, 1]) for q in cv.pts)                  # 꼭짓점에서 떨어진 점들
    return j if inner > vy[j] + 1e-6 and np.min(cv.poly[:, 1]) >= vy[j] - 1e-12 else None


def proof_steps(cv, j):
    """명제 5.6.12 증명의 4-7단계를 곡선 cv(회전지수 -1)의 가장 낮은 꼭짓점 j에서 수치로 따라간다.

    돌려주는 값: (α, β, ε, δ, 경우, 검사 결과 dict)
    """
    nP = len(cv.pts)
    p = cv.pos(j, 0.0)[0]
    y0 = p[1]
    tp = unit(cv.der(j, 0.0)[0])                      # 나가는 조각 j의 시작
    tm = unit(cv.der((j - 1) % nP, 1.0)[0])           # 들어오는 조각 j-1의 끝
    eps0 = float(ang(tm, tp))
    beta = math.atan2(tp[1], tp[0]) % TWO_PI
    alpha = math.atan2(-tm[1], -tm[0]) % TWO_PI
    delta = min(math.pi / 8, (beta - alpha) / 4) if beta > alpha else float("nan")
    res = {"(5.6.7) 0 ≤ α < β ≤ π": 0 <= alpha < beta <= math.pi,
           "(5.6.7) β - α = π + ε": abs(beta - alpha - (math.pi + eps0)) < 1e-12}
    caseA = beta >= math.pi / 2 + 2 * delta - 1e-12
    caseB = alpha <= math.pi / 2 - 2 * delta + 1e-12
    res["8단계: 6단계나 7단계의 경우가 성립"] = caseA or caseB
    us = cv.us
    # 5단계: r (u 단위, 두 조각에서 같은 길이)과 d
    Tout = np.array([unit(v) for v in cv.der(j, us)])
    Tin = np.array([unit(v) for v in cv.der((j - 1) % nP, us)])
    ok_out = Tout @ tp > math.cos(delta)
    ok_in = (Tin @ tm > math.cos(delta))[::-1]         # 꼭짓점에서 거꾸로
    n_r = min(np.argmin(ok_out) if not ok_out.all() else len(us), np.argmin(ok_in) if not ok_in.all() else len(us))
    n_r = min(n_r - 1, len(us) // 2 - 1)
    r = us[n_r]
    rest = [cv.pts[j][n_r:], cv.pts[(j - 1) % nP][:len(us) - n_r]] + [cv.pts[i] for i in range(nP) if i not in (j, (j - 1) % nP)]
    d = min(np.min(np.linalg.norm(q - p[None], axis=1)) for q in rest)
    res["5단계: d > 0"] = d > 0
    sd = math.sin(delta)
    # 6단계(나가는 조각) 또는 7단계(들어오는 조각)
    if caseA:
        idx = np.arange(1, n_r)
        seg_pts, seg_der = cv.pts[j][idx], cv.der(j, us[idx])
        other = cv.pts[(j - 1) % nP][len(us) - n_r:-1]                # 들어오는 조각의 [a - r, a)
        m = e_(beta - 2 * delta + math.pi / 2)
        case = "6단계"
    else:
        idx = np.arange(len(us) - 2, len(us) - n_r - 1, -1)          # 꼭짓점에서 거꾸로
        seg_pts, seg_der = cv.pts[(j - 1) % nP][idx], cv.der((j - 1) % nP, us[idx])
        other = cv.pts[j][1:n_r + 1]                                   # 나가는 조각의 (a, a + r]
        m = e_(alpha + 2 * delta - math.pi / 2)
        case = "7단계"
    rho = np.linalg.norm(seg_pts - p[None], axis=1)
    good = np.nonzero(rho * (1 + 1 / sd) < 0.9 * d)[0]
    k1 = good[-1]
    x1, T1 = seg_pts[k1], unit(seg_der[k1])
    nu = _J(T1)
    rho1 = rho[k1]
    res["y성분: <Jt1, e2> < -sin δ"] = nu[1] < -sd
    res["<γ(t) - p, m> < 0 (다른 조각)"] = bool(np.all((other - p[None]) @ m < 0))
    lam = np.linspace(0, rho1 / sd, 400)
    ray = x1[None] + lam[:, None] * nu[None]
    res["<x_λ - p, m> > 0 (0 ≤ λ ≤ ρ1/sin δ)"] = bool(np.all((ray - p[None]) @ m > 0))
    res["|x_λ - p| < d (0 ≤ λ ≤ ρ1/sin δ)"] = bool(np.all(np.linalg.norm(ray - p[None], axis=1) < d))
    arc_der = np.array([unit(v) for v in (cv.der(j, us[1:n_r + 1]) if caseA else cv.der((j - 1) % nP, us[len(us) - n_r - 1:-1]))])
    res["f' = <γ', t1> > 0 (고른 조각)"] = bool(np.all(arc_der @ T1 > 0))
    lam_end = (y0 - 1.0 - x1[1]) / nu[1]
    res["반직선이 곡선과 만나지 않음"] = not crosses(x1 + 1e-7 * nu, x1 + lam_end * nu, cv.poly)
    res["반직선이 y < y0로 들어감"] = x1[1] + (rho1 / sd) * nu[1] < y0
    q_left = x1 + min(2e-3, 0.3 * rho1) * nu
    res["γ(t1) + η Jγ'(t1)은 Ω 밖 (감는 수 0)"] = abs(cv.winding(q_left)) < 1e-6
    return alpha, beta, eps0, delta, case, res


concave12 = [(edge(P[i], P[(i + 1) % 3], 0.12), 0, 1) for i in range(3)]   # 그림 5.6.3, 5.6.4의 곡선
for hgt, ccw_pieces in ((0.25, concave), (0.12, concave12)):
    rev_pieces = [(e.subs(t, 1 - t), 0, 1) for e, _, _ in reversed(ccw_pieces)]
    area_exact = 3 * math.sqrt(3) / 4 - 3 * math.sqrt(3) * hgt * 2 / math.pi   # 곧은 삼각형 - 볼록한 세 조각
    cv_p, cv_n = PWCurve(ccw_pieces), PWCurve(rev_pieces)
    close(f"명제 5.6.12: 오목한 곡선 삼각형(h = {hgt}, 반시계)의 부호 있는 넓이 = 3√3/4 - 6√3h/π (그린 정리)",
          cv_p.area(), area_exact, tol=1e-8)
    close(f"명제 5.6.12: 같은 곡선을 시계 방향으로 돌면 부호 있는 넓이 = -(넓이)", cv_n.area(), -area_exact, tol=1e-8)
    pos_ok = all(abs(cv_p.winding(cv_p.left_point(i, u0)) - 1) < 1e-6 for i in range(3) for u0 in (0.1, 0.3, 0.5, 0.7, 0.9))
    neg_out = all(abs(cv_n.winding(cv_n.left_point(i, u0))) < 1e-6 for i in range(3) for u0 in (0.1, 0.3, 0.5, 0.7, 0.9))
    check(f"명제 5.6.12(h = {hgt}): 반시계 방향은 양의 방향 (15개 매끄러운 점에서 γ + ηJγ'의 감는 수 1)", pos_ok)
    check(f"명제 5.6.12(h = {hgt}): 시계 방향은 양의 방향이 아님 (같은 점에서 감는 수 0)", neg_out)
    tot_p, _, _ = polygon_total(ccw_pieces)
    tot_n, _, _ = polygon_total(rev_pieces)
    close(f"명제 5.6.12(h = {hgt}): 양의 방향(넓이 > 0)이면 Σ∫κ_s + Σε = +2π", tot_p, TWO_PI, tol=1e-7)
    close(f"명제 5.6.12(h = {hgt}): 반대 방향(넓이 < 0)이면 -2π", tot_n, -TWO_PI, tol=1e-7)
    # 평면을 돌려도: 가장 낮은 점은 늘 꼭짓점, 부호와 넓이의 부호가 맞고, 반대 방향에서 증명의 단계가 성립
    all_vertex, sign_ok, steps_ok, cases = True, True, True, set()
    failed_steps = []
    for kdeg in range(0, 360, 5):
        R = rot(math.radians(kdeg))
        cp, cn = cv_p.rotated(R, 4001), cv_n.rotated(R, 4001)
        jp, jn = lowest_vertex(cp), lowest_vertex(cn)
        if jp is None or jn is None:
            all_vertex = False
            continue
        tp_, tm_ = unit(cp.der(jp, 0.0)[0]), unit(cp.der((jp - 1) % 3, 1.0)[0])
        sign_ok &= contains_e1(tm_, tp_, float(ang(tm_, tp_))) and cp.area() > 0      # 정리 5.6.11: +1
        alpha_, beta_, eps_, delta_, case_, res_ = proof_steps(cn, jn)
        cases.add(case_)
        bad = [k_ for k_, v_ in res_.items() if not v_]
        if bad:
            steps_ok = False
            failed_steps.append((kdeg, bad))
    check(f"명제 5.6.12 동기(h = {hgt}): 평면을 5°씩 72번 돌려도 가장 낮은 점은 늘 꼭짓점", all_vertex)
    check(f"명제 5.6.12(h = {hgt}): 72번 모두 반시계 방향은 넓이 > 0이고 가장 낮은 꼭짓점의 호가 e1을 지난다 (+1)", sign_ok)
    check(f"명제 5.6.12 증명(h = {hgt}): 72번 모두 시계 방향 곡선에서 4-8단계의 부등식과 반직선이 성립", steps_ok,
          f"실패: {failed_steps[:3]}")
    check(f"명제 5.6.12 증명(h = {hgt}): 6단계와 7단계의 경우가 모두 나타난다", cases == {"6단계", "7단계"}, f"{cases}")

# 그림 5.6.4의 값: 시계 방향, 돌리지 않은 곡선 (h = 0.12)
cv_fig = PWCurve([(e.subs(t, 1 - t), 0, 1) for e, _, _ in reversed(concave12)])
a_fig, b_fig, e_fig, d_fig, c_fig, r_fig = proof_steps(cv_fig, 0)
close("그림 5.6.4: α ≈ 72.3°, β ≈ 107.7°", (round(math.degrees(a_fig), 1), round(math.degrees(b_fig), 1)), (72.3, 107.7), tol=1e-9)
check("그림 5.6.4: 좌우 대칭이라 α = π - β (6단계의 조건이 경계에서 성립)", abs(a_fig + b_fig - math.pi) < 1e-12
      and abs(b_fig - (math.pi / 2 + 2 * d_fig)) < 1e-12)
check("그림 5.6.4: 증명의 단계가 모두 성립", all(r_fig.values()), f"{r_fig}")

# 매끄러운 점이 가장 낮은 경우 (3단계): 콩 모양 곡선과 그 반대 방향
bean_p = bean
bean_n = bean.subs(t, -t)
for name_, curve_, want in (("반시계", bean_p, 1), ("시계", bean_n, 0)):
    cvb = PWCurve([(curve_.subs(t, TWO_PI * t), 0, 1)], n=40001)
    iy = int(np.argmin(cvb.pts[0][:, 1]))
    ul = cvb.us[iy]
    for _ in range(30):                                                # 가장 낮은 점을 뉴턴법으로
        dd = cvb.der(0, ul)[0][1]
        h_ = 1e-6
        ul -= dd / ((cvb.der(0, ul + h_)[0][1] - cvb.der(0, ul - h_)[0][1]) / (2 * h_))
    q_ = cvb.left_point(0, ul)
    below = q_[1] < cvb.pos(0, ul)[0][1]
    check(f"명제 5.6.12 증명 3단계: 콩 모양({name_}): 가장 낮은 점의 왼쪽 점은 감는 수 {want}"
          + (" (y < y0이므로 Ω 밖)" if want == 0 else ""), abs(cvb.winding(q_) - want) < 1e-6 and (below == (want == 0)))

# 연습 5.6.8(a): 시계 방향 원은 Jγ' = γ, 반시계 방향 원은 Jγ' = -γ
rr_ = sp.symbols("r", positive=True)
cw = sp.Matrix([rr_ * sp.cos(t), -rr_ * sp.sin(t)])
cc = sp.Matrix([rr_ * sp.cos(t), rr_ * sp.sin(t)])
Jm = sp.Matrix([[0, -1], [1, 0]])
sym_equal("연습 5.6.8(a): 시계 방향 원에서 Jγ'(t) = γ(t) (왼쪽 점 (1 + η)γ는 원판 밖)", Jm * sp.diff(cw, t), cw)
sym_equal("연습 5.6.8(a): 반시계 방향 원에서 Jγ'(t) = -γ(t) (왼쪽 점 (1 - η)γ는 원판 안)", Jm * sp.diff(cc, t), -cc)
check("연습 5.6.8(b): 방향을 뒤집으면 회전지수의 부호가 바뀐다 (오목한 곡선 삼각형, 위의 ±2π)", abs(tot_r + tot_c) < 1e-6)
check("연습 5.6.8(b): ∠(-u, -v) = ∠(u, v), ∠(v, u) = -∠(u, v) (무작위 100개)", all(
    abs(ang(-uu, -vv) - ang(uu, vv)) < 1e-12 and abs(ang(vv, uu) + ang(uu, vv)) < 1e-12
    for uu, vv in ((e_(a1), e_(a1 + b1)) for a1, b1 in zip(rng.uniform(-4, 4, 100), rng.uniform(-3, 3, 100)))))

# 명제 5.6.12 증명의 삼각함수 계산
al, be, de, th1, lam0, ep = sp.symbols("alpha beta delta theta_1 lambda_0 varepsilon", real=True)
E = lambda a_: sp.Matrix([sp.cos(a_), sp.sin(a_)])
dot = lambda x_, y_: (x_.T * y_)[0]
sym_equal("명제 5.6.12 증명 0단계: J e(φ) = e(φ + π/2)", Jm * E(th1), E(th1 + sp.pi / 2))
sym_equal("명제 5.6.12 증명 4단계: t⁻ = e(π - λ0ε)이면 -t⁻ = e(α), α = -λ0ε", -E(sp.pi - lam0 * ep), E(-lam0 * ep))
sym_equal("명제 5.6.12 증명 4단계: β - α = π + ε", (sp.pi + (1 - lam0) * ep) - (-lam0 * ep), sp.pi + ep)
sym_equal("명제 5.6.12 증명 4단계: t⁻, t⁺의 y성분 = sin(λ0ε), -sin((1-λ0)ε)",
          [E(sp.pi - lam0 * ep)[1], E(sp.pi + (1 - lam0) * ep)[1]], [sp.sin(lam0 * ep), -sp.sin((1 - lam0) * ep)])
m6 = E(be - 2 * de + sp.pi / 2)
sym_equal("명제 5.6.12 증명 6단계: <e(θ), m> = sin(θ - β + 2δ)", dot(E(th1), m6), sp.sin(th1 - be + 2 * de))
sym_equal("명제 5.6.12 증명 6단계: <J e(θ1), m> = cos(θ1 - β + 2δ)", dot(Jm * E(th1), m6), sp.cos(th1 - be + 2 * de))
sym_equal("명제 5.6.12 증명 6단계: <J e(θ1), e2> = cos θ1 = -sin(θ1 - π/2)", [(Jm * E(th1))[1], sp.cos(th1)], [sp.cos(th1), -sp.sin(th1 - sp.pi / 2)])
m7 = E(al + 2 * de - sp.pi / 2)
sym_equal("명제 5.6.12 증명 7단계: J(-e(θ1)) = e(θ1 - π/2)", Jm * (-E(th1)), E(th1 - sp.pi / 2))
sym_equal("명제 5.6.12 증명 7단계: <e(θ), m> = sin(α + 2δ - θ)", dot(E(th1), m7), sp.sin(al + 2 * de - th1))
sym_equal("명제 5.6.12 증명 7단계: <e(θ1 - π/2), m> = cos(θ1 - α - 2δ)", dot(E(th1 - sp.pi / 2), m7), sp.cos(th1 - al - 2 * de))
# 구간 주장: 무작위 0 ≤ α < β ≤ π와 δ = min(π/8, (β - α)/4)
ok_rng, ok_case, ok_bounds = True, True, True
for _ in range(4000):
    a1, b1 = sorted(rng.uniform(0, math.pi, 2))
    if b1 - a1 < 1e-9:
        continue
    d1 = min(math.pi / 8, (b1 - a1) / 4)
    ok_case &= (b1 >= math.pi / 2 + 2 * d1) or (a1 <= math.pi / 2 - 2 * d1)
    lo, hi = a1 - b1 + d1, a1 - b1 + 3 * d1
    ok_rng &= (-math.pi < lo) and (hi <= 0) and (3 * d1 < math.pi / 2)
    xs_s = np.linspace(d1, math.pi / 2 + d1, 2001)[1:-1]
    xs_c = np.linspace(-d1, math.pi / 2 - d1, 2001)[1:-1]
    ok_bounds &= bool(np.all(np.sin(xs_s) > math.sin(d1)) and np.all(np.cos(xs_c) > math.sin(d1)))
check("명제 5.6.12 증명 8단계: 0 ≤ α < β ≤ π이면 β ≥ π/2 + 2δ 또는 α ≤ π/2 - 2δ (무작위 4000개)", ok_case)
check("명제 5.6.12 증명 6·7단계: (α - β + δ, α - β + 3δ) ⊆ (-π, 0), 3δ < π/2", ok_rng)
check("명제 5.6.12 증명 6·7단계: (δ, π/2 + δ)에서 sin > sin δ, (-δ, π/2 - δ)에서 cos > sin δ", ok_bounds)

summary()
