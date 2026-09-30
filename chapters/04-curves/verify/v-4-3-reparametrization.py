"""4.3절 재매개화: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/verify/v-4-3-reparametrization.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
t, u, s = sp.symbols("t u s", real=True)


def speed2(g, var):
    d = sp.Matrix(g).diff(var)
    return sp.simplify(sp.expand(d.dot(d)))


# 예 4.3.3 ---------------------------------------------------------------------------
e = EX["circle"]
(tc,) = e["coords"]
(r,) = e["params"]
g = e["expr"]
sym_equal("예 4.3.3(b): γ(−u) = (r cos u, −r sin u, 0)", g.subs(tc, -u), sp.Matrix([r * sp.cos(u), -r * sp.sin(u), 0]))
x = sp.symbols("x", real=True)
hx = sp.acos(x / r)
sym_equal("예 4.3.3(c): h(x) = arccos(x/r)의 도함수 −1/√(r²−x²)", sp.diff(hx, x), -1 / sp.sqrt(r ** 2 - x ** 2), {x: (-0.4, 0.4), r: (0.5, 2)})
sym_equal("예 4.3.3(c): γ(arccos(x/r)) = (x, √(r²−x²), 0)", g.subs(tc, hx), sp.Matrix([x, sp.sqrt(r ** 2 - x ** 2), 0]),
          {x: (-0.4, 0.4), r: (0.5, 2)})

# 비예 4.3.4(a): h(u) = u³ ---------------------------------------------------------------
bet = sp.Matrix([u ** 3, u ** 3, 0])
check("비예 4.3.4(a): (u³, u³)은 u = 0에서 특이", bet.diff(u).subs(u, 0) == sp.zeros(3, 1))

# 명제 4.3.5 (a), (d): 일반 함수로 --------------------------------------------------------
G = sp.Matrix([sp.Function(f"g{i}")(t) for i in range(1, 4)])
hh = sp.Function("h")(u)
sym_equal("명제 4.3.5(a): (γ∘h)' = h'·γ'(h)", G.subs(t, hh).diff(u), hh.diff(u) * G.diff(t).subs(t, hh))

# 정리 4.3.6: 길이의 재매개화 불변성 (구체적 예) ------------------------------------------------
e = EX["helix"]
(th,) = e["coords"]
a, b = e["params"]
H = e["expr"]
# 방향을 보존하는 h(u) = u³ + u : (0, 1) → (0, 2), 뒤집는 h(u) = 2 − u² : (0, 1) → (1, 2)
for hname, hu, (u1, u2), (t1, t2) in (("u³ + u", u ** 3 + u, (0, 1), (0, 2)), ("2 − u²", 2 - u ** 2, (0, 1), (1, 2))):
    beta = H.subs(th, hu)
    Lb = sp.integrate(sp.sqrt(speed2(beta, u)).simplify(), (u, u1, u2))
    Lg = sp.integrate(sp.sqrt(a ** 2 + b ** 2), (th, t1, t2))
    sym_equal(f"정리 4.3.6: 나선, h(u) = {hname}에서 L(β) = L(γ)", Lb, Lg, e["domain"])

# 예 4.3.8, 4.3.10, 4.3.11: 호의 길이 매개화 -------------------------------------------------
circ_s = g.subs(tc, s / r)
sym_equal("예 4.3.8: (r cos(s/r), r sin(s/r), 0)은 단위속력", speed2(circ_s, s), 1)
cc = sp.sqrt(a ** 2 + b ** 2)
hel_s = H.subs(th, s / cc)
sym_equal("예 4.3.10: 나선의 호의 길이 매개화는 단위속력", speed2(hel_s, s), 1, e["domain"])
sym_equal("예 4.3.10: 나선의 호의 길이 s(t) = √(a²+b²) t", sp.integrate(sp.sqrt(a ** 2 + b ** 2), (th, 0, t)),
          sp.sqrt(a ** 2 + b ** 2) * t, e["domain"])
cat_s = sp.Matrix([sp.asinh(s), sp.sqrt(1 + s ** 2)])
sym_equal("예 4.3.11: 현수선의 호의 길이 s(t) = sinh t", sp.integrate(sp.cosh(t), (t, 0, t)), sp.sinh(t))
sym_equal("예 4.3.11: cosh(arsinh s) = √(1+s²)", sp.cosh(sp.asinh(s)), sp.sqrt(1 + s ** 2))
sym_equal("예 4.3.11: (arsinh s, √(1+s²))은 단위속력", speed2(cat_s, s), 1)

# 예 4.3.12: 타원의 둘레 -------------------------------------------------------------------
ell_speed = sp.sqrt(4 * sp.sin(t) ** 2 + sp.cos(t) ** 2)
sym_equal("예 4.3.12: 타원 (2cos t, sin t)의 속력² = 1 + 3 sin²t", speed2(sp.Matrix([2 * sp.cos(t), sp.sin(t)]), t),
          1 + 3 * sp.sin(t) ** 2)
Lell = sp.Integral(ell_speed, (t, 0, 2 * sp.pi)).evalf(20)
close("예 4.3.12: 둘레 L ≈ 9.6884", float(Lell), 9.68844822054768, 1e-10)

# 연습 4.3.1 ---------------------------------------------------------------------------
check("연습 4.3.1(a): tan' = 1 + tan² > 0", sp.simplify(sp.diff(sp.tan(u), u) - (1 + sp.tan(u) ** 2)) == 0)
check("연습 4.3.1(d): (1/u)' = −1/u² < 0", sp.diff(1 / u, u) == -1 / u ** 2)
check("연습 4.3.1(e): sin' = cos는 ±π/2에서 0", sp.cos(sp.pi / 2) == 0)

# 연습 4.3.2: 원뿔 나선 ----------------------------------------------------------------------
con = sp.Matrix([sp.exp(t) * sp.cos(t), sp.exp(t) * sp.sin(t), sp.exp(t)])
sym_equal("연습 4.3.2: 속력² = 3e^{2t}", speed2(con, t), 3 * sp.exp(2 * t))
sym_equal("연습 4.3.2: s(t) = √3(e^t − 1)", sp.integrate(sp.sqrt(3) * sp.exp(t), (t, 0, t)), sp.sqrt(3) * (sp.exp(t) - 1))
w = 1 + s / sp.sqrt(3)
con_s = sp.Matrix([w * sp.cos(sp.log(w)), w * sp.sin(sp.log(w)), w])
sym_equal("연습 4.3.2: β(s)는 단위속력 (s > −√3)", speed2(con_s, s), 1, {s: (-1.5, 3)})

# 연습 4.3.3: 반대 방향 곡선 --------------------------------------------------------------------
gb = hel_s.subs(s, -u)
sym_equal("연습 4.3.3: 나선의 반대 방향 곡선도 단위속력", speed2(gb, u), 1, e["domain"])
sym_equal("연습 4.3.3: 단위접벡터가 반대: γ̄'(u) = −γ'(−u)", gb.diff(u), -hel_s.diff(s).subs(s, -u), e["domain"])

# 연습 4.3.7: 뾰족점 곡선의 호의 길이 매개화 (t > 0) ------------------------------------------------
tp = sp.symbols("t", positive=True)
sp_ = sp.symbols("s", positive=True)
s_of_t = sp.integrate(sp.Symbol("v", positive=True) * sp.sqrt(4 + 9 * sp.Symbol("v", positive=True) ** 2),
                      (sp.Symbol("v", positive=True), 0, tp))
sym_equal("연습 4.3.7: s(t) = ((4 + 9t²)^{3/2} − 8)/27", s_of_t, ((4 + 9 * tp ** 2) ** sp.Rational(3, 2) - 8) / 27)
h_of_s = sp.sqrt(((27 * sp_ + 8) ** sp.Rational(2, 3) - 4) / 9)
sym_equal("연습 4.3.7: h(s(t)) = t", h_of_s.subs(sp_, s_of_t), tp)
cusp_s = sp.Matrix([h_of_s ** 2, h_of_s ** 3])
sym_equal("연습 4.3.7: β(s) = γ(h(s))는 단위속력", speed2(cusp_s, sp_), 1, {sp_: (0.01, 5)})
check("연습 4.3.7: h'(s) → ∞ (s → 0+)", sp.limit(sp.diff(h_of_s, sp_), sp_, 0, "+") == sp.oo)

# 연습 4.3.8: 직교변환은 길이를 보존 ----------------------------------------------------------------
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
cvec = rng.normal(size=3)
tt = np.linspace(0, 2 * np.pi, 20001)
P = np.stack([np.cos(tt), np.sin(tt), 0.3 * tt], axis=1)
Pm = P @ Q.T + cvec
lenP = np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))
lenPm = np.sum(np.linalg.norm(np.diff(Pm, axis=0), axis=1))
close("연습 4.3.8: 강체운동 후 나선 한 바퀴의 (꺾은선) 길이 불변", lenPm, lenP, 1e-10)
close("연습 4.3.8: 그 값 ≈ 2π√(1 + 0.09)", lenP, 2 * np.pi * np.sqrt(1.09), 1e-6)

summary()
