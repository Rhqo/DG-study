"""7.5절 곡면의 방향: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/verify/v-7-5-orientation.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
u, v, s, t = sp.symbols("u v s t", real=True)


def det3(a, b, c):
    return sp.Matrix.hstack(a, b, c).det()


# 명제 7.5.3(b): N_y = sgn(det Dh) N_x — 구면 기준 매개화와 변수를 바꾼 매개화 ------------------------
sph = EX["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
Xs = sph["expr"]
Nx = dgsym.unit_normal(Xs, th, ph, sph["positive"])
sym_equal("예 7.5.2(a): 구면 N_x = x/r (바깥쪽)", Nx, Xs / r, sph["domain"])
Xsw = Xs.subs({th: s, ph: t}).subs({s: t, t: s}, simultaneous=True)   # x̃(s, t) = x(t, s)
nsw = Xsw.diff(s).cross(Xsw.diff(t))
nx = (Xs.diff(th).cross(Xs.diff(ph))).subs({th: t, ph: s})
check("명제 7.5.3(b): 변수를 바꾼 매개화의 외적 = −(x_θ × x_φ) (det Dh = −1)", sp.simplify(nsw + nx) == sp.zeros(3, 1))

# 예 7.5.2: 기준 예제의 단위법벡터 ------------------------------------------------------------
for key in ("cylinder", "torus"):
    e = EX[key]
    a, b = e["coords"]
    sym_equal(f"예 7.5.2: {e['name']} N_x ({e['normal']})", dgsym.unit_normal(e["expr"], a, b, e["positive"]), e["expected"]["normal"], e["domain"])
rev = EX["revolution"]
ru, rv = rev["coords"]
rf, zf = rev["functions"]
nrev = rev["expr"].diff(ru).cross(rev["expr"].diff(rv))
check("예 7.5.2(d): 회전면 x_u × x_v = ρ(−z' cos v, −z' sin v, ρ')",
      sp.simplify(nrev - rf(ru) * sp.Matrix([-sp.diff(zf(ru), ru) * sp.cos(rv), -sp.diff(zf(ru), ru) * sp.sin(rv), sp.diff(rf(ru), ru)])) == sp.zeros(3, 1))
fg = sp.Function("f")(u, v)
ng = sp.Matrix([u, v, fg]).diff(u).cross(sp.Matrix([u, v, fg]).diff(v))
check("예 7.5.2(e): 그래프 x_u × x_v = (−f_u, −f_v, 1) (위쪽)", sp.simplify(ng - sp.Matrix([-sp.diff(fg, u), -sp.diff(fg, v), 1])) == sp.zeros(3, 1))
# 원환면: 기울기는 바깥쪽 = −N_x
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
x, y, z = sp.symbols("x y z", real=True)
rho = sp.sqrt(x ** 2 + y ** 2)
ft = (rho - R) ** 2 + z ** 2
gt = sp.Matrix([ft]).jacobian([x, y, z]).T.subs({x: tor["expr"][0], y: tor["expr"][1], z: tor["expr"][2]})
sym_equal("예 7.5.8(b): 원환면 grad f/|grad f| = −N_x (바깥쪽)", gt / sp.sqrt(gt.dot(gt)), -tor["expected"]["normal"], tor["domain"])

# 정리 7.5.5: det(x_u, x_v, N_x) = |x_u × x_v| > 0, 기저변환 규칙 -----------------------------------------
f1, f2, f3 = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3"))
Xg = sp.Matrix([f1, f2, f3])
ngen = Xg.diff(u).cross(Xg.diff(v))
check("정리 7.5.5: det(x_u, x_v, x_u × x_v) = |x_u × x_v|² (일반)", sp.simplify(det3(Xg.diff(u), Xg.diff(v), ngen) - ngen.dot(ngen)) == 0)
B = sp.Matrix(2, 2, sp.symbols("b11 b12 b21 b22", real=True))
w1 = B[0, 0] * Xg.diff(u) + B[1, 0] * Xg.diff(v)
w2 = B[0, 1] * Xg.diff(u) + B[1, 1] * Xg.diff(v)
nvec = sp.Matrix(sp.symbols("n1:4", real=True))
check("정리 7.5.5: det(w₁, w₂, N) = det B · det(x_u, x_v, N)", sp.expand(det3(w1, w2, nvec) - B.det() * det3(Xg.diff(u), Xg.diff(v), nvec)) == 0)

# 예 7.5.9, 명제 7.5.10: 뫼비우스 띠 ----------------------------------------------------------------
Xm = sp.Matrix([(1 + v * sp.cos(u / 2)) * sp.cos(u), (1 + v * sp.cos(u / 2)) * sp.sin(u), v * sp.sin(u / 2)])
check("(7.5.6): x(u + 2π, v) = x(u, −v)", sp.simplify(Xm.subs(u, u + 2 * sp.pi) - Xm.subs(v, -v)) == sp.zeros(3, 1))
Em, Fm, Gm = dgsym.first_ff(Xm, u, v)
check("예 7.5.9: E = (1 + v cos(u/2))² + v²/4, F = 0, G = 1",
      (sp.simplify(Em - ((1 + v * sp.cos(u / 2)) ** 2 + v ** 2 / 4)), Fm, Gm) == (0, 0, 1))
nm = Xm.diff(u).cross(Xm.diff(v))
sym_equal("예 7.5.9: |x_u × x_v|² = EG − F²", nm.dot(nm), Em * Gm - Fm ** 2, {u: (0, 12), v: (-0.49, 0.49)})
vals = [((1 + vv * np.cos(uu / 2)) ** 2 + vv ** 2 / 4) for uu in np.linspace(0, 4 * np.pi, 81) for vv in np.linspace(-0.5, 0.5, 41)]
check("예 7.5.9: |v| < 1/2에서 |x_u × x_v|² ≥ 1/4 > 0 (격자 확인)", min(vals) >= 0.25 - 1e-12)
check("(7.5.7): (x_u × x_v)(u + 2π, v) = −(x_u × x_v)(u, −v)", sp.simplify(nm.subs(u, u + 2 * sp.pi) + nm.subs(v, -v)) == sp.zeros(3, 1))
check("명제 7.5.10: 중심원에서 x_u × x_v = (sin(u/2) cos u, sin(u/2) sin u, −cos(u/2)) (단위벡터)",
      sp.simplify(nm.subs(v, 0) - sp.Matrix([sp.sin(u / 2) * sp.cos(u), sp.sin(u / 2) * sp.sin(u), -sp.cos(u / 2)])) == sp.zeros(3, 1))
check("명제 7.5.10: N_x(2π, 0) = −N_x(0, 0) = (0, 0, 1)", nm.subs({u: 2 * sp.pi, v: 0}) == sp.Matrix([0, 0, 1]) and nm.subs({u: 0, v: 0}) == sp.Matrix([0, 0, -1]))
# 역사상의 식: ρ − 1 = v cos(u/2), z = v sin(u/2) ⇒ v = (ρ − 1)cos(u/2) + z sin(u/2)
rho_m = sp.sqrt(Xm[0] ** 2 + Xm[1] ** 2)
sym_equal("예 7.5.9: ρ = 1 + v cos(u/2) (|v| < 1/2)", rho_m, 1 + v * sp.cos(u / 2), {u: (0, 12), v: (-0.49, 0.49)})
sym_equal("예 7.5.9: v = (ρ − 1)cos(u/2) + z sin(u/2)", (1 + v * sp.cos(u / 2) - 1) * sp.cos(u / 2) + Xm[2] * sp.sin(u / 2), v)
# 겹치는 곳의 매개변수 변환 (비고): ū ∈ (2π, 3π)이면 h(ū, v̄) = (ū − 2π, −v̄), det = −1
hm = sp.Matrix([u - 2 * sp.pi, -v])
check("비고 (7.5절): 두 번째 성분에서 매개변수 변환의 행렬식 = −1", hm.jacobian([u, v]).det() == -1)

# 예 7.5.12: 반사와 대척사상 ---------------------------------------------------------------------
Ns = Xs / r
q_hat = {th: sp.pi - th}
xt, xp = Xs.diff(th), Xs.diff(ph)
Sig = sp.diag(1, 1, -1)
check("예 7.5.12(a): dσ(x_θ(q)) = −x_θ(q̂), dσ(x_φ(q)) = x_φ(q̂) (예 6.4.7)",
      sp.simplify(Sig * xt + xt.subs(q_hat)) == sp.zeros(3, 1) and sp.simplify(Sig * xp - xp.subs(q_hat)) == sp.zeros(3, 1))
dsig = det3(Sig * xt, Sig * xp, Ns.subs(q_hat))
sym_equal("예 7.5.12(a): det(dσ x_θ, dσ x_φ, N(σ(p))) = −r² sin θ < 0 (뒤집는다)", dsig, -r ** 2 * sp.sin(th), sph["domain"])
dA = det3(-xt, -xp, -Ns)
sym_equal("예 7.5.12(b): 대척사상 det(−x_θ, −x_φ, N(−p)) = −r² sin θ < 0 (뒤집는다)", dA, -r ** 2 * sp.sin(th), sph["domain"])
al = sp.symbols("alpha", real=True)
Rz = sp.Matrix([[sp.cos(al), -sp.sin(al), 0], [sp.sin(al), sp.cos(al), 0], [0, 0, 1]])
dR = det3(Rz * xt, Rz * xp, (Rz * Xs) / r)
sym_equal("예 7.5.12(c): 회전은 보존 det = r² sin θ > 0", dR, r ** 2 * sp.sin(th), sph["domain"])

# 연습 7.5.1: 원환면의 바깥쪽 방향에 대한 양의 매개화 ----------------------------------------------------
Xt = tor["expr"]
Xswap = Xt.subs({tu: v, tv: u}, simultaneous=True)       # y(u, v) = x(v, u)
nswap = Xswap.diff(u).cross(Xswap.diff(v))
out = sp.Matrix([sp.cos(v) * sp.cos(u), sp.cos(v) * sp.sin(u), sp.sin(v)])
sym_equal("연습 7.5.1: y(u, v) = x(v, u)의 y_u × y_v = r(R + r cos v)·(바깥쪽 단위벡터)", nswap, rr * (R + rr * sp.cos(v)) * out,
          {u: (0, 6), v: (0, 6), R: (1.5, 2), rr: (0.2, 0.9)})

# 연습 7.5.3: 두 겹 쌍곡면 ---------------------------------------------------------------------------
fh = -x ** 2 - y ** 2 + z ** 2
gh = sp.Matrix([fh]).jacobian([x, y, z]).T
up = sp.Matrix([u, v, sp.sqrt(1 + u ** 2 + v ** 2)])
nup = up.diff(u).cross(up.diff(v))
gup = gh.subs({x: up[0], y: up[1], z: up[2]})
sym_equal("연습 7.5.3: 위쪽 판에서 <x_u × x_v, grad f> = 2(1 + 2u² + 2v²)/√(1 + u² + v²) > 0", nup.dot(gup),
          2 * (1 + 2 * u ** 2 + 2 * v ** 2) / sp.sqrt(1 + u ** 2 + v ** 2), {u: (-2, 2), v: (-2, 2)})

# 연습 7.5.4: 원기둥의 두 사상 ---------------------------------------------------------------------
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Xc = cyl["expr"]
Nc = cyl["expected"]["normal"]
Sz = sp.diag(1, 1, -1)
Rpi = sp.diag(-1, -1, 1)
d1 = det3(Sz * Xc.diff(cu), Sz * Xc.diff(cv), Nc)            # σ_z(p)에서의 바깥쪽 N은 같은 식 (cos u, sin u, 0)
d2 = det3(Rpi * Xc.diff(cu), Rpi * Xc.diff(cv), Rpi * Nc)
check("연습 7.5.4: σ_z는 원기둥의 바깥쪽 방향을 뒤집는다 (det = −r)", sp.simplify(d1 + cr) == 0)
check("연습 7.5.4: z축 둘레 π 회전은 보존한다 (det = r)", sp.simplify(d2 - cr) == 0)

# 연습 7.5.5: σ⁻¹은 바깥쪽 방향에 대해 음의 매개화 --------------------------------------------------------
st = dgsym.stereographic(2)
U1, U2 = st["chart_coords"]
inv = st["sigma_inv"]
w = U1 ** 2 + U2 ** 2 + 1
ns_inv = inv.diff(U1).cross(inv.diff(U2))
check("연습 7.5.5: σ⁻¹_u × σ⁻¹_v = −(4/w²)σ⁻¹(u, v) (안쪽)", sp.simplify(ns_inv + 4 / w ** 2 * inv) == sp.zeros(3, 1))
south = st["sigma_south_inv"]
ns_s = south.diff(U1).cross(south.diff(U2))
check("연습 7.5.5: 남극 입체사영의 역사상은 σ̃⁻¹_u × σ̃⁻¹_v = +(4/w²)σ̃⁻¹ (바깥쪽)", sp.simplify(ns_s - 4 / w ** 2 * south) == sp.zeros(3, 1))

# 연습 7.5.6: 중심원을 뺀 뫼비우스 띠 --------------------------------------------------------------------
check("연습 7.5.6: (x_u × x_v)(u + 4π, v) = (x_u × x_v)(u, v)", sp.simplify(nm.subs(u, u + 4 * sp.pi) - nm) == sp.zeros(3, 1))
check("연습 7.5.6: x(u + 4π, v) = x(u, v)", sp.simplify(Xm.subs(u, u + 4 * sp.pi) - Xm) == sp.zeros(3, 1))

# 연습 7.5.7: 원환면의 두 사상 (방향 보존 여부는 상과 원래 기저의 det 부호 비교) ---------------------------------------------------------------------
Nout = -tor["expected"]["normal"]
xu_, xv_ = Xt.diff(tu), Xt.diff(tv)
base = det3(xu_, xv_, Nout)                                           # < 0: x는 바깥쪽 방향에 대해 음의 매개화
e1 = det3(Sz * xu_, Sz * xv_, Nout.subs(tu, -tu))                     # σ_z(x(u, v)) = x(−u, v)
check("연습 7.5.7: σ_z(x(u, v)) = x(−u, v) (원환면)", sp.simplify(Sz * Xt - Xt.subs(tu, -tu)) == sp.zeros(3, 1))
sym_equal("연습 7.5.7: σ_z는 원환면의 바깥쪽 방향을 뒤집는다 (det 비 = −1)", e1 / base, -1, {tu: (0.2, 6), tv: (0.2, 6), R: (1.5, 2), rr: (0.2, 0.9)})
e2 = det3(Rpi * xu_, Rpi * xv_, Rpi * Nout)
sym_equal("연습 7.5.7: z축 둘레 π 회전은 보존한다 (det 비 = +1)", e2 / base, 1, {tu: (0.2, 6), tv: (0.2, 6), R: (1.5, 2), rr: (0.2, 0.9)})

summary()
