"""8.3절 주곡률과 주방향: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/verify/v-8-3-principal-curvatures.py``
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
lam = sp.symbols("lambda", real=True)

# (8.3.1): 2×2 특성다항식 ------------------------------------------------------------------------------
a, b, c, d = sp.symbols("a b c d", real=True)
A = sp.Matrix([[a, b], [c, d]])
check("(8.3.1): det(A − λI) = λ² − (tr A)λ + det A", sp.expand((A - lam * sp.eye(2)).det() - (lam ** 2 - A.trace() * lam + A.det())) == 0)

# 비예 8.3.2 -----------------------------------------------------------------------------------------------
J = sp.Matrix([[0, -1], [1, 0]])
check("비예 8.3.2(a): J의 특성다항식 λ² + 1 (실근 없음)", sp.expand((J - lam * sp.eye(2)).det()) == lam ** 2 + 1 and sp.solve(lam ** 2 + 1, lam) == [])
Sh = sp.Matrix([[1, 1], [0, 1]])
check("비예 8.3.2(b): 층밀림의 고유공간 = Span{e₁} (1차원)", (Sh - sp.eye(2)).nullspace() == [sp.Matrix([1, 0])])

# 정리 8.3.3: 판별식, 정규직교 고유기저 (일반 내적에서 수치) ----------------------------------------------------
check("정리 8.3.3: 판별식 (a + c)² − 4(ac − b²) = (a − c)² + 4b²", sp.expand((a + c) ** 2 - 4 * (a * c - b ** 2) - ((a - c) ** 2 + 4 * b ** 2)) == 0)
rng = np.random.default_rng(11)
ok = True
for _ in range(50):
    Bm = rng.normal(size=(2, 2))
    Gm = Bm.T @ Bm + 0.1 * np.eye(2)                  # 그람 행렬 (양의 정부호)
    Sym = rng.normal(size=(2, 2)); Sym = Sym + Sym.T
    Amat = np.linalg.solve(Gm, Sym)                    # G·[A] = Sym 대칭 → 자기수반 (명제 8.1.7(b))
    ev, V = np.linalg.eig(Amat)
    ok &= np.all(np.abs(ev.imag) < 1e-12)
    ok &= abs(V[:, 0] @ Gm @ V[:, 1]) < 1e-9          # 고유벡터가 g에 대해 수직
    # (c): 단위원 위 Q(v) = g(Av, v)의 최대·최소 = 고윳값
    th = np.linspace(0, 2 * np.pi, 20001)
    L = np.linalg.cholesky(Gm)
    Linv = np.linalg.inv(L.T)                          # v = Linv·(cos, sin)은 g-단위벡터
    vs = Linv @ np.stack([np.cos(th), np.sin(th)])
    Q = np.einsum("in,ij,jn->n", vs, Gm @ Amat, vs)
    ok &= abs(Q.max() - ev.real.max()) < 1e-6 and abs(Q.min() - ev.real.min()) < 1e-6
check("정리 8.3.3: 무작위 자기수반 변환 50개 — 실고윳값, g-직교 고유벡터, 최대·최소 = 고윳값", ok)

# 예 8.3.6: 기준 예제의 주곡률 ----------------------------------------------------------------------------------
sph = EX["sphere"]
th_, ph_ = sph["coords"]
(r,) = sph["params"]
sym_equal("예 8.3.6(b): 구면 W = −(1/r) I (바깥쪽), κ₁ = κ₂ = −1/r", dgsym.shape_operator(sph["expr"], th_, ph_, sph["positive"]),
          sp.diag(*sph["expected"]["kappa12"]), sph["domain"])
cyl = EX["cylinder"]
cu, cv = cyl["coords"]
(rc,) = cyl["params"]
Wc = dgsym.shape_operator(cyl["expr"], cu, cv)
sym_equal("예 8.3.6(c): 원기둥 W = diag(−1/r, 0) (x_u, x_v; 바깥쪽)", Wc, sp.diag(-1 / rc, 0), cyl["domain"])
check("예 8.3.6(c): 고윳값 {0, −1/r}", set(Wc.eigenvals().keys()) == {0, -1 / rc})
sad = EX["saddle"]
u, v = sad["coords"]
sym_equal("예 8.3.6(d): 안장면 원점 W = diag(2, −2)", dgsym.shape_operator(sad["expr"], u, v).subs({u: 0, v: 0}), sp.diag(2, -2))
tor = EX["torus"]
tu, tv = tor["coords"]
R, rr = tor["params"]
Wt = dgsym.shape_operator(tor["expr"], tu, tv, tor["positive"])
k1t, k2t = 1 / rr, sp.cos(tu) / (R + rr * sp.cos(tu))
sym_equal("(8.3.3): 원환면 W = diag(1/r, cos u/(R + r cos u)) (N_x 안쪽)", Wt, sp.diag(k1t, k2t), tor["domain"])
sym_equal("예 8.3.6(e): κ₁ − κ₂ = R/(r(R + r cos u)) > 0", k1t - k2t, R / (rr * (R + rr * sp.cos(tu))), tor["domain"])
for u0, want in ((0, 1 / (R + rr)), (sp.pi / 2, 0), (sp.pi, -1 / (R - rr))):
    sym_equal(f"예 8.3.6(e): u = {u0}에서 κ₂ = {want}", k2t.subs(tu, u0), want, tor["domain"])
# 오일러 공식 (8.3.2): 원환면에서 직접
beta = sp.symbols("beta", real=True)
E, F, G = dgsym.first_ff(tor["expr"], tu, tv, tor["positive"])
cvec = sp.Matrix([sp.cos(beta) / rr, sp.sin(beta) / (R + rr * sp.cos(tu))])       # w = cos β e₁ + sin β e₂
Imat = sp.Matrix([[E, F], [F, G]])
kn = (cvec.T * Imat * Wt * cvec)[0]
sym_equal("(8.3.2): 원환면 κ_n(w) = κ₁cos²β + κ₂sin²β", kn, k1t * sp.cos(beta) ** 2 + k2t * sp.sin(beta) ** 2, {**tor["domain"], beta: (0, 6)})

# 예 8.3.8: 포물면의 원점 ---------------------------------------------------------------------------------------
P = sp.Matrix([u, v, u ** 2 + v ** 2])
sym_equal("예 8.3.8(b): z = u² + v²의 원점 W = 2 I (배꼽점)", dgsym.shape_operator(P, u, v).subs({u: 0, v: 0}), 2 * sp.eye(2))
Wp1 = dgsym.shape_operator(P, u, v).subs({u: 1, v: 0})
check("예 8.3.8(b): z = u² + v²의 (1, 0)은 배꼽점이 아님 (고윳값이 다름)", len(Wp1.eigenvals()) == 2)

# 정리 8.3.9 ----------------------------------------------------------------------------------------------------
f1, f2, f3, kf = (sp.Function(n)(u, v) for n in ("f1", "f2", "f3", "kappa"))
Xg = sp.Matrix([f1, f2, f3])
Nu, Nv = -kf * Xg.diff(u), -kf * Xg.diff(v)
check("정리 8.3.9 1단계: N_u = −κx_u, N_v = −κx_v이면 N_uv − N_vu = −κ_v x_u + κ_u x_v",
      sp.simplify(Nu.diff(v) - Nv.diff(u) - (-kf.diff(v) * Xg.diff(u) + kf.diff(u) * Xg.diff(v))) == sp.zeros(3, 1))
Nsph = sph["expr"] / r
ksph = -1 / r
sym_equal("정리 8.3.9 4단계: 구면 c(q) = q + N/κ = 0 (중심)", sph["expr"] + Nsph / ksph, sp.zeros(3, 1), sph["domain"])
sym_equal("정리 8.3.9 4단계: N → −N이면 κ → −κ, N/κ 불변", (-Nsph) / (-ksph), Nsph / ksph, sph["domain"])

# 명제 8.3.12: 회전면 (일반 ρ, z) --------------------------------------------------------------------------------
rev = EX["revolution"]
ru, rv = rev["coords"]
rho, zf = rev["functions"]
Xr = rev["expr"]
Nr = dgsym.unit_normal(Xr, ru, rv)
sp1 = sp.sqrt(sp.diff(rho(ru), ru) ** 2 + sp.diff(zf(ru), ru) ** 2)
rp, rpp = sp.diff(rho(ru), ru), sp.diff(rho(ru), ru, 2)
zp, zpp = sp.diff(zf(ru), ru), sp.diff(zf(ru), ru, 2)
sym_equal("명제 8.3.12: N_v = −z'/(ρ|α'|) x_v", Nr.diff(rv), -zp / (rho(ru) * sp1) * Xr.diff(rv))
IIu = Xr.diff(ru, 2).dot(Nr)
sym_equal("명제 8.3.12: II(x_u) = ⟨x_uu, N⟩ = (ρ'z'' − ρ''z')/|α'|", IIu, (rp * zpp - rpp * zp) / sp1)
Wr = dgsym.shape_operator(Xr, ru, rv)
sym_equal("(8.3.5): 회전면 W = diag((ρ'z'' − ρ''z')/|α'|³, z'/(ρ|α'|))", Wr, sp.diag((rp * zpp - rpp * zp) / sp1 ** 3, zp / (rho(ru) * sp1)))
# 곡률선: 나선은 원기둥의 곡률선이 아니다
Wv = Wc * sp.Matrix([1, sp.symbols("b", positive=True)])
check("본문: 나선 속도 x_u + b x_v의 W 상 = −x_u/r (평행 아님)", Wv == sp.Matrix([-1 / rc, 0]))

# 예 8.3.13 --------------------------------------------------------------------------------------------------------
def rev_curvatures(rho_e, z_e, s):
    rp_, rpp_ = sp.diff(rho_e, s), sp.diff(rho_e, s, 2)
    zp_, zpp_ = sp.diff(z_e, s), sp.diff(z_e, s, 2)
    sp_ = sp.sqrt(rp_ ** 2 + zp_ ** 2)
    return sp.simplify((rp_ * zpp_ - rpp_ * zp_) / sp_ ** 3), sp.simplify(zp_ / (rho_e * sp_))


thq = sp.symbols("theta", positive=True)
km, kp = rev_curvatures(r * sp.sin(thq), r * sp.cos(thq), thq)
check("예 8.3.13(a): 구면 경선·위도원 주곡률 = −1/r", sp.simplify(km + 1 / r) == 0 and sp.simplify(sp.refine(kp, sp.Q.positive(sp.sin(thq))) + 1 / r) == 0)
km, kp = rev_curvatures(R + rr * sp.cos(tu), rr * sp.sin(tu), tu)
sym_equal("예 8.3.13(b): 원환면 경선 1/r, 위도원 cos u/(R + r cos u)", sp.Matrix([km, kp]), sp.Matrix([k1t, k2t]), tor["domain"])
uq = sp.symbols("u", real=True)
km, kp = rev_curvatures(sp.cosh(uq), uq, uq)
sym_equal("예 8.3.13(c): 현수면 경선 −1/cosh²u, 위도원 1/cosh²u, 합 0", sp.Matrix([km, kp, km + kp]),
          sp.Matrix([-1 / sp.cosh(uq) ** 2, 1 / sp.cosh(uq) ** 2, 0]), {uq: (-2, 2)})

# 연습 8.3.1: 원기둥을 회전면으로 ---------------------------------------------------------------------------------
Y = sp.Matrix([rc * sp.cos(v), rc * sp.sin(v), u])
sym_equal("연습 8.3.1: N_y = (−cos v, −sin v, 0) (안쪽)", dgsym.unit_normal(Y, u, v), sp.Matrix([-sp.cos(v), -sp.sin(v), 0]))
sym_equal("연습 8.3.1: W = diag(0, 1/r) (모선, 둘레)", dgsym.shape_operator(Y, u, v), sp.diag(0, 1 / rc))

# 연습 8.3.3: 안장면 x(1/2, 1/2) ------------------------------------------------------------------------------------
half = {u: sp.Rational(1, 2), v: sp.Rational(1, 2)}
Ws = sp.simplify(dgsym.shape_operator(sad["expr"], u, v).subs(half))
cst = 2 / (3 * sp.sqrt(3))
sym_equal("연습 8.3.3: W 행렬 = (2/(3√3))[[2, −1], [1, −2]]", Ws, cst * sp.Matrix([[2, -1], [1, -2]]))
check("연습 8.3.3: 고윳값 ±2/3", set(sp.simplify(k) for k in Ws.eigenvals().keys()) == {sp.Rational(2, 3), -sp.Rational(2, 3)})
w1 = sp.Matrix([1, 2 - sp.sqrt(3)])
w2 = sp.Matrix([1, 2 + sp.sqrt(3)])
sym_equal("연습 8.3.3: W w₁ = (2/3) w₁", Ws * w1, sp.Rational(2, 3) * w1)
sym_equal("연습 8.3.3: W w₂ = −(2/3) w₂", Ws * w2, -sp.Rational(2, 3) * w2)
Es, Fs, Gs = (x.subs(half) for x in dgsym.first_ff(sad["expr"], u, v))
Is = sp.Matrix([[Es, Fs], [Fs, Gs]])
check("연습 8.3.3: ⟨w₁, w₂⟩_I = 0, 표준 내적은 2", sp.simplify((w1.T * Is * w2)[0]) == 0 and sp.simplify(w1.dot(w2)) == 2)
sym_equal("연습 8.3.3: det W = −4/9", Ws.det(), -sp.Rational(4, 9))

# 연습 8.3.4 --------------------------------------------------------------------------------------------------------
k1n, k2n = 1 / 0.8, -1 / 1.2
bz = np.arctan(np.sqrt(-k1n / k2n))
close("연습 8.3.4: u = π에서 κ_n = 0 인 β = arctan √1.5 ≈ 0.886", [k1n * np.cos(bz) ** 2 + k2n * np.sin(bz) ** 2, bz], [0.0, 0.8861], tol=1e-4)

# 연습 8.3.5: 원환면의 경선은 평면 곡선, N이 그 평면에 평행 → 로드리게스 ----------------------------------------------------
Nt = dgsym.unit_normal(tor["expr"], tu, tv, tor["positive"])
v0 = sp.Rational(3, 5)
m = sp.Matrix([-sp.sin(v0), sp.cos(v0), 0])            # 경선 v = v0이 놓인 평면의 법선
mer = tor["expr"].subs(tv, v0)
Nm = Nt.subs(tv, v0)
check("연습 8.3.5: 경선과 N은 평면 ⟂ m 안에 있다", sp.simplify(mer.dot(m)) == 0 and sp.simplify(Nm.dot(m)) == 0)
sym_equal("연습 8.3.5: (N∘γ)' = −(1/r) γ' (로드리게스, λ = κ₁)", Nm.diff(tu), -mer.diff(tu) / rr, tor["domain"])

summary()
