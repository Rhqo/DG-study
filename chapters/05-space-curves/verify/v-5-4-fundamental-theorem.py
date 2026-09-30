"""5.4절 곡선론의 기본정리: 본문과 연습문제의 계산 검증 (GUIDELINES.md §13).

강체운동의 성질, 곡률·비틀림률의 보존, 존재 증명의 그람 행렬 방정식, 상수 κ, τ의 나선, 수치 구성(RK4),
κ > 0 가정이 필요한 비예를 확인한다. 나선은 dgsym.EXAMPLES["helix"]에서 가져온다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/verify/v-5-4-fundamental-theorem.py``
"""

import math

import numpy as np
import sympy as sp

import dgnum
import dgsym
from dgcheck import check, close, sym_equal, summary

EX = dgsym.EXAMPLES
s, t, al, th = sp.symbols("s t alpha theta", real=True)


def cayley(w1, w2, w3):
    """케일리 변환 (I - W)^{-1}(I + W): W가 반대칭이면 직교행렬이고 행렬식이 1 (유리수 성분의 시험용 회전)."""
    W = sp.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
    return sp.simplify((sp.eye(3) - W).inv() * (sp.eye(3) + W))


A = cayley(sp.Rational(1, 2), -sp.Rational(1, 3), sp.Rational(2, 5))
u = sp.Matrix(sp.symbols("u1:4", real=True))
v = sp.Matrix(sp.symbols("v1:4", real=True))

# 명제 5.4.2: 직교행렬의 성질 ------------------------------------------------------------------
check("명제 5.4.2: 시험 행렬 A는 AᵀA = I", sp.simplify(A.T * A) == sp.eye(3))
check("명제 5.4.2(b): det A = 1", sp.simplify(A.det()) == 1)
check("명제 5.4.2(a): <Au, Av> = <u, v>", sp.simplify((A * u).dot(A * v) - u.dot(v)) == 0)
check("명제 5.4.2(c): det A = 1이면 A(u×v) = Au × Av", sp.simplify(A * u.cross(v) - (A * u).cross(A * v)) == sp.zeros(3, 1))
Am = -A
check("명제 5.4.2(c): det(-A) = -1이면 (-A)(u×v) = -((-A)u × (-A)v)",
      sp.simplify(Am * u.cross(v) + (Am * u).cross(Am * v)) == sp.zeros(3, 1))
check("명제 5.4.2(d): A⁻¹ = Aᵀ", sp.simplify(A.inv() - A.T) == sp.zeros(3, 3))

# 명제 5.4.4: 평면의 강체운동 ---------------------------------------------------------------------
Ra = sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]])
Rblk = sp.diag(Ra, 1)
check("명제 5.4.4(a): diag(R_α, 1)은 직교행렬", sp.simplify(Rblk.T * Rblk) == sp.eye(3))
check("명제 5.4.4(a): det diag(R_α, 1) = 1", sp.simplify(Rblk.det()) == 1)
p_, q_ = sp.symbols("p q", real=True)
B2 = sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]])
check("명제 5.4.4(c): 첫 열 (cos α, sin α)에 수직인 단위벡터는 ±(-sin α, cos α)이고 det = 1이면 +",
      sp.simplify(sp.Matrix([[sp.cos(al), -sp.sin(al)], [sp.sin(al), sp.cos(al)]]).det()) == 1
      and sp.simplify(sp.Matrix([[sp.cos(al), sp.sin(al)], [sp.sin(al), -sp.cos(al)]]).det()) == -1)

# 예 5.4.5와 연습 5.4.1: 나선의 나사 운동 ------------------------------------------------------------
eh = EX["helix"]
(tt,) = eh["coords"]
a, b = eh["params"]
Rz = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
screw = Rz * eh["expr"] + sp.Matrix([0, 0, b * th])
check("예 5.4.5: 나사 운동 ψ_θ(γ(t)) = γ(t + θ)", sp.simplify(screw - eh["expr"].subs(tt, tt + th)) == sp.zeros(3, 1))
check("예 5.4.5: R_θ (z축 회전)는 직교행렬, det = 1", sp.simplify(Rz.T * Rz) == sp.eye(3) and sp.simplify(Rz.det()) == 1)

# 비예 5.4.6: 반사 ------------------------------------------------------------------------------
rho = sp.diag(1, 1, -1)
check("비예 5.4.6: ρ = diag(1,1,-1)은 직교행렬이지만 det = -1", rho.T * rho == sp.eye(3) and rho.det() == -1)
check("비예 5.4.6: ρ는 외적의 부호를 뒤집는다: ρ(e1×e2) = -ρe1×ρe2",
      rho * sp.Matrix([1, 0, 0]).cross(sp.Matrix([0, 1, 0])) == -(rho * sp.Matrix([1, 0, 0])).cross(rho * sp.Matrix([0, 1, 0])))

# 명제 5.4.7: 강체운동은 κ, τ 보존 ----------------------------------------------------------------
cub = sp.Matrix([t, t ** 2, t ** 3])
cvec = sp.Matrix([1, -2, 5])
k0, t0 = dgsym.curvature_torsion(cub, t)
k1, t1 = dgsym.curvature_torsion(A * cub + cvec, t)
sym_equal("명제 5.4.7: 강체운동 뒤 κ 보존 (꼬인 삼차곡선)", k1, k0, {t: (-2, 2)})
sym_equal("명제 5.4.7: 강체운동 뒤 τ 보존 (꼬인 삼차곡선)", t1, t0, {t: (-2, 2)})
k2, t2 = dgsym.curvature_torsion(-A * cub + cvec, t)
sym_equal("명제 5.4.7: det = -1인 직교변환은 κ 보존", k2, k0, {t: (-2, 2)})
sym_equal("명제 5.4.7: det = -1인 직교변환은 τ를 뒤집음", t2, -t0, {t: (-2, 2)})
T0, N0, B0 = dgsym.frenet_frame(cub, t)
T1, N1, B1 = dgsym.frenet_frame(A * cub + cvec, t)
sym_equal("명제 5.4.7: 틀은 A로 옮겨진다 (t̃, ñ, b̃) = (At, An, Ab)", sp.Matrix.hstack(T1, N1, B1),
          A * sp.Matrix.hstack(T0, N0, B0), {t: (-2, 2)})

# 정리 5.4.8 증명: 그람 행렬의 방정식 --------------------------------------------------------------
kk, tu = sp.symbols("kappa tau", real=True)
K = sp.Matrix([[0, -kk, 0], [kk, 0, -tu], [0, tu, 0]])
check("정리 5.4.8 증명: G = I는 G' = KᵀG + GK의 해 (Kᵀ + K = 0)", sp.simplify(K.T * sp.eye(3) + sp.eye(3) * K) == sp.zeros(3, 3))
Gs = sp.Matrix(3, 3, sp.symbols("g1:10", real=True))
Fs = sp.Matrix(3, 3, sp.symbols("f1:10", real=True))
lhs = (Fs * K).T * Fs + Fs.T * (Fs * K)
check("정리 5.4.8 증명: F' = FK이면 (FᵀF)' = Kᵀ(FᵀF) + (FᵀF)K", sp.simplify(lhs - (K.T * (Fs.T * Fs) + (Fs.T * Fs) * K)) == sp.zeros(3, 3))
# (b)의 강체운동 A = F̃0 F0ᵀ: 나선과 그 회전·평행이동 사본
sub = {a: 1, b: sp.Rational(3, 10)}
hel = eh["expr"].subs(sub)
Th, Nh, Bh = dgsym.frenet_frame(hel, tt)
F0 = sp.Matrix.hstack(Th, Nh, Bh).subs(tt, 0)
hel2 = A * hel + cvec
Th2, Nh2, Bh2 = dgsym.frenet_frame(hel2, tt)
F0t = sp.simplify(sp.Matrix.hstack(Th2, Nh2, Bh2).subs(tt, 0))
Arec = sp.simplify(F0t * F0.T)
check("정리 5.4.8(b) 증명: A = F̃0 F0ᵀ가 원래의 회전을 되살린다", sp.simplify(Arec - A) == sp.zeros(3, 3))
crec = sp.simplify(hel2.subs(tt, 0) - Arec * hel.subs(tt, 0))
check("정리 5.4.8(b) 증명: c = γ̃(s0) - Aγ(s0)", sp.simplify(crec - cvec) == sp.zeros(3, 1))

# 따름정리 5.4.9: 상수 κ, τ의 나선 ---------------------------------------------------------------
kp = sp.symbols("kappa0", positive=True)
tp = sp.symbols("tau0", real=True, nonzero=True)
aa = kp / (kp ** 2 + tp ** 2)
bb = tp / (kp ** 2 + tp ** 2)
sym_equal("따름정리 5.4.9: a/(a²+b²) = κ", sp.simplify(aa / (aa ** 2 + bb ** 2)), kp, {kp: (0.2, 3), tp: (-3, 3)})
sym_equal("따름정리 5.4.9: b/(a²+b²) = τ", sp.simplify(bb / (aa ** 2 + bb ** 2)), tp, {kp: (0.2, 3), tp: (-3, 3)})
sym_equal("연습 5.4.3: κ = τ = 1이면 a = b = 1/2", (aa.subs({kp: 1, tp: 1}), bb.subs({kp: 1, tp: 1})), (sp.Rational(1, 2), sp.Rational(1, 2)))

# 예 5.4.10: 수치 구성 (RK4) = 정확한 나선을 강체운동한 것 ---------------------------------------------------
K0, TT0 = 1.0, 0.4


def frenet_ode(kap, tau):
    def f(s_, y):
        Tv, Nv, Bv = y[3:6], y[6:9], y[9:12]
        return np.concatenate([Tv, kap(s_) * Nv, -kap(s_) * Tv + tau(s_) * Bv, -tau(s_) * Nv])
    return f


ts = np.linspace(0, 12, 2401)
y0 = np.concatenate([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
Y = dgnum.rk4(frenet_ode(lambda _: K0, lambda _: TT0), y0, ts)
av, bv = K0 / (K0 ** 2 + TT0 ** 2), TT0 / (K0 ** 2 + TT0 ** 2)
cv = math.hypot(av, bv)
beta = sp.Matrix([a * sp.cos(s / sp.sqrt(a ** 2 + b ** 2)), a * sp.sin(s / sp.sqrt(a ** 2 + b ** 2)),
                  b * s / sp.sqrt(a ** 2 + b ** 2)]).subs({a: av, b: bv})
Tb = beta.diff(s)
Nb = beta.diff(s, 2) / sp.sqrt(beta.diff(s, 2).dot(beta.diff(s, 2)))
Bb = Tb.cross(Nb)
Fb0 = np.array(sp.Matrix.hstack(Tb, Nb, Bb).subs(s, 0).evalf(), float)
Aex = np.eye(3) @ Fb0.T                                # 목표 틀 = 표준기저
bf = sp.lambdify(s, list(beta), "numpy")
exact = np.array([Aex @ (np.array(bf(x), float) - np.array(bf(0.0), float)) for x in ts])
close("예 5.4.10: RK4 해 = 정확한 나선(a≈0.862, b≈0.345)을 강체운동한 것", Y[:, :3], exact, tol=1e-8)
close("예 5.4.10: a = 1/1.16", av, 1 / 1.16, tol=1e-15)
Fn = Y[:, 3:12].reshape(-1, 3, 3).transpose(0, 2, 1)
close("정리 5.4.8 증명: 수치해의 틀은 정규직교로 남는다", np.einsum("kij,kil->kjl", Fn, Fn), np.broadcast_to(np.eye(3), Fn.shape), tol=1e-9)
close("정리 5.4.8 증명: 수치해의 틀의 행렬식 = 1", np.linalg.det(Fn), np.ones(len(ts)), tol=1e-9)

# 비예 5.4.11: 곡선 A(평면)와 B(두 평면) ----------------------------------------------------------
f = lambda x: math.exp(-1 / x) if x > 0 else 0.0
fp = lambda x: math.exp(-1 / x) / x ** 2 if x > 0 else 0.0
fpp = lambda x: math.exp(-1 / x) * (1 - 2 * x) / x ** 4 if x > 0 else 0.0


def curv(d1, d2):
    return np.linalg.norm(np.cross(d1, d2)) / np.linalg.norm(d1) ** 3


for x in (-1.3, -0.4, 0.3, 0.9, 1.6):
    dA1 = np.array([1, fp(x) - fp(-x), 0]); dA2 = np.array([0, fpp(x) + fpp(-x), 0])
    dB1 = np.array([1, -fp(-x), fp(x)]); dB2 = np.array([0, fpp(-x), fpp(x)])
    close(f"비예 5.4.11: t = {x}에서 두 곡선의 속력이 같다", np.linalg.norm(dA1), np.linalg.norm(dB1), tol=1e-14)
    close(f"비예 5.4.11: t = {x}에서 두 곡선의 곡률이 같다", curv(dA1, dA2), curv(dB1, dB2), tol=1e-12)
PA = [np.array([x, f(x) + f(-x), 0.0]) for x in (-1.0, 0.5, 2.0)]
check("비예 5.4.11: A는 xy평면에 있다", all(P[2] == 0 for P in PA))
P1, P2, Q = (np.array([x, f(-x), f(x)]) for x in (-1.0, -2.0, 1.0))
check("비예 5.4.11: B는 한 평면에 있지 않다 (det ≠ 0)", abs(np.linalg.det(np.stack([P1, P2, Q]))) > 1e-2)

# 연습 5.4.6: xy평면의 원과 방향을 뒤집은 원 -------------------------------------------------------
r = sp.symbols("r", positive=True)
circ = sp.Matrix([r * sp.cos(s / r), r * sp.sin(s / r), 0])
circ_bar = sp.Matrix([r * sp.cos(s / r), -r * sp.sin(s / r), 0])
Ax = sp.diag(1, -1, -1)
check("연습 5.4.6: A = diag(1,-1,-1)은 직교행렬, det = 1", Ax.T * Ax == sp.eye(3) and Ax.det() == 1)
check("연습 5.4.6: Aγ = γ̄", sp.simplify(Ax * circ - circ_bar) == sp.zeros(3, 1))
check("연습 5.4.6: 부호곡률은 1/r, -1/r로 다르다",
      sp.simplify(dgsym.signed_curvature(circ, s) - 1 / r) == 0 and sp.simplify(dgsym.signed_curvature(circ_bar, s) + 1 / r) == 0)

summary()
