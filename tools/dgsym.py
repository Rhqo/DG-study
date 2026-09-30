"""DG-study 기호계산 모듈: GUIDELINES.md §6의 규약을 한 곳에 구현한다.

verify 스크립트와 그림 스크립트는 곡률 공식을 새로 짜지 말고 이 모듈을 쓴다 (§13).
기준 예제(§7)의 **유일한 코드 정의**는 ``EXAMPLES``다 (§3.6).

규약 요약 (§6):

* 곡선: 프레네 틀 (t, n, b), b' = -τ n  (do Carmo와 τ 부호 반대).
  κ = |γ'×γ''| / |γ'|³,  τ = <γ'×γ'', γ'''> / |γ'×γ''|².
* 곡면: N = (x_u × x_v)/|x_u × x_v|,  W_p = -dN_p,
  (E, F, G) = 제1기본형식,  (L, M, N) = <x_uu, N>, <x_uv, N>, <x_vv, N>,
  K = (LN - M²)/(EG - F²),  H = (EN - 2FM + GL) / (2(EG - F²)).
* 리만 (Lee IRM): ∇_{∂_i}∂_j = Γ^k_{ij} ∂_k,
  R(X,Y)Z = ∇_X∇_Y Z - ∇_Y∇_X Z - ∇_{[X,Y]} Z,
  R(∂_i,∂_j)∂_k = R_{ijk}^l ∂_l,  R_{ijkl} = g_{lm} R_{ijk}^m,
  Rc_{jk} = R_{ijk}^i,  S = g^{jk} Rc_{jk},
  K(v,w) = Rm(v,w,w,v) / (|v|²|w|² - <v,w>²).

인덱스 규칙: 이 모듈이 돌려주는 중첩 리스트는 모두 **수식의 인덱스 순서 그대로**다.

* ``christoffel(g, q)[k][i][j]``  = Γ^k_{ij}
* ``riemann(g, q)[i][j][k][l]``   = R_{ijk}^l
* ``riemann_lower(g, q)[i][j][k][l]`` = R_{ijkl}
* ``ricci(g, q)``                 = sympy Matrix, 성분 (j, k) = Rc_{jk}

``python3 tools/dgsym.py``를 실행하면 §7의 모든 값을 재현하는 자체 테스트가 돈다.
"""

import sympy as sp

__all__ = [
    "simp", "curvature_torsion", "frenet_frame",
    "first_ff", "unit_normal", "second_ff", "shape_operator", "K_H",
    "induced_metric", "christoffel", "riemann", "riemann_lower", "ricci",
    "scalar", "sectional", "R_apply", "round_sphere_metric", "EXAMPLES",
]


# ---------------------------------------------------------------------------
# 단순화 헬퍼
# ---------------------------------------------------------------------------

def simp(expr, positive=()):
    """sympy 식(또는 행렬)을 단순화한다.

    ``positive``에는 정의역에서 양수인 식들을 준다 (예: θ ∈ (0, π)이면 ``sin(θ)``).
    ``sqrt(sin(θ)**2)`` 같은 식이 ``sin(θ)``로 정리된다.
    """
    if isinstance(expr, sp.MatrixBase):
        return expr.applyfunc(lambda e: simp(e, positive))
    e = sp.simplify(expr)
    if positive:
        assumptions = sp.And(*[sp.Q.positive(p) for p in positive])
        e = sp.simplify(sp.refine(e, assumptions))
        e = sp.simplify(sp.refine(sp.factor(e), assumptions))
    return e


def _norm(v, positive=()):
    return simp(sp.sqrt(v.dot(v)), positive)


# ---------------------------------------------------------------------------
# 곡선 (§6.2)
# ---------------------------------------------------------------------------

def curvature_torsion(gamma, t, positive=()):
    """일반 매개변수 곡선 ``gamma(t)`` (3×1 Matrix)의 (κ, τ).

    κ = |γ'×γ''|/|γ'|³,  τ = <γ'×γ'', γ'''>/|γ'×γ''|²  (b' = -τ n 규약).
    γ'×γ'' = 0인 점(직선 부분)에서는 τ가 정의되지 않는다.
    """
    gamma = sp.Matrix(gamma)
    d1, d2, d3 = (gamma.diff(t, k) for k in (1, 2, 3))
    c = d1.cross(d2)
    kappa = simp(sp.sqrt(c.dot(c)) / sp.sqrt(d1.dot(d1)) ** 3, positive)
    tau = simp(c.dot(d3) / c.dot(c), positive)
    return kappa, tau


def frenet_frame(gamma, t, positive=()):
    """프레네 틀 (t, n, b) (각각 3×1 Matrix). b = t × n."""
    gamma = sp.Matrix(gamma)
    d1, d2 = gamma.diff(t), gamma.diff(t, 2)
    T = simp(d1 / sp.sqrt(d1.dot(d1)), positive)
    c = d1.cross(d2)
    B = simp(c / sp.sqrt(c.dot(c)), positive)
    Nn = simp(B.cross(T), positive)
    return T, Nn, B


# ---------------------------------------------------------------------------
# 곡면 (§6.3)
# ---------------------------------------------------------------------------

def first_ff(X, u, v, positive=()):
    """제1기본형식 계수 (E, F, G)."""
    X = sp.Matrix(X)
    Xu, Xv = X.diff(u), X.diff(v)
    return tuple(simp(a.dot(b), positive) for a, b in ((Xu, Xu), (Xu, Xv), (Xv, Xv)))


def unit_normal(X, u, v, positive=()):
    """단위법벡터 N = (x_u × x_v)/|x_u × x_v| (3×1 Matrix)."""
    X = sp.Matrix(X)
    n = X.diff(u).cross(X.diff(v))
    return simp(n / sp.sqrt(simp(n.dot(n), positive)), positive)


def second_ff(X, u, v, positive=()):
    """제2기본형식 계수 (L, M, N) = (<x_uu,N>, <x_uv,N>, <x_vv,N>)."""
    X = sp.Matrix(X)
    Nv = unit_normal(X, u, v, positive)
    return tuple(simp(Nv.dot(w), positive)
                 for w in (X.diff(u, 2), X.diff(u).diff(v), X.diff(v, 2)))


def shape_operator(X, u, v, positive=()):
    """형태작용소 W_p = -dN_p의 행렬 (기저 x_u, x_v): I^{-1} II."""
    E, F, G = first_ff(X, u, v, positive)
    L, M, N = second_ff(X, u, v, positive)
    I = sp.Matrix([[E, F], [F, G]])
    II = sp.Matrix([[L, M], [M, N]])
    return simp(I.inv() * II, positive)


def K_H(X, u, v, positive=()):
    """가우스 곡률 K와 평균곡률 H (법벡터는 unit_normal의 방향)."""
    E, F, G = first_ff(X, u, v, positive)
    L, M, N = second_ff(X, u, v, positive)
    det = E * G - F ** 2
    K = simp((L * N - M ** 2) / det, positive)
    H = simp((E * N - 2 * F * M + G * L) / (2 * det), positive)
    return K, H


# ---------------------------------------------------------------------------
# 리만 기하 (§6.5, Lee IRM 규약)
# ---------------------------------------------------------------------------

def induced_metric(X, coords, positive=()):
    """매개화 X(q^1, …, q^n) ⊂ ℝ^N이 유도하는 계량 g_ij = <∂_i X, ∂_j X>."""
    X = sp.Matrix(X)
    D = [X.diff(q) for q in coords]
    n = len(coords)
    return sp.Matrix(n, n, lambda i, j: simp(D[i].dot(D[j]), positive))


def christoffel(g, coords, positive=()):
    """레비치비타 접속의 크리스토펠 기호. 반환값 ``G[k][i][j]`` = Γ^k_{ij}.

    Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij}).
    """
    g = sp.Matrix(g)
    n = len(coords)
    ginv = simp(g.inv(), positive)
    G = [[[0] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = sum(ginv[k, l] * (sp.diff(g[j, l], coords[i])
                                      + sp.diff(g[i, l], coords[j])
                                      - sp.diff(g[i, j], coords[l]))
                        for l in range(n)) / 2
                G[k][i][j] = simp(s, positive)
    return G


def riemann(g, coords, positive=(), Gamma=None):
    """곡률 연산자의 성분. 반환값 ``R[i][j][k][l]`` = R_{ijk}^l.

    R(∂_i,∂_j)∂_k = R_{ijk}^l ∂_l, 즉
    R_{ijk}^l = ∂_iΓ^l_{jk} − ∂_jΓ^l_{ik} + Γ^m_{jk}Γ^l_{im} − Γ^m_{ik}Γ^l_{jm}.
    """
    n = len(coords)
    Gm = Gamma if Gamma is not None else christoffel(g, coords, positive)
    R = [[[[0] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    s = sp.diff(Gm[l][j][k], coords[i]) - sp.diff(Gm[l][i][k], coords[j])
                    s += sum(Gm[m][j][k] * Gm[l][i][m] - Gm[m][i][k] * Gm[l][j][m]
                             for m in range(n))
                    R[i][j][k][l] = simp(s, positive)
    return R


def riemann_lower(g, coords, positive=(), R=None):
    """곡률텐서 Rm의 성분. 반환값 ``Rl[i][j][k][l]`` = R_{ijkl} = g_{lm} R_{ijk}^m."""
    g = sp.Matrix(g)
    n = len(coords)
    R = R if R is not None else riemann(g, coords, positive)
    return [[[[simp(sum(g[l, m] * R[i][j][k][m] for m in range(n)), positive)
               for l in range(n)] for k in range(n)] for j in range(n)] for i in range(n)]


def ricci(g, coords, positive=(), R=None):
    """리치 곡률 Rc_{jk} = R_{ijk}^i (sympy Matrix)."""
    n = len(coords)
    R = R if R is not None else riemann(g, coords, positive)
    return sp.Matrix(n, n, lambda j, k: simp(sum(R[i][j][k][i] for i in range(n)), positive))


def scalar(g, coords, positive=(), R=None):
    """스칼라곡률 S = g^{jk} Rc_{jk}."""
    g = sp.Matrix(g)
    n = len(coords)
    Rc = ricci(g, coords, positive, R)
    ginv = g.inv()
    return simp(sum(ginv[j, k] * Rc[j, k] for j in range(n) for k in range(n)), positive)


def sectional(g, coords, v, w, positive=(), R=None):
    """단면곡률 K(v,w) = Rm(v,w,w,v) / (|v|²|w|² − <v,w>²).

    ``v``, ``w``는 좌표기저에 대한 성분 리스트.
    """
    g = sp.Matrix(g)
    n = len(coords)
    Rl = riemann_lower(g, coords, positive, R)
    v, w = list(v), list(w)
    num = sum(Rl[i][j][k][l] * v[i] * w[j] * w[k] * v[l]
              for i in range(n) for j in range(n) for k in range(n) for l in range(n))
    ip = lambda a, b: sum(g[i, j] * a[i] * b[j] for i in range(n) for j in range(n))
    den = ip(v, v) * ip(w, w) - ip(v, w) ** 2
    return simp(num / den, positive)


def R_apply(R, X, Y, Z):
    """벡터 R(X,Y)Z의 성분 리스트: (R(X,Y)Z)^l = R_{ijk}^l X^i Y^j Z^k."""
    n = len(X)
    return [sp.simplify(sum(R[i][j][k][l] * X[i] * Y[j] * Z[k]
                            for i in range(n) for j in range(n) for k in range(n)))
            for l in range(n)]


def round_sphere_metric(n, r):
    """반지름 r인 Sⁿ의 둥근 계량을 초구면좌표로 준다.

    좌표 ψ_1, …, ψ_n: ψ_1, …, ψ_{n−1} ∈ (0, π), ψ_n ∈ (0, 2π).
    g = r²(dψ_1² + sin²ψ_1 dψ_2² + sin²ψ_1 sin²ψ_2 dψ_3² + ⋯).
    반환값: (g, coords, domain, positive)
    """
    coords = sp.symbols(f"psi1:{n + 1}", real=True)
    diag = []
    fac = sp.Integer(1)
    for i in range(n):
        diag.append(r ** 2 * fac)
        fac = fac * sp.sin(coords[i]) ** 2
    domain = {c: (0, sp.pi) for c in coords[:-1]}
    domain[coords[-1]] = (0, 2 * sp.pi)
    positive = tuple(sp.sin(c) for c in coords[:-1])
    return sp.diag(*diag), coords, domain, positive


# ---------------------------------------------------------------------------
# 기준 예제 (§7) — 유일한 코드 정의
# ---------------------------------------------------------------------------

def _examples():
    t = sp.symbols("t", real=True)
    r = sp.symbols("r", positive=True)
    a = sp.symbols("a", positive=True)
    b = sp.symbols("b", real=True, nonzero=True)
    R = sp.symbols("R", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    u, v = sp.symbols("u v", real=True)
    x, y = sp.symbols("x y", real=True)
    yp = sp.symbols("y", positive=True)
    rho = sp.Function("rho", positive=True)
    z = sp.Function("z", real=True)
    f = sp.Function("f", real=True)

    ex = {}

    ex["circle"] = dict(
        kind="curve", name="원 (반지름 r)", params=(r,), coords=(t,),
        domain={t: (0, 2 * sp.pi), r: (0.5, 2)},
        expr=sp.Matrix([r * sp.cos(t), r * sp.sin(t), 0]),
        positive=(),
        expected=dict(kappa=1 / r, tau=sp.Integer(0)),
    )

    ex["helix"] = dict(
        kind="curve", name="나선 (a > 0, b ≠ 0)", params=(a, b), coords=(t,),
        domain={t: (0, 2 * sp.pi), a: (0.5, 2), b: (0.3, 1.5)},
        expr=sp.Matrix([a * sp.cos(t), a * sp.sin(t), b * t]),
        positive=(),
        note="b > 0이면 오른손 나선이고 τ > 0",
        expected=dict(kappa=a / (a ** 2 + b ** 2), tau=b / (a ** 2 + b ** 2)),
    )

    ex["sphere"] = dict(
        kind="surface", name="구면 S²(r)", params=(r,), coords=(th, ph),
        domain={th: (0, sp.pi), ph: (0, 2 * sp.pi), r: (0.5, 2)},
        expr=sp.Matrix([r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph), r * sp.cos(th)]),
        positive=(sp.sin(th),),
        normal="바깥쪽: N = x/r",
        expected=dict(
            E=r ** 2, F=sp.Integer(0), G=r ** 2 * sp.sin(th) ** 2,
            L=-r, M=sp.Integer(0), N=-r * sp.sin(th) ** 2,
            kappa12=(-1 / r, -1 / r), K=1 / r ** 2, H=-1 / r,
            normal=sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)]),
            # Γ^k_{ij}: 인덱스 0 = θ, 1 = φ. 나머지는 0.
            christoffel={(0, 1, 1): -sp.sin(th) * sp.cos(th),
                         (1, 0, 1): sp.cos(th) / sp.sin(th),
                         (1, 1, 0): sp.cos(th) / sp.sin(th)},
        ),
    )

    ex["cylinder"] = dict(
        kind="surface", name="원기둥 (반지름 r)", params=(r,), coords=(u, v),
        domain={u: (0, 2 * sp.pi), v: (-2, 2), r: (0.5, 2)},
        expr=sp.Matrix([r * sp.cos(u), r * sp.sin(u), v]),
        positive=(),
        normal="바깥쪽: N = (cos u, sin u, 0)",
        expected=dict(E=r ** 2, F=sp.Integer(0), G=sp.Integer(1),
                      L=-r, M=sp.Integer(0), N=sp.Integer(0),
                      K=sp.Integer(0), H=-1 / (2 * r),
                      normal=sp.Matrix([sp.cos(u), sp.sin(u), 0])),
    )

    ex["torus"] = dict(
        kind="surface", name="원환면 (0 < r < R)", params=(R, r), coords=(u, v),
        domain={u: (0, 2 * sp.pi), v: (0, 2 * sp.pi), r: (0.2, 0.9), R: (1.1, 2.0)},
        expr=sp.Matrix([(R + r * sp.cos(u)) * sp.cos(v), (R + r * sp.cos(u)) * sp.sin(v), r * sp.sin(u)]),
        positive=(R + r * sp.cos(u),),
        normal="x_u × x_v 방향 = 안쪽(관의 중심원 쪽): N = −(cos u cos v, cos u sin v, sin u)",
        expected=dict(E=r ** 2, F=sp.Integer(0), G=(R + r * sp.cos(u)) ** 2,
                      K=sp.cos(u) / (r * (R + r * sp.cos(u))),
                      normal=-sp.Matrix([sp.cos(u) * sp.cos(v), sp.cos(u) * sp.sin(v), sp.sin(u)])),
    )

    ex["revolution"] = dict(
        kind="surface", name="회전면 (ρ > 0)", params=(), coords=(u, v),
        domain={},
        expr=sp.Matrix([rho(u) * sp.cos(v), rho(u) * sp.sin(v), z(u)]),
        positive=(),
        functions=(rho, z),
        expected=dict(E=sp.diff(rho(u), u) ** 2 + sp.diff(z(u), u) ** 2,
                      F=sp.Integer(0), G=rho(u) ** 2),
    )

    ex["graph"] = dict(
        kind="surface", name="그래프 곡면 z = f(u, v)", params=(), coords=(u, v),
        domain={},
        expr=sp.Matrix([u, v, f(u, v)]),
        positive=(),
        functions=(f,),
        normal="위쪽: N = (−f_u, −f_v, 1)/√(1 + f_u² + f_v²)",
        expected=dict(
            K=(sp.diff(f(u, v), u, 2) * sp.diff(f(u, v), v, 2) - sp.diff(f(u, v), u, v) ** 2)
            / (1 + sp.diff(f(u, v), u) ** 2 + sp.diff(f(u, v), v) ** 2) ** 2),
    )

    ex["saddle"] = dict(
        kind="surface", name="안장면 z = x² − y²", params=(), coords=(u, v),
        domain={u: (-1, 1), v: (-1, 1)},
        expr=sp.Matrix([u, v, u ** 2 - v ** 2]),
        positive=(),
        normal="위쪽 (그래프 곡면과 같음)",
        point={u: 0, v: 0},
        expected=dict(K_at_origin=sp.Integer(-4), H_at_origin=sp.Integer(0)),
    )

    ex["hyperbolic_plane"] = dict(
        kind="metric", name="쌍곡평면 𝕌² (상반평면 모형)", params=(), coords=(x, yp),
        domain={x: (-2, 2), yp: (0.2, 3)},
        expr=sp.diag(1 / yp ** 2, 1 / yp ** 2),
        positive=(),
        expected=dict(
            K=sp.Integer(-1),
            # Γ^k_{ij}: 인덱스 0 = x, 1 = y. 나머지는 0.
            christoffel={(0, 0, 1): -1 / yp, (0, 1, 0): -1 / yp,
                         (1, 0, 0): 1 / yp, (1, 1, 1): -1 / yp},
        ),
    )
    return ex


EXAMPLES = _examples()


# ---------------------------------------------------------------------------
# 자체 테스트
# ---------------------------------------------------------------------------

def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dgcheck import check, sym_equal, summary

    ex = EXAMPLES

    # 곡선
    for key in ("circle", "helix"):
        e = ex[key]
        (t,) = e["coords"]
        kappa, tau = curvature_torsion(e["expr"], t, e["positive"])
        sym_equal(f"§7 {e['name']}: κ", kappa, e["expected"]["kappa"], e["domain"])
        sym_equal(f"§7 {e['name']}: τ", tau, e["expected"]["tau"], e["domain"])

    # 나선: b' = −τ|γ'| n (b′ = −τn 부호 규약 확인)
    e = ex["helix"]
    (t,) = e["coords"]
    T, Nn, B = frenet_frame(e["expr"], t)
    speed = sp.sqrt(e["expr"].diff(t).dot(e["expr"].diff(t)))
    _, tau = curvature_torsion(e["expr"], t)
    sym_equal("§6.2 나선: db/dt = −τ|γ'| n (b′ = −τn 규약)", B.diff(t), -tau * speed * Nn, e["domain"])
    kappa, _ = curvature_torsion(e["expr"], t)
    sym_equal("§6.2 나선: dt/dt = κ|γ'| n", T.diff(t), kappa * speed * Nn, e["domain"])

    # 곡면
    for key in ("sphere", "cylinder", "torus", "revolution"):
        e = ex[key]
        u, v = e["coords"]
        X, pos, dom, exp = e["expr"], e["positive"], e["domain"], e["expected"]
        E, F, G = first_ff(X, u, v, pos)
        for nm, val in zip("EFG", (E, F, G)):
            sym_equal(f"§7 {e['name']}: {nm}", val, exp[nm], dom)
        if key == "revolution":
            continue
        L, M, N = second_ff(X, u, v, pos)
        for nm, val in zip("LMN", (L, M, N)):
            if nm in exp:
                sym_equal(f"§7 {e['name']}: {nm}", val, exp[nm], dom)
        K, H = K_H(X, u, v, pos)
        sym_equal(f"§7 {e['name']}: K", K, exp["K"], dom)
        if "H" in exp:
            sym_equal(f"§7 {e['name']}: H", H, exp["H"], dom)
        if "normal" in exp:
            sym_equal(f"§7 {e['name']}: 법벡터 방향 ({e['normal']})",
                      unit_normal(X, u, v, pos), exp["normal"], dom)
        if "kappa12" in exp:
            W = shape_operator(X, u, v, pos)
            sym_equal(f"§7 {e['name']}: W_p = κ Id, κ₁ = κ₂", W,
                      sp.diag(*exp["kappa12"]), dom)

    # 원환면 K의 부호: 바깥쪽 적도 u = 0에서 K > 0, 안쪽 u = π에서 K < 0
    e = ex["torus"]
    u, v = e["coords"]
    R_, r_ = e["params"]
    Kt = e["expected"]["K"]
    check("§7 원환면: u = 0에서 K > 0", bool(Kt.subs(u, 0).is_positive))
    check("§7 원환면: u = π에서 K < 0", bool(Kt.subs({u: sp.pi, R_: 2, r_: 1}) < 0))

    # 그래프 곡면: K 공식 (일반 f)
    e = ex["graph"]
    u, v = e["coords"]
    K, _ = K_H(e["expr"], u, v)
    sym_equal("§7 그래프 곡면: K = (f_uu f_vv − f_uv²)/(1 + f_u² + f_v²)²", K, e["expected"]["K"])

    # 안장면: 원점에서 K = −4, H = 0
    e = ex["saddle"]
    u, v = e["coords"]
    K, H = K_H(e["expr"], u, v)
    sym_equal("§7 안장면: 원점에서 K = −4", K.subs(e["point"]), e["expected"]["K_at_origin"])
    sym_equal("§7 안장면: 원점에서 H = 0", H.subs(e["point"]), e["expected"]["H_at_origin"])

    # 구면: 크리스토펠 기호 (유도된 계량)와 K의 두 경로 교차검증
    e = ex["sphere"]
    th, ph = e["coords"]
    (r,) = e["params"]
    g = induced_metric(e["expr"], (th, ph), e["positive"])
    sym_equal("구면: 유도된 계량 = diag(E, G)", g,
              sp.diag(e["expected"]["E"], e["expected"]["G"]), e["domain"])
    Gm = christoffel(g, (th, ph), e["positive"])
    for k in range(2):
        for i in range(2):
            for j in range(2):
                want = e["expected"]["christoffel"].get((k, i, j), 0)
                sym_equal(f"§7 구면: Γ^{('θ', 'φ')[k]}_{('θ', 'φ')[i]}{('θ', 'φ')[j]}",
                          Gm[k][i][j], want, e["domain"])
    Ksec = sectional(g, (th, ph), [1, 0], [0, 1], e["positive"])
    Ksurf, _ = K_H(e["expr"], th, ph, e["positive"])
    sym_equal("구면: 단면곡률(리만 공식) = 1/r²", Ksec, 1 / r ** 2, e["domain"])
    sym_equal("구면: 단면곡률(리만) = 가우스 곡률(곡면 공식) — 부호 규약 교차검증",
              Ksec, Ksurf, e["domain"])

    # S²(r): 상수곡률 항등식 R(X,Y)Z = c(<Y,Z>X − <X,Z>Y), c = 1/r²
    Rs = riemann(g, (th, ph), e["positive"])
    c = 1 / r ** 2
    basis = ([1, 0], [0, 1])
    ok = True
    for X_ in basis:
        for Y_ in basis:
            for Z_ in basis:
                lhs = R_apply(Rs, X_, Y_, Z_)
                ip = lambda A, B: sum(g[i, j] * A[i] * B[j] for i in range(2) for j in range(2))
                rhs = [c * (ip(Y_, Z_) * X_[l] - ip(X_, Z_) * Y_[l]) for l in range(2)]
                ok &= all(sp.simplify(sp.refine(lhs[l] - rhs[l], sp.Q.positive(sp.sin(th)))) == 0
                          for l in range(2))
    check("S²(r): R(X,Y)Z = (1/r²)(<Y,Z>X − <X,Z>Y) (좌표기저 전부)", ok)

    # 쌍곡평면
    e = ex["hyperbolic_plane"]
    q = e["coords"]
    Gm = christoffel(e["expr"], q)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                want = e["expected"]["christoffel"].get((k, i, j), 0)
                sym_equal(f"§7 𝕌²: Γ^{'xy'[k]}_{'xy'[i]}{'xy'[j]}", Gm[k][i][j], want, e["domain"])
    Kh = sectional(e["expr"], q, [1, 0], [0, 1])
    sym_equal("§7 𝕌²: K ≡ −1", Kh, e["expected"]["K"], e["domain"])
    sym_equal("𝕌²: 스칼라곡률 S = 2K = −2", scalar(e["expr"], q), -2, e["domain"])

    # S³(r): Rc = (2/r²) g, S = 6/r²
    r = sp.symbols("r", positive=True)
    g3, q3, dom3, pos3 = round_sphere_metric(3, r)
    dom3 = dict(dom3)
    dom3[r] = (0.5, 2)
    R3 = riemann(g3, q3, pos3)
    sym_equal("S³(r): Rc = (2/r²) g", ricci(g3, q3, pos3, R3), (2 / r ** 2) * g3, dom3)
    sym_equal("S³(r): S = 6/r²", scalar(g3, q3, pos3, R3), 6 / r ** 2, dom3)
    sym_equal("S³(r): K(∂_1, ∂_3) = 1/r²", sectional(g3, q3, [1, 0, 0], [0, 0, 1], pos3, R3),
              1 / r ** 2, dom3)

    summary()


if __name__ == "__main__":
    _selftest()
