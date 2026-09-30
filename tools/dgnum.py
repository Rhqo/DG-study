"""DG-study 수치 헬퍼 (numpy만 사용; 이 환경에서는 scipy를 import할 수 없다 — 부록 E).

* ``rk4(f, y0, ts)``: 고정 격자 ts 위의 4차 룽게-쿠타. f(t, y) → dy/dt.
* ``numeric_christoffel(g, coords, params=None)``: dgsym.christoffel을 lambdify해
  ``Gamma(x) → ndarray Γ[k, i, j]`` (= Γ^k_{ij})를 돌려준다.
* ``integrate_geodesic(Gamma, x0, v0, ts)``: 측지선 방정식
  ẍ^k + Γ^k_{ij} ẋ^i ẋ^j = 0.  반환값 (x, v): 모양 (len(ts), n)씩.
* ``parallel_transport(Gamma, x_of_t, dx_of_t, V0, ts)``: 곡선을 따른 평행이동
  V̇^k + Γ^k_{ij}(x(t)) ẋ^i V^j = 0.  반환값 V: 모양 (len(ts), n).

인덱스 규칙은 dgsym과 같다: ``Gamma(x)[k, i, j]`` = Γ^k_{ij}.
"""

import numpy as np


def rk4(f, y0, ts):
    """고정 격자 ``ts``에서 y' = f(t, y)를 푼다. 반환값: 모양 (len(ts), dim)."""
    ts = np.asarray(ts, dtype=float)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    ys = np.empty((len(ts), y0.size))
    ys[0] = y0
    y = y0.copy()
    for n in range(len(ts) - 1):
        t, h = ts[n], ts[n + 1] - ts[n]
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float)
        k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[n + 1] = y
    return ys


def numeric_christoffel(g, coords, params=None, positive=()):
    """기호 계량 ``g``(sympy Matrix)로부터 수치 함수 Gamma(x) → Γ[k, i, j]를 만든다.

    ``params``: 계량에 들어 있는 매개변수의 값 {기호: 수} (예: {r: 1}).
    """
    import sympy as sp
    import dgsym

    g = sp.Matrix(g)
    if params:
        g = g.subs(params)
    n = len(coords)
    G = dgsym.christoffel(g, coords, positive)
    flat = [G[k][i][j] for k in range(n) for i in range(n) for j in range(n)]
    fn = sp.lambdify(coords, flat, modules="numpy")

    def Gamma(x):
        vals = fn(*np.asarray(x, dtype=float))
        return np.array([np.broadcast_to(v, ()) for v in vals], dtype=float).reshape(n, n, n)

    return Gamma


def integrate_geodesic(Gamma, x0, v0, ts):
    """측지선 방정식을 푼다. 반환값 (x, v), 각각 모양 (len(ts), n)."""
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    n = x0.size

    def f(t, y):
        x, v = y[:n], y[n:]
        acc = -np.einsum("kij,i,j->k", Gamma(x), v, v)
        return np.concatenate([v, acc])

    ys = rk4(f, np.concatenate([x0, v0]), ts)
    return ys[:, :n], ys[:, n:]


def parallel_transport(Gamma, x_of_t, dx_of_t, V0, ts):
    """곡선 x(t)를 따라 V0를 평행이동한다. 반환값 V: 모양 (len(ts), n).

    ``x_of_t(t)``, ``dx_of_t(t)``: 곡선과 그 속도(좌표 성분)를 주는 함수.
    """

    def f(t, V):
        return -np.einsum("kij,i,j->k", Gamma(x_of_t(t)), dx_of_t(t), V)

    return rk4(f, V0, ts)
