"""dgnum 수치 테스트. 실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 tools/tests/test_dgnum.py``."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import sympy as sp

import dgnum
from dgcheck import check, close, summary
from dgsym import EXAMPLES

sph = EXAMPLES["sphere"]
th, ph = sph["coords"]
(r,) = sph["params"]
g = sp.diag(sph["expected"]["E"], sph["expected"]["G"])
Gamma = dgnum.numeric_christoffel(g, (th, ph), params={r: 1})

# rk4 자체: y' = y, y(0) = 1 → e
ts = np.linspace(0, 1, 201)
close("rk4: y' = y의 해 y(1) = e", dgnum.rk4(lambda t, y: y, [1.0], ts)[-1, 0], np.e, tol=1e-9)

# 1. 위도원 평행이동: 회전각 2π cos θ0 (mod 2π), 길이 보존 (§7 평행이동 검증 예)
ts = np.linspace(0, 2 * np.pi, 4001)
for th0 in (np.pi / 6, np.pi / 3, 0.9 * np.pi / 2, 2 * np.pi / 3):
    x_of_t = lambda t, th0=th0: np.array([th0, t])
    dx_of_t = lambda t: np.array([0.0, 1.0])
    V = dgnum.parallel_transport(Gamma, x_of_t, dx_of_t, np.array([1.0, 0.0]), ts)
    # 정규직교 틀 (e_θ, e_φ) = (∂_θ, ∂_φ / sin θ0)에서의 성분
    a, b = V[:, 0], np.sin(th0) * V[:, 1]
    close(f"S²(1) 위도원 θ0 = {th0:.4f}: 길이 보존", np.hypot(a, b), np.ones_like(a), tol=1e-8)
    # 해석해: (a, b) = (cos(t cos θ0), −sin(t cos θ0)), 즉 틀 (e_θ, e_φ)에서 시계 방향으로 t cos θ0만큼 회전
    close(f"S²(1) 위도원 θ0 = {th0:.4f}: 한 바퀴 후 벡터 = 회전각 2π cos θ0의 회전",
          [a[-1], b[-1]], [np.cos(2 * np.pi * np.cos(th0)), -np.sin(2 * np.pi * np.cos(th0))], tol=1e-8)
    angle = (-np.arctan2(b[-1], a[-1])) % (2 * np.pi)
    close(f"S²(1) 위도원 θ0 = {th0:.4f}: 회전각 ≡ 2π cos θ0 (mod 2π)",
          angle, (2 * np.pi * np.cos(th0)) % (2 * np.pi), tol=1e-8)
    # 가우스-보네와의 관계: 2π cos θ0 ≡ −2π(1 − cos θ0) = −(극관의 넓이)·K (mod 2π)
    close(f"S²(1) 위도원 θ0 = {th0:.4f}: 2π cos θ0 ≡ −(극관 넓이)·K (mod 2π)",
          (2 * np.pi * np.cos(th0)) % (2 * np.pi), (-2 * np.pi * (1 - np.cos(th0))) % (2 * np.pi),
          tol=1e-12)

# 2. 적도에서 출발한 측지선은 대원 위에 머문다
def embed(q):
    q = np.atleast_2d(q)
    return np.stack([np.sin(q[:, 0]) * np.cos(q[:, 1]),
                     np.sin(q[:, 0]) * np.sin(q[:, 1]),
                     np.cos(q[:, 0])], axis=1)


alpha = np.deg2rad(40)                       # 적도와 이루는 각 (극을 지나지 않게 < 90°)
x0 = np.array([np.pi / 2, 0.0])
v0 = np.array([-np.sin(alpha), np.cos(alpha)])  # 적도에서 g = diag(1, 1)이므로 단위속력
ts = np.linspace(0, 2 * np.pi, 4001)
xs, vs = dgnum.integrate_geodesic(Gamma, x0, v0, ts)
P = embed(xs)
p0 = embed(x0)[0]
# ℝ³에서의 초기 속도: dP = P_θ v^θ + P_φ v^φ
P_th = np.array([np.cos(x0[0]) * np.cos(x0[1]), np.cos(x0[0]) * np.sin(x0[1]), -np.sin(x0[0])])
P_ph = np.array([-np.sin(x0[0]) * np.sin(x0[1]), np.sin(x0[0]) * np.cos(x0[1]), 0.0])
w0 = P_th * v0[0] + P_ph * v0[1]
nrm = np.cross(p0, w0)
nrm /= np.linalg.norm(nrm)
close("S²(1) 측지선: 모든 점이 대원의 평면 위 (<P(t), n> = 0)", P @ nrm, np.zeros(len(ts)), tol=1e-8)
speed2 = vs[:, 0] ** 2 + np.sin(xs[:, 0]) ** 2 * vs[:, 1] ** 2
close("S²(1) 측지선: 속력 보존", speed2, np.ones(len(ts)), tol=1e-8)
close("S²(1) 측지선: 길이 2π 후 출발점으로 돌아옴", P[-1], p0, tol=1e-7)
check("S²(1) 측지선: 극을 지나지 않음 (sin θ > 0.5)", bool(np.min(np.sin(xs[:, 0])) > 0.5))

summary()
