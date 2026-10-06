"""Figure 0.5.1: 왜곡 없는 지도는 없다 — Mercator(conformal)와 Lambert/Archimedes(넓이 보존) (0.5절).

unit sphere의 standard parametrization x(θ, φ) (dgsym.EXAMPLES["sphere"], r = 1).
위: Mercator projection ψ(x(θ, φ)) = (φ, log cot(θ/2)) (Section 7.4의 Example 7.4.10).
아래: Lambert cylindrical equal-area = Archimedes' projection ψ(x(θ, φ)) = (φ, cos θ) (Example 7.3.12).
graticule: 경도 30°, 위도 30° 간격(회색). Tissot indicatrix: sphere 위의 같은 크기의 작은 원(반지름 ε = 0.15)의
상 dψ_p(ε 단위원) (하늘색). 경도 φ = k·π/3, 위도 0°, ±30°, ±60°에 그린다.
Mercator에서는 반지름 ε/sin θ인 원(모양 보존, 크기 변화), Lambert에서는 반축 ε/sin θ(가로), ε sin θ(세로)인
ellipse(넓이 πε² 일정, 모양 변화).
자기검사: 두 map의 dψ를 sympy로 미분해 위의 반축과 넓이를 확인한다. Mercator는 위도 ±75°에서 자른다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-5-1-map-projections.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Ellipse

import dgsym

C = dgfig.COLORS
EPS = 0.15
ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
E, F, G = (e.subs(rs, 1) for e in dgsym.first_ff(ex["expr"], th, ph, ex["positive"]))

maps = {
    "mercator": sp.Matrix([ph, sp.log(sp.cot(th / 2))]),
    "lambert": sp.Matrix([ph, sp.cos(th)]),
}
semi = {}
for name, psi in maps.items():
    J = psi.jacobian([th, ph])                         # dψ in (θ, φ) coordinates
    # orthonormal frame on the sphere: e_θ = x_θ/√E, e_φ = x_φ/√G
    A = sp.simplify(J * sp.diag(1 / sp.sqrt(E), 1 / sp.sqrt(G)).subs(sp.sqrt(sp.sin(th) ** 2), sp.sin(th)))
    A = sp.simplify(A.subs(sp.Abs(sp.sin(th)), sp.sin(th)))
    semi[name] = sp.lambdify(th, A, "numpy")
    if name == "mercator":
        assert sp.simplify(A.T * A - sp.eye(2) / sp.sin(th) ** 2) == sp.zeros(2, 2)      # conformal
    else:
        assert sp.simplify(sp.Abs(A.det()) - 1) == 0                                     # 넓이 보존
        assert sp.simplify(A.T * A - sp.diag(sp.sin(th) ** 2, 1 / sp.sin(th) ** 2)) == sp.zeros(2, 2)

fig = plt.figure(figsize=(5.2, 5.4))
axM = fig.add_axes([0.0, 0.36, 1.0, 0.64])
axL = fig.add_axes([0.0, 0.0, 1.0, 0.32])
lat_cut = np.deg2rad(75)


def to_plane(name, theta, phi):
    if name == "mercator":
        return phi, np.log(1 / np.tan(theta / 2))
    return phi, np.cos(theta)


for name, ax in (("mercator", axM), ("lambert", axL)):
    tmin = np.pi / 2 - lat_cut if name == "mercator" else 1e-3
    tmax = np.pi - tmin
    for lat in np.deg2rad(np.arange(-60, 61, 30)):
        x_, y_ = to_plane(name, (np.pi / 2 - lat) * np.ones(2), np.array([0, 2 * np.pi]))
        ax.plot(x_, y_, color="#A0A0A0", lw=0.5, zorder=1)
    for k in range(13):
        tt = np.linspace(tmin, tmax, 100)
        x_, y_ = to_plane(name, tt, k * np.pi / 6 * np.ones_like(tt))
        ax.plot(x_, y_, color="#A0A0A0", lw=0.5, zorder=1)
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        t0 = np.pi / 2 - lat
        A = np.array(semi[name](t0), float)
        sv = np.linalg.svd(A, compute_uv=False)
        if name == "mercator":
            assert np.allclose(sv, 1 / np.sin(t0))
        else:
            assert np.isclose(sv[0] * sv[1], 1) and np.isclose(sv.max(), 1 / np.sin(t0))
        for k in range(6):
            p0 = k * np.pi / 3 + np.pi / 6
            cx, cy = to_plane(name, np.array(t0), np.array(p0))
            # A의 열: e_θ, e_φ의 상. 화면 좌표 (가로 φ, 세로 y)
            w_, h_ = EPS * abs(A[0, 1]), EPS * abs(A[1, 0])
            ax.add_patch(Ellipse((float(cx), float(cy)), 2 * w_, 2 * h_, fc=C["region"], ec=C["tangent"],
                                 lw=0.7, alpha=0.9, zorder=3))
    x0, y0 = to_plane(name, np.array([tmin, tmax]), np.array([0, 0]))
    ax.plot([0, 2 * np.pi, 2 * np.pi, 0, 0], [y0[0], y0[0], y0[1], y0[1], y0[0]], color=C["main"], lw=0.8, zorder=2)
    dgfig.schematic_axes(ax, (-0.1, 2 * np.pi + 0.1), (min(y0) - 0.1, max(y0) + 0.1))

dgfig.save(fig, __file__)
