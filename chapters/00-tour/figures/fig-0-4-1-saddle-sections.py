"""Figure 0.4.1: saddle surface의 normal section과 principal curvature (0.4절, Example 0.4.4). 정적 SVG + 인터랙티브.

saddle surface z = x² − y² (dgsym.EXAMPLES["saddle"], 그래프 곡면 x(u, v) = (u, v, u² − v²)), u² + v² ≤ 1.
원점에서 N = (0, 0, 1) (위쪽), tangent plane z = 0 (하늘색), principal direction e₁ = (1, 0, 0), e₂ = (0, 1, 0) (파랑).
normal section: 평면 y = 0과의 교선 z = x² (주황, N 쪽으로 휜다, κ_n = 2),
               평면 x = 0과의 교선 z = −y² (주황 점선, N 반대쪽으로 휜다, κ_n = −2).
회색 점선: surface에 통째로 놓인 두 직선 x = ±y (κ_n = 0인 방향).
자기검사: dgsym.K_H로 원점에서 K = −4, H = 0 (§7). shape operator 행렬 = diag(2, −2) (Hessian).
Euler's formula κ_n(θ) = 2cos²θ − 2sin²θ가 θ = π/4에서 0.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-4-1-saddle-sections.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
ex = dgsym.EXAMPLES["saddle"]
u, v = ex["coords"]
X = ex["expr"]
K, H = dgsym.K_H(X, u, v)
pt = ex["point"]
assert sp.simplify(K.subs(pt) - ex["expected"]["K_at_origin"]) == 0
assert sp.simplify(H.subs(pt) - ex["expected"]["H_at_origin"]) == 0
W0 = dgsym.shape_operator(X, u, v).subs(pt)
assert W0 == sp.diag(2, -2)
th = sp.symbols("theta", real=True)
w = sp.Matrix([sp.cos(th), sp.sin(th)])
kn = sp.simplify((w.T * W0 * w)[0])
assert sp.simplify(kn - (2 * sp.cos(th) ** 2 - 2 * sp.sin(th) ** 2)) == 0 and kn.subs(th, sp.pi / 4) == 0

rho, phi_ = np.meshgrid(np.linspace(0, 1, 26), np.linspace(0, 2 * np.pi, 97), indexing="ij")
Xg, Yg = rho * np.cos(phi_), rho * np.sin(phi_)
Zg = Xg ** 2 - Yg ** 2

fig = plt.figure(figsize=(5.4, 4.4))
ax = dgfig.axes3d(fig, elev=24, azim=-58)
ax.set_proj_type("ortho")
dgfig.surface(ax, Xg, Yg, Zg, alpha=0.4, grid=True, grid_every=5, zorder=1)
dgfig.tangent_plane(ax, np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), size=0.45, zorder=3)
s = np.linspace(-1, 1, 200)
sec1 = np.stack([s, 0 * s, s ** 2], axis=1)
sec2 = np.stack([0 * s, s, -s ** 2], axis=1)
dgfig.curve3d(ax, sec1, role="accent", lw=2.0, zorder=6)
dgfig.curve3d(ax, sec2, role="accent", lw=2.0, ls=(0, (4, 2)), zorder=6)
d = np.linspace(-1, 1, 50) / np.sqrt(2)
for sg in (1, -1):
    line = np.stack([d, sg * d, 0 * d], axis=1)
    assert np.allclose(line[:, 2], line[:, 0] ** 2 - line[:, 1] ** 2)
    dgfig.curve3d(ax, line, role="aux", lw=1.0, ls=(0, (2, 2)), zorder=5)
dgfig.arrow3d(ax, np.zeros(3), np.array([0, 0, 1.0]), role="normal", scale=0.7, label=r"$\mathbf{N}$",
              label_offset=(0, 0, 0.08), zorder=10)
dgfig.arrow3d(ax, np.zeros(3), np.array([1.0, 0, 0]), role="tangent", scale=0.45, label=r"$\mathbf{e}_1$",
              label_offset=(0.02, -0.08, -0.06), zorder=10)
dgfig.arrow3d(ax, np.zeros(3), np.array([0, 1.0, 0]), role="tangent", scale=0.45, label=r"$\mathbf{e}_2$",
              label_offset=(0.06, 0.0, -0.08), zorder=10)
dgfig.point3d(ax, np.zeros(3), zorder=11)
ax.text(1.0, 0.0, 1.12, r"$\kappa_n = 2$", color=C["accent"], fontsize=12, zorder=12)
ax.text(0.0, 1.05, -1.25, r"$\kappa_n = -2$", color=C["accent"], fontsize=12, zorder=12)
dgfig.equal_aspect(ax, np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3), zoom=1.2)
dgfig.save(fig, __file__)

# 인터랙티브 판 -----------------------------------------------------------------------------
import plotly.graph_objects as go

pfig = go.Figure()
pfig.add_trace(go.Surface(x=Xg, y=Yg, z=Zg, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]], showscale=False,
                          opacity=0.8, hoverinfo="skip", lighting=dict(ambient=0.75, diffuse=0.4)))
pfig.add_trace(go.Scatter3d(x=sec1[:, 0], y=sec1[:, 1], z=sec1[:, 2], mode="lines",
                            line=dict(color="#E69F00", width=6), name="z = x² (κ_n = 2)"))
pfig.add_trace(go.Scatter3d(x=sec2[:, 0], y=sec2[:, 1], z=sec2[:, 2], mode="lines",
                            line=dict(color="#E69F00", width=6, dash="dash"), name="z = −y² (κ_n = −2)"))
for sg in (1, -1):
    pfig.add_trace(go.Scatter3d(x=d, y=sg * d, z=0 * d, mode="lines", line=dict(color="#777777", width=3, dash="dot"),
                                name="line x = %sy (κ_n = 0)" % ("" if sg > 0 else "−")))
pfig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.7], mode="lines", line=dict(color="#D55E00", width=6),
                            name="N"))
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0, y=1),
                   title=dict(text="Saddle surface: normal sections at the origin", x=0.5,
                              font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-4-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
