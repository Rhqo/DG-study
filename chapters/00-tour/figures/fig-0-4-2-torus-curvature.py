"""Figure 0.4.2: torus of revolution의 Gaussian curvature의 부호 (0.4절, Example 0.4.6, Exercise 0.4.3). 정적 + 인터랙티브.

torus of revolution (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8. K = cos u / (r(R + r cos u)) (dgsym.K_H, §7).
diverging colormap RdBu_r, 0이 중앙(TwoSlopeNorm): 빨강 K > 0 (elliptic), 파랑 K < 0 (hyperbolic),
검은 원 u = ±π/2 (parabolic, K = 0).
인터랙티브: 같은 식, hover에 u, v, K.
자기검사: K_H의 K가 §7과 같다. K(0) = 1/(r(R + r)) ≈ 0.446, K(π) = −1/(r(R − r)) ≈ −1.042.
∫∫ K dA = 0 (Section 0.5의 Gauss–Bonnet와 맞는다, 수치).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-4-2-torus-curvature.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.colors import TwoSlopeNorm

import dgsym

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
vals = {Rs: RV, rs: rV}
Ksym, Hsym = dgsym.K_H(ex["expr"], u, v, ex["positive"])
assert sp.simplify(Ksym - ex["expected"]["K"]) == 0
Kf = sp.lambdify(u, Ksym.subs(vals), "numpy")
X = sp.lambdify((u, v), list(ex["expr"].subs(vals)), "numpy")
assert np.isclose(Kf(0.0), 1 / (rV * (RV + rV))) and np.isclose(Kf(np.pi), -1 / (rV * (RV - rV)))
assert abs(Kf(np.pi / 2)) < 1e-15
uu = np.linspace(0, 2 * np.pi, 4001)
fu = Kf(uu) * rV * (RV + rV * np.cos(uu))
total = np.sum(0.5 * (fu[1:] + fu[:-1]) * np.diff(uu)) * 2 * np.pi
assert abs(total) < 1e-9


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


Ug, Vg = np.meshgrid(np.linspace(-np.pi, np.pi, 121), np.linspace(0, 2 * np.pi, 181), indexing="ij")
S = xmap(Ug, Vg)
Kc = Kf(0.5 * (Ug[:-1, :-1] + Ug[1:, 1:]))
norm = TwoSlopeNorm(vmin=Kf(np.pi), vcenter=0.0, vmax=Kf(0.0))
cmap = plt.get_cmap("RdBu_r")

fig = plt.figure(figsize=(6.0, 3.6))
ax = dgfig.axes3d(fig, elev=36, azim=-60)
ax.set_position([0.0, 0.0, 0.8, 1.0])
ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], facecolors=cmap(norm(Kc)), rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=1)
view = dgfig.view_vector(ax)
tt = np.linspace(0, 2 * np.pi, 400)
for u0 in (np.pi / 2, -np.pi / 2):
    c = xmap(u0 * np.ones_like(tt), tt)
    nrm = np.stack([np.cos(u0) * np.cos(tt), np.cos(u0) * np.sin(tt), np.sin(u0) * np.ones_like(tt)], axis=1)
    dgfig.curve3d(ax, c, role="main", lw=1.4, visible=nrm @ view > 0, hidden=None, zorder=5)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
cax = fig.add_axes([0.8, 0.22, 0.022, 0.56])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_ticks([Kf(np.pi), 0, Kf(0.0)])
cb.set_ticklabels([r"$-1.04$", r"$0$", r"$0.45$"])
cb.set_label(r"$K$", rotation=0, labelpad=4)
dgfig.save(fig, __file__)

# 인터랙티브 판 -----------------------------------------------------------------------------
import plotly.graph_objects as go

Ui, Vi = np.meshgrid(np.linspace(-np.pi, np.pi, 73), np.linspace(0, 2 * np.pi, 109), indexing="ij")
Si = xmap(Ui, Vi)
Ki = Kf(Ui)
kmin, kmax = float(Kf(np.pi)), float(Kf(0.0))
kgrid = np.linspace(kmin, kmax, 21)               # 정적 그림과 같은 TwoSlopeNorm 색
cscale = [[float((k - kmin) / (kmax - kmin)),
           "rgb(%d,%d,%d)" % tuple(int(round(255 * c)) for c in cmap(norm(k))[:3])] for k in kgrid]
pfig = go.Figure(go.Surface(x=Si[..., 0], y=Si[..., 1], z=Si[..., 2], surfacecolor=Ki, colorscale=cscale,
                            cmin=kmin, cmax=kmax,
                            colorbar=dict(title="K"),
                            customdata=np.round(np.stack([Ui, Vi, Ki], axis=-1), 3),
                            hovertemplate="u = %{customdata[0]}<br>v = %{customdata[1]}<br>K = %{customdata[2]}<extra></extra>"))
for u0 in (np.pi / 2, -np.pi / 2):
    c = xmap(u0 * np.ones_like(tt), tt)
    pfig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="lines", line=dict(color="black", width=4),
                                showlegend=False, hoverinfo="skip"))
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=30, b=0),
                   title=dict(text="Torus of revolution (R = 2, r = 0.8): Gaussian curvature K",
                              x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-4-2-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
