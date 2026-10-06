"""Figure 0.3.1: parametrization = UV 좌표 (0.3절). 정적 SVG와 인터랙티브 HTML을 함께 만든다.

torus of revolution T_{R,r} (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8,
x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u), U = (0, 2π) × (0, 2π).
왼쪽: U의 checkerboard(u 방향 8칸, v 방향 16칸). 오른쪽: 같은 checkerboard를 x로 옮긴 것(texture mapping).
점 q = (π/3, −π/6 + 2π)와 p = x(q)에서 x_u, x_v (파랑, 길이 0.6배).
같은 칸이 바깥쪽(u ≈ 0)에서는 넓고 안쪽(u ≈ π)에서는 좁다: G = (R + r cos u)².
인터랙티브: 같은 식, 같은 checkerboard (plotly, fig-0-3-1-torus-uv-interactive.html).
자기검사: 점들이 implicit equation (√(x²+y²) − R)² + z² = r²를 만족한다. x_u ⟂ x_v (F = 0).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-3-1-torus-uv.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.colors import to_rgba

import dgsym

C = dgfig.COLORS
ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
vals = {Rs: RV, rs: rV}
Xe = ex["expr"].subs(vals)
X = sp.lambdify((u, v), list(Xe), "numpy")
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")
E, F, G = dgsym.first_ff(ex["expr"], u, v, ex["positive"])
assert F == 0


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


NU, NV = 8, 16
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 8 * NU + 1), np.linspace(0, 2 * np.pi, 8 * NV + 1), indexing="ij")
S = xmap(Ug, Vg)
assert np.allclose((np.hypot(S[..., 0], S[..., 1]) - RV) ** 2 + S[..., 2] ** 2, rV ** 2)
Uc = 0.5 * (Ug[:-1, :-1] + Ug[1:, 1:])
Vc = 0.5 * (Vg[:-1, :-1] + Vg[1:, 1:])
checker = ((np.floor(Uc / (2 * np.pi / NU)) + np.floor(Vc / (2 * np.pi / NV))) % 2).astype(int)
col_a, col_b = np.array(to_rgba(C["region"])), np.array(to_rgba("#FFFFFF"))
col_a[:3] = 0.55 * col_a[:3] + 0.45          # 연한 하늘색
fc = np.where(checker[..., None] == 1, col_a, col_b)

q = (np.pi / 3, 4.6)
p = np.array(X(*q), float)
xu, xv = np.array(Xu(*q), float), np.array(Xv(*q), float)
assert abs(xu @ xv) < 1e-12

fig = plt.figure(figsize=(6.8, 3.0))
axU = fig.add_axes([0.03, 0.12, 0.33, 0.8])
for i in range(NU):
    for j in range(NV):
        if (i + j) % 2 == 1:
            axU.add_patch(plt.Rectangle((j * 2 * np.pi / NV, i * 2 * np.pi / NU), 2 * np.pi / NV, 2 * np.pi / NU,
                                        fc=col_a, ec="none"))
axU.add_patch(plt.Rectangle((0, 0), 2 * np.pi, 2 * np.pi, fill=False, ec=C["main"], lw=0.8))
axU.plot(q[1], q[0], "o", color=C["main"], ms=4, zorder=5)
axU.text(q[1] - 0.45, q[0] - 0.55, r"$q$", fontsize=12)
axU.annotate("", xy=(q[1], q[0] + 0.9), xytext=(q[1], q[0]),
             arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=10, shrinkA=0, shrinkB=0))
axU.annotate("", xy=(q[1] + 0.9, q[0]), xytext=(q[1], q[0]),
             arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=10, shrinkA=0, shrinkB=0))
axU.set_xlim(-0.1, 2 * np.pi + 0.1)
axU.set_ylim(-0.1, 2 * np.pi + 0.1)
axU.set_aspect("equal")
axU.set_xticks([0, np.pi, 2 * np.pi])
axU.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axU.set_yticks([0, np.pi, 2 * np.pi])
axU.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axU.set_xlabel(r"$v$", labelpad=1)
axU.set_ylabel(r"$u$", rotation=0, labelpad=6)
for sp_ in axU.spines.values():
    sp_.set_visible(False)
axU.text(np.pi, 2 * np.pi + 0.45, r"$U$", fontsize=12, ha="center")

ax = fig.add_axes([0.38, -0.05, 0.62, 1.1], projection="3d", computed_zorder=False)
ax.view_init(elev=38, azim=-62)
ax.set_axis_off()
ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], facecolors=fc, rstride=1, cstride=1, linewidth=0,
                antialiased=False, shade=True, rasterized=True, zorder=1)
dgfig.arrow3d(ax, p, xu, role="tangent", scale=0.4, label=r"$\mathbf{x}_u$", label_offset=(0, 0, 0.12), zorder=10)
dgfig.arrow3d(ax, p, xv, role="tangent", scale=0.4, label=r"$\mathbf{x}_v$", label_offset=(0.1, 0, 0.1), zorder=10)
dgfig.point3d(ax, p, label=r"$p$", label_offset=(0.1, -0.1, -0.35), zorder=11)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.28)

dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(1.02, 0.8), xy_to=(0.22, 0.78), rad=-0.3,
                label_offset=(0.0, -0.02))
dgfig.save(fig, __file__)

# 인터랙티브 판 (plotly) -----------------------------------------------------------------------
import plotly.graph_objects as go

Ui, Vi = np.meshgrid(np.linspace(0, 2 * np.pi, 8 * NU + 1), np.linspace(0, 2 * np.pi, 8 * NV + 1), indexing="ij")
Si = xmap(Ui, Vi)
cell = ((np.floor(np.clip(Ui, 0, 2 * np.pi - 1e-9) / (2 * np.pi / NU))
         + np.floor(np.clip(Vi, 0, 2 * np.pi - 1e-9) / (2 * np.pi / NV))) % 2)
pfig = go.Figure(go.Surface(x=Si[..., 0], y=Si[..., 1], z=Si[..., 2], surfacecolor=cell,
                            colorscale=[[0, "#FFFFFF"], [1, "#9ED2F0"]], showscale=False,
                            hovertemplate="u = %{customdata[0]:.2f}<br>v = %{customdata[1]:.2f}<extra></extra>",
                            customdata=np.stack([Ui, Vi], axis=-1), lighting=dict(ambient=0.7, diffuse=0.5)))
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=30, b=0),
                   title=dict(text="Torus of revolution (R = 2, r = 0.8) with a UV checkerboard", x=0.5,
                              font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-3-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
