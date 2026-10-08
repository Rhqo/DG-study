"""Figure 30.2.2: curvature of the torus of revolution read off implicit functions (Section 30.2).

T_{R,r}, R = 2, r = 0.8.
(a) color = Delta f for the exact SDF f = sqrt((rho - R)^2 + z^2) - r, i.e. -2H with the outward normal
    (Proposition 30.2.4). Values from (R + 2r cos u)/(r(R + r cos u)) range over [0.417, 1.607].
(b) color = Gaussian curvature K from the bordered-Hessian formula (30.2.6) applied to the polynomial defining function
    f = (rho - R)^2 + z^2 - r^2 (not an SDF); range [-1.042, 0.446]; black circles u = +-pi/2: K = 0.
Both colorings use the diverging map RdBu_r centered at 0 (GUIDELINES §12.3, curvature sign).
The values are computed from the implicit functions with finite differences and checked against the parametric
formulas of Example 0.4.6 (assert). Also writes an interactive plotly version with the same data.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-2-2-torus-curvature.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import proj3d

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
R0, r0 = 2.0, 0.8


def f_sdf(P):
    return np.hypot(np.hypot(P[..., 0], P[..., 1]) - R0, P[..., 2]) - r0


def f_poly(P):
    return (np.hypot(P[..., 0], P[..., 1]) - R0) ** 2 + P[..., 2] ** 2 - r0 ** 2


def fd_grad_hess(fun, P, h=1e-4):
    E = np.eye(3)
    g = np.stack([(fun(P + h * E[i]) - fun(P - h * E[i])) / (2 * h) for i in range(3)], -1)
    Hm = np.empty(P.shape[:-1] + (3, 3))
    for i in range(3):
        for j in range(3):
            Hm[..., i, j] = (fun(P + h * E[i] + h * E[j]) - fun(P + h * E[i] - h * E[j])
                             - fun(P - h * E[i] + h * E[j]) + fun(P - h * E[i] - h * E[j])) / (4 * h * h)
    return g, Hm


nu, nv = 73, 109
u = np.linspace(0, 2 * np.pi, nu)
v = np.linspace(0, 2 * np.pi, nv)
U, V = np.meshgrid(u, v, indexing="ij")
P = np.stack([(R0 + r0 * np.cos(U)) * np.cos(V), (R0 + r0 * np.cos(U)) * np.sin(V), r0 * np.sin(U)], -1)

_, Hs = fd_grad_hess(f_sdf, P)
LAP = np.trace(Hs, axis1=-2, axis2=-1)
gp, Hp = fd_grad_hess(f_poly, P)
B = np.zeros(P.shape[:-1] + (4, 4))
B[..., :3, :3] = Hp
B[..., :3, 3] = gp
B[..., 3, :3] = gp
KK = -np.linalg.det(B) / np.sum(gp ** 2, -1) ** 2

lap_ref = (R0 + 2 * r0 * np.cos(U)) / (r0 * (R0 + r0 * np.cos(U)))
K_ref = np.cos(U) / (r0 * (R0 + r0 * np.cos(U)))
assert np.max(np.abs(LAP - lap_ref)) < 1e-4, np.max(np.abs(LAP - lap_ref))
assert np.max(np.abs(KK - K_ref)) < 1e-4, np.max(np.abs(KK - K_ref))
print(f"Delta f range [{LAP.min():.3f}, {LAP.max():.3f}], K range [{KK.min():.3f}, {KK.max():.3f}]")

fig = plt.figure(figsize=(5.8, 8.0))
panels = [(LAP, TwoSlopeNorm(vmin=-1.7, vcenter=0, vmax=1.7), r"(a) $\Delta f = -2H$ (SDF, outward $\mathbf{N}$)",
           r"$\Delta f$"),
          (KK, TwoSlopeNorm(vmin=-1.1, vcenter=0, vmax=1.1), r"(b) $K$ from the bordered Hessian", r"$K$")]
for k, (val, norm, title, lab) in enumerate(panels):
    ax = fig.add_subplot(2, 1, k + 1, projection="3d", computed_zorder=False)
    ax.view_init(elev=38, azim=-62)
    ax.set_axis_off()
    fc = cm.RdBu_r(norm(val))
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=fc, linewidth=0, antialiased=False, shade=False,
                    rstride=1, cstride=1, rasterized=True)
    if k == 1:                                   # parabolic circle u = pi/2 (top); u = -pi/2 is underneath, hidden
        vv = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R0 * np.cos(vv), R0 * np.sin(vv), r0 + 0 * vv, color=C["main"], lw=1.4, zorder=10)
    dgfig.equal_aspect(ax, P.reshape(-1, 3), zoom=1.45)
    ax.set_title(title, fontsize=13.5, y=0.95)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.0, shrink=0.75)
    cb.set_label(lab)
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5] if k == 0 else [-1, -0.5, 0, 0.5, 1])
    # arrows to the outer equator (u = 0, the point facing the viewer) and to the inner equator (u = pi, the far
    # side of the hole, which is visible from above); the values are the extremes printed above
    az = np.radians(-62)
    p_out = ((R0 + r0) * np.cos(az), (R0 + r0) * np.sin(az), 0.0)
    p_in = ((R0 - r0) * np.cos(az + np.pi), (R0 - r0) * np.sin(az + np.pi), 0.0)
    v_out, v_in = (r"$1.607$", r"$0.417$") if k == 0 else (r"$0.446$", r"$-1.042$")
    for pt, txt, xyt in ((p_out, "outer equator " + v_out, (0.03, 0.04)),
                         (p_in, "inner equator " + v_in, (0.03, 0.86))):
        x2, y2, _ = proj3d.proj_transform(*pt, ax.get_proj())
        ax.annotate(txt, xy=(x2, y2), xycoords="data", xytext=xyt, textcoords="axes fraction", fontsize=11,
                    ha="left", va="center", zorder=20, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]))
fig.subplots_adjust(hspace=0.02, left=0.0, right=0.95, top=0.97, bottom=0.02)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pfig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                     subplot_titles=("Laplacian of the SDF = -2H (outward N)", "Gaussian curvature K (bordered Hessian)"))
for k, (val, rng_, name) in enumerate([(LAP, 1.7, "Δf"), (KK, 1.1, "K")]):
    pfig.add_trace(go.Surface(x=P[..., 0], y=P[..., 1], z=P[..., 2], surfacecolor=val, cmin=-rng_, cmax=rng_,
                              colorscale="RdBu", reversescale=True,
                              colorbar=dict(title=name, x=0.45 if k == 0 else 1.0, len=0.7),
                              customdata=np.stack([np.degrees(U), val], -1),
                              hovertemplate="u = %{customdata[0]:.0f}°<br>" + name + " = %{customdata[1]:.3f}<extra></extra>"),
                   row=1, col=k + 1)
scene = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
pfig.update_layout(scene=scene, scene2=scene, margin=dict(l=0, r=0, t=60, b=0),
                   title=dict(text="Torus of revolution R = 2, r = 0.8: curvature from implicit functions", x=0.5,
                              font=dict(size=14)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-30-2-2-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
