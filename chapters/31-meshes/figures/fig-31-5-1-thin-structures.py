"""Figure 31.5.1: the extracted mesh has the topology of the sampled field, not of the shape (Section 31.5).

Shape: two balls of radius 0.5 centred at (-0.8, 0, 0) and (0.8, 0, 0), joined by a rod of radius 0.05 whose axis
goes from (-0.8, 0, -0.07) to (0.8, 0, 0.07) (slightly tilted). Exact SDF (union by min), sampled on a grid of
spacing h whose y- and z-lines sit at +-h/2, +-3h/2, ... (none on the rod axis), level set extracted by marching
tetrahedra (each cube split into six tetrahedra, linear interpolation on edges).
(a) h = 0.04, (b) h = 0.08, (c) h = 0.14, seen from the side (-y direction, z up), same scale.
Self-check: (a) 1 component, chi = 2; (b) 2 components, chi = 4, the rod breaks between x = -0.11 and 0.11 and each ball keeps a stub;
(c) 2 components, chi = 4, no rod. Every mesh is a closed manifold mesh.
Also writes an interactive version with a menu over h = 0.04, 0.08, 0.14 (mesh edges drawn for h = 0.08, 0.14).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-5-1-thin-structures.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
cam = dm.Ortho(elev=8, azim=-90)
base = dm.hex_to_rgb(C["surface"])
fig, axs = plt.subplots(3, 1, figsize=(7.0, 6.9))
expect = {0.04: (1, 2), 0.08: (2, 4), 0.14: (2, 4)}
for ax, (lab, h) in zip(axs, (("(a)", 0.04), ("(b)", 0.08), ("(c)", 0.14))):
    V, F = dm.dumbbell_mesh(h)
    comps = dm.components(F)
    chi = dm.euler_characteristic(F)
    assert (len(comps), chi) == expect[h]
    assert dm.is_manifold(F) and not dm.boundary_edges(F)
    if h == 0.08:   # the rod breaks in the middle: each ball keeps a stub (gap between x = -0.11 and 0.11)
        ext = sorted((V[np.unique(F[c])][:, 0].min(), V[np.unique(F[c])][:, 0].max()) for c in comps)
        assert abs(ext[0][1] + 0.113) < 0.002 and abs(ext[1][0] - 0.114) < 0.002
    if h == 0.14:
        xmin = sorted(V[np.unique(F[c])][:, 0].min() for c in comps)
        assert xmin[1] > 0.29
    cols = dm.shade(V, F, base, light=(0.4, -0.7, 0.6), ambient=0.4)
    P = dm.draw_mesh_2d(ax, V, F, cam, cols, edgecolor="none", lw=0, cull=True, rasterized=True)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.62, 0.62)
    ax.set_aspect("equal")
    ax.set_axis_off()
    word = "component" if len(comps) == 1 else "components"
    ax.text(-1.4, 0.6, rf"{lab} $h = {h}$", fontsize=12, va="top")
    ax.text(1.4, 0.6, rf"{len(comps)} {word},  $\chi = {chi}$", fontsize=11.5, va="top", ha="right",
            color=C["tangent"])
ax = axs[0]
ax.annotate("rod, radius 0.05", xy=(0.0, -0.06), xytext=(-0.35, -0.5), fontsize=10.5,
            arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
axs[1].annotate("gap", xy=(0.0, 0.0), xytext=(-0.12, -0.5), fontsize=10.5,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, hspace=0.05)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: menu over h
import plotly.graph_objects as go  # noqa: E402



def edge_trace(V, F):
    """Mesh edges as one line trace (float32 arrays with NaN separators keep the file small)."""
    E = np.array(dm.edges(F))
    seg = np.full((len(E), 3, 3), np.nan, dtype=np.float32)
    seg[:, 0], seg[:, 1] = V[E[:, 0]], V[E[:, 1]]
    seg = seg.reshape(-1, 3)
    return go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", line=dict(color="#555555", width=1.0),
                        hoverinfo="skip", showlegend=False, connectgaps=False)


traces, hs_ = [], []
for h in (0.04, 0.08, 0.14):
    V, F = dm.dumbbell_mesh(h)
    comps = dm.components(F)
    tr = dm.plotly_mesh(V.astype(np.float32), F.astype(np.int32), color="#C8C8C8", name=f"h = {h}")
    tr.lighting.facenormalsepsilon = 0      # small faces would otherwise get a zero normal (drawn dark)
    tr.lighting.vertexnormalsepsilon = 0
    tr.lightposition = dict(x=3e4, y=-1e5, z=6e4)
    traces.append(tr)
    if h > 0.05:     # the finest mesh is drawn without edges (file size)
        traces.append(edge_trace(V, F))
        hs_.append((h, len(comps), dm.euler_characteristic(F)))
    hs_.append((h, len(comps), dm.euler_characteristic(F)))
for t_, (h, _, _) in zip(traces, hs_):
    t_.visible = (h == 0.04)
buttons = []
for h, nc, chi in sorted(set(hs_)):
    buttons.append(dict(label=f"h = {h}", method="update",
                        args=[{"visible": [hh == h for (hh, _, _) in hs_]},
                              {"title.text": f"Marching tetrahedra of the dumbbell SDF, h = {h}: "
                                             f"{nc} component{'s' if nc > 1 else ''}, chi = {chi}"}]))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                              camera=dict(eye=dict(x=0.0, y=-2.8, z=0.7))),
                   margin=dict(l=0, r=0, t=60, b=0), showlegend=False,
                   updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.0, xanchor="left", yanchor="top")],
                   title=dict(text="Marching tetrahedra of the dumbbell SDF, h = 0.04: 1 component, chi = 2", x=0.5,
                              font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-31-5-1-interactive", config={"displaylogo": False})
print(f"saved {out}")
