"""Figure 31.1.3: counting n0 - n1 + n2 on three closed meshes (Section 31.1).

(a) icosphere (icosahedron subdivided once): 42 - 120 + 80 = 2 (genus 0).
(b) torus of revolution T_{R,r} with R = 2, r = 0.8 on an 8 x 16 grid: 128 - 384 + 256 = 0 (genus 1).
(c) a genus-2 surface (smooth union of two tori, marching tetrahedra with grid spacing 0.16): chi = -2.
Self-check: the counts are computed from the meshes, the meshes are manifold and consistently oriented,
and chi = 2 - 2g.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-1-3-euler.py``
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
base = dm.hex_to_rgb(C["surface"])

meshes = [
    ("(a) sphere", dm.icosphere(1), 0, dict(elev=20, azim=-60), True),
    ("(b) torus", dm.torus_mesh(2.0, 0.8, 8, 16), 1, dict(elev=38, azim=-60), True),
    ("(c) genus 2", dm.genus2_mesh(h=0.16), 2, dict(elev=42, azim=-70), False),
]

fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.3), gridspec_kw=dict(width_ratios=[0.8, 1.15, 1.25]))
for ax, (title, (V, F), g, view, edges) in zip(axs, meshes):
    n0, n1, n2 = dm.counts(F)
    chi = n0 - n1 + n2
    assert chi == 2 - 2 * g == dm.euler_characteristic(F)
    assert dm.is_manifold(F) and dm.oriented_consistently(F) and not dm.boundary_edges(F)
    assert len(dm.components(F)) == 1 and dm.signed_volume(V, F) > 0
    assert 3 * n2 == 2 * n1  # closed triangle mesh
    cam = dm.Ortho(**view)
    cols = dm.shade(V, F, base, light=(0.3, -0.5, 0.9), ambient=0.45)
    P = dm.draw_mesh_2d(ax, V, F, cam, cols, edgecolor="#4a4a4a" if edges else "none",
                        lw=0.5 if edges else 0.0, rasterized=not edges, cull=True)
    ax.set_xlim(P[:, 0].min() - 0.1, P[:, 0].max() + 0.1)
    ax.set_ylim(P[:, 1].min() - 0.1, P[:, 1].max() + 0.1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    ax.text(0.5, -0.10, rf"${n0} - {n1} + {n2} = {chi}$", transform=ax.transAxes, ha="center", fontsize=11.5)
    ax.text(0.5, -0.24, rf"$g = {g}$", transform=ax.transAxes, ha="center", fontsize=11.5,
            color=C["tangent"])

fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.08)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version: the three meshes side by side
tr = []
offsets = [np.array([-4.4, 0, 0]), np.array([0.3, 0, 0]), np.array([5.6, 0, 0])]
for (title, (V, F), g, view, edges), off in zip(meshes, offsets):
    n0, n1, n2 = dm.counts(F)
    lab = f"{title}: {n0} - {n1} + {n2} = {n0 - n1 + n2}"
    tr.append(dm.plotly_mesh(V + off, F, color="#D0D0D0", name=lab))
    if edges:
        tr.append(dm.plotly_edges(V + off, F))
dm.plotly_save(tr, __file__, "n0 - n1 + n2 on a sphere (2), a torus (0) and a genus-2 surface (-2)",
               "fig-31-1-3-interactive")
