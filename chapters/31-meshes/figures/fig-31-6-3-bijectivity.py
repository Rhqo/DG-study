"""Figure 31.6.3: which flattenings are guaranteed not to flip triangles (Section 31.6).

A small disk-like mesh: the 4 x 4 grid on [0, 1]^2 (diagonals alternating), the four interior vertices moved by
uniform random offsets of at most 0.45 grid cells (numpy default_rng(8921)), and every vertex lifted to
z = a sin(f_x x) sin(f_y y) with a = 1.09, f_x = 4.14, f_y = 2.66 drawn from the same generator (a crumpled patch,
18 triangles, smallest angle 9 degrees). Three flattenings:
(b) Tutte / uniform weights, boundary on the unit circle by arc length (a convex combination map),
(c) cotan (discrete harmonic) weights, the same boundary,
(d) LSCM [Levy02] with the two opposite corners x_0 and x_15 pinned (free boundary).
(a) The patch (orthographic view); vermillion: the edge x_9 x_13, whose cotan weight is -1.75.
In (b)-(d) the vermillion faces are flipped (negative signed area in UV).
Self-check: Tutte has no flipped face (Floater's Theorem 7); the cotan map flips the two faces (x_8, x_9, x_5) and
(x_5, x_9, x_10); LSCM flips one face for this pin choice and at least one face for every one of the 66 choices
of two boundary pins.
Also writes an interactive version of (a).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-6-3-bijectivity.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

C = dgfig.COLORS


def crumpled_patch(seed=8921):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 6))
    xs = np.linspace(0, 1, n)
    V, F = dm.grid_mesh(xs, xs, pattern="alternate")
    V = V.copy()
    inner = [i for i in range(len(V)) if 0 < V[i, 0] < 1 and 0 < V[i, 1] < 1]
    V[inner, :2] += rng.uniform(-0.45, 0.45, size=(len(inner), 2)) / (n - 1)
    amp = rng.uniform(0, 1.2)
    fx, fy = rng.uniform(1, 5, 2)
    V[:, 2] = amp * np.sin(fx * V[:, 0]) * np.sin(fy * V[:, 1])
    return V, F, (amp, fx, fy)


V, F, (amp, fx, fy) = crumpled_patch()
assert len(V) == 16 and len(F) == 18 and dm.euler_characteristic(F) == 1
assert round(amp, 2) == 1.09 and round(fx, 2) == 4.14 and round(fy, 2) == 2.66
w = dm.cotan_weights(V, F)
assert round(w[(9, 13)], 2) == -1.75
UV_t = dm.fixed_boundary_map(V, F, "uniform")
UV_c = dm.fixed_boundary_map(V, F, "cotan")
UV_l = dm.lscm(V, F, (0, 15), [(0.0, 0.0), (1.0, 0.0)])
flips = {}
for name, UV in (("tutte", UV_t), ("cotan", UV_c), ("lscm", UV_l)):
    _, det = dm.uv_distortion(V, F, UV)
    flips[name] = np.where(det < 0)[0]
assert len(flips["tutte"]) == 0
assert sorted(map(sorted, F[flips["cotan"]].tolist())) == [[5, 8, 9], [5, 9, 10]]
assert len(flips["lscm"]) == 1
loop = dm.boundary_loop(F)
worst = []
for a in range(len(loop)):
    for b in range(a + 1, len(loop)):
        _, det = dm.uv_distortion(V, F, dm.lscm(V, F, (loop[a], loop[b]), [(0.0, 0.0), (1.0, 0.0)]))
        worst.append((det < 0).sum())
assert len(worst) == 66 and min(worst) >= 1
print("LSCM flipped face:", F[flips["lscm"]].tolist(), " flips over all pin pairs:", np.bincount(worst))


def draw_uv(ax, UV, flipped, title, pins=None, circle=True):
    for f, tri in enumerate(F):
        bad = f in set(flipped)
        ax.add_patch(Polygon(UV[tri], closed=True, fc=C["normal"] if bad else C["surface"], alpha=0.85 if bad else 0.6,
                             ec="#333333", lw=0.9, zorder=2 if bad else 1))
    if circle:
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=C["aux"], lw=0.6, ls=(0, (3, 3)), zorder=0))
    ax.plot(*UV[9], "o", color=C["normal"], ms=5, zorder=4)
    ax.plot(*UV[13], "o", color=C["main"], ms=4, zorder=4)
    if pins is not None:
        for p in pins:
            ax.plot(*UV[p], "s", color=C["tangent"], ms=6, zorder=5)
    lo, hi = UV.min(0), UV.max(0)
    pad = 0.08 * (hi - lo).max()
    c = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() + pad
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)


fig, axs = plt.subplots(2, 2, figsize=(7.0, 7.0), gridspec_kw=dict(hspace=0.16, wspace=0.06))
ax = axs[0, 0]
cam = dm.Ortho(elev=35, azim=-60)
cols = dm.shade(V, F, dm.hex_to_rgb(C["surface"]), light=(0.3, -0.6, 0.8), ambient=0.5)
order = np.argsort(cam.depth(V[F].mean(1)))
P2 = cam(V)
for f in order:
    ax.add_patch(Polygon(P2[F[f]], closed=True, fc=cols[f], ec="#333333", lw=0.8))
ax.plot(*P2[[9, 13]].T, color=C["normal"], lw=2.6, zorder=5)
ax.plot(*P2[9], "o", color=C["normal"], ms=5, zorder=6)
ax.plot(*P2[13], "o", color=C["main"], ms=4, zorder=6)
for i, off in ((9, (-0.13, 0.03)), (13, (0.03, -0.09)), (0, (-0.1, -0.06)), (15, (0.03, 0.02))):
    ax.text(P2[i, 0] + off[0], P2[i, 1] + off[1], rf"$x_{{{i}}}$", fontsize=11.5,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6), zorder=7)
ax.set_aspect("equal")
ax.autoscale_view()
ax.set_axis_off()
ax.set_title("(a) crumpled patch (18 faces)", fontsize=12)
draw_uv(axs[0, 1], UV_t, flips["tutte"], "(b) Tutte (uniform): no flip")
draw_uv(axs[1, 0], UV_c, flips["cotan"], f"(c) cotan (harmonic): {len(flips['cotan'])} flipped")
draw_uv(axs[1, 1], UV_l, flips["lscm"], f"(d) LSCM, pins $x_0, x_{{15}}$: {len(flips['lscm'])} flipped",
        pins=(0, 15), circle=False)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: the patch
tr = dm.plotly_mesh(V, F, color="#C8C8C8", name="patch")
import plotly.graph_objects as go  # noqa: E402

et = dm.plotly_edges(V, F, color="#333333", width=2)
bad = go.Scatter3d(x=V[[9, 13], 0], y=V[[9, 13], 1], z=V[[9, 13], 2], mode="lines+markers+text",
                   line=dict(color=C["normal"], width=8), marker=dict(size=4, color=C["normal"]),
                   text=["x_9", "x_13"], textposition="top center", showlegend=False)
dm.plotly_save([tr, et, bad], __file__, "Crumpled patch of Figure 31.6.3; vermillion: edge x_9 x_13 with cotan weight -1.75",
               "fig-31-6-3-interactive")
