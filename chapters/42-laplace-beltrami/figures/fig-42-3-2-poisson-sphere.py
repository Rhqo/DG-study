"""Figure 42.3.2: the P1 finite element solution of Delta u = -2z on sphere meshes converges (Section 42.3,
Example 42.3.12).

Meshes: dgmesh.icosphere(level), levels 1-4, and the same with tangential noise (ch42.jittered_sphere, s = 0.15,
seed 42: obtuse triangles and negative cotan weights). Systems: L u = M^G f (Galerkin mass, blue) and L u = M f with the
lumped (barycentric) mass (orange), f = -2z at the vertices; the mean of the right-hand side is removed and the
solution is normalized to M-weighted mean 0 (Section 42.6). Error: area-weighted RMS of u - (z - mean z) at the
vertices. Solid: regular icosphere; dashed: jittered. Horizontal axis: mean edge length. Gray dotted: slope 2.
Self-check: all four curves have fitted slopes between 1.8 and 2.6 over the last three levels.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-3-2-poisson-sphere.py``
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch42  # noqa: E402

col = dgfig.COLORS


def solve(V, F, kind):
    L = dm.cotan_laplacian(V, F)
    z = V[:, 2]
    if kind == "galerkin":
        MG = ch42.galerkin_mass(V, F)
        b, m = MG @ (-2 * z), MG.sum(1)
    else:
        m = ch42.mass(V, F, kind)
        b = m * (-2 * z)
    b = b - m * (b.sum() / m.sum())
    u = np.linalg.solve(-L + np.outer(m, m), -b)
    e = u - (z - (m @ z) / m.sum())
    return np.sqrt((e ** 2 * m).sum() / m.sum())


rows = {}
for mesh in ("regular", "jittered"):
    for kind in ("galerkin", "barycentric"):
        hs, es = [], []
        for lv in range(1, 5):
            V, F = dm.icosphere(lv) if mesh == "regular" else ch42.jittered_sphere(lv, s=0.15, seed=42)
            hs.append(dd.primal_edge_lengths(V, F).mean())
            es.append(solve(V, F, kind))
        rows[(mesh, kind)] = (np.array(hs), np.array(es))
        sl = np.polyfit(np.log(hs[1:]), np.log(es[1:]), 1)[0]
        assert 1.8 < sl < 2.6, (mesh, kind, sl)
        print(mesh, kind, [f"{e:.2e}" for e in es], f"slope {sl:.2f}")

fig, ax = plt.subplots(figsize=(4.6, 3.4))
for (mesh, kind), (hs, es) in rows.items():
    c = col["tangent"] if kind == "galerkin" else col["accent"]
    ls = "-" if mesh == "regular" else "--"
    lab = ("Galerkin $M^{\\mathrm{G}}$" if kind == "galerkin" else "lumped $M$") + (", regular" if mesh == "regular" else ", jittered")
    ax.loglog(hs, es, ls, color=c, marker="o", ms=3.5, lw=1.4, label=lab)
hs = rows[("regular", "galerkin")][0]
ref = lambda h: 0.9 * rows[("regular", "galerkin")][1][-1] * (h / hs[-1]) ** 2 / 3  # noqa: E731
ax.loglog(hs, ref(hs), ":", color=col["aux"], lw=1.0)
ax.set_xticks([0.08, 0.15, 0.3, 0.6], ["0.08", "0.15", "0.3", "0.6"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(r"mean edge length $\bar\ell$")
ax.set_ylabel("RMS error")
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.15)
# label the reference line just above it, between the blue and the orange lines, rotated along the line
x0, x1 = 0.19, 0.26
p0, p1 = ax.transData.transform([(x0, ref(x0)), (x1, ref(x1))])
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(x0, ref(x0) * 1.3, "slope 2", fontsize=9, color=col["aux"], rotation=ang, rotation_mode="anchor")
dgfig.save(fig, __file__)
