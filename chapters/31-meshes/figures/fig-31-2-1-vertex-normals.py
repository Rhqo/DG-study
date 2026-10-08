"""Figure 31.2.1: three vertex normals on the same mesh (Section 31.2).

(a) Top view of an irregular one-ring: the vertex sits at the north pole of the unit sphere and its 7 neighbours
    lie on the sphere at longitudes (0, 25, 50, 75, 160, 235, 300) degrees and colatitudes
    (0.40, 0.30, 0.38, 0.30, 0.62, 0.50, 0.55). The exact normal is (0, 0, 1). The arrows are the horizontal
    components of the three vertex normals, magnified so that 1 degree of tilt = 0.065 units.
(b) Mean angle error (degrees) of the three normals against the exact sphere normal on jittered icospheres
    (levels 1..5; every vertex moved by 0.1 h in a random direction and projected back to the sphere, seed 1),
    against the mean edge length h. No triangle is flipped.
Self-check: the tilts are 1.24, 6.50 and 4.22 degrees; all three errors decrease by about half per level.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-2-1-vertex-normals.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
KINDS = [("uniform", "uniform", C["tangent"]), ("area", "area-weighted", C["third"]),
         ("angle", "angle-weighted", C["accent"])]


def one_ring():
    lon = np.radians([0, 25, 50, 75, 160, 235, 300])
    col = np.array([0.40, 0.30, 0.38, 0.30, 0.62, 0.50, 0.55])
    P = np.stack([np.sin(col) * np.cos(lon), np.sin(col) * np.sin(lon), np.cos(col)], 1)
    V = np.vstack([[0, 0, 1.0], P])
    F = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
    return V, F


fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.3))

# ---------------------------------------------------------------- (a)
V, F = one_ring()
areas = dm.face_areas(V, F)
for f, tri in enumerate(F):
    P = V[tri][:, :2]
    ax.add_patch(Polygon(P, closed=True, fc=C["surface"], ec="#555555", lw=0.8, alpha=0.75, zorder=1))
ax.plot(0, 0, "o", color=C["main"], ms=5, zorder=8)
SCALE = 0.065
tilts = {}
for kind, label, col in KINDS:
    N = dm.vertex_normals(V, F, kind)[0]
    tilt = np.degrees(np.arccos(N[2]))
    tilts[kind] = tilt
    d = N[:2] / np.linalg.norm(N[:2]) * tilt * SCALE
    ax.add_patch(FancyArrowPatch((0, 0), d, arrowstyle="-|>", mutation_scale=12, color=col, lw=2.2,
                                 zorder=7 if kind == "uniform" else 5))
assert np.allclose([tilts["uniform"], tilts["area"], tilts["angle"]], [1.239, 6.501, 4.222], atol=2e-3)
ax.text(0.05, 0.07, "vertex; exact\nnormal (0, 0, 1)", fontsize=10.5, ha="left",
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=6)
leg = [plt.Line2D([], [], color=col, lw=2.2, label=f"{label}: {tilts[k]:.1f}°") for k, label, col in KINDS]
ax.legend(handles=leg, loc="upper center", fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.0), ncol=1)
ax.set_aspect("equal")
ax.set_xlim(-0.62, 0.55)
ax.set_ylim(-0.62, 0.52)
ax.set_axis_off()
ax.set_title("(a) one vertex, three normals\n(top view, tilt magnified)", fontsize=11.5)

# ---------------------------------------------------------------- (b)
rng = np.random.default_rng(1)
hs, errs = [], {k: [] for k, _, _ in KINDS}
for lv in range(1, 6):
    V0, F0 = dm.icosphere(lv)
    h = np.mean([np.linalg.norm(V0[a] - V0[b]) for a, b in dm.edges(F0)])
    J = V0 + 0.1 * h * rng.normal(size=V0.shape)
    J /= np.linalg.norm(J, axis=1, keepdims=True)
    fn = dm.face_normals(J, F0)
    assert (np.einsum("ij,ij->i", fn, J[F0].mean(1)) > 0).all()  # no flipped triangle
    hs.append(h)
    for kind, _, _ in KINDS:
        N = dm.vertex_normals(J, F0, kind)
        errs[kind].append(np.degrees(np.arccos(np.clip((N * J).sum(1), -1, 1))).mean())
hs = np.array(hs)
for kind, label, col in KINDS:
    e = np.array(errs[kind])
    axb.loglog(hs, e, "o-", color=col, lw=1.6, ms=5, label=label)
    rate = np.log(e[-2] / e[-1]) / np.log(hs[-2] / hs[-1])
    assert 0.8 < rate < 1.2
axb.loglog(hs, 3.0 * hs / hs[0] * 1.0, ls=(0, (4, 3)), color=C["aux"], lw=1.0)
axb.text(hs[1] * 1.0, 3.0 * hs[1] / hs[0] * 0.62, "slope 1", color=C["aux"], fontsize=10.5,
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
axb.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
axb.yaxis.set_major_formatter(ScalarFormatter())
axb.yaxis.set_minor_formatter(NullFormatter())
axb.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5]))
axb.xaxis.set_major_formatter(ScalarFormatter())
axb.xaxis.set_minor_formatter(NullFormatter())
axb.set_xlabel(r"mean edge length $h$")
axb.set_ylabel("mean normal error (degrees)")
axb.legend(fontsize=10.5, frameon=False, loc="lower right")
axb.set_title("(b) jittered icospheres", fontsize=11.5)
axb.grid(True, which="both", lw=0.3, alpha=0.5)

fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.3)
dgfig.save(fig, __file__)
