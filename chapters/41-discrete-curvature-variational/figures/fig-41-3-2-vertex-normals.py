"""Figure 41.3.2: the sphere-inscribed normal is exact when the one-ring lies on a sphere.

Angle (degrees, log scale) between a vertex normal and the exact unit normal at every vertex, sorted:
(a) a jittered unit sphere (icosphere(3), tangential jitter 0.15 x mean edge length, seed 7, projected back to the sphere):
    uniform, area-weighted, angle-weighted and sphere-inscribed normals; the last is at the rounding level.
(b) a bumpy surface of revolution, profile (rho(t), z(t)) = (2 + 0.8 cos t + 0.2 cos 2t, 0.6 sin t), sampled on the
    24 x 48 grid of torus_mesh (same connectivity): no normal is exact.
Points below 1e-12 degrees are drawn at 1e-12.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-3-2-vertex-normals.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS


def ang(A, B):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(A, B), axis=1), np.sum(A * B, 1)))


Vj, Fj = ddg41.jittered_sphere(3, 0.15, 7)
assert (np.sum(dm.face_cross(Vj, Fj) * Vj[Fj].mean(1), 1) < 0).sum() == 0
Vb, Fb, Nb = ddg41.bumpy_revolution()
cases = [(Vj, Fj, Vj), (Vb, Fb, Nb)]
kinds = [("uniform", C["tangent"], "-"), ("area", C["third"], "--"), ("angle", C["accent"], "-"), ("sphere", C["normal"], "-")]
res = []
for V, F, Nex in cases:
    d = {k: ang(dm.vertex_normals(V, F, kind=k), Nex) for k, _, _ in kinds[:3]}
    d["sphere"] = ang(ddg41.sphere_inscribed_normals(V, F), Nex)
    res.append(d)
assert res[0]["sphere"].max() < 1e-11 and res[0]["angle"].max() > 1.0
assert res[1]["sphere"].max() > 1.0

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, d, title in zip(axes, res, ("(a)", "(b)")):
    for k, col, ls in kinds:
        y = np.maximum(np.sort(d[k])[::-1], 1e-12)
        ax.semilogy(np.arange(len(y)) / (len(y) - 1), y, color=col, ls=ls, lw=1.6 if k == "sphere" else 1.2, label=k)
    ax.set_xlabel("vertices, sorted by error")
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(5e-13, 30)
    ax.set_title(title, loc="left", fontsize=12)
axes[0].set_ylabel("angle to the exact normal (deg)")
axes[1].legend(fontsize=9.5, frameon=False, loc="lower left")
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.9, wspace=0.08)
dgfig.save(fig, __file__)
print({k: round(float(v.max()), 4) for k, v in res[0].items()}, {k: round(float(v.max()), 4) for k, v in res[1].items()})
