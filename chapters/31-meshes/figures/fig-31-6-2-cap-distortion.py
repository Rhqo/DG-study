"""Figure 31.6.2: a curved cap cannot be flattened without distortion; conformal flattening pays in area (Section 31.6).

Spherical caps {polar angle < Theta} of the unit sphere: the faces of the level-4 icosphere (2562 vertices) whose
centroid has z > cos Theta. Each cap is flattened by LSCM [Levy02] with two boundary vertices pinned (the farthest
pair). For every face, sigma_1 >= sigma_2 are the singular values of the Jacobian of the map UV -> surface; sigma_1 sigma_2 is the
surface area covered by a unit UV area (the inverse of the texel density in area).
(a) The UV images for Theta = 60, 90, 120 degrees (each scaled to the same width), faces coloured by
    log2(sigma_1 sigma_2 / median).
(b) The ratio max/min of sigma_1 sigma_2 over the faces against Theta (dots), and sec^4(Theta/2) (curve), the ratio of the
    area factors of the stereographic projection from the antipode between the rim and the centre of the cap.
    Top axis: the total Gaussian curvature of the cap, 2 pi (1 - cos Theta).
Self-check: no flipped triangles; the conformal distortion sigma_1/sigma_2 has median below 1.08; the area ratio follows
sec^4(Theta/2) within a factor 1.5 (the cap boundary is jagged).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-6-2-cap-distortion.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

C = dgfig.COLORS
V, F = dm.icosphere(4)
CEN = V[F].mean(1)


def farthest_pair(Vc, loop):
    P = Vc[loop]
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    return loop[a], loop[b]


def flatten_cap(theta_deg):
    m = CEN[:, 2] > np.cos(np.radians(theta_deg))
    Vc, Fc, _ = dm.submesh(V, F, m)
    assert dm.euler_characteristic(Fc) == 1 and dm.is_manifold(Fc)
    loop = dm.boundary_loop(Fc)
    UV = dm.lscm(Vc, Fc, farthest_pair(Vc, loop), [(0.0, 0.0), (1.0, 0.0)])
    s, det = dm.uv_distortion(Vc, Fc, UV)
    return Vc, Fc, UV, s, det


thetas = [30, 45, 60, 75, 90, 105, 120, 135, 150, 160, 170]
ratio, conf = [], []
keep = {}
for th in thetas:
    Vc, Fc, UV, s, det = flatten_cap(th)
    assert (det > 0).all()
    a = s[:, 0] * s[:, 1]
    ratio.append(a.max() / a.min())
    conf.append(np.median(s[:, 0] / s[:, 1]))
    if th in (60, 90, 120):
        keep[th] = (Fc, UV, a)
ratio = np.array(ratio)
sec4 = 1 / np.cos(np.radians(np.array(thetas)) / 2) ** 4
print("median s1/s2:", np.round(conf, 3))
assert max(conf) < 1.08
print("ratio / sec4:", np.round(ratio / sec4, 3))
assert np.all(np.abs(np.log(ratio / sec4)) < np.log(1.5))
print("Theta:", thetas)
print("max/min of s1 s2:", np.round(ratio, 2))
print("sec^4(Theta/2):  ", np.round(sec4, 2))

fig = plt.figure(figsize=(7.4, 7.0))
norm = TwoSlopeNorm(vcenter=0.0, vmin=-2.5, vmax=2.5)
cmap = plt.get_cmap("RdBu_r")
for col, th in enumerate((60, 90, 120)):
    ax = fig.add_axes([0.04 + col * 0.32, 0.665, 0.28, 0.26])
    Fc, UV, a = keep[th]
    W = UV - 0.5 * (UV.min(0) + UV.max(0))
    W /= np.ptp(W, axis=0).max()
    val = np.log2(a / np.median(a))
    pc = PolyCollection(W[Fc], facecolors=cmap(norm(val)), edgecolors="none", rasterized=True)
    ax.add_collection(pc)
    ax.set_xlim(-0.53, 0.53)
    ax.set_ylim(-0.53, 0.53)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(rf"$\Theta = {th}^\circ$: ratio {a.max() / a.min():.1f}", fontsize=11.5)
fig.text(0.5, 0.965, "(a) LSCM flattening of spherical caps (UV, same size)", ha="center", fontsize=12)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = fig.add_axes([0.2, 0.615, 0.6, 0.018])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label(r"$\log_2(\sigma_1 \sigma_2 / \mathrm{median})$: red = a texel covers more surface", fontsize=10)
cb.ax.tick_params(labelsize=9)

ax = fig.add_axes([0.11, 0.08, 0.86, 0.31])
tt = np.linspace(5, 172, 300)
ax.semilogy(tt, 1 / np.cos(np.radians(tt) / 2) ** 4, color=C["main"], lw=1.6, label=r"$\sec^4(\Theta/2)$ (stereographic)")
ax.semilogy(thetas, ratio, "o", color=C["normal"], ms=6, label=r"LSCM: $\max \sigma_1\sigma_2 / \min \sigma_1\sigma_2$")
ax.set_xlabel(r"cap size $\Theta$ (degrees)")
ax.set_ylabel("area ratio")
ax.set_xlim(0, 180)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.grid(True, which="major", lw=0.3, alpha=0.5)
ax.legend(fontsize=10.5, frameon=False, loc="upper left")
top = ax.secondary_xaxis("top", functions=(lambda t: 2 * np.pi * (1 - np.cos(np.radians(t))),
                                            lambda k: np.degrees(np.arccos(np.clip(1 - k / (2 * np.pi), -1, 1)))))
top.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
top.set_xticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
top.set_xlabel(r"total curvature $\int K\,dA = 2\pi(1 - \cos\Theta)$", fontsize=10.5, labelpad=8)
fig.text(0.5, 0.505, "(b) area distortion grows with the curvature inside the chart", ha="center", fontsize=12)
dgfig.save(fig, __file__)
