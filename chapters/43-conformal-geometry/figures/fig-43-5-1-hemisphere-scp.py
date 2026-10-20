"""Figure 43.5.1: the spectral conformal parametrization of the hemisphere is (nearly) its stereographic projection
(Example 43.5.11, Implementation 43.5).

Mesh: ch43.hemisphere(3) = the faces of dgmesh.icosphere(3) with centroid below z = 0 (337 vertices, 624 faces,
outward normals). z = spectral conformal parametrization with B = M (barycentric vertex areas), by the inverse power
iteration of ch43.spectral_conformal; z is rotated so that its best fit a conj(sigma) has a real positive a.
(a) the mesh seen from below, faces coloured by sigma1/sigma2 of the face map (sequential colours, 1 = white).
(b) the image z(mesh) with the same colours; the dashed polygon is the best fit a conj(sigma) + c evaluated at the
    boundary vertices (the stereographic image of the mesh boundary), the black dot is z(south pole).
Self-check: lambda = 2.194e-3, residual < 1e-10, no flipped face, max sigma1/sigma2 < 1.1, relative distance to the
best fit a conj(sigma) + c below 0.01; the boundary vertices lie within 3 % of |a| of the dashed polygon.
The interactive version shows the hemisphere with the same colours.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-5-1-hemisphere-scp.py``
"""

import pathlib
import sys

import dgfig  # noqa: E402
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch43  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

col = dgfig.COLORS
V, F = ch43.hemisphere(3)
m = dm.mass_barycentric(V, F)
z, lam, res = ch43.spectral_conformal(V, F, m)
sig = ch43.stereo_conj(V)
a, err = ch43.best_fit(z, sig, m)
z = z * np.exp(-1j * np.angle(a))                      # rotate: best fit has a > 0
a, err = ch43.best_fit(z, sig, m)
cen = (m @ (z - a * sig)) / m.sum()
q = ch43.quasi_conformal(V, F, z)
flips = (ch43.face_signed_areas(F, z) < 0).sum()
loop = dm.boundary_loop(F)
target = a * sig + cen                                 # best-fit stereographic image
rb = np.abs(z[loop] - target[loop]) / abs(a)
south = np.argmin(V[:, 2])
assert abs(lam - 2.194e-3) < 2e-6 and res < 1e-10 and flips == 0 and q.max() < 1.1 and err < 0.01
assert rb.max() < 0.03
print("n0 %d, n2 %d, lambda %.4e, max qc %.4f, median qc %.4f, fit error %.4f, |a| %.4f, max boundary deviation %.4f"
      % (len(V), len(F), lam, q.max(), np.median(q), err, abs(a), rb.max()))

norm = colors.Normalize(1.0, 1.1)
cmap = plt.colormaps["YlOrRd"]
fc = cmap(norm(q))
fig = plt.figure(figsize=(6.6, 3.2))
ax = dgfig.axes3d(fig, 121, elev=-55, azim=-60)
ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolor="#555555", linewidths=0.15))
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 0]]), zoom=1.3)
ax.set_title("(a) hemisphere mesh", fontsize=11)
ax2 = fig.add_subplot(122)
P = np.stack([z.real, z.imag], -1)[F]
ax2.add_collection(PolyCollection(P, facecolors=fc, edgecolors="#555555", linewidths=0.15))
cc = target[np.r_[loop, loop[:1]]]
ax2.plot(cc.real, cc.imag, ls="--", color=col["main"], lw=0.9)
ax2.plot([z[south].real], [z[south].imag], "o", color=col["main"], ms=3)
ax2.set_aspect("equal")
ax2.autoscale_view()
ax2.set_axis_off()
ax2.set_title(r"(b) image $z$ ($B = M$)", fontsize=11)
cax = fig.add_axes([0.92, 0.18, 0.015, 0.6])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$\sigma_1/\sigma_2$", fontsize=10)
cb.ax.tick_params(labelsize=8)
fig.subplots_adjust(left=0.0, right=0.9, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

hover = [f"sigma1/sigma2 = {v:.3f}" for v in q]
tr = dm.plotly_mesh(V, F, intensity=q, colorscale="YlOrRd", cmin=1.0, cmax=1.1, name="hemisphere", showscale=True,
                    colorbar_title="sigma1/sigma2")
dm.plotly_save([tr, dm.plotly_edges(V, F, color="#555555", width=1)], __file__,
               "hemisphere (icosphere(3), z < 0): sigma1/sigma2 of the spectral conformal parametrization (B = M)", "fig-43-5-1")
