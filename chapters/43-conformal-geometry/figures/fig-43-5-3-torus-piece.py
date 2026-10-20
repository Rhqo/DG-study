"""Figure 43.5.3: fixing the boundary on a circle destroys conformality at the corners; the free (spectral)
boundary keeps the right angles (Example 43.5.12, Theorem 43.4.10).

Mesh: ch43.torus_piece(24, 48): the faces of dgmesh.torus_mesh(2, 0.8, 24, 48) with |u| < pi/2 and 0 < v < pi/2
(169 vertices, 288 faces; conformally a rectangle, Example 43.3.14(c)). Two flattenings, faces coloured by
sigma1/sigma2 on a common log scale:
(a) boundary fixed on the unit circle by arc length, interior discrete harmonic (dgmesh.fixed_boundary_map, cotan
    weights; this is the minimizer of E_C with that boundary, Theorem 43.4.10),
(b) spectral conformal parametrization with B = M (ch43.spectral_conformal).
The four corner vertices (where u = +-pi/2 meets v = 0, pi/2) are black dots.
Self-check: no flipped faces; (a) median sigma1/sigma2 > 1.6 and max > 10; (b) median < 1.05, max < 1.08; the interior
angle of the image boundary at the four corners is within 5 deg of 90 in (b) and within 10 deg of 180 in (a)
(172.5 = the angle of a regular 48-gon).

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-5-3-torus-piece.py``
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

V, F, u, v = ch43.torus_piece(24, 48)
loop = dm.boundary_loop(F)
UV = dm.fixed_boundary_map(V, F, "cotan")
zf = UV[:, 0] + 1j * UV[:, 1]
zs, lam, res = ch43.spectral_conformal(V, F, dm.mass_barycentric(V, F))
corners = [int(np.argmin((u - cu) ** 2 + (v - cv) ** 2)) for cu, cv in ((-np.pi / 2, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2))]
d01 = zs[corners[3]] - zs[corners[0]]                  # rotate (b) so that the side v = 0 ... v = pi/2 at u = -pi/2 is horizontal
zs = zs * np.conj(d01) / abs(d01)


def corner_angles(z):
    out = []
    for k in corners:
        p = list(loop).index(k)
        a, b, c = z[loop[p - 1]], z[k], z[loop[(p + 1) % len(loop)]]
        out.append(abs(np.degrees(np.angle((a - b) / (c - b)))))
    return np.array(out)


stats = {}
for name, z in (("fixed", zf), ("free", zs)):
    q = ch43.quasi_conformal(V, F, z)
    stats[name] = (np.median(q), q.max(), corner_angles(z), (ch43.face_signed_areas(F, z) < 0).sum(), q)
assert stats["fixed"][3] == 0 and stats["free"][3] == 0
assert stats["fixed"][0] > 1.6 and stats["fixed"][1] > 10 and stats["free"][0] < 1.05 and stats["free"][1] < 1.08
assert np.all(np.abs(stats["free"][2] - 90) < 5) and np.all(np.abs(stats["fixed"][2] - 180) < 10)
for k, s in stats.items():
    print(k, "median %.4f max %.4f corners %s" % (s[0], s[1], np.round(s[2], 2)))
print("n0", len(V), "n2", len(F), "lambda %.4e" % lam)

norm = colors.LogNorm(1.0, 20.0)
cmap = plt.colormaps["YlOrRd"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
for ax, (name, z), title in zip(axes, (("fixed", zf), ("free", zs)), ("(a) boundary fixed on a circle", r"(b) spectral, $B = M$")):
    P = np.stack([z.real, z.imag], -1)[F]
    ax.add_collection(PolyCollection(P, facecolors=cmap(norm(stats[name][4])), edgecolors="#555555", linewidths=0.2))
    ax.plot(z[corners].real, z[corners].imag, "o", color="black", ms=3.5)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
cax = fig.add_axes([0.92, 0.18, 0.015, 0.6])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$\sigma_1/\sigma_2$", fontsize=10)
cb.ax.tick_params(labelsize=8)
fig.subplots_adjust(left=0.01, right=0.9, top=0.88, bottom=0.03, wspace=0.12)
dgfig.save(fig, __file__)
