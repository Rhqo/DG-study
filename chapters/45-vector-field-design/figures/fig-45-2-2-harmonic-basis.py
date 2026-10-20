"""Figure 45.2.2: the harmonic basis of a torus mesh from its tree-cotree generators (Section 45.2, Example 45.2.15).

Mesh: dgmesh.torus_mesh(2, 0.8, 24, 48). ch45.generators (BFS from vertex 0 and face 0) gives the primal loops c_1, c_2
and the dual loops c*_1, c*_2; omega_a = c*_a (the closed 1-form that is +-1 on the edges crossed by c*_a) and
gamma_a = its harmonic part (ch45.harmonic_basis, one Cholesky factorization of -L + a a^T).
(a), (b): the mesh seen from elev 30, azim -60 (painter's algorithm, back faces culled), with the edges of the primal
loop c_a (black) and blue arrows gamma_a^sharp (Whitney interpolant at the barycenters; 1.5 times the true length in (a),
4 times in (b), where |gamma_2| = 1/(2 pi (R + r cos u)) is small) on the faces 2 (i nv + j) with i, j even that face
the viewer.
Self-checks: c_b . gamma_a = delta_ab; d1 gamma_a = 0 and d0^T star1 gamma_a = 0 to 1e-12; the Gram matrix
<gamma_a, gamma_b>_1 equals c*_a . star1 gamma_b and is symmetric positive definite; each gamma_a is a combination of
the harmonic parts of R_1(du) and R_1(dv) (to 1e-12).
Interactive version: both fields on the same mesh (blue gamma_1, green gamma_2).

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-2-2-harmonic-basis.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
R, r, nu, nv = 2.0, 0.8, 24, 48
V, F = dm.torus_mesh(R, r, nu, nv)
D0, D1, s1 = dd.d0(F, len(V)), dd.d1(F), dd.hodge1(V, F)
tc, C, Cs = ch45.generators(F)
G = ch45.harmonic_basis(D0, s1, Cs)
assert np.allclose(C @ G.T, np.eye(2), atol=1e-10)
assert np.abs(D1 @ G.T).max() < 1e-12 and np.abs(D0.T @ (s1[:, None] * G.T)).max() < 1e-12
gram = G @ (s1[:, None] * G.T)
assert np.allclose(gram, Cs @ (s1[:, None] * G.T), atol=1e-10) and np.linalg.eigvalsh(gram).min() > 0
u = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]) - R)
v = np.arctan2(V[:, 1], V[:, 0])
H2 = ch45.harmonic_basis(D0, s1, np.array([ch45.wrap(D0 @ u), ch45.wrap(D0 @ v)]))
coef = np.linalg.lstsq(H2.T, G.T, rcond=None)[0]
assert np.abs(H2.T @ coef - G.T).max() < 1e-12
print("gamma_a in terms of (harm du, harm dv), times 2 pi:\n", np.round(2 * np.pi * coef.T, 6))
print("Gram matrix <gamma_a, gamma_b>_1:\n", np.round(gram, 4))

cam = dm.Ortho(elev=30, azim=-60)
Nf = dm.face_normals(V, F)
cen = V[F].mean(1)
vis = Nf @ cam.view > 0
order = np.argsort(cam.depth(cen))
order = order[vis[order]]
light = np.clip(0.6 + 0.4 * (Nf @ ch45.unit(np.array([0.3, -0.4, 0.9]))), 0.35, 1.0)
fc = np.tile(np.array([0.86, 0.86, 0.86, 1.0]), (len(F), 1))
fc[:, :3] *= light[:, None]
fi, fj = (np.arange(len(F)) // 2) // nv, (np.arange(len(F)) // 2) % nv
sel = vis & (Nf @ cam.view > 0.25) & (np.arange(len(F)) % 2 == 0) & (fi % 2 == 0) & (fj % 2 == 0)
E = tc["E"]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
SCALE = (1.5, 4.0)                                  # arrow scale per panel
for a, ax in enumerate(axs):
    X = dd.whitney_at_barycenters(V, F, G[a])
    ax.add_collection(PolyCollection(cam(V[F[order]]), facecolors=fc[order], edgecolors="#A8A8A8", linewidths=0.15))
    loop = [cam(V[E[e]]) for e in np.where(C[a] != 0)[0]
            if (Nf[tc["left"][e]] @ cam.view > 0) or (Nf[tc["right"][e]] @ cam.view > 0)]
    ax.add_collection(LineCollection(loop, colors=col["main"], linewidths=1.6, zorder=3))
    for f in np.where(sel)[0]:
        ax.annotate("", xy=cam(cen[f] + SCALE[a] * X[f]), xytext=cam(cen[f]),
                    arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=0.8, mutation_scale=7), zorder=4)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    ax.set_title(rf"({'ab'[a]}) $\gamma_{a + 1}$ and the loop $c_{a + 1}$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.04)
dgfig.save(fig, __file__)

traces = [dm.plotly_mesh(V, F, color="#D8D8D8", name="torus"), dm.plotly_edges(V, F, color="#A8A8A8", width=0.6)]
for a, cc in enumerate((col["tangent"], col["third"])):
    X = dd.whitney_at_barycenters(V, F, G[a])
    traces += dm.plotly_arrows(cen[::2], 1.5 * X[::2], color=cc, width=3, cone=0.05, name=f"gamma_{a + 1}")
dm.plotly_save(traces, __file__, "torus_mesh(2, 0.8, 24, 48): harmonic basis gamma_1 (blue), gamma_2 (green), "
               "arrows 1.5 x true length", "fig-45-2-2")
