"""Figure 36.4.2: closed but not exact discrete 1-forms on the torus mesh (Section 36.4, Example 36.4.14).

Mesh: dgmesh.torus_mesh(2, 0.8, 12, 16), i.e. T_{2,0.8} with 12 vertices around the tube (angle u) and 16 around
the z-axis (angle v); vertex 16 i + j sits at u = 2 pi i / 12, v = 2 pi j / 16.
Discrete 1-forms: alpha = R_1(dv) with dv = (x dy - y dx)/(x^2 + y^2) (angle around the z-axis) and
beta = R_1(du) with u = atan2(z, rho - 2), rho = sqrt(x^2 + y^2) (angle around the core circle). Both smooth forms
are closed where they are defined, so d1 alpha = d1 beta = 0.
(a) the torus (gray, painter's algorithm) with the longitude loop i = 0 (blue, around the z-axis) and the
    meridian loop j = 0 (orange, around the tube); hidden parts are dashed.
(b) the parameter domain [0, 2 pi]^2 (horizontal v, vertical u) with the triangulation and the same two loops;
    the table of loop sums.
Self-check: d1 alpha = d1 beta = 0 to 1e-12; <alpha, longitude> = 2 pi, <alpha, meridian> = 0,
<beta, longitude> = 0, <beta, meridian> = 2 pi; both loops are cycles; least squares d0 u = alpha leaves a residual;
first Betti number n1 - rank d0 - rank d1 = 2. Writes the interactive version
fig-36-4-2-torus-closed-not-exact-interactive.html (plotly).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-4-2-torus-closed-not-exact.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

nu, nv = 12, 16
V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
E, index = dd.edge_list(F)
D0, D1 = dd.d0(F, len(V)), dd.d1(F)


def dv(p):
    return np.array([-p[1], p[0], 0.0]) / (p[0] ** 2 + p[1] ** 2)


def du(p):
    rho = np.hypot(p[0], p[1])
    q = (rho - 2.0) ** 2 + p[2] ** 2
    drho = np.array([p[0], p[1], 0.0]) / rho
    return (-p[2] * drho + (rho - 2.0) * np.array([0.0, 0.0, 1.0])) / q


alpha = dd.derham1(dv, V, E, order=10)
beta = dd.derham1(du, V, E, order=10)
assert np.abs(D1 @ alpha).max() < 1e-12 and np.abs(D1 @ beta).max() < 1e-12


def loop_chain(verts):
    c = np.zeros(len(E))
    for a, b in zip(verts, verts[1:] + verts[:1]):
        c[index[(min(a, b), max(a, b))]] += 1 if a < b else -1
    return c


lon = [0 * nv + j for j in range(nv)]                 # i = 0: outer equator, around the z-axis
mer = [i * nv + 0 for i in range(nu)]                 # j = 0: around the tube
c_lon, c_mer = loop_chain(lon), loop_chain(mer)
assert not np.any(D0.T @ c_lon) and not np.any(D0.T @ c_mer)
sums = np.array([[alpha @ c_lon, alpha @ c_mer], [beta @ c_lon, beta @ c_mer]])
assert np.allclose(sums, [[2 * np.pi, 0], [0, 2 * np.pi]], atol=1e-12)
res = alpha - D0 @ np.linalg.lstsq(D0, alpha, rcond=None)[0]
assert np.linalg.norm(res) > 0.1
assert len(E) - np.linalg.matrix_rank(D0) - np.linalg.matrix_rank(D1) == 2

C = dgfig.COLORS
BLUE, ORANGE = C["tangent"], C["accent"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 3.3), gridspec_kw={"width_ratios": [1.25, 1.0]})
cam = dm.Ortho(elev=38, azim=-62)
cols = dm.shade(V, F, dm.hex_to_rgb("#D8D8D8"))
dm.draw_mesh_2d(axa, V, F, cam, cols, edgecolor="#9A9A9A", lw=0.25)
ef = dm.edge_faces(F)
normals = dm.face_normals(V, F)
for verts, col in ((lon, BLUE), (mer, ORANGE)):
    for a, b in zip(verts, verts[1:] + verts[:1]):
        fs = ef[(min(a, b), max(a, b))]
        visible = any(normals[f] @ cam.view > 0 for f in fs)
        P = cam(V[[a, b]])
        # lift visible segments slightly toward the viewer so that they are drawn over the faces
        axa.plot(*P.T, color=col, lw=2.4 if visible else 1.3, ls="-" if visible else (0, (2, 2)),
                 zorder=5 if visible else 3, alpha=1.0 if visible else 0.8)
axa.set_aspect("equal")
axa.set_axis_off()
axa.autoscale_view()
axa.set_title("(a) two loops on the torus mesh", fontsize=11)

# (b) parameter domain: draw each face once with unwrapped indices
for i in range(nu):
    for j in range(nv):
        a, d, b, c = (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)
        for tri in ((a, d, b), (b, d, c)):
            P = np.array([[2 * np.pi * jj / nv, 2 * np.pi * ii / nu] for ii, jj in tri])
            axb.fill(*P.T, facecolor="#F4F4F4", edgecolor="#BDBDBD", lw=0.4)
axb.plot([0, 2 * np.pi], [0, 0], color=BLUE, lw=2.6, solid_capstyle="butt")
axb.plot([0, 0], [0, 2 * np.pi], color=ORANGE, lw=2.6, solid_capstyle="butt")
axb.annotate("", xy=(np.pi + 0.5, 0), xytext=(np.pi - 0.3, 0),
             arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0, mutation_scale=14))
axb.annotate("", xy=(0, np.pi + 0.5), xytext=(0, np.pi - 0.3),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, mutation_scale=14))
axb.set_xlabel(r"$v$")
axb.set_ylabel(r"$u$")
axb.set_xticks([0, np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
axb.set_yticks([0, np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
axb.set_aspect("equal")
axb.set_title("(b) parameter domain", fontsize=11)
txt = (r"$\langle\widehat{dv},\,$" + "blue" + r"$\rangle = 2\pi$,  " + r"$\langle\widehat{dv},\,$" + "orange"
       + r"$\rangle = 0$" + "\n" + r"$\langle\widehat{du},\,$" + "blue" + r"$\rangle = 0$,  "
       + r"$\langle\widehat{du},\,$" + "orange" + r"$\rangle = 2\pi$")
axb.text(np.pi, -1.9, txt, fontsize=9.5, ha="center", va="top")
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.24, wspace=0.12)
dgfig.save(fig, __file__)

# interactive version
import plotly.graph_objects as go  # noqa: E402

traces = [dm.plotly_mesh(V, F, color="#D8D8D8", name="torus T<sub>2,0.8</sub>"),
          dm.plotly_edges(V, F, color="#888888", width=1)]
for verts, col, name in ((lon, BLUE, "longitude: \u27e8dv, c\u27e9 = 2\u03c0"),
                         (mer, ORANGE, "meridian: \u27e8du, c\u27e9 = 2\u03c0")):
    P = V[verts + verts[:1]] * 1.01
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=col, width=7),
                               name=name, hoverinfo="name"))
dm.plotly_save(traces, __file__,
               "Torus mesh T<sub>2,0.8</sub> (12 \u00d7 16): the longitude (blue) and meridian (orange) loops",
               "fig-36-4-2")
