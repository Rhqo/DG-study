"""Figure 45.3.1: Levi-Civita transport across an edge, folded and unfolded (Section 45.3, Definition 45.3.2).

The hinge of Example 38.6.4: x0 = (0, 0, 0), x1 = (1, 0, 0), x2 = (0.5, 0.8, 0), x3 = (0.3, -0.9 cos psi, 0.9 sin psi)
with psi = 70 degrees; f = (0, 1, 2) contains the halfedge 0 -> 1 (left face of e = [0, 1]), g = (1, 0, 3) the
halfedge 1 -> 0 (right face), so the dual edge e* runs g -> f (Remark 36.5.1).
w = unit vector in g at angle 50 degrees from the reference direction r_g = unit(x0 - x1) (counterclockwise about N_g);
P_e w = <w, e> e + <w, N_g x e> N_f x e (44.3.4) with e = unit(x1 - x0).
(a) the folded hinge (orthographic view elev 26, azim 28, looking obliquely along the edge into the hinge): f light gray, g light blue, the edge black, N_f and N_g
    vermilion at the face centroids (length 0.35), w (blue) at the centroid of g and P_e w (orange) at the centroid of f.
(b) the two faces unfolded into the plane of f (g rotated about the edge into z = 0): w and P_e w are the same plane
    vector, drawn at the two centroids (blue and orange), the reference directions r_f, r_g gray.
Self-checks: |P_e w| = 1, <P_e w, N_f> = 0, P_e fixes e, the unfolded copy of w equals P_e w, and
rho_e = angle(P_e r_g) - angle(r_f) agrees with ch45.transport_angles.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-3-1-hinge-transport.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
psi = np.radians(70)
V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0], [0.3, -0.9 * np.cos(psi), 0.9 * np.sin(psi)]], float)
F = np.array([[0, 1, 2], [1, 0, 3]])
f, g = 0, 1
N = dm.face_normals(V, F)
eh = ch45.unit(V[1] - V[0])
r, b, _ = ch45.face_frames3(V, F)
ang = np.radians(50)
w = np.cos(ang) * r[g] + np.sin(ang) * b[g]


def P(x):
    return (x @ eh) * eh + (x @ np.cross(N[g], eh)) * np.cross(N[f], eh)


Pw = P(w)
assert abs(np.linalg.norm(Pw) - 1) < 1e-12 and abs(Pw @ N[f]) < 1e-12 and np.allclose(P(eh), eh)
# unfold g into the plane of f: rotate about the edge axis by the angle taking N_g to N_f
th = np.arctan2(np.cross(N[g], N[f]) @ eh, N[g] @ N[f])
K = np.array([[0, -eh[2], eh[1]], [eh[2], 0, -eh[0]], [-eh[1], eh[0], 0]])
Rot = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
assert np.allclose(Rot @ N[g], N[f]) and np.allclose(Rot @ w, Pw)          # Proposition 45.3.5
x3u = Rot @ V[3]
assert abs(x3u[2]) < 1e-12
rho = ch45.transport_angles(np.vstack([V[:3], [V[3]]]), F)
_, idx = __import__("dgdec").edge_list(F)
rho01 = rho[idx[(0, 1)]]
Prg = P(r[g])
assert abs(np.arctan2(Prg @ b[f], Prg @ r[f]) - rho01) < 1e-12
print(f"bending angle {np.degrees(th):.2f} deg, rho_e = {np.degrees(rho01):.2f} deg, "
      f"angle of P_e w from r_f = {np.degrees(np.arctan2(Pw @ b[f], Pw @ r[f])):.2f} deg")

cf, cg = V[F[f]].mean(0), V[F[g]].mean(0)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.05, 1.0]})
cam = dm.Ortho(elev=26, azim=28)
for tri, fc in sorted(((F[g], "#CFE6F5"), (F[f], "#E4E4E4")), key=lambda t: cam.depth(V[t[0]].mean(0))):
    axa.add_patch(Polygon(cam(V[tri]), closed=True, fc=fc, ec="#8A8A8A", lw=0.8, zorder=1))
axa.plot(*cam(V[[0, 1]]).T, color=col["main"], lw=2.0, zorder=3)


def arrow(ax, p, q, color, lw=1.6, z=4):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11), zorder=z)


s = 0.35
arrow(axa, cam(cf), cam(cf + s * N[f]), col["normal"])
arrow(axa, cam(cg), cam(cg + s * N[g]), col["normal"])
arrow(axa, cam(cg), cam(cg + 0.38 * w), col["tangent"], 2.0)
arrow(axa, cam(cf), cam(cf + 0.38 * Pw), col["accent"], 2.0)
for p, t, dx in ((cf + s * N[f], r"$\mathbf{N}_f$", (0.03, -0.05)), (cg + s * N[g], r"$\mathbf{N}_g$", (0.03, 0.02)),
                 (cg + 0.38 * w, r"$w$", (0.02, -0.07)), (cf + 0.38 * Pw, r"$P_e w$", (-0.2, 0.0))):
    axa.text(*(cam(p) + dx), t, fontsize=11)
axa.text(*(cam(0.5 * (V[0] + V[1])) + [0.0, -0.1]), r"$e$", fontsize=11)
axa.text(*cam(V[2] + [0.04, 0.02, 0]), r"$f$", fontsize=11)
axa.text(*cam(V[3] + [0.04, 0, 0.02]), r"$g$", fontsize=11)
axa.set_title(r"(a) folded: $P_e$ carries $w$ from $g$ to $f$", fontsize=11)
# (b) unfolded into the plane of f (z = 0): coordinates (x, y)
Pf = V[F[f]][:, :2]
Pg = np.array([V[1], V[0], x3u])[:, :2]
axb.add_patch(Polygon(Pg, closed=True, fc="#CFE6F5", ec="#8A8A8A", lw=0.8, zorder=1))
axb.add_patch(Polygon(Pf, closed=True, fc="#E4E4E4", ec="#8A8A8A", lw=0.8, zorder=1))
axb.plot([0, 1], [0, 0], color=col["main"], lw=2.0, zorder=3)
cgu = Pg.mean(0)
cfu = Pf.mean(0)
wu = (Rot @ w)[:2]
arrow(axb, cgu, cgu + 0.38 * wu, col["tangent"], 2.0)
arrow(axb, cfu, cfu + 0.38 * wu, col["accent"], 2.0)
rg = (Rot @ r[g])[:2]
arrow(axb, cgu, cgu + 0.25 * rg, col["aux"], 1.0)
arrow(axb, cfu, cfu + 0.25 * r[f][:2], col["aux"], 1.0)
axb.text(*(cgu + 0.38 * wu + [0.02, -0.06]), r"$w$", fontsize=11)
axb.text(*(cfu + 0.38 * wu + [-0.2, 0.03]), r"$P_e w$", fontsize=11)
axb.text(*(cgu + 0.25 * rg + [-0.02, -0.09]), r"$\mathbf{r}_g$", fontsize=10, color=col["aux"])
axb.text(*(cfu + 0.25 * r[f][:2] + [0.01, -0.07]), r"$\mathbf{r}_f$", fontsize=10, color=col["aux"])
axb.text(0.5, 0.05, r"$e$", fontsize=11)
axb.set_title(r"(b) unfolded: $w$ is only translated", fontsize=11)
for ax in (axa, axb):
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)
