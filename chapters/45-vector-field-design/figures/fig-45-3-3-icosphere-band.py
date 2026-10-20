"""Figure 45.3.3: Levi-Civita holonomy around a latitude band of the icosphere (Section 45.3, Example 45.3.13).

Mesh: dgmesh.icosphere(3) (642 vertices on the unit sphere, outward faces). For a colatitude theta_0, S is the set of
vertices with colatitude < theta_0 and c* = -d0 1_S is the boundary of the union of their dual cells: a dual cycle
through the band of faces that have vertices on both sides, traversed with S on the left (longitude increasing).
(a) theta_0 = pi/3, seen from elev 25, azim -60 (back faces culled): the faces with all three vertices in S gray, the
    band light blue; a unit vector started in one band face (at the meridian v = 0, pointing south) and carried once
    around the band by psi -> psi +- rho_e (blue arrows on the front band faces, 0.12 long); in the start face the
    vector w_0 (blue, thick) and the returned vector w_1 (orange, thick).
(b) the holonomy c*.rho mod 2 pi (in [0, 2 pi)) against theta_0 for 400 values of theta_0 in (0.05, pi - 0.05):
    icosphere(3) (black dots) and icosphere(4) (gray dots), and the smooth value 2 pi (1 - cos theta_0) mod 2 pi
    (Example 0.5.7; orange, wide); vertical gray dashed line theta_0 = pi/3.
Self-checks: on both meshes and for every theta_0, c*.rho = sum_{i in S} delta_i mod 2 pi (to 1e-10) (Corollary
45.3.12); the transported vector returns rotated by that angle; at theta_0 = pi/3 on icosphere(3) the holonomy is
within 0.03 of pi.
Interactive version: (a) with all band faces.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-3-3-icosphere-band.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
TWO_PI = 2 * np.pi


def band_data(level):
    V, F = dm.icosphere(level)
    D0 = dd.d0(F, len(V))
    rho = ch45.transport_angles(V, F)
    delta = dm.angle_defect(V, F)
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    return V, F, D0, rho, delta, th


ths = np.linspace(0.05, np.pi - 0.05, 400)
curves = {}
for level in (3, 4):
    V, F, D0, rho, delta, th = band_data(level)
    hol, sd = [], []
    for t0 in ths:
        S = (th < t0).astype(float)
        cs = -(D0 @ S)
        hol.append(np.mod(cs @ rho, TWO_PI))
        sd.append(np.mod(delta @ S, TWO_PI))
    hol, sd = np.array(hol), np.array(sd)
    d = np.abs(ch45.wrap(hol - sd))
    assert d.max() < 1e-10, d.max()
    curves[level] = hol

V, F, D0, rho, delta, th = band_data(3)
t0 = np.pi / 3
S = (th < t0).astype(float)
cs = -(D0 @ S)
E, left, right = ch45.left_right_faces(F)
band_edges = np.where(cs != 0)[0]
# order the band: from face to face across the dual edges of the cycle, in the direction of c*
nxt = {}
for e in band_edges:
    a, b = (right[e], left[e]) if cs[e] > 0 else (left[e], right[e])
    nxt[int(a)] = (int(b), int(e), 1.0 if cs[e] > 0 else -1.0)
cen = V[F].mean(1)
start = min(nxt, key=lambda f: (abs(np.arctan2(cen[f, 1], cen[f, 0])) + 0.0 * cen[f, 2]))
r, b, N = ch45.face_frames3(V, F)
south = ch45.unit(np.array([0, 0, -1.0]) - (np.array([0, 0, -1.0]) @ N[start]) * N[start])
psi0 = np.arctan2(south @ b[start], south @ r[start])
seq, psis, f, p = [start], [psi0], start, psi0
while True:
    g, e, sg = nxt[f]
    p = p + sg * rho[e]
    if g == start:
        break
    seq.append(g)
    psis.append(p)
    f = g
turn = np.mod(p - psi0, TWO_PI)
assert abs(ch45.wrap(turn - np.mod(cs @ rho, TWO_PI))) < 1e-10 and len(seq) == len(band_edges)
assert abs(turn - np.pi) < 0.03
print(f"icosphere(3), theta_0 = pi/3: {len(seq)} band faces, holonomy {turn:.4f} (pi = {np.pi:.4f}), "
      f"sum of delta over S = {np.mod(delta @ S, TWO_PI):.4f}")

cam = dm.Ortho(elev=25, azim=-60)
Nf = dm.face_normals(V, F)
vis = Nf @ cam.view > 0
order = np.argsort(cam.depth(cen))
order = order[vis[order]]
inside = S[F].min(1) > 0
inband = np.zeros(len(F), bool)
inband[seq] = True
light = np.clip(0.6 + 0.4 * (Nf @ ch45.unit(np.array([0.3, -0.4, 0.9]))), 0.35, 1.0)
base = np.where(inside[:, None], np.array([0.62, 0.62, 0.62]), np.where(inband[:, None], np.array([0.81, 0.90, 0.96]),
                                                                          np.array([0.92, 0.92, 0.92])))
fc = np.concatenate([base * light[:, None], np.ones((len(F), 1))], 1)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [0.9, 1.1]})
axa.add_collection(PolyCollection(cam(V[F[order]]), facecolors=fc[order], edgecolors="#A8A8A8", linewidths=0.15))


def arrow(ax, p, q, color, lw=1.0, z=5, ms=6):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms), zorder=z)


for f, ps in zip(seq, psis):
    if Nf[f] @ cam.view > 0.2 and f != start:
        w = np.cos(ps) * r[f] + np.sin(ps) * b[f]
        arrow(axa, cam(cen[f]), cam(cen[f] + 0.12 * w), col["tangent"])
w0 = np.cos(psi0) * r[start] + np.sin(psi0) * b[start]
w1 = np.cos(p) * r[start] + np.sin(p) * b[start]
arrow(axa, cam(cen[start]), cam(cen[start] + 0.3 * w0), col["tangent"], 1.8, ms=9)
arrow(axa, cam(cen[start]), cam(cen[start] + 0.3 * w1), col["accent"], 1.8, ms=9)
axa.text(*(cam(cen[start] + 0.3 * w0) + [0.02, -0.08]), r"$w_0$", fontsize=10)
axa.text(*(cam(cen[start] + 0.3 * w1) + [0.02, 0.02]), r"$w_1$", fontsize=10)
axa.set_aspect("equal")
axa.autoscale_view()
axa.set_axis_off()
axa.set_title(r"(a) band at $\theta_0 = \pi/3$", fontsize=11)
axb.plot(ths, curves[4], ".", color="#9A9A9A", ms=1.6, zorder=2, label="icosphere(4)")
axb.plot(ths, curves[3], ".", color=col["main"], ms=1.6, zorder=3, label="icosphere(3)")
sm = np.mod(TWO_PI * (1 - np.cos(ths)), TWO_PI)
jump = np.where(np.abs(np.diff(sm)) > np.pi)[0]
sm_plot = sm.copy().astype(float)
sm_plot[jump] = np.nan
axb.plot(ths, sm_plot, color=col["accent"], lw=3.4, alpha=0.75, zorder=1, label=r"$2\pi(1 - \cos\theta_0)$ mod $2\pi$")
axb.axvline(np.pi / 3, color=col["aux"], lw=0.8, ls="--")
axb.set_xticks([0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
axb.set_xticklabels([r"$0$", r"$\pi/3$", r"$\pi/2$", r"$2\pi/3$", r"$\pi$"])
axb.set_yticks([0, np.pi, TWO_PI])
axb.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axb.set_xlabel(r"colatitude $\theta_0$ of the band")
axb.set_ylabel("holonomy mod $2\\pi$")
axb.legend(fontsize=8, loc="upper left", frameon=False, markerscale=4)
axb.set_title("(b) holonomy against $\\theta_0$", fontsize=11)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.17, wspace=0.22)
dgfig.save(fig, __file__)

traces = [dm.plotly_mesh(V, F, color="#E0E0E0", name="icosphere(3)"), dm.plotly_edges(V, F, color="#A8A8A8", width=0.6)]
W = np.array([np.cos(ps) * r[f] + np.sin(ps) * b[f] for f, ps in zip(seq, psis)])
traces += dm.plotly_arrows(cen[seq] * 1.01, 0.12 * W, color=col["tangent"], width=4, cone=0.03, name="transport")
traces += dm.plotly_arrows(cen[[start]] * 1.01, 0.3 * w1[None], color=col["accent"], width=6, cone=0.05, name="returned")
dm.plotly_save(traces, __file__, "icosphere(3): a vector carried once around the band at colatitude pi/3 "
               "(orange: returned vector, holonomy close to pi)", "fig-45-3-3")
