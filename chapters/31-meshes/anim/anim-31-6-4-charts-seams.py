"""Animation 31.6.4: cutting the sphere into more charts lowers the area distortion and lengthens the seams (Figure 31.6.4).

Data (as fig-31-6-4-charts-seams.py): the unit sphere as the level-4 icosphere (2562 vertices, 5120 faces), cut by face
centroid c into 1 chart (the sphere minus the cap c_z < -cos 10 deg), 2 charts (hemispheres), 6 charts (cube faces) and
20 charts (icosahedron faces); every chart flattened by LSCM with its farthest pair of boundary vertices pinned.
Each chart's UV is rescaled so that its UV area equals its surface area; then sigma_1 sigma_2 = 1 means the average
texel density of that chart. Colour = log2(sigma_1 sigma_2) on [-2.5, 2.5] (RdBu_r, red = sparse texels), the same on the
sphere (left, seen from below, elev -30, so that the 1-chart hole at the south pole is visible) and on the UV pieces
(middle, laid out in a grid, same scale for every cut). Black: seams (on the sphere) and the chart boundaries in UV.
Right: total seam length against the worst chart's max sigma_1 sigma_2 / min sigma_1 sigma_2 (log axis).
Self-check (as the figure): every chart is a disk (chi = 1) without flipped faces; worst ratios 14470, 4.84, 1.47, 1.10
(within 1%); seam lengths 1.52, 7.06, 17.74, 33.21.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-6-4-charts-seams.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

V, F = dm.icosphere(4)
CEN = V[F].mean(1)
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)
Vi, Fi = dm.icosahedron()
Ci = Vi[Fi].mean(1)
Ci /= np.linalg.norm(Ci, axis=1, keepdims=True)
AX_ = np.argmax(np.abs(CEN), 1)
CUTS = {
    1: np.where(CEN[:, 2] < -np.cos(np.radians(10)), -1, 0),
    2: (CEN[:, 2] < 0).astype(int),
    6: 2 * AX_ + (CEN[np.arange(len(CEN)), AX_] < 0),
    20: np.argmax(CEN @ Ci.T, 1),
}


def farthest_pair(Vc, loop):
    P = Vc[loop]
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    return loop[a], loop[b]


def seam_edges(lab):
    out = []
    for (i, j), fs in dm.edge_faces(F).items():
        if len(fs) == 2 and lab[fs[0]] != lab[fs[1]]:
            out.append((i, j))
    return np.array(out)


RES = {}
for k, lab in CUTS.items():
    charts, ratios = [], []
    val = np.full(len(F), np.nan)
    for c in np.unique(lab):
        if c < 0:
            continue
        idx = np.where(lab == c)[0]
        Vc, Fc, _ = dm.submesh(V, F, lab == c)
        assert dm.euler_characteristic(Fc) == 1 and dm.is_manifold(Fc)
        loop = dm.boundary_loop(Fc)
        UV = dm.lscm(Vc, Fc, farthest_pair(Vc, loop), [(0.0, 0.0), (1.0, 0.0)])
        s, det = dm.uv_distortion(Vc, Fc, UV)
        assert (det > 0).all()
        a = s[:, 0] * s[:, 1]
        ratios.append(a.max() / a.min())
        A_surf = dm.face_areas(Vc, Fc).sum()
        A_uv = det.sum()
        UVn = (UV - UV.mean(0)) * np.sqrt(A_surf / A_uv)       # UV area = surface area
        a_n = a * A_uv / A_surf
        val[idx] = np.log2(a_n)
        charts.append((Fc, UVn))
    E = seam_edges(lab)
    length = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).sum()
    RES[k] = dict(lab=lab, charts=charts, ratio=max(ratios), seam=length, E=E, val=val)
for k, r in {1: 14470.06, 2: 4.84, 6: 1.47, 20: 1.10}.items():
    assert abs(RES[k]["ratio"] / r - 1) < 0.01
assert [round(RES[k]["seam"], 2) for k in (1, 2, 6, 20)] == [1.52, 7.06, 17.74, 33.21]


def layout(charts, cell, cols):
    """Place the (rescaled) UV pieces on a grid; returns a list of (Fc, 2D points)."""
    out = []
    for n, (Fc, UV) in enumerate(charts):
        # rotate each piece so that its longest direction is horizontal (display only)
        Wc = UV - UV.mean(0)
        _, _, vt = np.linalg.svd(Wc, full_matrices=False)
        Wc = Wc @ vt.T
        r, c = divmod(n, cols)
        rows = (len(charts) + cols - 1) // cols
        off = np.array([(c - (cols - 1) / 2) * cell, ((rows - 1) / 2 - r) * cell])
        out.append((Fc, Wc + off))
    return out


LAYOUT = {1: (0, 1), 2: (3.1, 2), 6: (1.85, 3), 20: (1.45, 5)}


class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=-30, azim=-55, scale=1.4, center=[-5.35, 0.45, 0])
        sph = MeshView(F, stroke_width=0.5, ambient=0.5)
        UVC, UVS = np.array([-0.75, 0.45]), 0.68
        ttl = title("More charts: less area distortion, longer seams", 34)
        lab_s = da.label("on the sphere", 30).move_to([-5.35, 2.75, 0])
        lab_u = da.label("UV pieces (same scale)", 30).move_to([-0.75, 2.75, 0])
        bar = colorbar(-2.9, -2.5, 2.0, 0.2, "RdBu_r", -2.5, 2.5, ticks=[(-2, "-2"), (0, "0"), (2, "2")], size=30)
        bar_lab = VGroup(mt(r"colour: $\log_2(\sigma_1\sigma_2)$", 30).move_to([-2.9, -1.95, 0], aligned_edge=LEFT),
                         da.label("red: sparse", 30).move_to([-0.65, -2.4, 0], aligned_edge=LEFT))
        P = Plot(3.6, -1.55, 3.0, 3.45, (0, 36), (1, 1e5), ylog=True)
        axes = P.axes(xticks=[(0, "0"), (15, "15"), (30, "30")],
                      yticks=[(1, "1"), (10, "10"), (100, mt(r"$10^2$", 30)), (1e3, mt(r"$10^3$", 30)),
                              (1e4, mt(r"$10^4$", 30)), (1e5, mt(r"$10^5$", 30))],
                      xlabel=da.label("seam length", 30))
        ylab = VGroup(da.label("worst chart:", 30), mt(r"$\max\sigma_1\sigma_2\,/\min\sigma_1\sigma_2$", 30))
        ylab.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([2.7, 3.05, 0], aligned_edge=UL)
        self.add(ttl, lab_s, lab_u, bar, bar_lab, axes, ylab)
        prev = None
        pts = []
        for k in (1, 2, 6, 20):
            R = RES[k]
            rgb = cmap_rgb(np.nan_to_num(R["val"], nan=0.0), -2.5, 2.5, "RdBu_r")
            rgb[np.isnan(R["val"])] = hex_rgb("#FFFFFF")
            sph.set_state(V, rgb, cam, stroke="seal")
            vis = cam.depth(V[R["E"]].mean(1)) > 0.05
            seams = VGroup()
            for (i, j) in R["E"][vis]:
                seams.add(Line(cam(V[i]), cam(V[j]), stroke_width=2.2, color=BLACK))
            seams.set_z_index(3)
            cell, cols = LAYOUT[k]
            pieces = Group()
            chart_ids = [c for c in np.unique(R["lab"]) if c >= 0]
            for (Fc, W), c in zip(layout(R["charts"], cell, cols), chart_ids):
                fm = FlatMeshView(Fc, stroke_width=0.5)
                v = R["val"][np.where(R["lab"] == c)[0]]       # submesh keeps the order of F[lab == c]
                fm.set_state(W, cmap_rgb(v, -2.5, 2.5, "RdBu_r"), center=UVC, scale=UVS, stroke="seal")
                pieces.add(fm)
                bl = dm.boundary_loop(Fc)
                pieces.add(polyline([[UVC[0] + UVS * q[0], UVC[1] + UVS * q[1], 0] for q in W[np.r_[bl, bl[:1]]]],
                                    color=BLACK, width=2.2))
            word = "chart" if k == 1 else "charts"
            head = da.label(f"{k} {word}", 34).move_to([-6.85, -1.55, 0], aligned_edge=LEFT)
            rr = R["ratio"]
            rtxt = f"{rr:.0f}" if rr > 100 else f"{rr:.2f}"
            info = VGroup(mt(rf"seam length ${R['seam']:.1f}$", 30).move_to([-6.85, -2.05, 0], aligned_edge=LEFT),
                          mt(rf"worst ratio ${rtxt}$", 30).move_to([-6.85, -2.5, 0], aligned_edge=LEFT))
            dot = Dot(P.p(R["seam"], rr), radius=0.09, color=C["normal"], stroke_color=BLACK, stroke_width=1.2)
            dlab = da.label(str(k), 30).next_to(dot, UP if k == 20 else UR, buff=0.05)
            pts.append(P.p(R["seam"], rr))
            line = polyline(pts, color=GREY_B, width=2) if len(pts) > 1 else VGroup()
            if prev is None:
                self.add(sph, seams, head, info)
                self.wait(1.2)
                self.play(FadeIn(pieces), run_time=0.8)
            else:
                old_seams, old_pieces, old_head, old_info = prev
                self.remove(old_seams, old_head, old_info)
                self.add(seams, head, info)
                self.play(FadeOut(old_pieces), run_time=0.4)
                self.wait(0.6)
                self.play(FadeIn(pieces), run_time=0.8)
            self.play(FadeIn(dot, scale=0.5), FadeIn(dlab), Create(line) if len(pts) > 1 else Wait(0.01), run_time=0.6)
            self.wait(2.0 if k != 20 else 2.6)
            prev = (seams, pieces, head, info)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)
