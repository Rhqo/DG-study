"""Animation 31.3.1: a cotan weight turns negative, the maximum principle fails, an edge flip repairs it (Section 31.3,
(31.3.5), Figure 31.3.2).

The planar fan of Figure 31.3.2(c): interior vertex x_0 = (0, 0) and boundary vertices (1.6, 0), (0.8, s), (0, 1),
(-1, 0.5), (-1, -0.5), (0, -1), (0.8, -s). Boundary values: u = 0 at the tip (1.6, 0), u = 1 elsewhere. u_0 is the
discrete harmonic value ((L u)_0 = 0, dgmesh.cotan_laplacian). s goes from 1.0 to 0.25 (the figure's value):
  - the edge e from x_0 to the tip has opposite angles alpha = beta at (0.8, +-s); w_e = (cot alpha + cot beta)/2;
  - at s = 0.8 the point (0.8, -s) crosses the circumcircle of x_0, (1.6, 0), (0.8, s) (a circle of radius 0.8 about
    (0.8, 0)): alpha + beta = 180 degrees, w_e = 0 and u_0 = 1 exactly;
  - for s < 0.8, w_e < 0 and u_0 > 1 = the largest boundary value; at s = 0.25, w_e = -1.44 and u_0 = 1.29.
Then e is flipped to the other diagonal ((0.8, 0.25), (0.8, -0.25)): all weights at x_0 are positive and u_0 = 1.
Colours: the piecewise-linear interpolant of u (RdBu_r, white = 1, as in the figure), drawn exactly by splitting each
triangle into 144 flat pieces coloured by the linear interpolant at their centroids.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-3-1-negative-weight.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, Arc, FadeOut, Polygon, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402

S0, S1 = 1.0, 0.25


def fan(s, flipped=False):
    B = np.array([[1.6, 0], [0.8, s], [0, 1], [-1, 0.5], [-1, -0.5], [0, -1], [0.8, -s]])
    V = np.c_[np.vstack([[0, 0], B]), np.zeros(8)]
    F = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
    if flipped:   # replace the edge (x_0, tip) by ((0.8, s), (0.8, -s))
        F = np.array([t for t in F.tolist() if not (0 in t and 1 in t)] + [[0, 7, 2], [7, 1, 2]])
    return V, F


U_B = np.array([0.0, 1, 1, 1, 1, 1, 1])


def solve(s, flipped=False):
    V, F = fan(s, flipped)
    L = dm.cotan_laplacian(V, F)
    u = np.r_[0.0, U_B]
    u[0] = -(L[0, 1:] @ u[1:]) / L[0, 0]
    ang = dm.corner_angles(V, F)
    return V, F, L, u, ang


for s in np.linspace(S1, S0, 31):
    V, F, L, u, ang = solve(s)
    assert (np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 2] > 0).all()     # no flipped triangle
    w = L[0, 1]
    al, be = ang[0, 2], ang[6, 1]                                  # opposite angles of edge (x_0, tip)
    assert np.isclose(w, 0.5 * (1 / np.tan(al) + 1 / np.tan(be)))
    assert np.isclose(w, np.sin(al + be) / (2 * np.sin(al) * np.sin(be)))          # (31.3.5)
    inside = np.linalg.norm(np.array([0.8, -s]) - np.array([0.8, 0.0])) < 0.8 - 1e-12   # circumcircle: centre (0.8, 0)
    assert (w < -1e-12) == inside == (u[0] > 1 + 1e-12)
    assert (L[0, 2:] > 0).all()                                   # only the edge to the tip can be negative
_, _, L1, u1, _ = solve(S1)
assert abs(L1[0, 1] - (-1.4437)) < 1e-4 and abs(u1[0] - 1.2882) < 1e-4          # Figure 31.3.2(c)
_, _, Lc, uc, _ = solve(0.8)
assert abs(Lc[0, 1]) < 1e-12 and abs(uc[0] - 1) < 1e-12
Vf, Ff, Lf, uf, _ = solve(S1, flipped=True)
assert Lf[0, 1] == 0 and (Lf[0, 2:][Lf[0, 2:] != 0] > 0).all() and abs(uf[0] - 1) < 1e-12
NSUB = 12
BARY = [(i, j) for i in range(NSUB) for j in range(NSUB - i)]


def sub_tris(V, F, u):
    """Each triangle split into NSUB^2 flat pieces: (corners, value at the centroid)."""
    out = []
    for t in F:
        P, w = V[t, :2], u[t]
        for i in range(NSUB):
            for j in range(NSUB - i):
                for tri in (((i, j), (i + 1, j), (i, j + 1)), ((i + 1, j), (i + 1, j + 1), (i, j + 1))):
                    if max(a + b for a, b in tri) > NSUB:
                        continue
                    lam = np.array([[1 - (a + b) / NSUB, a / NSUB, b / NSUB] for a, b in tri])
                    pts = lam @ P
                    out.append((pts, float(lam.mean(0) @ w)))
    return out


class Anim(hc.Scene2D):
    def construct(self):
        # the label of (0, -1) stays above the bottom 12% of the frame (paused control bar, GUIDELINES §12.8)
        SC, O = 2.5, np.array([-3.65, 0.0, 0])

        def P(x):
            x = np.asarray(x, float)
            return O + SC * np.r_[x[:2], 0.0]
        s = ValueTracker(S0)
        state = {"flipped": False}

        def picture():
            V, F, L, u, ang = solve(s.get_value(), state["flipped"])
            g = VGroup()
            for pts, val in sub_tris(V, F, u):
                c = hc.cmap_rgb([val], 0.0, 1.3, center=1.0)[0]
                g.add(Polygon(*[P(p) for p in pts], fill_color=hc.ManimColor(tuple(c)), fill_opacity=1,
                              stroke_color=hc.ManimColor(tuple(c)), stroke_width=0.6))
            E = dm.edges(F)
            for a, b in E:
                if {a, b} == {0, 1}:
                    continue
                g.add(hc.seg(P(V[a]), P(V[b]), color="#555555", width=1.6))
            if not state["flipped"]:
                w = L[0, 1]
                g.add(hc.seg(P(V[0]), P(V[1]), color=hc.C["tangent"] if w >= 0 else hc.C["normal"], width=7))
            else:
                g.add(hc.seg(P(V[2]), P(V[7]), color=hc.C["third"], width=7))
            return g
        pic = always_redraw(picture)
        self.add(pic)

        # labels of the boundary values and of x_0
        V0, _ = fan(S0)
        vals = VGroup()
        for k in range(1, 8):
            vals.add(always_redraw(lambda k=k: hc.text(f"{U_B[k - 1]:.0f}", size=30).move_to(
                P(fan(s.get_value())[0][k] * 1.0) + SC * 0.13 * np.r_[
                    fan(s.get_value())[0][k][:2] / np.linalg.norm(fan(s.get_value())[0][k][:2]), 0])))
        x0 = hc.dot(P([0, 0]), color="#FFFFFF", r=0.09)
        x0.set_stroke(hc.BLACK, width=2).set_z_index(6)
        x0l = hc.math(r"$x_0$", size=34).move_to(P([-0.2, 0.18])).set_z_index(6)
        self.add(vals, x0, x0l)

        # circumcircle of x_0, tip, (0.8, s): centre (0.8, 0) only when ... computed in general
        def circle():
            """Dashed circumcircle of x_0, the tip and (0.8, s), drawn only inside the fan's part of the frame."""
            V, F = fan(s.get_value())
            a, b, c = V[0, :2], V[1, :2], V[2, :2]
            A = 2 * np.array([b - a, c - a])
            o = np.linalg.solve(A, np.array([b @ b - a @ a, c @ c - a @ a]))
            r = np.linalg.norm(a - o)
            t = np.linspace(0, 2 * np.pi, 241)
            pts = [P(o + r * np.array([np.cos(v), np.sin(v)])) for v in t]
            g = VGroup()
            for k in range(0, 240, 2):
                p0, p1 = pts[k], pts[k + 1]
                if all(-6.9 < q[0] < 1.1 and -3.9 < q[1] < 3.0 for q in (p0, p1)):
                    g.add(hc.seg(p0, p1, color=hc.C["aux"], width=2.5))
            return g
        circ = always_redraw(circle)

        def arcs():
            V, F, L, u, ang = solve(s.get_value())
            g = VGroup()
            for k, sgn in ((2, 1), (7, -1)):
                p = V[k, :2]
                d1, d2 = V[0, :2] - p, V[1, :2] - p
                a1, a2 = np.arctan2(d1[1], d1[0]), np.arctan2(d2[1], d2[0])
                lo = a1 if (a2 - a1) % (2 * np.pi) < np.pi else a2
                span = ((a2 - a1) % (2 * np.pi)) if lo == a1 else ((a1 - a2) % (2 * np.pi))
                g.add(Arc(radius=0.42, start_angle=lo, angle=span, color=hc.C["accent"], stroke_width=5)
                      .move_arc_center_to(P(p)))
            return g
        arc = always_redraw(arcs)
        self.add(circ, arc)
        al = always_redraw(lambda: hc.math(r"$\alpha$", size=36, color=hc.C["accent"]).move_to(
            P(fan(s.get_value())[0][2]) + np.array([-0.72, 0.0, 0])))
        bl = always_redraw(lambda: hc.math(r"$\beta$", size=36, color=hc.C["accent"]).move_to(
            P(fan(s.get_value())[0][7]) + np.array([-0.72, 0.0, 0])))
        self.add(al, bl)

        # ---- right panel: live numbers
        XR = 1.45
        title = hc.text("one edge, its weight, the harmonic value", size=32).move_to([0, 3.45, 0])
        self.add(title)

        def panel():
            V, F, L, u, ang = solve(s.get_value(), state["flipped"])
            if state["flipped"]:
                rows = [("edge flipped:", hc.C["third"]), ("all weights at x₀ > 0", hc.C["third"]),
                        (f"u₀ = {u[0]:.2f}", hc.BLACK)]
            else:
                w = L[0, 1]
                sab = np.degrees(ang[0, 2] + ang[6, 1])
                col = hc.C["tangent"] if w >= 0 else hc.C["normal"]
                rows = [(f"α + β = {sab:.1f}°", hc.C["accent"]), (f"w = {w:+.2f}".replace("-", "−"), col),
                        (f"u₀ = {u[0]:.2f}", hc.C["normal"] if u[0] > 1 + 1e-9 else hc.BLACK)]
            return VGroup(*[hc.text(t, size=34, color=c).move_to([XR, 2.3 - 0.75 * k, 0], aligned_edge=LEFT)
                            for k, (t, c) in enumerate(rows)])
        pan = always_redraw(panel)
        self.add(pan)

        def status(lines, color):
            return VGroup(*[hc.text(t, size=30, color=color).move_to([XR, -1.15 - 0.55 * k, 0], aligned_edge=LEFT)
                            for k, t in enumerate(lines)])
        st = status(["Delaunay: w > 0", "u₀ < 1 (an average)"], hc.C["tangent"])
        self.add(st)
        self.wait(2.5)
        # move (0.8, +-s) toward the axis; pause on the circle
        self.play(s.animate.set_value(0.8), run_time=3.0, rate_func=smooth)
        st2 = status(["on the circle:", "α + β = 180°", "w = 0, u₀ = 1"], hc.BLACK)
        self.play(*hc.swap(st, st2), run_time=0.6)
        self.wait(1.6)
        self.play(s.animate.set_value(S1), run_time=4.5, rate_func=smooth)
        st3 = status(["inside the circle: w < 0", "u₀ > all boundary values", "maximum principle fails"],
                     hc.C["normal"])
        self.play(*hc.swap(st2, st3), run_time=0.6)
        self.wait(2.8)
        # flip the edge
        st4 = status(["Delaunay flip of the edge", "u₀ = 1 again"], hc.C["third"])
        self.play(FadeOut(circ), FadeOut(arc), FadeOut(al), FadeOut(bl), run_time=0.6)
        state["flipped"] = True
        self.play(*hc.swap(st3, st4), run_time=1.0)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=14.5)
