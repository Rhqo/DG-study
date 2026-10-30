"""Animation 41.5.3: under mean curvature flow a torus mesh shrinks to a circle while its triangles become slivers
(Section 41.5, Remark 41.5.18, Example 41.5.5).

Data (as Remark 41.5.18): dgmesh.torus_mesh(2, 0.8, 24, 48) (R = 2, r = 0.8, faces outward). Explicit discrete mean
curvature flow x <- x + h (L x)_i / A_i (ddg41.mcf_velocity, A_i the mixed area) with h = 0.2 * 2 / lambda_max,
lambda_max recomputed every 50 steps (ddg41.largest_eigenvalue), until the tube radius is below 0.08 (t = 0.333).
Tube radius r = mean of the distances of the vertices from the core circle, core radius = mean distance from the
z axis (the same measures as the text).
Left: the mesh (elevation 35, azimuth -60, painter's algorithm with back faces culled), fixed zoom.
Right top: r^2 against t; orange = the mesh, black dashed = the cylinder estimate r_0^2 - 2t (Example 41.5.5).
Right bottom: the face whose smallest angle is smallest at the end, drawn in its own plane, rotated so that its longest
edge is horizontal and rescaled to a fixed length; readout of the smallest angle of the whole mesh.
The video time is linear in t (13 s for t from 0 to 0.333); frames between two steps interpolate linearly.
Self-check (Remark 41.5.18): r^2 = 0.487, 0.331, 0.182 at t = 0.083, 0.165, 0.243 (cylinder estimate 0.474, 0.310,
0.155); at t = 0.333 r = 0.079 and the core radius 1.83; the smallest angle goes from 29.8 to 2.9 degrees; the area
decreases at every step.

Render: bash tools/render_anims.sh chapters/41-discrete-curvature-variational/anim/anim-41-5-3-torus-collapse.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg41  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import DashedVMobject, FadeIn, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
V, F = dm.torus_mesh(2.0, 0.8, 24, 48)


def measures(V):
    rho = np.hypot(V[:, 0], V[:, 1])
    Rc = rho.mean()
    return Rc, np.hypot(rho - Rc, V[:, 2]).mean()


T, X = [0.0], [V.copy()]
h = 0.2 * 2 / ddg41.largest_eigenvalue(V, F)
k = 0
while True:                                   # the loop of verify/v-41-5-surface-flows.py (Remark 41.5.18)
    if k % 50 == 0:
        if measures(X[-1])[1] < 0.08:
            break
        if k > 0:
            h = 0.2 * 2 / ddg41.largest_eigenvalue(X[-1], F, iters=60)
    X.append(X[-1] + h * ddg41.mcf_velocity(X[-1], F))
    T.append(T[-1] + h)
    k += 1
T, X = np.array(T), np.array(X)
REC = np.array([(T[j], *measures(X[j])) for j in range(0, len(X), 50)])      # the records of the text
assert len(X) - 1 == 50 * (len(REC) - 1)
assert np.allclose(REC[1:4, 0], [0.083, 0.165, 0.243], atol=6e-4)
assert np.allclose(REC[1:4, 2] ** 2, [0.487, 0.331, 0.182], atol=6e-4)
assert np.allclose(0.64 - 2 * REC[1:4, 0], [0.474, 0.310, 0.155], atol=1.1e-3)
assert np.allclose([REC[-1, 0], REC[-1, 2], REC[-1, 1]], [0.333, 0.079, 1.83], atol=6e-3)
ANG = np.array([np.degrees(dm.corner_angles(Vk, F).min()) for Vk in X])
assert abs(ANG[0] - 29.8) < 0.051 and abs(ANG[-1] - 2.9) < 0.051
AREA = np.array([dm.face_areas(Vk, F).sum() for Vk in X])
assert np.all(np.diff(AREA) < 0)
R2 = np.array([measures(Vk)[1] ** 2 for Vk in X])
CORE = np.array([measures(Vk)[0] for Vk in X])
FSTAR = int(np.argmin(dm.corner_angles(X[-1], F).min(1)))
T_END = float(T[-1])


def at(t):
    j = int(np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2))
    w = (t - T[j]) / (T[j + 1] - T[j])
    w = float(np.clip(w, 0, 1))
    return j, w


def lerp(A, t):
    j, w = at(t)
    return (1 - w) * A[j] + w * A[j + 1]


def flat_triangle(P):
    """the triangle P (3, 3) in its own plane: longest edge from (0, 0) to (1, 0), third vertex above"""
    L = [np.linalg.norm(P[(a + 1) % 3] - P[a]) for a in range(3)]
    a = int(np.argmax(L))
    p, q, r = P[a], P[(a + 1) % 3], P[(a + 2) % 3]
    e = (q - p) / L[a]
    x = (r - p) @ e
    y = np.linalg.norm((r - p) - x * e)
    return np.array([[0, 0], [1, 0], [x / L[a], y / L[a]]])


class Anim(hp.Scene2D):
    def construct(self):
        tt = ValueTracker(0.0)
        cam = hp.Cam(elev=35, azim=-60, scale=1.15, origin=(-3.4, -0.95))
        mv = hp.MeshView(X[0], F, cam, edge_width=0.35, edge_color="#555555", cull=True, sort=True, ambient=0.55)
        mv.add_updater(lambda m: m.set_mesh(lerp(X, tt.get_value())))
        mv.update()
        title = hp.text("torus under mean curvature flow", 30).move_to([-3.4, 3.45, 0])
        rr = always_redraw(lambda: VGroup(
            hp.text(f"tube radius {np.sqrt(lerp(R2, tt.get_value())):.3f}", 30).move_to([-3.4, 2.9, 0]),
            hp.text(f"core radius {lerp(CORE, tt.get_value()):.2f}", 30).move_to([-3.4, 2.4, 0])))

        # ---- r^2 against t
        P = hp.Plot(1.75, 0.45, 4.6, 2.3, (0, 0.35), (0, 0.7))
        ax = P.axes(xticks=[(0, "0"), (0.1, "0.1"), (0.2, "0.2"), (0.3, "0.3")],
                    yticks=[(0, "0"), (0.32, "0.32"), (0.64, "0.64")])
        yl = hp.math(r"$r^2$", 32).next_to(P.p(0, 0.7), np.array([0, 1.0, 0]), buff=0.12)
        xl = hp.math(r"$t$", 32).next_to(P.p(0.35, 0), np.array([1.0, 0, 0]), buff=0.15)
        ts = np.linspace(0, 0.32, 50)
        cyl = DashedVMobject(P.curve(ts, 0.64 - 2 * ts, color=hp.C["main"], width=2.5), num_dashes=30)
        leg1 = hp.math(r"$r_0^2 - 2t$", 30).move_to(P.p(0.27, 0.32))
        tr = always_redraw(lambda: VGroup(
            P.curve(np.r_[T[:at(tt.get_value())[0] + 1], tt.get_value()],
                    np.r_[R2[:at(tt.get_value())[0] + 1], lerp(R2, tt.get_value())], color=hp.C["accent"], width=5),
            hp.dot(P.p(tt.get_value(), lerp(R2, tt.get_value())), color=hp.C["accent"], r=0.07)))
        tl = hp.math(r"$t =$", 32).move_to([2.6, 3.4, 0])
        tn = always_redraw(lambda: hp.text(f"{tt.get_value():.3f}", 32).next_to(tl, np.array([1.0, 0, 0]), buff=0.15))

        # ---- one triangle, rescaled
        S0 = np.array([2.5, -2.7, 0.0])
        W = 3.0

        def tri():
            j, w = at(tt.get_value())
            Pt = (1 - w) * X[j][F[FSTAR]] + w * X[j + 1][F[FSTAR]]
            Q = flat_triangle(Pt) * W
            pts = np.c_[Q, np.zeros(3)] + S0
            return hp.filled(pts, hp.C["surface"], opacity=1.0, stroke_width=3, stroke_color=hp.C["main"])
        tg = always_redraw(tri)
        al = always_redraw(lambda: hp.text(f"smallest angle {lerp(ANG, tt.get_value()):.1f}°", 30)
                           .move_to([4.0, -0.5, 0]))
        cap = hp.text("one triangle, rescaled", 30, color=hp.C["aux"]).move_to([4.0, -1.0, 0])

        self.add(mv, title, rr, ax, yl, xl, cyl, leg1, tr, tl, tn, tg, al, cap)
        for m in (rr, tr, tn, tg, al):
            m.update()
        self.wait(1.5)
        self.play(tt.animate.set_value(T_END), run_time=13.0, rate_func=linear)
        e1 = hp.text("a circle, and slivers", 30, color=hp.C["normal"]).move_to([-3.4, 1.85, 0])
        self.play(FadeIn(e1), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)
