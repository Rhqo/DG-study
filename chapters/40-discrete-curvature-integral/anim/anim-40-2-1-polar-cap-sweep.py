"""Animation 40.2.1: growing a discrete polar cap trades boundary turning for interior angle defect (Section 40.2,
Theorem 40.2.6, Example 40.2.8, Figure 40.2.1).

Cap(theta0) = the faces of dgmesh.icosphere(3) (unit sphere, 642 vertices) whose three vertices have colatitude
<= theta0 (lib/ddg40.py icosphere_cap). theta0 grows from 0.25 to pi - 0.15 (the range of Figure 40.2.1(b)), with a
stop at theta0 = 2 pi / 3 (Example 40.2.8).
Left: the sphere (azimuth -60, elevation going from +35 to -35 so that the boundary stays in front), cap faces dark
grey, the rest light grey; vermilion dots = interior vertices, green polygon = boundary edges, filled green dots =
boundary vertices with tau_i = pi - Theta_i > 0, open dots = tau_i < 0 (front half only).
Right: the interior sum of delta_i (vermilion), the boundary sum of tau_i (green) and the total (black) traced
against theta0, with the smooth polar cap terms 2 pi (1 - cos theta0) and 2 pi cos theta0 (thin dashed, (39.4.3)).
Self-check: every cap is a disk (chi = 1, manifold); the total is 2 pi to 1e-10 for every cap; at 2 pi / 3 the cap has
481 vertices, 42 on the boundary (28 with tau > 0, 14 with tau < 0), the two sums are 8.5710 and -2.2878; every jump
of the interior sum is cancelled by an equal drop of the boundary sum.

Render: bash tools/render_anims.sh chapters/40-discrete-curvature-integral/anim/anim-40-2-1-polar-cap-sweep.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg40  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import WHITE, DashedVMobject, FadeIn, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
LEVEL = 3
V, F = dm.icosphere(LEVEL)
COL = np.arccos(np.clip(V[:, 2], -1, 1))
ANG = dm.corner_angles(V, F)
T0, T1, TSTOP = 0.25, np.pi - 0.15, 2 * np.pi / 3
# the cap changes only when theta0 passes the largest colatitude of a face: list those break points
FACE_MAX = COL[F].max(1)
BREAKS = np.unique(np.round(FACE_MAX, 12))


def cap_state(theta0):
    keep = FACE_MAX <= theta0 + 1e-12
    Fk = F[keep]
    Theta = np.zeros(len(V))
    for c in range(3):
        np.add.at(Theta, Fk[:, c], ANG[keep, c])
    bnd_edges = np.array(dm.boundary_edges(Fk)) if len(Fk) else np.zeros((0, 2), int)
    bnd = np.unique(bnd_edges) if len(bnd_edges) else np.zeros(0, int)
    used = np.unique(Fk) if len(Fk) else np.zeros(0, int)
    inner = np.setdiff1d(used, bnd)
    delta = 2 * np.pi - Theta[inner]
    tau = np.pi - Theta[bnd]
    return dict(keep=keep, edges=bnd_edges, inner=inner, bnd=bnd, tau=tau, sd=float(delta.sum()), st=float(tau.sum()))


STATES = {}
prev = None
for b in BREAKS[(BREAKS >= T0 - 1e-9) & (BREAKS <= T1)]:
    st = cap_state(b)
    Fk = F[st["keep"]]
    assert dm.euler_characteristic(Fk) == 1 and dm.is_manifold(Fk)                        # a disk
    assert abs(st["sd"] + st["st"] - 2 * np.pi) < 1e-10                                     # Theorem 40.2.6
    if prev is not None:
        assert abs((st["sd"] - prev["sd"]) + (st["st"] - prev["st"])) < 1e-10              # a jump is cancelled
    STATES[float(b)] = st
    prev = st
KEYS = np.array(sorted(STATES))
FIRST = cap_state(T0)
assert len(F[FIRST["keep"]]) > 0 and abs(FIRST["sd"] + FIRST["st"] - 2 * np.pi) < 1e-10
STOP = cap_state(TSTOP)
Vs, Fs = ddg40.icosphere_cap(LEVEL, TSTOP)
assert len(Vs) == 481 and len(STOP["bnd"]) == 42
assert (STOP["tau"] > 0).sum() == 28 and (STOP["tau"] < 0).sum() == 14
assert abs(STOP["sd"] - 8.5710) < 5e-5 and abs(STOP["st"] + 2.2878) < 5e-5


def state(theta0):
    k = np.searchsorted(KEYS, theta0 + 1e-12, side="right") - 1
    return FIRST if k < 0 else STATES[float(KEYS[k])]


# the traced curves on a fine grid (the plot)
TS = np.linspace(T0, T1, 480)
SD = np.array([state(t)["sd"] for t in TS])
ST = np.array([state(t)["st"] for t in TS])
assert np.allclose(SD + ST, 2 * np.pi, atol=1e-10)
assert abs(SD[-1] - 12.316) < 5e-4 and abs(ST[-1] + 6.032) < 5e-4          # the end values quoted in the caption

DARK = hp.rgb("#9C9C9C")
LIGHT = hp.rgb("#E6E6E6")


class Anim(hp.Scene2D):
    def construct(self):
        th = ValueTracker(T0)
        cam = hp.Cam(elev=35, azim=-60, scale=2.65, origin=(-3.55, -0.15))

        def elev(t):
            return 35 - 70 * (t - T0) / (T1 - T0)

        cols = np.tile(LIGHT, (len(F), 1))
        mv = hp.MeshView(V, F, cam, colors=cols, edge_width=0.5, edge_color="#8a8a8a", cull=True, sort=False,
                         ambient=0.62)
        last = {"key": None}

        def upd_mesh(m):
            t = th.get_value()
            cam.set(elev(t), -60)
            st = state(t)
            if last["key"] is not st:
                last["key"] = st
                m.colors = np.where(st["keep"][:, None], DARK, LIGHT)
            m.set_mesh(cam=cam)
        mv.add_updater(upd_mesh)
        mv.update()

        def overlay():
            st = state(th.get_value())
            g = VGroup()
            front = (V @ cam.view) > 0.02
            for a, b in st["edges"]:
                if front[a] and front[b]:
                    g.add(hp.seg(cam(V[a] * 1.003), cam(V[b] * 1.003), color=hp.C["third"], width=4))
            for i in st["inner"]:
                if front[i]:
                    g.add(hp.dot(cam(V[i]), color=hp.C["normal"], r=0.028))
            for i, tv in zip(st["bnd"], st["tau"]):
                if front[i]:
                    d = hp.dot(cam(V[i] * 1.003), color=hp.C["third"], r=0.055)
                    if tv < 0:
                        d.set_fill(WHITE, opacity=1).set_stroke(hp.C["third"], width=2.5)
                    g.add(d)
            return g
        ov = always_redraw(overlay)
        title = hp.text("polar cap of icosphere(3)", size=32).move_to([-3.55, 3.4, 0])

        # ---- the plot
        P = hp.Plot(1.75, -2.15, 4.5, 4.0, (0, np.pi), (-2 * np.pi - 0.6, 4 * np.pi + 0.6))
        ax = P.axes(xticks=[(0, "0"), (np.pi / 2, hp.math(r"$\pi/2$", 30)), (np.pi, hp.math(r"$\pi$", 30))],
                    yticks=[(-2 * np.pi, hp.math(r"$-2\pi$", 30)), (0, "0"), (2 * np.pi, hp.math(r"$2\pi$", 30)),
                            (4 * np.pi, hp.math(r"$4\pi$", 30))])
        xl = hp.math(r"$\theta_0$", 32).next_to(P.p(np.pi, -2 * np.pi - 0.6), np.array([1.0, 0, 0]), buff=0.15)
        zero = hp.seg(P.p(0, 0), P.p(np.pi, 0), color=hp.C["aux"], width=1.2)
        tt = np.linspace(0, np.pi, 200)
        dashed = VGroup(*[DashedVMobject(P.curve(tt, y, color=c, width=2), num_dashes=40).set_stroke(opacity=0.6)
                          for y, c in ((2 * np.pi * (1 - np.cos(tt)), hp.C["normal"]),
                                       (2 * np.pi * np.cos(tt), hp.C["third"]))])

        def traces():
            t = th.get_value()
            n = max(2, int(np.searchsorted(TS, t, side="right")))
            g = VGroup(P.curve(TS[:n], SD[:n], color=hp.C["normal"], width=5),
                       P.curve(TS[:n], ST[:n], color=hp.C["third"], width=5),
                       P.curve(TS[:n], SD[:n] + ST[:n], color=hp.C["main"], width=5))
            x = float(P.X(t))
            g.add(hp.seg([x, P.y0, 0], [x, P.y0 + P.h, 0], color=hp.C["aux"], width=1.5))
            return g
        tr = always_redraw(traces)

        # ---- readouts (top right)
        r1 = hp.math(r"interior $\Sigma\,\delta_i =$", 32, color=hp.C["normal"]).move_to([2.35, 3.45, 0])
        r2 = hp.math(r"boundary $\Sigma\,\tau_i =$", 32, color=hp.C["third"]).move_to([2.35, 2.85, 0])
        r3 = hp.text("total =", 32).move_to([2.35, 2.25, 0])
        for r in (r2, r3):
            r.align_to(r1, np.array([1.0, 0, 0]))
        n1 = always_redraw(lambda: hp.text(hp.num(state(th.get_value())["sd"]), 32, color=hp.C["normal"])
                           .next_to(r1, np.array([1.0, 0, 0]), buff=0.2))
        n2 = always_redraw(lambda: hp.text(hp.num(state(th.get_value())["st"]), 32, color=hp.C["third"])
                           .next_to(r2, np.array([1.0, 0, 0]), buff=0.2))
        n3 = always_redraw(lambda: hp.text(hp.num(state(th.get_value())["sd"] + state(th.get_value())["st"]), 32)
                           .next_to(r3, np.array([1.0, 0, 0]), buff=0.2))
        n3b = always_redraw(lambda: hp.math(r"$= 2\pi$", 32).next_to(n3, np.array([1.0, 0, 0]), buff=0.2))

        self.add(mv, ov, title, ax, xl, zero, dashed, tr, r1, r2, r3, n1, n2, n3, n3b)
        for m in (ov, tr, n1, n2, n3, n3b):
            m.update()
        self.wait(1.5)
        self.play(th.animate.set_value(TSTOP), run_time=9.0, rate_func=linear)
        note = hp.math(r"$\theta_0 = 2\pi/3$", 32).move_to([-3.55, 2.85, 0])
        self.play(FadeIn(note), run_time=0.4)
        self.wait(2.0)
        self.play(note.animate.set_opacity(0), run_time=0.4)
        self.remove(note)
        self.play(th.animate.set_value(T1), run_time=4.5, rate_func=linear)
        end = hp.text("each jump up = an equal drop", 32).move_to([-3.35, 2.85, 0])
        self.play(FadeIn(end), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=11.0)
