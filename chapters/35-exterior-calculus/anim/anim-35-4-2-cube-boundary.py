"""Animation 35.4.2: folding the boundary of a cube -- every edge gets two opposite arrows, so the boundary of the
boundary is empty (Section 35.4, Proposition 35.4.3, the Derivation of d o d = 0, Figure 35.4.3).

Q = [0, 1]^3. Its six faces carry the Stokes orientation (outward normal first, Proposition 35.4.3), and each face's
boundary carries the Stokes orientation of the face: a loop of four arrows, counterclockwise seen from outside
(drawn inside the face; colours as in Figure 35.4.3: x-faces blue, y-faces green, z-faces orange).
Part 1: the cube is cut open into a net in the plane z = 1 (top face in the middle, the four side faces around it,
the bottom face beyond the y = 0 face), every face seen from outside. The 5 edges inside the net already carry two
opposite arrows (highlighted).
Part 2: the net folds up into the cube (each side face turns 90 degrees about its edge with the top face, the bottom
face turns 90 degrees more about its edge with the y = 0 face). The 7 cut edges meet again, and on each of them the
two arrows run in opposite directions (the three visible ones are highlighted).
Part 3: the camera turns 180 degrees about the vertical axis to show the other side.
Self-check: the folded faces are the six faces of Q with outward normals; every loop is counterclockwise about its
outward normal; each of the 12 edges of Q is traversed exactly twice, once in each direction (5 inside the net,
7 glued by folding), so the sum of the 24 directed edges is 0.

Render: ``bash tools/render_anims.sh chapters/35-exterior-calculus/anim/anim-35-4-2-cube-boundary.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, Polygon, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402


def rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


# net squares in the plane z = 1, corners counterclockwise seen from +z (= from outside after folding)
SQ = lambda x0, y0: np.array([[x0, y0, 1], [x0 + 1, y0, 1], [x0 + 1, y0 + 1, 1], [x0, y0 + 1, 1]], float)  # noqa: E731
NET = {"T": SQ(0, 0), "E": SQ(1, 0), "W": SQ(-1, 0), "N": SQ(0, 1), "S": SQ(0, -1), "B": SQ(0, -2)}
FCOL = {"T": hp.C["accent"], "B": hp.C["accent"], "E": hp.C["tangent"], "W": hp.C["tangent"],
        "N": hp.C["third"], "S": hp.C["third"]}
# hinge: (point on the axis, axis direction, sign) such that +90 degrees folds the face downward (to z < 1)
HINGE = {"E": ([1, 0, 1], [0, 1, 0], +1), "W": ([0, 0, 1], [0, 1, 0], -1),
         "N": ([0, 1, 1], [1, 0, 0], -1), "S": ([0, 0, 1], [1, 0, 0], +1)}


def place(name, f):
    """corners of face `name` after folding by f in [0, 1] (0: net, 1: cube)."""
    P = NET[name].copy()
    phi = f * np.pi / 2
    if name == "T":
        return P
    if name == "B":                                    # first about its edge with S (y = -1 in the net), then with S
        R = rot([1, 0, 0], phi)
        P = (P - [0, -1, 1]) @ R.T + [0, -1, 1]
        name = "S"
    h, d, s = HINGE[name]
    R = rot(d, s * phi)
    return (np.asarray(P) - h) @ R.T + h


def normal(P):
    n = np.cross(P[1] - P[0], P[3] - P[0])
    return n / np.linalg.norm(n)



def key(a, b):
    return frozenset((tuple(np.round(a, 6)), tuple(np.round(b, 6))))


def net_key(nm, k):
    P = NET[nm]
    return key(P[k], P[(k + 1) % 4])


def world_key(nm, k):
    P = place(nm, 1.0)
    return key(P[k], P[(k + 1) % 4])


CUBE = {"T": (2, 1), "B": (2, 0), "E": (0, 1), "W": (0, 0), "N": (1, 1), "S": (1, 0)}   # axis, value
for nm, (ax, val) in CUBE.items():
    P = place(nm, 1.0)
    assert np.allclose(P[:, ax], val)                                     # it is the face x_ax = val of Q
    out = np.zeros(3)
    out[ax] = 1 if val == 1 else -1
    assert np.allclose(normal(P), out)                                    # outward, and the loop is ccw about it
    assert np.all((P >= -1e-12) & (P <= 1 + 1e-12)) and np.allclose(normal(place(nm, 0.0)), [0, 0, 1])
directed = [(tuple(np.round(place(nm, 1.0)[k], 6)), tuple(np.round(place(nm, 1.0)[(k + 1) % 4], 6)))
            for nm in NET for k in range(4)]
und = {}
for a_, b_ in directed:
    und.setdefault(frozenset((a_, b_)), []).append((a_, b_))
assert len(directed) == 24 and len(und) == 12
assert all(len(v) == 2 and v[0] == v[1][::-1] for v in und.values())          # every edge twice, opposite
NETCOUNT = {}
for nm in NET:
    for k in range(4):
        NETCOUNT[net_key(nm, k)] = NETCOUNT.get(net_key(nm, k), 0) + 1
INNER_NET = {kk for kk, c in NETCOUNT.items() if c == 2}
assert len(INNER_NET) == 5 and sum(c == 1 for c in NETCOUNT.values()) == 14           # 5 inside, 7 cut pairs
for nm in NET:                                   # in the net the two arrows of an inner edge are already opposite
    for k in range(4):
        if net_key(nm, k) in INNER_NET:
            P = NET[nm]
            m, j = [(m, j) for m in NET for j in range(4) if (m, j) != (nm, k) and net_key(m, j) == net_key(nm, k)][0]
            Q = NET[m]
            assert np.allclose(P[k], Q[(j + 1) % 4]) and np.allclose(P[(k + 1) % 4], Q[j])

CENTER = np.array([0.5, 0.5, 0.5])
ELEV, AZ0 = 32.0, -58.0
SCALE0, SCALE1 = 1.3, 2.3                       # zoom in while folding (the net is large, the cube small)
ORIGIN2 = (-2.3, 0.35)
_cam0 = hp.Cam(ELEV, AZ0, SCALE1, ORIGIN2)
FRONT0 = {nm for nm in NET if normal(place(nm, 1.0)) @ _cam0.view > 0}
assert FRONT0 == {"T", "E", "S"}
VISIBLE = {world_key(nm, k) for nm in FRONT0 for k in range(4)
           if sum(world_key(m, j) == world_key(nm, k) for m in FRONT0 for j in range(4)) == 2}
_inner_world = {world_key(m, j) for m in NET for j in range(4) if net_key(m, j) in INNER_NET}
assert len(VISIBLE) == 3 and len(VISIBLE - _inner_world) == 1          # T-E, T-S were inside the net; E-S is glued


def edge_arrows(P):
    """the four loop arrows of a face: (start, end), inset toward the face centre."""
    c = P.mean(0)
    out = []
    for k in range(4):
        a, b = P[k], P[(k + 1) % 4]
        a2, b2 = a + 0.17 * (c - a), b + 0.17 * (c - b)
        out.append((a2 + 0.18 * (b2 - a2), a2 + 0.82 * (b2 - a2)))
    return out


for _f in (0.0, 0.5, 1.0):                          # layout: everything stays inside the left part of the frame
    for _az in (AZ0, AZ0 + 90, AZ0 + 180):
        _cam = hp.Cam(ELEV, _az, SCALE0 + (SCALE1 - SCALE0) * _f, ORIGIN2)
        _S = _cam(np.vstack([place(nm, _f) for nm in NET]) - CENTER)
        assert _S[:, 0].min() > -6.8 and _S[:, 0].max() < 1.7 and _S[:, 1].min() > -2.9 and _S[:, 1].max() < 3.6, \
            (_f, _az, _S[:, 0].min(), _S[:, 0].max(), _S[:, 1].min(), _S[:, 1].max())


class Anim(hp.Scene2D):
    def construct(self):
        fold = ValueTracker(0.0)
        az = ValueTracker(AZ0)
        glow = ValueTracker(0.0)                         # highlight of the 5 net edges
        glow2 = ValueTracker(0.0)                        # highlight of the 3 visible cube edges

        def scene():
            f = fold.get_value()
            cam = hp.Cam(ELEV, az.get_value(), SCALE0 + (SCALE1 - SCALE0) * f, ORIGIN2)
            items = []
            for nm in NET:
                P = place(nm, f)
                front = normal(P) @ cam.view > 0
                g = VGroup(Polygon(*cam(P - CENTER), fill_color="#E4E4E4" if front else "#F4F4F4",
                                   fill_opacity=0.95 if front else 0.6, stroke_color="#8A8A8A", stroke_width=2))
                if front:
                    for k, (p0, p1) in enumerate(edge_arrows(P)):
                        hi = glow.get_value() if net_key(nm, k) in INNER_NET else 0.0
                        if world_key(nm, k) in VISIBLE:
                            hi = max(hi, glow2.get_value())
                        g.add(hp.arrow(cam(p0 - CENTER), cam(p1 - CENTER), color=FCOL[nm], width=4.5 + 5 * hi,
                                       tip=0.2 + 0.07 * hi))
                items.append((cam.depth(P.mean(0) - CENTER), g))
            items.sort(key=lambda t: t[0])
            return VGroup(*[g for _, g in items])
        self.add(always_redraw(scene))
        XL = 2.0
        t1 = hp.text("faces of ∂Q and their", size=30).move_to([XL, 3.2, 0], aligned_edge=np.array([-1, 0, 0]))
        t2 = hp.text("boundary loops", size=30).move_to([XL, 2.7, 0], aligned_edge=np.array([-1, 0, 0]))
        self.add(t1, t2)
        self.wait(2.0)
        s1 = hp.text("5 edges in the net:", size=30).move_to([XL, 1.7, 0], aligned_edge=np.array([-1, 0, 0]))
        s2 = hp.text("two opposite arrows", size=30, color=hp.C["normal"]).move_to([XL, 1.2, 0],
                                                                                aligned_edge=np.array([-1, 0, 0]))
        self.play(glow.animate.set_value(1.0), FadeIn(s1), FadeIn(s2), run_time=1.0)
        self.wait(2.0)
        self.play(glow.animate.set_value(0.0), run_time=0.5)
        s3 = hp.text("fold: 7 edges glued", size=30).move_to([XL, 0.3, 0], aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(s3), run_time=0.5)
        self.play(fold.animate.set_value(1.0), run_time=6.0, rate_func=smooth)
        s4 = hp.text("opposite again", size=30, color=hp.C["normal"]).move_to([XL, -0.2, 0],
                                                                                    aligned_edge=np.array([-1, 0, 0]))
        self.play(glow2.animate.set_value(1.0), FadeIn(s4), run_time=1.0)
        self.wait(1.5)
        self.play(glow2.animate.set_value(0.0), run_time=0.4)
        self.play(az.animate.set_value(AZ0 + 180), run_time=6.0, rate_func=smooth)
        fin = VGroup(hp.math(r"$\partial(\partial Q) = 0$:", size=32),
                     hp.text("all 12 edges: twice,", size=30),
                     hp.text("in opposite directions", size=30)) \
            .arrange(np.array([0, -1.0, 0]), buff=0.18, aligned_edge=np.array([-1, 0, 0])) \
            .move_to([XL, -1.6, 0], aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.5)
