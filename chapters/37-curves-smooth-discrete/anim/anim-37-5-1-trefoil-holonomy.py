"""Animation 37.5.1: the discrete Bishop frame carried once around the trefoil does not come back (Section 37.5,
Definition 37.5.10, Definition 37.5.12, Proposition 37.5.14, Example 37.5.15; smooth version Proposition 37.3.13).

Curve: the trefoil of Example 37.3.14, gamma(s) = ((2 + cos 3s) cos 2s, (2 + cos 3s) sin 2s, sin 3s), sampled at
s_j = 2 pi j / 60 (60 vertices, all curvature angles > 0). Frame: t_0 on the first edge and u_0 orthogonal to it
(dgdec.parallel_transport_frames' default), moved along each edge and turned at every vertex i by the minimal rotation
P_i = rotation_between(t_{i-1}, t_i) (about b_i by theta_i), drawn as a smooth rotation. After 60 vertices the
transported vector u_60 = P_0 u_59 sits on the first edge again, turned about t_0 by the holonomy
Delta = 3.1307 rad (dgdec.bishop_holonomy; Example 37.5.15), = -sum psi_i (mod 2 pi). Smooth value 3.0270.
Self-check: the drawn frames equal dgdec.parallel_transport_frames; |u| = 1, u . t = 0 throughout; the final angle
equals bishop_holonomy and -sum psi (mod 2 pi), and rounds to 3.1307.

Render: bash tools/render_anims.sh chapters/37-curves-smooth-discrete/anim/anim-37-5-1-trefoil-holonomy.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _helpers as hp  # noqa: E402
import dganim as da  # noqa: E402
import dgdec as dd  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
N = 60
S = 2 * np.pi * np.arange(N) / N
X = np.stack([(2 + np.cos(3 * S)) * np.cos(2 * S), (2 + np.cos(3 * S)) * np.sin(2 * S), np.sin(3 * S)], 1)
assert dd.turning_angles(X, True).min() > 0
T, U, W = dd.parallel_transport_frames(X, closed=True)
PSI = dd.torsion_angles(X, True)
HOL = dd.bishop_holonomy(X)
assert round(HOL, 4) == 3.1307
assert abs(np.angle(np.exp(1j * (HOL + PSI.sum())))) < 1e-9
U_END = dd.rotation_between(T[-1], T[0]) @ U[-1]
assert np.isclose(np.arctan2(np.dot(np.cross(U[0], U_END), T[0]), np.dot(U[0], U_END)), HOL)


def rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def turn(i_from, i_to):
    """axis and angle of the minimal rotation taking T[i_from] to T[i_to]."""
    a, b = T[i_from], T[i_to]
    ax = np.cross(a, b)
    return ax / np.linalg.norm(ax), float(np.arctan2(np.linalg.norm(ax), a @ b))


# frames along the way: (position, t, u, running -sum psi)
FR = []
cum = 0.0
for i in range(N):
    nxt = (i + 1) % N
    p0, p1 = X[i], X[nxt]
    for s in np.linspace(0.15, 0.85, 4):                       # slide along edge i
        FR.append((p0 + s * (p1 - p0), T[i], U[i], cum))
    cum -= PSI[i]                                              # edge i done: psi_i (b_i -> b_{i+1}) joins the sum,
    ax, ang = turn(i, nxt)                                     # so "vertex k" shows -(psi_0 + ... + psi_{k-1})
    u = U[i] if i < N - 1 else U[-1]
    for s in np.linspace(0.0, 1.0, 4)[1:]:                     # turn at vertex i + 1
        R = rot(ax, s * ang)
        FR.append((p1, R @ T[i], R @ u, cum))
    uu = U[nxt] if nxt > 0 else U_END
    assert np.allclose(rot(ax, ang) @ u, uu, atol=1e-9) and np.allclose(rot(ax, ang) @ T[i], T[nxt], atol=1e-12)
for p, t, u, c in FR:
    assert abs(np.linalg.norm(u) - 1) < 1e-9 and abs(u @ t) < 1e-9
CUM_END = cum
assert FR[-1][3] == CUM_END                                    # the last "vertex 60" readout is the full sum
assert np.isclose(np.angle(np.exp(1j * CUM_END)), HOL)

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=38, azim=-70, scale=1.08, origin=(-3.15, -0.4))
FL = 1.15                                                    # frame vector length (world units)


def curve_mobs():
    out = []
    for i in range(N):
        a, b = X[i], X[(i + 1) % N]
        d = CAM.depth(0.5 * (a + b))
        halo = hp.seg(CAM(a), CAM(b), color="#FFFFFF", width=12)
        line = hp.seg(CAM(a), CAM(b), color=hp.BLACK, width=3.5)
        out.append((d, VGroup(halo, line)))
    out.sort(key=lambda x: x[0])
    return [m for _, m in out] + [hp.dot(CAM(X[0]), r=0.06)]


CURVE = curve_mobs()


def frame_mobs(p, t, u, ghost=True):
    out = []
    if ghost:
        out.append(hp.DashedVMobject(hp.arrow(CAM(FR[0][0]), CAM(FR[0][0] + FL * U[0]), color=hp.C["accent"],
                                              width=4, tip=0.2), num_dashes=7))
    out.append(hp.arrow(CAM(p), CAM(p + FL * t), color=hp.C["tangent"], width=6, tip=0.22))
    out.append(hp.arrow(CAM(p), CAM(p + FL * u), color=hp.C["accent"], width=6, tip=0.22))
    out.append(hp.dot(CAM(p), r=0.07))
    return out


RX = 1.45


def readout(k_edge, cum, extra=None):
    out = [hp.text(f"vertex {k_edge} of {N}", 32).move_to([RX, 1.4, 0], aligned_edge=hp.LEFT),
           hp.text(f"−Σψ so far = {hp.num(cum, 3)}", 32).move_to([RX, 0.75, 0], aligned_edge=hp.LEFT)]
    if extra:
        out += extra
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("parallel transport around the trefoil does not close up", 32).move_to([0, 3.55, 0])
        key = VGroup(hp.mt(r"blue $\mathbf{t}_i$, orange $\mathbf{u}_i$", 30),
                     hp.text("dashed: u\u2080 (the start)", 30, color=hp.C["accent"]),
                     hp.mt(r"at vertex $i$: turn $\mathbf{t}_{i-1} \to \mathbf{t}_i$", 30))
        key.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.12).move_to([RX, 2.55, 0], aligned_edge=hp.LEFT)
        self.add(title, *CURVE)
        self.add(key)
        g = VGroup()
        self.add(g)
        St = hp.setter
        p, t, u, c = FR[0]
        St(g, frame_mobs(p, t, u) + readout(0, 0.0))(1)
        self.mark("start")
        self.wait(2.0)
        ev, tt = [], 0.0
        per = 7                                                  # frames per edge (4 slide + 3 turn)
        for k, (p, t, u, c) in enumerate(FR):
            ev.append((tt, tt, (lambda st: (lambda q: St(g, st)(1)))(frame_mobs(p, t, u) + readout(k // per + (1 if k % per >= 4 else 0), c))))
            tt += 1 / 30
        self.play(hp.timeline(g, ev, tt + 0.5))
        self.mark("loop done")
        # the comparison on the first edge
        p0 = FR[0][0]
        ang = np.linspace(0, HOL, 40)
        arc = [p0 + 0.55 * FL * (np.cos(a) * U[0] + np.sin(a) * np.cross(T[0], U[0])) for a in ang]
        extra = [hp.text(f"holonomy \u0394 = {HOL:.4f}", 32, color=hp.C["normal"]),
                 hp.text(f"= {np.degrees(HOL):.1f}\u00b0 = \u2212\u03a3\u03c8 (mod 2\u03c0)", 30, color=hp.C["normal"]),
                 hp.text("smooth trefoil: 3.0270", 30, color=hp.HEX["aux"])]
        for i, m in enumerate(extra):
            m.move_to([RX, -0.35 - 0.6 * i, 0], aligned_edge=hp.LEFT)
        final = frame_mobs(p0, T[0], U_END) + [hp.polyline([CAM(q) for q in arc], color=hp.C["normal"], width=4)]
        ev = [(0.0, 0.0, St(g, final + readout(N, CUM_END, extra))), (0.0, 0.8, hp.fade(extra))]
        self.play(hp.timeline(g, ev, 1.2))
        self.mark("holonomy")
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=20.5)
