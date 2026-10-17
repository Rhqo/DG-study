"""Animation 31.2.2: cutting a vertex star along one edge and unfolding it into the plane (Section 31.2, (31.2.2),
Figure 31.2.2).

The three stars of Figure 31.2.2: apex at the origin, neighbours p_k = (cos t_k, sin t_k, z_k), t_k = 2 pi k / 6, with
(a) z_k = -0.55 (cone), (b) z_k = 0 (flat), (c) z_k = 0.32 (-1)^k (saddle). The star is cut along the edge to p_0 and
unfolded as a chain of rigid triangles hinged along the edges to p_1, ..., p_5: every hinge angle goes linearly to
zero, and the first triangle is turned (along a single rotation) into the horizontal plane. Triangles keep their
shape at every moment (an isometric motion), so the final layout is the planar development of Figure 31.2.2
(bottom row). The two black rays are the two copies of the cut edge. The angle between them is
delta = 2 pi - (sum of the angles at the apex): a gap for the cone (+0.84), nothing for the flat star (0), an overlap
for the saddle (-0.93). The camera rises from 22 to 90 degrees elevation while the star unfolds.
Self-check: the chain reproduces the 3D star at the start and the planar development at the end; edge lengths are
preserved at intermediate times; delta matches dgmesh.corner_angles.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-2-2-unfold-defect.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (Create, FadeIn, Polygon, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-12:
        return np.array([0, 0, 1.0]), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax, ang


def frame(p, q):
    e1 = p / np.linalg.norm(p)
    e2 = q - (q @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


class Star:
    """A vertex star cut along the edge to p_0 and unfolded as a hinged chain."""

    def __init__(self, z):
        t = 2 * np.pi * np.arange(6) / 6
        self.P = np.stack([np.cos(t), np.sin(t), z], 1)
        V = np.vstack([[0, 0, 0], self.P])
        F = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
        self.theta = dm.corner_angles(V, F)[:, 0]
        self.delta = 2 * np.pi - self.theta.sum()
        r = np.linalg.norm(self.P, axis=1)
        c = np.concatenate([[0], np.cumsum(self.theta)])
        self.c = c
        # flat development Q_0..Q_6 (Q_6 is the second copy of p_0)
        self.Q = np.stack([r[np.arange(7) % 6] * np.cos(c), r[np.arange(7) % 6] * np.sin(c), np.zeros(7)], 1)
        P7 = np.vstack([self.P, self.P[:1]])
        self.G1 = frame(P7[0], P7[1]) @ frame(self.Q[0], self.Q[1]).T
        self.g_axis, self.g_ang = axis_angle(self.G1)
        # hinge angles beta_k (k = 1..5) so that the chain reproduces the 3D star
        self.beta = np.zeros(6)
        R = self.G1.copy()
        for k in range(1, 6):
            x = R @ self.Q[k + 1]
            pk = P7[k] / np.linalg.norm(P7[k])
            xp, yp = x - (x @ pk) * pk, P7[k + 1] - (P7[k + 1] @ pk) * pk
            self.beta[k] = np.arctan2(np.cross(xp, yp) @ pk, xp @ yp)
            R = R @ rot(self.Q[k], self.beta[k])

    def tris(self, lam):
        """The six triangles (6, 3, 3) at bending parameter lam (1 = the 3D star, 0 = flat development)."""
        R = rot(self.g_axis, lam * self.g_ang)
        out = []
        for k in range(6):
            if k >= 1:
                R = R @ rot(self.Q[k], lam * self.beta[k])
            out.append([np.zeros(3), R @ self.Q[k], R @ self.Q[k + 1]])
        return np.array(out)


STARS = [("(a) cone", Star(np.full(6, -0.55))), ("(b) flat", Star(np.zeros(6))),
         ("(c) saddle", Star(0.32 * (-1.0) ** np.arange(6)))]
for name, S in STARS:
    T1, T0 = S.tris(1.0), S.tris(0.0)
    P7 = np.vstack([S.P, S.P[:1]])
    assert np.allclose(T1[:, 1], P7[:6], atol=1e-12) and np.allclose(T1[:, 2], P7[1:], atol=1e-12)   # 3D star
    assert np.allclose(T0[:, 1], S.Q[:6], atol=1e-12) and np.allclose(T0[:, 2], S.Q[1:], atol=1e-12)  # development
    for lam in (0.25, 0.5, 0.8):                                                       # rigid triangles
        T = S.tris(lam)
        for k in range(6):
            assert np.isclose(np.linalg.norm(T[k, 1] - T[k, 2]), np.linalg.norm(P7[k] - P7[k + 1]))
            assert np.isclose(np.linalg.norm(T[k, 1]), np.linalg.norm(P7[k]))
        assert np.allclose(T[1:, 1], T[:-1, 2])                                         # still hinged
    assert abs((2 * np.pi - S.c[-1]) - S.delta) < 1e-12
assert round(STARS[0][1].delta, 2) == 0.84 and abs(STARS[1][1].delta) < 1e-12 and round(STARS[2][1].delta, 2) == -0.93
XS = [-4.65, 0.0, 4.65]


class Anim(hc.Scene2D):
    def construct(self):
        tt = ValueTracker(0.0)
        cams = [hc.Cam(elev=22, azim=-90, scale=1.62, origin=(x, 0.5)) for x in XS]
        views = []
        for (name, S), cam, x in zip(STARS, cams, XS):
            Fi = np.arange(18).reshape(6, 3)
            mv = hc.MeshView(S.tris(1.0).reshape(-1, 3), Fi, cam, edge_width=1.4, opacity=0.85, ambient=0.55)

            def upd(m, S=S, cam=cam):
                t = tt.get_value()
                cam.set(22 + (90 - 22) * smooth(min(1.0, t * 1.15)), -90)
                m.set_mesh(S.tris(1 - t).reshape(-1, 3), cam=cam)
            mv.add_updater(upd)
            views.append(mv)
            self.add(mv)
            self.add(hc.text(name, size=36).move_to([x, 3.35, 0]))

        def cut_rays(S, cam):
            T = S.tris(1 - tt.get_value())
            return VGroup(hc.seg(cam(T[0, 0]), cam(T[0, 1]), width=6), hc.seg(cam(T[5, 0]), cam(T[5, 2]), width=6))
        rays = [always_redraw(lambda S=S, cam=cam: cut_rays(S, cam).set_z_index(4)) for (_, S), cam in zip(STARS, cams)]
        apex = [always_redraw(lambda cam=cam: hc.dot(cam([0, 0, 0]), r=0.07).set_z_index(5)) for cam in cams]
        self.add(*apex)
        self.wait(1.5)
        note = hc.text("cut along one edge", size=32).move_to([0, -2.75, 0])
        self.play(*[Create(r) for r in rays], FadeIn(note), run_time=1.2)
        self.wait(1.3)
        note2 = hc.text("unfold: every triangle keeps its shape", size=32).move_to([0, -2.75, 0])
        self.play(*hc.swap(note, note2), run_time=0.6)
        self.play(tt.animate.set_value(1.0), run_time=10.0, rate_func=smooth)
        self.wait(0.8)

        # wedges and delta
        res = []
        for (name, S), cam, x in zip(STARS, cams, XS):
            rr = 1.2
            if S.delta > 1e-9:
                a = np.linspace(S.c[-1], 2 * np.pi, 30)
                col, txt = hc.C["accent"], f"gap: δ = +{S.delta:.2f}"
            elif S.delta < -1e-9:
                a = np.linspace(2 * np.pi, S.c[-1], 30)
                col, txt = hc.C["tangent"], f"overlap: δ = {S.delta:.2f}".replace("-", "−")
            else:
                a, col, txt = None, hc.BLACK, "closes: δ = 0"
            g = VGroup()
            if a is not None:
                pts = [[0, 0, 0]] + [[rr * np.cos(v), rr * np.sin(v), 0] for v in a]
                g.add(Polygon(*cam(np.array(pts)), fill_color=col, fill_opacity=0.45, stroke_color=col, stroke_width=2)
                      .set_z_index(3))
            lab = hc.text(txt, size=32, color=col).move_to([x, -2.1, 0])
            res.append((g, lab))
        note3 = hc.text("δ = 2π − (sum of the angles at the vertex)", size=32).move_to([0, -2.82, 0])  # above the control bar
        self.play(*[FadeIn(g) for g, _ in res], *hc.swap(note2, note3), run_time=1.0)
        self.play(*[FadeIn(lab) for _, lab in res], run_time=0.8)
        self.wait(3.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)
