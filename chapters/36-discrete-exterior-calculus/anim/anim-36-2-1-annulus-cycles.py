"""Animation 36.2.1: a cycle that bounds and a cycle that does not (Section 36.2, Example 36.2.15, Figure 36.2.2).

Mesh: the annulus 1 <= r <= 2 of Example 36.2.15, lib/ch36.annulus(3, 12) (rings of radius 1, 1.5, 2 with 12 vertices
each, 48 counterclockwise triangles). 1-cochain: the de Rham image of d theta = (-y dx + x dy)/(x^2 + y^2),
dgdec.derham1 with order 12 as in Figure 36.2.2: the signed angle that the edge subtends at the origin, i.e. pi/6 on
a ring edge run counterclockwise, 0 on a radial edge, +-pi/6 on a diagonal.
Part 1: the 2-chain b grows face by face over the sector 0 <= theta <= pi/3 (8 faces); <hat dtheta, partial b> = 0.
Part 2: the cycle z = inner circle (counterclockwise) has <hat dtheta, z> = 12 pi/6 = 2 pi. Sweeping the 48 faces
in order, z_k = z + partial(b_k) moves out to the middle and then the outer circle; the pairing stays 2 pi because
every face adds <hat dtheta, partial f> = 0 (closed form, Stokes on each triangle).
Self-check: derham1 values equal the exact subtended angles (1e-9); every face boundary pairs to 0; every z_k is a
cycle with coefficients in {-1, 0, 1} and pairs to 2 pi; z_24 and z_48 are the middle and outer circles.

Render: bash tools/render_anims.sh chapters/36-discrete-exterior-calculus/anim/anim-36-2-1-annulus-cycles.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ch36  # noqa: E402
import dganim as da  # noqa: E402
import dgdec as dd  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, VGroup  # noqa: E402

# ---------------------------------------------------------------- data
NT = 12
V, F = ch36.annulus(3, NT)
E, EIDX = dd.edge_list(F)
D1 = dd.d1(F).astype(int)
B2 = D1.T                                              # boundary of 2-chains (n1 x n2)
dtheta = lambda p: np.array([-p[1], p[0]]) / (p[0] ** 2 + p[1] ** 2)  # noqa: E731
A = dd.derham1(dtheta, V, E, order=12)
ang = np.arctan2(V[:, 1], V[:, 0])
exact = np.angle(np.exp(1j * (ang[E[:, 1]] - ang[E[:, 0]])))
assert np.allclose(A, exact, atol=1e-9)
UNITS = np.rint(A / (np.pi / 6)).astype(int)            # values in units of pi/6
assert np.allclose(A, UNITS * np.pi / 6, atol=1e-9) and set(np.unique(UNITS)) <= {-1, 0, 1}
assert np.allclose(D1 @ A, 0, atol=1e-12)               # every face: <hat dtheta, partial f> = 0


def fidx(i, j, half):
    """Face index of lib/ch36.annulus: ring i, angular slot j, half 0 = (i,j),(i+1,j),(i+1,j+1), 1 = the other."""
    return 2 * (i * NT + j % NT) + half


def ring_cycle(i):
    """The counterclockwise circle of ring i as a 1-chain (coefficients on edge_list)."""
    c = np.zeros(len(E), int)
    for j in range(NT):
        a, b = i * NT + j, i * NT + (j + 1) % NT
        c[EIDX[(min(a, b), max(a, b))]] += 1 if a < b else -1
    return c


# part 1: the sector 0 <= theta <= pi/3
SECTOR = [fidx(0, 0, 0), fidx(0, 0, 1), fidx(1, 0, 0), fidx(1, 0, 1),
          fidx(0, 1, 0), fidx(0, 1, 1), fidx(1, 1, 0), fidx(1, 1, 1)]
PART1 = []
b = np.zeros(len(F), int)
for f in SECTOR:
    b[f] = 1
    bd = B2 @ b
    assert abs(A @ bd) < 1e-12 and set(np.unique(bd)) <= {-1, 0, 1}
    PART1.append((b.copy(), bd))
assert abs(PART1[-1][1]).sum() == 8                     # the sector's boundary: 2 + 2 ring edges, 4 radial

# part 2: sweep all faces, ring by ring, counterclockwise
Z0 = ring_cycle(0)
assert abs(A @ Z0 - 2 * np.pi) < 1e-9
SWEEP = [fidx(i, j, h) for i in range(2) for j in range(NT) for h in range(2)]
PART2 = [Z0.copy()]
b = np.zeros(len(F), int)
for f in SWEEP:
    b[f] = 1
    z = Z0 + B2 @ b
    assert set(np.unique(z)) <= {-1, 0, 1} and not (dd.d0(F).T @ z).any()
    assert abs(A @ z - 2 * np.pi) < 1e-9
    PART2.append(z)
assert (PART2[2 * NT] == ring_cycle(1)).all() and (PART2[4 * NT] == ring_cycle(2)).all()

# ---------------------------------------------------------------- layout
C0, SC = np.array([-3.55, 0.15, 0.0]), 1.45
PV = np.array([C0 + SC * np.array([x, y, 0.0]) for x, y in V])
PX = 0.25
J = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def chain_arrows(coef, color, side=0.0):
    """Arrows along the edges with nonzero coefficient, in the direction of the coefficient."""
    out = []
    for e in np.nonzero(coef)[0]:
        i, j = E[e]
        a, b = (i, j) if coef[e] > 0 else (j, i)
        pa, pb = PV[a], PV[b]
        d = (pb - pa) / np.linalg.norm(pb - pa)
        off = side * (J @ d)
        out.append(hp.arrow(pa + 0.12 * (pb - pa) + off, pa + 0.88 * (pb - pa) + off, color=color, width=5, tip=0.17))
    return out


def faces_fill(fs, color, opacity=0.35):
    return [hp.poly_fill(PV[F[f]], color, opacity=opacity) for f in fs]


def unit_label(u):
    return {1: r"$+\pi/6$", -1: r"$-\pi/6$", 0: r"$0$"}[int(u)]


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("a cycle that bounds and a cycle that does not", 34).move_to([0, 3.52, 0])
        mesh = VGroup(*[hp.seg(PV[i], PV[j], color=hp.HEX["aux"], width=1.6) for i, j in E],
                      hp.mt(r"$+$", 30).move_to(C0))
        cap = hp.mt(r"$\widehat{d\theta}_e$ = angle of $e$ seen from $0$", 30).move_to([PX, 2.65, 0],
                                                                                         aligned_edge=hp.LEFT)
        cap2 = hp.mt(r"ring edge $\pm\pi/6$, radial $0$", 30, color=hp.C["aux"]).move_to([PX, 2.1, 0],
                                                                                         aligned_edge=hp.LEFT)
        self.add(title, mesh, cap, cap2)
        fill_g, arr_g, dyn_g, txt_g = VGroup(), VGroup(), VGroup(), VGroup()
        world = VGroup(fill_g, arr_g, dyn_g, txt_g)
        self.add(world)
        S = hp.setter

        def readout(lhs_tex, value, color):
            row = hp.mt(lhs_tex, 36).move_to([PX, 1.15, 0], aligned_edge=hp.LEFT)
            val = hp.mt(value, 36, color=color).next_to(row, hp.RIGHT, buff=0.2)
            return [row, val]

        # ---- part 1: a boundary
        self.mark("part 1")
        head1 = hp.text("1. the boundary of a 2-chain b", 30, color=hp.C["tangent"]).move_to([PX, 0.35, 0],
                                                                                              aligned_edge=hp.LEFT)
        f0 = SECTOR[0]
        tri = F[f0]
        labels = []
        for c in range(3):
            a, bb = int(tri[c]), int(tri[(c + 1) % 3])
            e = EIDX[(min(a, bb), max(a, bb))]
            u = UNITS[e] * (1 if a < bb else -1)
            pa, pb = PV[a], PV[bb]
            d = (pb - pa) / np.linalg.norm(pb - pa)
            frac = {0: 0.5, 1: 0.3, 2: 0.7}[c]            # keep the labels near the shared corner apart
            labels.append(hp.mt(unit_label(u), 30, color=hp.C["accent"]).move_to(pa + frac * (pb - pa) - 0.58 * (J @ d)))
        sumline = hp.mt(r"one face: $0 + \pi/6 - \pi/6 = 0$", 30, color=hp.C["accent"]).move_to(
            [PX, -0.35, 0], aligned_edge=hp.LEFT)
        first_arrows = chain_arrows(PART1[0][1], hp.C["accent"])
        ev = [(0.0, 0.0, S(txt_g, readout(r"$\langle\widehat{d\theta},\ \partial b\rangle =$", r"$0$", hp.C["tangent"])
                                + [head1])),
              (0.0, 0.0, S(fill_g, faces_fill([f0], hp.C["region"]))),
              (0.0, 0.0, S(dyn_g, first_arrows + labels + [sumline])),
              (0.0, 0.6, hp.fade(first_arrows + labels + [sumline]))]
        self.play(hp.timeline(world, ev, 3.4))
        # grow the sector
        ev = [(0.0, 0.0, S(dyn_g, []))]
        t = 0.0
        for k in range(len(SECTOR)):
            bvec, bd = PART1[k]
            ev += [(t, t, S(fill_g, faces_fill(np.nonzero(bvec)[0], hp.C["region"]))),
                   (t, t, S(arr_g, chain_arrows(bd, hp.C["tangent"])))]
            t += 0.65
        note1 = VGroup(hp.text("each face adds 0,", 30, color=hp.C["normal"]),
                       hp.text("so every boundary gives 0", 30, color=hp.C["normal"]))
        note1.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.12).move_to([PX, -1.25, 0], aligned_edge=hp.LEFT)
        ev += [(t, t, S(dyn_g, [note1])), (t, t + 0.5, hp.fade([note1]))]
        self.play(hp.timeline(world, ev, t + 2.2))

        # ---- part 2: the inner circle
        self.mark("part 2")
        head2 = hp.text("2. the inner circle z", 30, color=hp.C["normal"]).move_to([PX, 0.35, 0],
                                                                                   aligned_edge=hp.LEFT)
        zline = hp.mt(r"$12 \times \pi/6 = 2\pi$", 30, color=hp.C["normal"]).move_to([PX, -0.35, 0],
                                                                                    aligned_edge=hp.LEFT)
        zarr = chain_arrows(PART2[0], hp.C["normal"])
        ev = [(0.0, 0.0, S(fill_g, [])), (0.0, 0.0, S(dyn_g, [])),
              (0.0, 0.0, S(arr_g, zarr)), (0.0, 0.6, hp.fade(zarr)),
              (0.0, 0.0, S(txt_g, readout(r"$\langle\widehat{d\theta},\ z\rangle =$", r"$2\pi$", hp.C["normal"])
                                + [head2, zline]))]
        self.play(hp.timeline(world, ev, 2.6))
        # sweep the faces: z + partial(b_k)
        self.mark("sweep")
        sweep_txt = hp.mt(r"$z + \partial b$: still $2\pi$", 30, color=hp.C["normal"]).move_to([PX, -1.0, 0],
                                                                                            aligned_edge=hp.LEFT)
        ev = [(0.0, 0.0, S(dyn_g, [sweep_txt])), (0.0, 0.4, hp.fade([sweep_txt]))]
        t = 0.0
        for k, f in enumerate(SWEEP):
            ev += [(t, t, S(arr_g, chain_arrows(PART2[k + 1], hp.C["normal"]))),
                   (t, t, S(fill_g, faces_fill(SWEEP[:k + 1], hp.C["accent"], opacity=0.22)))]
            t += 0.25 if k != 2 * NT - 1 else 1.2              # pause at the middle circle
        self.play(hp.timeline(world, ev, t + 0.8))
        self.mark("outer circle")
        end = VGroup(hp.text("no face can change the 2π,", 30, color=hp.C["normal"]),
                     hp.text("so z bounds no 2-chain b", 30, color=hp.C["normal"]))
        end.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.12).move_to([PX, -1.95, 0], aligned_edge=hp.LEFT)
        self.play(FadeIn(end), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=20.0)
