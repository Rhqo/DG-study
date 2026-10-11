"""Animation 31.6.3: Tutte relaxation untangles a UV map; turning the weights into cotan weights flips faces.

Data (Figure 31.6.3): the crumpled 4 x 4 patch (seed 8921, 16 vertices, 18 faces); boundary on the unit circle by arc
length (fixed); interior vertices x_5, x_6, x_9, x_10.
Phase 1: interior vertices start at random points in the disk (numpy default_rng(3), radius < sqrt 0.8; 3 faces
flipped) and are moved again and again to the average of their neighbours (Jacobi iteration of (31.6.3) with
lambda_ij = 1/deg i). The iteration converges to the Tutte map of Figure 31.6.3(b).
Phase 2: the weights are blended, w_ij(s) = (1 - s) * 1 + s * (cotan weight), s from 0 (Tutte) to 1 (the discrete
harmonic map of Figure 31.6.3(c)); the UV map solves (31.6.3) with the normalised weights at every s.
Vermillion faces are flipped (negative signed UV area); the dashed polygon is the ring of x_9's neighbours.
Self-check: the Jacobi iterates converge to dm.fixed_boundary_map(..., "uniform") (error < 1e-6 after 30 sweeps)
with no flipped face from sweep 1 on; along the blend no face flips while all weights are positive (the first
weight to turn negative is w_{9,13}, at s = 0.363); at s = 1 the map equals dm.fixed_boundary_map(..., "cotan") with the
two flipped faces (x_8, x_9, x_5), (x_5, x_9, x_10), and x_9 lies outside its ring (oriented angle sum 0).

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-6-3-tutte-cotan.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403


def crumpled_patch(seed=8921):
    """The patch of Figure 31.6.3 (same construction as fig-31-6-3-bijectivity.py)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 6))
    xs = np.linspace(0, 1, n)
    V, F = dm.grid_mesh(xs, xs, pattern="alternate")
    V = V.copy()
    inner = [i for i in range(len(V)) if 0 < V[i, 0] < 1 and 0 < V[i, 1] < 1]
    V[inner, :2] += rng.uniform(-0.45, 0.45, size=(len(inner), 2)) / (n - 1)
    amp = rng.uniform(0, 1.2)
    fx, fy = rng.uniform(1, 5, 2)
    V[:, 2] = amp * np.sin(fx * V[:, 0]) * np.sin(fy * V[:, 1])
    return V, F


V, F = crumpled_patch()
N0 = len(V)
assert N0 == 16 and len(F) == 18
LOOP = dm.boundary_loop(F)
IS_B = np.zeros(N0, bool)
IS_B[LOOP] = True
INN = np.where(~IS_B)[0]
NB = dm.neighbors(F, N0)
UV_T = dm.fixed_boundary_map(V, F, "uniform")
UV_C = dm.fixed_boundary_map(V, F, "cotan")
WCOT = dm.cotan_weights(V, F)
assert round(WCOT[(9, 13)], 2) == -1.75


def flipped(UV):
    return np.where(dm.uv_distortion(V, F, UV)[1] < 0)[0]


# phase 1: Jacobi sweeps from random interior positions
rng = np.random.default_rng(3)
UV0 = UV_T.copy()
r_ = np.sqrt(rng.uniform(0, 0.8, len(INN)))
a_ = rng.uniform(0, 2 * np.pi, len(INN))
UV0[INN] = np.stack([r_ * np.cos(a_), r_ * np.sin(a_)], 1)
ITERS = [UV0]
for _ in range(30):
    U = ITERS[-1]
    new = U.copy()
    for i in INN:
        new[i] = U[list(NB[i])].mean(0)
    ITERS.append(new)
assert len(flipped(ITERS[0])) == 3 and all(len(flipped(U)) == 0 for U in ITERS[1:])
assert np.abs(ITERS[-1] - UV_T).max() < 1e-6

# phase 2: weight blend
WU = np.zeros((N0, N0))
WC = np.zeros((N0, N0))
for i, js in enumerate(NB):
    WU[i, list(js)] = 1.0
for (i, j), w in WCOT.items():
    WC[i, j] = WC[j, i] = w


def solve(W):
    UV = UV_T.copy()
    A = np.diag(W[INN].sum(1)) - W[np.ix_(INN, INN)]
    rhs = W[np.ix_(INN, np.where(IS_B)[0])] @ UV_T[IS_B]
    UV[INN] = np.linalg.solve(A, rhs)
    return UV


SS = np.linspace(0, 1, 201)
BLEND = [solve((1 - s) * WU + s * WC) for s in SS]
EDGES = [(i, j) for i in range(N0) for j in range(i + 1, N0) if WU[i, j] > 0]
NEG = np.array([sum(((1 - s) * WU[i, j] + s * WC[i, j]) < 0 for i, j in EDGES) for s in SS])
NFLIP = np.array([len(flipped(U)) for U in BLEND])
assert np.all(NFLIP[NEG == 0] == 0)                       # Theorem 31.6.5 along the blend
s_neg = 1 / (1 - WCOT[(9, 13)])                            # w_{9,13}(s) = 0
assert abs(s_neg - 0.363) < 1e-3 and SS[np.argmax(NEG > 0)] >= s_neg - 1e-9
assert np.allclose(BLEND[-1], UV_C) and sorted(map(sorted, F[flipped(UV_C)].tolist())) == [[5, 8, 9], [5, 9, 10]]
def ring_of(i):
    """Neighbours of interior vertex i in the cyclic order of its faces (from dm.vertex_link)."""
    nxt = dict(dm.vertex_link(F, i))
    a0 = next(iter(nxt))
    ring = [a0]
    while nxt[ring[-1]] != a0:
        ring.append(nxt[ring[-1]])
    assert len(ring) == len(nxt)
    return ring


RING9 = ring_of(9)


def winding(UV, i, ring):
    ang = 0.0
    for a, b in zip(ring, list(ring[1:]) + [ring[0]]):
        p, q = UV[a] - UV[i], UV[b] - UV[i]
        ang += np.arctan2(p[0] * q[1] - p[1] * q[0], p @ q)
    return ang


assert abs(winding(UV_T, 9, RING9)) > 6.0 and abs(winding(UV_C, 9, RING9)) < 1e-9


class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=35, azim=-60, scale=1.8, center=[-4.6, 0.75, 0], pivot=(0.5, 0.5, 0.0))
        patch = MeshView(F, stroke_width=1.2, stroke_rgb=(0.2, 0.2, 0.2), cull=False, ambient=0.5)
        patch.set_state(V, hex_rgb("#D0D0D0"), cam)
        e913 = Line(cam(V[9]), cam(V[13]), stroke_width=6, color=C["normal"]).set_z_index(3)
        plabs = VGroup()
        for m_ in (mt(r"$x_9$", 32, color=C["normal"]).next_to(cam(V[9]), LEFT, buff=0.12),
                   mt(r"$x_{13}$", 32).next_to(cam(V[13]), DR, buff=0.05)):
            plabs.add(BackgroundRectangle(m_, color=WHITE, fill_opacity=0.85, buff=0.04), m_)
        plabs.set_z_index(4)
        ptitle = da.label("the patch (18 faces)", 30).move_to([-4.6, 2.75, 0])
        wnote = mt(r"cotan weight of $x_9x_{13}$: $-1.75$", 30, color=C["normal"]).move_to([-6.85, -1.55, 0], aligned_edge=LEFT)

        UC, US = np.array([3.25, -0.2]), 2.05

        def u2s(p):
            return np.array([UC[0] + US * p[0], UC[1] + US * p[1], 0.0])

        circle = DashedVMobject(Circle(radius=US, stroke_width=1.5, color=C["aux"]), num_dashes=60).move_to(u2s((0, 0)))
        state = {"UV": ITERS[0]}

        def uvmesh():
            UV = state["UV"]
            fl = set(flipped(UV).tolist())
            g = VGroup()
            for f, tri in enumerate(F):
                bad = f in fl
                g.add(Polygon(*[u2s(UV[k]) for k in tri], stroke_width=1.6, color=BLACK,
                              fill_color=C["normal"] if bad else GREY_A, fill_opacity=0.8 if bad else 0.55)
                      .set_z_index(2 if bad else 1))
            ring = [u2s(UV[k]) for k in RING9]
            g.add(DashedVMobject(Polygon(*ring, stroke_width=3, color=C["accent"]), num_dashes=40).set_z_index(3))
            g.add(Dot(u2s(UV[9]), radius=0.09, color=C["normal"]).set_z_index(4))
            g.add(Dot(u2s(UV[13]), radius=0.07, color=BLACK).set_z_index(4))
            return g

        uvm = always_redraw(uvmesh)
        ttl = title("Tutte map vs. cotan weights", 34)
        uvtitle = da.label("UV: boundary on the circle", 30).move_to([3.25, 2.75, 0])
        self.add(patch, e913, plabs, ptitle, circle, uvm, ttl, uvtitle)

        def flips_text():
            return f"flipped faces: {len(flipped(state['UV']))}"

        fread = readout(flips_text, [-0.6, 2.2, 0], size=32, color=C["normal"])
        step = da.label("start: interior vertices at random", 30).move_to([-0.6, -2.7, 0], aligned_edge=LEFT)
        self.add(fread, step)
        self.wait(2.0)
        new = da.label("Tutte: each to its neighbours' mean", 30).move_to([-0.6, -2.7, 0], aligned_edge=LEFT)
        self.play(FadeTransform(step, new), run_time=0.5)
        step = new
        it = ValueTracker(0.0)

        def set_iter(mob):
            x = it.get_value()
            k = int(np.floor(x))
            t = x - k
            k1 = min(k + 1, len(ITERS) - 1)
            state["UV"] = (1 - t) * ITERS[k] + t * ITERS[k1]

        anchor = Mobject()
        anchor.add_updater(set_iter)
        self.add(anchor)
        for k in range(1, 7):
            self.play(it.animate.set_value(k), run_time=0.75 if k < 4 else 0.45, rate_func=smooth)
            self.wait(0.45 if k < 4 else 0.2)
        self.play(it.animate.set_value(30), run_time=0.6)
        anchor.clear_updaters()
        state["UV"] = UV_T
        new = mt(r"converged: the Tutte map, no flip", 30).move_to([-0.6, -2.7, 0], aligned_edge=LEFT)
        self.play(FadeTransform(step, new), run_time=0.5)
        step = new
        self.wait(1.6)

        # ---------------------------------------------------------- phase 2
        sv = ValueTracker(0.0)

        def set_s(mob):
            s = sv.get_value()
            j = s * (len(SS) - 1)
            k = int(np.floor(j))
            t = j - k
            k1 = min(k + 1, len(SS) - 1)
            state["UV"] = (1 - t) * BLEND[k] + t * BLEND[k1]

        anchor.add_updater(set_s)
        blend = mt(r"blend: $w_{ij}(s) = (1-s) + s\,w^{\cot}_{ij}$", 30).move_to([-0.6, -2.7, 0],
                                                                                         aligned_edge=LEFT)
        self.play(FadeTransform(step, blend), FadeIn(wnote), run_time=0.6)
        sread = readout(lambda: f"s = {sv.get_value():.2f}", [-6.85, -1.97, 0], size=30)
        wread = readout(lambda: f"w(x\u2089, x\u2081\u2083) = {(1 - sv.get_value()) + sv.get_value() * WCOT[(9, 13)]:+.2f}",
                        [-6.85, -2.39, 0], size=30, color=C["normal"])
        self.add(sread, wread)
        self.play(sv.animate.set_value(s_neg), run_time=2.6, rate_func=linear)
        self.wait(1.0)
        self.play(sv.animate.set_value(1.0), run_time=4.4, rate_func=linear)
        anchor.clear_updaters()
        state["UV"] = UV_C
        out = da.label("x\u2089 leaves its ring", 30, color=C["normal"]).move_to([-6.85, -2.81, 0], aligned_edge=LEFT)
        self.play(FadeIn(out), run_time=0.5)
        self.wait(2.4)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.0)
