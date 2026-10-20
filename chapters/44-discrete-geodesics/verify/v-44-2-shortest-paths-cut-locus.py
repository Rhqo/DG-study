"""Verification for Section 44.2 (Discrete Shortest Paths, Cut Locus, and Medial Axis).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-2-shortest-paths-cut-locus.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgsym  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-2-shortest-paths-cut-locus.html"

# ---------------------------------------------------------------- Implementation 44.2.1
check("Implementation 44.2: at most 30 lines", dgcode.code_lines(PAGE, "code-44-2-1") <= 30)
ns, out = ch44.run_printed(PAGE, "code-44-2-1")
check("Implementation 44.2: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-44-2-1").strip(), out)
V, F = dm.icosphere(3)
d_impl = ns["dijkstra"](V, F, 7)
d_lib, _ = ch44.dijkstra(V, F, 7)
close("Implementation 44.2: dijkstra agrees with lib/ch44.dijkstra (icosphere(3))", d_impl, d_lib, 1e-12)

# ---------------------------------------------------------------- Proposition 44.2.1, (44.2.1), Exercise 44.2.1


def formula(a, b, h):
    if a * b >= 0:
        return h * (np.sqrt(2) * min(abs(a), abs(b)) + abs(abs(a) - abs(b)))
    return h * (abs(a) + abs(b))


n, h = 12, 1 / 12
xs = np.linspace(-0.5, 0.5, n + 1)
Vg, Fg = dm.grid_mesh(xs, xs, pattern="same")
c = (n // 2) * (n + 1) + n // 2
dg = ns["dijkstra"](Vg, Fg, c)
worst = 0.0
for i in range(n + 1):
    for j in range(n + 1):
        a, b = i - n // 2, j - n // 2
        worst = max(worst, abs(dg[i * (n + 1) + j] - formula(a, b, h)))
check("Prop 44.2.1: (44.2.1) equals Dijkstra for all grid vertices (n = 12)", worst < 1e-12, f"{worst}")
check("Prop 44.2.1: diagonals of pattern 'same' run along (1, 1)", all(
    abs((Vg[f[2]] - Vg[f[0]])[0] - (Vg[f[2]] - Vg[f[0]])[1]) < 1e-12 for f in Fg[::2]))
close("Prop 44.2.1: ratio toward (2, 1) = (sqrt2 + 1)/sqrt5 = 1.0797", formula(2, 1, 1) / np.sqrt(5), 1.0797, 5e-5)
close("Prop 44.2.1: ratio toward (1, -1) = sqrt 2", formula(1, -1, 1) / np.sqrt(2), np.sqrt(2), 1e-12)
close("Exercise 44.2.1: (3, -2) ratio 1.3868", formula(3, -2, 1) / np.sqrt(13), 1.3868, 5e-5)
close("Exercise 44.2.1: (2, 5) d_G = 5.8284, ratio 1.0823", [formula(2, 5, 1), formula(2, 5, 1) / np.sqrt(29)],
      [5.8284, 1.0823], 5e-5)
# dual certificate y = (1, sqrt2 - 1): <y, step> <= length for every step
steps = [np.array(s_) for s_ in ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))]
lens = [1, 1, np.sqrt(2)] * 2
y = np.array([1, np.sqrt(2) - 1])
check("Prop 44.2.1 proof: <y, step> <= length for y = (1, sqrt2 - 1)", all(y @ s_ <= L_ + 1e-12 for s_, L_ in zip(steps, lens)))
y2 = np.array([1.0, -1.0])
check("Prop 44.2.1 proof: <y, step> <= length for y = (1, -1)", all(y2 @ s_ <= L_ + 1e-12 for s_, L_ in zip(steps, lens)))

# ---------------------------------------------------------------- Example 44.2.4 (cube), Figure 44.2.1
p = np.array([0.2, 0.3, 0.0])
q = np.array([0.9, 0.6, 1.0])
dpq = ch44.cube_distance(p, q[None])[0]
close("Example 44.2.4: d(p, q) = sqrt(1.9^2 + 0.3^2) = 1.9235", dpq, np.sqrt(1.9 ** 2 + 0.3 ** 2), 1e-12)
close("Example 44.2.4: = 1.9235 (4 digits)", dpq, 1.9235, 5e-5)
pn, qn = np.array([0.2, 0.3]), np.array([2.1, 0.6])
heights = [pn[1] + (xe - pn[0]) / (qn[0] - pn[0]) * (qn[1] - pn[1]) for xe in (1.0, 2.0)]
close("Example 44.2.4: crossing heights 0.426, 0.584", heights, [0.426, 0.584], 5e-4)
cands = [np.hypot(0.3, 2.1), np.hypot(0.7, 1.9), np.hypot(0.7, 2.1)]
close("Example 44.2.4: other single side faces 2.1213, 2.0248, 2.2136", cands, [2.1213, 2.0248, 2.2136], 5e-5)
check("Example 44.2.4: the exact distance is the minimum (all candidates longer)", all(c_ > dpq for c_ in cands))
# independent bounds: Euclidean <= exact <= graph distance on a fine cube mesh
Vc, Fc = ch44.unit_cube(16)
s = int(np.argmin(np.linalg.norm(Vc - [0.25, 0.3125, 0], axis=1)))
dG, _ = ch44.dijkstra(Vc, Fc, s)
ex_c = ch44.cube_distance(Vc[s], Vc)
check("Example 44.2.4: Euclidean <= exact <= graph distance (unit_cube(16))",
      (ex_c >= np.linalg.norm(Vc - Vc[s], axis=1) - 1e-12).all() and (ex_c <= dG + 1e-12).all())
rng = np.random.default_rng(3)


def rand_pt():
    a = rng.integers(3)
    x = rng.random(3)
    x[a] = rng.integers(2)
    return x


bad = 0
for _ in range(150):
    A, B, C = rand_pt(), rand_pt(), rand_pt()
    ab, bc, ac = (ch44.cube_distance(A, B[None])[0], ch44.cube_distance(B, C[None])[0], ch44.cube_distance(A, C[None])[0])
    bad += (ac > ab + bc + 1e-9) + (abs(ab - ch44.cube_distance(B, A[None])[0]) > 1e-9)
check("Example 44.2.4: cube distance is symmetric and satisfies the triangle inequality (150 triples)", bad == 0)
# Exercise 44.2.2: corner to corner sqrt 5, six shortest paths
o = np.zeros(3)
wins = ch44.cube_windows(o)
target = np.ones(3)
hits = set()
for k, src, wa, wb, seq in wins:
    if abs(target[ch44.CUBE_FACES[k][0]] - ch44.CUBE_FACES[k][1]) > 1e-12:
        continue
    if wa is not None and not ch44._in_cone(src, wa, wb, target, 1e-9):
        continue
    if abs(np.linalg.norm(target - src) - np.sqrt(5)) < 1e-9:
        hits.add(seq[:2])
close("Exercise 44.2.2: d((0,0,0), (1,1,1)) = sqrt 5", ch44.cube_distance(o, target[None])[0], np.sqrt(5), 1e-12)
check("Exercise 44.2.2: six shortest paths (pairs of first two faces)", len(hits) == 6, str(hits))

# ---------------------------------------------------------------- Proposition 44.2.5, Corollary 44.2.6 (exact cones)
eps = 0.3
for Theta, name in ((5 * np.pi / 3, "cone"), (2 * np.pi, "flat"), (8 * np.pi / 3, "saddle")):
    ok = True
    for tl in np.linspace(0.05, Theta - 0.05, 400):
        d, _ = ch44.cone_distance(eps, 0.0, eps, tl, Theta)
        shortest_through_vertex = abs(d - 2 * eps) < 1e-12
        cond = (tl >= np.pi - 1e-12) and (Theta - tl >= np.pi - 1e-12)
        ok &= (shortest_through_vertex == cond)
        if tl < np.pi and tl <= Theta - tl:
            ok &= abs(d - 2 * eps * np.sin(tl / 2)) < 1e-12
    check(f"Prop 44.2.5: {name} (Theta = {Theta / np.pi:.3f} pi): through-vertex path shortest iff theta_l, theta_r >= pi", ok)
    width = max(0.0, Theta - 2 * np.pi)
    check(f"Cor 44.2.6: {name}: admissible interval width {width:.4f} = max(0, -delta)", abs(width - max(0, -(2 * np.pi - Theta))) < 1e-12)
for m in (5, 6, 8):
    Vs, Fs = ch44.equilateral_star(m)
    L_ = np.linalg.norm(Vs[dm.edges(Fs)[:, 0]] - Vs[dm.edges(Fs)[:, 1]], axis=1)
    check(f"Figure 44.2.2: equilateral star m = {m}: all edges 1, angle sum m pi/3",
          np.allclose(L_, 1) and abs(dm.angle_sums(Vs, Fs)[0] - m * np.pi / 3) < 1e-9)
# the lifted-angle argument: |Delta| >= min(theta_l, theta_r) for all k
Theta = 8 * np.pi / 3
for tl in (np.pi, 1.2 * np.pi, Theta - np.pi):
    check(f"Prop 44.2.5 proof: min_k |theta_l + k Theta| = min(theta_l, theta_r) (theta_l = {tl:.3f})",
          abs(min(abs(tl + k * Theta) for k in range(-3, 4)) - min(tl, Theta - tl)) < 1e-12)

# ---------------------------------------------------------------- Remark 44.2.7, Exercise 44.2.5
check("Remark 44.2.7: cube belt splits the corners 4 + 4, defect 4 (pi/2) = 2 pi on each side", abs(4 * np.pi / 2 - 2 * np.pi) < 1e-12)
A_, B_, C_ = np.array([1.0, 1, 1]), np.array([1.0, -1, -1]), np.array([-1.0, 1, -1])
D_ = np.array([-1.0, -1, 1])
sc = 1 / np.linalg.norm(A_ - B_)
A_, B_, C_, D_ = A_ * sc, B_ * sc, C_ * sc, D_ * sc
mids = [(A_ + C_) / 2, (C_ + B_) / 2, (B_ + D_) / 2, (D_ + A_) / 2]
close("Exercise 44.2.5: closed path length 2", sum(np.linalg.norm(mids[(k + 1) % 4] - mids[k]) for k in range(4)), 2.0, 1e-12)
# straightness at m2 (edge CB): unfold face BCD onto the plane of ABC across BC
e = (B_ - C_) / np.linalg.norm(B_ - C_)


def unfold_point(X, P0, e_, plane_ref):
    """rotate X about the line (P0, e_) into the plane of plane_ref, on the opposite side of plane_ref"""
    v = X - P0
    along = (v @ e_) * e_
    perp = np.linalg.norm(v - along)
    w = plane_ref - P0
    w = w - (w @ e_) * e_
    w /= np.linalg.norm(w)
    return P0 + along - perp * w


Dp = unfold_point(D_, C_, e, A_)
check("Exercise 44.2.5: D' - C = B - A after unfolding (rhombus)", np.allclose(Dp - C_, B_ - A_, atol=1e-12))
m3p = (B_ + Dp) / 2
dir1, dir2 = mids[1] - mids[0], m3p - mids[1]
check("Exercise 44.2.5: the two segments are collinear at m2", np.linalg.norm(np.cross(dir1, dir2)) < 1e-12 and dir1 @ dir2 > 0)
Vt = np.array([A_, B_, C_, D_])
Ft = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
close("Exercise 44.2.5: regular tetrahedron angle defect pi", dm.angle_defect(Vt, Ft), [np.pi] * 4, 1e-12)

# ---------------------------------------------------------------- Proposition 44.2.9, Example 44.2.10
V3, F3 = dm.icosphere(3)
delta3 = dm.angle_defect(V3, F3)
check("Example 44.2.10(a): every icosphere(3) vertex is a cone vertex", (delta3 > 0).all())
s3 = int(np.argmax(V3[:, 2]))
close("Example 44.2.10(a): delta at the vertex nearest the north pole = 0.0196", delta3[s3], 0.0196, 5e-5)
nb = dm.neighbors(F3, len(V3))[s3]
close("Example 44.2.10(a): shortest neighbouring edge 0.1383", np.linalg.norm(V3[nb] - V3[s3], axis=1).min(), 0.1383, 5e-5)
corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
cd = ch44.cube_distance(p, corners)
close("Example 44.2.10(b): nearest corner (0,0,0) at sqrt(0.13) = 0.3606", cd.min(), np.sqrt(0.13), 1e-12)
nn = 120
g = (np.arange(nn) + 0.5) / nn
pts = []
for k, (a, sd) in enumerate(ch44.CUBE_FACES):
    ax = [i for i in range(3) if i != a]
    U, W = np.meshgrid(g, g, indexing="ij")
    P = np.zeros((nn * nn, 3))
    P[:, a], P[:, ax[0]], P[:, ax[1]] = sd, U.ravel(), W.ravel()
    pts.append(P)
P = np.vstack(pts)
best, second = ch44.cube_distance_vec(p, P)
cut = (second - best) < 1.2 / nn
close("Example 44.2.10(b): nearest sampled cut point at 0.362", best[cut].min(), 0.362, 1e-3)
check("Figure 44.2.1: every corner is within 2.5/120 of a sampled cut point",
      all(np.linalg.norm(P[cut] - c_, axis=1).min() < 2.5 / nn for c_ in corners))
close("Figure 44.2.1: farthest point at distance 2.045", best.max(), 2.045, 5e-4)

# ---------------------------------------------------------------- Proposition 44.2.12 (rectangle)
a_, b_ = 1.5, 1.0
rect = np.array([[-a_, -b_], [a_, -b_], [a_, b_], [-a_, b_]])
Pm, rm = ch44.polygon_medial_samples(rect, 0.01)
on_axis = (np.abs(Pm[:, 1]) < 0.011) & (np.abs(Pm[:, 0]) <= a_ - b_ + 0.011)
on_diag = np.abs(np.abs(Pm[:, 0]) - np.abs(Pm[:, 1]) - (a_ - b_)) < 0.02
check("Prop 44.2.12: sampled medial axis of [-1.5,1.5]x[-1,1] lies on the predicted segments", (on_axis | on_diag).all())
check("Prop 44.2.12: the middle segment is covered", np.abs(Pm[on_axis, 0]).max() > a_ - b_ - 0.02)
sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
Ps, _ = ch44.polygon_medial_samples(sq, 0.01)
check("Prop 44.2.12: square: medial axis on the diagonals", (np.abs(np.abs(Ps[:, 0]) - np.abs(Ps[:, 1])) < 0.02).all())

# ---------------------------------------------------------------- Proposition 44.2.13 (ellipse)
xs_ = np.array([0.0, 0.7, 1.2, 1.49, -1.3])
d_, k_ = ch44.ellipse_closest(np.stack([xs_, 0 * xs_], 1))
check("Prop 44.2.13: two closest points on the open segment |x| < 3/2", (k_ == 2).all())
close("Prop 44.2.13: radius function sqrt(1 - x^2/3) on the axis", d_, np.sqrt(1 - xs_ ** 2 / 3), 1e-7)
d2, k2 = ch44.ellipse_closest(np.array([[1.6, 0.0], [1.9, 0.0], [0.4, 0.3], [-1.1, -0.2], [0.0, 0.5]]))
check("Prop 44.2.13: unique closest point off the segment", (k2 == 1).all())
close("Prop 44.2.13: radius at (3/2, 0) is 1/2", ch44.ellipse_closest(np.array([[1.5, 0.0]]))[0][0], 0.5, 1e-9)
t = sp.symbols("t", real=True)
kap, _ = dgsym.curvature_torsion(sp.Matrix([2 * sp.cos(t), sp.sin(t), 0]), t)
close("Prop 44.2.13: curvature of the ellipse at (2, 0) is a/b^2 = 2", float(kap.subs(t, 0)), 2.0, 1e-12)
cc = sp.symbols("c", real=True)
xx = sp.symbols("x", real=True)
check("Prop 44.2.13 proof: |e(t) - (x,0)|^2 = 3c^2 - 4xc + x^2 + 1",
      sp.simplify((2 * sp.cos(t) - xx) ** 2 + sp.sin(t) ** 2 - (3 * sp.cos(t) ** 2 - 4 * xx * sp.cos(t) + xx ** 2 + 1)) == 0)

# ---------------------------------------------------------------- Proposition 44.2.14, Example 44.2.15 (L-shape)
tk = 2 * np.pi * np.arange(16) / 16
poly = np.stack([2 * np.cos(tk), np.sin(tk)], 1)
P16, _ = ch44.polygon_medial_samples(poly, 0.004)
check("Prop 44.2.14: every vertex of the inscribed 16-gon is within 0.03 of its medial axis",
      all(np.linalg.norm(P16 - v, axis=1).min() < 0.03 for v in poly))
Lsh = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
PL, rL = ch44.polygon_medial_samples(Lsh, 0.0025)
sel = (PL[:, 0] > 0.62) & (PL[:, 0] < 0.97) & (PL[:, 1] < 0.8) & (PL[:, 1] > 0.4)
check("Example 44.2.15: sampled branch on the parabola (1 - x)^2 = 2y - 1 (residual < 3e-3)",
      sel.sum() > 50 and np.abs((1 - PL[sel, 0]) ** 2 - (2 * PL[sel, 1] - 1)).max() < 3e-3)
x0 = 2 - np.sqrt(2)
check("Example 44.2.15: (1 - x)^2 = 2x - 1 at x = 2 - sqrt 2 (meets the diagonal)", abs((1 - x0) ** 2 - (2 * x0 - 1)) < 1e-12)
close("Example 44.2.15: meeting point equidistant (bottom, left, reflex vertex) = 0.5858",
      [x0, x0, np.hypot(1 - x0, 1 - x0)], [0.5858] * 3, 5e-5)
for cvx in ([2, 0], [2, 1], [1, 2], [0, 2], [0, 0]):
    check(f"Prop 44.2.14: convex corner {cvx} of the L-shape touches the medial axis",
          np.linalg.norm(PL - np.array(cvx, float), axis=1).min() < 0.02)
check("Example 44.2.15: the reflex vertex (1, 1) does not (nearest sample > 0.1)",
      np.linalg.norm(PL - np.array([1.0, 1.0]), axis=1).min() > 0.1)

# ---------------------------------------------------------------- Exercises 44.2.3, 44.2.4
close("Exercise 44.2.3: ratios sin(3 pi/8) = 0.9239, sin(pi/4) = 0.7071", [np.sin(3 * np.pi / 8), np.sin(np.pi / 4)],
      [0.9239, 0.7071], 5e-5)
close("Exercise 44.2.4: inradius of the unit equilateral triangle 0.2887", 1 / (2 * np.sqrt(3)), 0.2887, 5e-5)
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
Pt, rt = ch44.polygon_medial_samples(tri, 0.004)
inc = tri.mean(0)
on_bis = [np.abs(np.cross(np.c_[Pt - v, 0 * Pt[:, 0]], np.r_[inc - v, 0])[:, 2]) / np.linalg.norm(inc - v) for v in tri]
check("Exercise 44.2.4: triangle medial axis on the three bisector segments", (np.min(on_bis, 0) < 0.01).all())

# ---------------------------------------------------------------- Exercise 44.2.6 (Coding)
check("Exercise 44.2.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-2-6") <= 30)
ns2, out2 = ch44.run_printed(PAGE, "code-ex-44-2-6", dict(ns))
check("Exercise 44.2.6: printed output matches the page", out2.strip() == ch44.printed_block(PAGE, "out-ex-44-2-6").strip(), out2)
close("Exercise 44.2.6: corner-to-corner graph distance 1 + sqrt 2", 1 + np.sqrt(2), 2.4142, 5e-5)
summary()
