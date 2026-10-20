"""Verification for Section 44.3 (Discrete Straightest Geodesics).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-3-straightest-geodesics.py
"""
import sys
from pathlib import Path

import dgmesh as dm
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-3-straightest-geodesics.html"

# ---------------------------------------------------------------- Implementation 44.3 (two blocks)
for bid in ("code-44-3-1", "code-44-3-2"):
    check(f"Implementation 44.3: {bid} at most 30 lines", dgcode.code_lines(PAGE, bid) <= 30)
ns, _ = ch44.run_printed(PAGE, "code-44-3-1")
ns, out = ch44.run_printed(PAGE, "code-44-3-2", ns)
check("Implementation 44.3: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-44-3-2").strip(), out)
trace = ns["trace"]
for level in (2, 3):
    V, F = dm.icosphere(level)
    p = V[F[5]].mean(0)
    d = ch44.unit(np.cross(dm.face_normals(V, F)[5], [0.3, -0.2, 1.0]))
    a = trace(V, F, 5, p, d, 2 * np.pi)
    b = ch44.trace_straightest(V, F, 5, p, d, 2 * np.pi)["points"]
    check(f"Implementation 44.3: trace agrees with lib trace_straightest (icosphere({level}))",
          a.shape == b.shape and np.abs(a - b).max() < 1e-10)
Vc, Fc = ch44.unit_cube(1)
pc = np.array([0.5, 0.2, 0.0])
c = np.array([1.0, 1.0, 0.0])
path = trace(Vc, Fc, ch44.locate_face(Vc, Fc, pc), pc, ch44.unit(c - pc), 1.6)
close("Implementation 44.3: cube path passes the corner and ends at (0.8524, 1, 0.6398)", path[1:], [c, [0.8524, 1, 0.6398]], 5e-5)
w = ch44.unit(pc - c)
ang1 = np.degrees(np.arccos(w @ np.array([-1.0, 0, 0])))
ang2 = np.degrees(np.arccos(w @ np.array([0, -1.0, 0])))
ang3 = np.degrees(np.arctan2(1 - path[2][0], path[2][2]))
close("Implementation 44.3: -w makes 58.0 and 32.0 degrees with the two bottom edges; exit 13.0 degrees from +z",
      [ang1, ang2, ang3], [58.0, 32.0, 13.0], 0.05)
close("Implementation 44.3: 32.0 + 90 + 13.0 = 135 = Theta/2 in degrees", ang2 + 90 + ang3, 135.0, 1e-6)
# the other way around gives the same exit (58.0 + 90 + (90 - 13.0) - ... = 135 on the other side)
close("Prop 44.3.6: turning Theta/2 the other way reaches the same direction (58 + 77 = 135)", ang1 + (90 - ang3), 135.0, 1e-6)
# a clockwise sweep implementation must give the same exit direction


def exit_cw(V, F, f, v, d):
    EF, N, Theta = dm.edge_faces(F), dm.face_normals(V, F), dm.angle_sums(V, F)
    other = lambda g, a, b: [h for h in EF[(min(a, b), max(a, b))] if h != g][0]  # noqa: E731
    w, acc = ch44.unit(-d), 0.0
    while True:
        na = F[f][(list(F[f]).index(v) + 1) % 3]          # clockwise edge (v, na) of f
        beta = np.arccos(np.clip(w @ ch44.unit(V[na] - V[v]), -1, 1))
        if acc + beta >= Theta[v] / 2:
            break
        acc, w, f = acc + beta, ch44.unit(V[na] - V[v]), other(f, v, na)
    phi = Theta[v] / 2 - acc
    return np.cos(phi) * w - np.sin(phi) * np.cross(N[f], w), f


vidx = int(np.argmin(np.linalg.norm(Vc - c, axis=1)))
dcw, fcw = exit_cw(Vc, Fc, ch44.locate_face(Vc, Fc, pc), vidx, ch44.unit(c - pc))
close("Prop 44.3.6: clockwise and counterclockwise sweeps give the same exit direction", dcw, ch44.unit(path[2] - c), 1e-9)
# saddle and cone stars: exit directions from both sweeps agree too
for m in (5, 8):
    Vs, Fs = ch44.equilateral_star(m)
    f0 = 0
    q0 = Vs[Fs[f0]].mean(0)
    dd = ch44.unit(Vs[0] - q0)
    r = ch44.trace_straightest(Vs, Fs, f0, q0, dd, np.linalg.norm(Vs[0] - q0) + 0.3)
    out_dir = ch44.unit(r["points"][-1] - Vs[0])
    dcw, _ = exit_cw(Vs, Fs, f0, 0, dd)
    close(f"Prop 44.3.6: star m = {m}: both sweeps agree", dcw, out_dir, 1e-9)

# ---------------------------------------------------------------- Definition 44.3.2, (44.3.2), Non-example 44.3.4
rng = np.random.default_rng(5)
ok = True
for _ in range(200):
    a_in, kappa = rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0)
    tl, tr = np.pi - kappa, np.pi + kappa                 # flat point turning left by kappa
    ok &= abs((tr - tl) / 2 - kappa) < 1e-12
check("Def 44.3.2: at a flat point (theta_r - theta_l)/2 is the turning angle", ok)
# planar polygon: turning angle from geometry equals (theta_r - theta_l)/2 with theta_l measured on the left
ok = True
for _ in range(200):
    u = ch44.unit(rng.normal(size=2))
    w2 = ch44.unit(rng.normal(size=2))
    turn = np.arctan2(u[0] * w2[1] - u[1] * w2[0], u @ w2)        # counterclockwise positive (Definition 32.3.1)
    back = -u
    tl = np.arctan2(w2[0] * back[1] - w2[1] * back[0], w2 @ back) % (2 * np.pi)   # from w2 ccw to -u: left side
    tr = 2 * np.pi - tl
    ok &= abs((tr - tl) / 2 - turn) < 1e-9
check("Def 44.3.2: planar corners: (theta_r - theta_l)/2 = signed turning angle (ccw positive = left)", ok)
thl, Th = sp.symbols("theta_l Theta", real=True)
dl = 2 * sp.pi - Th
kap = ((Th - thl) - thl) / 2
sym_equal("(44.3.2): kappa_g = Theta/2 - theta_l", kap, Th / 2 - thl)
sym_equal("(44.3.2): kappa_g = (pi - theta_l) - delta/2", kap, (sp.pi - thl) - dl / 2)
sym_equal("(44.3.2): kappa_g = (theta_r - pi) + delta/2", kap, ((Th - thl) - sp.pi) + dl / 2)
sym_equal("Non-example 44.3.4: (pi - theta_l) + (pi - theta_r) = delta", (sp.pi - thl) + (sp.pi - (Th - thl)), dl)
close("Example 44.3.3(b): theta_l = pi/2, theta_r = pi at the cube corner gives kappa_g = pi/4",
      (np.pi - np.pi / 2) / 2, np.pi / 4, 1e-15)
check("Example 44.3.3(b): one-sided values pi/2 and 0 differ by delta = pi/2", abs((np.pi - np.pi / 2) - 0 - np.pi / 2) < 1e-15)
check("Exercise 44.3.3: reversing swaps theta_l, theta_r and flips kappa_g", sp.simplify(((thl - (Th - thl)) / 2) + kap) == 0)

# ---------------------------------------------------------------- (44.3.4): unfolding across an edge keeps the line straight
ok = True
for _ in range(100):
    A, B = rng.normal(size=3), rng.normal(size=3)
    Cf, Cg = rng.normal(size=3), rng.normal(size=3)
    e = ch44.unit(B - A)
    Nf = ch44.unit(np.cross(B - A, Cf - A))                # face f = (A, B, Cf)
    Ng = ch44.unit(np.cross(A - B, Cg - B))                # face g = (B, A, Cg): opposite traversal of AB
    # make the configuration a valid hinge: Cf and Cg on opposite sides of AB within their planes
    if (np.cross(Nf, e) @ (Cf - A)) <= 0 or (np.cross(Ng, -e) @ (Cg - A)) <= 0:
        continue
    wd = ch44.unit(rng.normal() * e - abs(rng.normal()) * np.cross(Nf, e))     # leaving f through AB
    wp = (wd @ e) * e + (wd @ np.cross(Nf, e)) * np.cross(Ng, e)
    ok &= abs(np.linalg.norm(wp) - 1) < 1e-12 and abs(wp @ Ng) < 1e-12 and abs(wp @ e - wd @ e) < 1e-12
    ok &= (wp @ np.cross(Ng, -e)) > 0                      # points into g
check("(44.3.4): the transported direction is a unit vector of g's plane pointing into g, same angle with the edge", ok)

# ---------------------------------------------------------------- Proposition 44.3.7, Corollary 44.3.8, Figure 44.3.1
q = path[-1]
dq = ch44.cube_distance(pc, q[None])[0]
close("Figure 44.3.1(a): d(p, q) = 1.4823 < 1.6", dq, 1.4823, 5e-5)
r0 = np.linalg.norm(c - pc)
close("Prop 44.3.7(a): d(p, q) equals the chord in a 3 pi/4 sector",
      dq, np.sqrt(r0 ** 2 + (1.6 - r0) ** 2 - 2 * r0 * (1.6 - r0) * np.cos(3 * np.pi / 4)), 1e-12)
eps = 0.2
for Theta, expect in ((5 * np.pi / 3, False), (2 * np.pi, True), (8 * np.pi / 3, True)):
    dd, _ = ch44.cone_distance(eps, 0.0, eps, Theta / 2, Theta)
    check(f"Prop 44.3.7: straightest through a vertex with Theta = {Theta / np.pi:.3f} pi is locally shortest: {expect}",
          (abs(dd - 2 * eps) < 1e-12) == expect)
close("Exercise 44.3.4(c): saddle Theta = 8 pi/3: admissible theta_l in [pi, 5 pi/3], straightest at 4 pi/3",
      [np.pi, 8 * np.pi / 3 - np.pi, 4 * np.pi / 3], [np.pi, 5 * np.pi / 3, 4 * np.pi / 3], 1e-12)

# ---------------------------------------------------------------- Theorem 44.3.9, Example 44.3.10, Exercise 44.3.2
OUT = {"bottom": np.array([0, 0, -1.0]), "x1": np.array([1.0, 0, 0]), "y1": np.array([0, 1.0, 0])}


def transport(w, e, Nf, Ng):
    e = e / np.linalg.norm(e)
    return (w @ e) * e + (w @ np.cross(Nf, e)) * np.cross(Ng, e)


def turn_at(t_in, t_out, e, Nf, Ng):
    """signed turning angle about the outward normal Ng of the face the curve leaves into ((44.3.4), then angle)."""
    w = transport(t_in, e, Nf, Ng)
    return np.arctan2(np.cross(w, t_out) @ Ng, w @ t_out)


for a in (0.2, 0.5, 0.9):
    A3, B3, C3 = np.array([1 - a, 1, 0.0]), np.array([1, 1, a]), np.array([1, 1 - a, 0.0])
    tAB, tBC, tCA = ch44.unit(B3 - A3), ch44.unit(C3 - B3), ch44.unit(A3 - C3)
    turn = [turn_at(tAB, tBC, np.array([0, 0, 1.0]), OUT["y1"], OUT["x1"]),
            turn_at(tBC, tCA, np.array([0, 1.0, 0]), OUT["x1"], OUT["bottom"]),
            turn_at(tCA, tAB, np.array([1.0, 0, 0]), OUT["bottom"], OUT["y1"])]
    close(f"Example 44.3.10: a = {a}: A -> B -> C turns +pi/2 (left, outward N) at each corner; delta + sum = 2 pi",
          turn + [sum(turn) + np.pi / 2], [np.pi / 2] * 3 + [2 * np.pi], 1e-12)
    rev = [turn_at(-tBC, -tAB, np.array([0, 0, 1.0]), OUT["x1"], OUT["y1"])]
    close(f"Example 44.3.10: a = {a}: the reversed curve turns -pi/2", rev, [-np.pi / 2], 1e-12)
# Theorem 44.3.9 on a curve through a cube corner: c2 = (1, 0, 0) -> P = (0.4, 1, 0) -> Q = (1, 1, 0.6) -> c2
c2v, Pq, Qq = np.array([1.0, 0, 0]), np.array([0.4, 1, 0]), np.array([1, 1, 0.6])


def angle(u, v):
    return np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))


th_c2 = angle(Pq - c2v, np.array([0, 1.0, 0])) + angle(np.array([0, 1.0, 0]), Qq - c2v)   # bottom + x = 1 wedges
Qu = np.array([Qq[0], 1 + Qq[2], 0.0])                     # y = 1 unfolded onto the plane of the bottom
th_P = angle(c2v - Pq, Qu - Pq)
Pu, c2u, Qc = np.array([Pq[0] - 1, Pq[2]]), np.array([1 - c2v[1], c2v[2]]), np.array([0.0, Qq[2]])
th_Q = angle(Pu - Qc, c2u - Qc)                            # y = 1 and x = 1 unfolded about the vertical edge
thetas = np.array([th_c2, th_P, th_Q])
Theta_c = np.array([1.5 * np.pi, 2 * np.pi, 2 * np.pi])
close("Theorem 44.3.9: curve through the corner (1, 0, 0) around c: first line = 2 pi",
      np.pi / 2 + np.sum(np.pi - thetas), 2 * np.pi, 1e-12)
close("Theorem 44.3.9: same curve, second line (half of delta at (1, 0, 0)) = 2 pi",
      np.pi / 2 + 0.5 * np.pi / 2 + np.sum(Theta_c / 2 - thetas), 2 * np.pi, 1e-12)
close("Theorem 44.3.9: the complementary disk (six corners inside) also gives 2 pi",
      6 * np.pi / 2 + np.sum(np.pi - (Theta_c - thetas)), 2 * np.pi, 1e-12)
kps = 2 * np.pi / Theta_c * (Theta_c / 2 - thetas)
close("Theorem 44.3.9 / Polthier98 normalization: delta + (theta_l/Theta) delta + sum kappa^PS = 2 pi",
      np.pi / 2 + thetas[0] / Theta_c[0] * np.pi / 2 + kps.sum(), 2 * np.pi, 1e-12)
check("Def 44.3.2 (iii): Polthier98's value is 2 pi/Theta times (44.3.1)",
      np.allclose(kps, 2 * np.pi / Theta_c * (Theta_c / 2 - thetas)) and abs(kps[1] - (np.pi - thetas[1])) < 1e-12)
# the unfolding used in Figure 44.3.1 is an isometry on each face (distances preserved within a face)
close("Exercise 44.3.2: bottom face boundary: 4 (pi - pi/2) = 2 pi", 4 * (np.pi - np.pi / 2), 2 * np.pi, 1e-15)
close("Exercise 44.3.2: kappa_g = pi/4 at each corner, (1/2) sum delta + sum kappa_g = 2 pi",
      [(np.pi - np.pi / 2) / 2, 0.5 * 4 * np.pi / 2 + 4 * np.pi / 4], [np.pi / 4, 2 * np.pi], 1e-15)
# Theorem 44.3.9 on a refined mesh: cut the cube along the triangle of Example 44.3.10 and apply Corollary 40.2.9
# (numerically: the region is the three corner squares [0, a]^2 pieces cut by the diagonal; its interior vertex is c)
close("Theorem 44.3.9: cube corner region: delta(c) + sum (pi - theta_l) = 2 pi",
      np.pi / 2 + 3 * (np.pi - np.pi / 2), 2 * np.pi, 1e-15)

# ---------------------------------------------------------------- Example 44.3.11, Figure 44.3.2
devs, closes = [], []
for level in (2, 3, 4):
    V, F = dm.icosphere(level)
    p = V[F[5]].mean(0)
    d = ch44.unit(np.cross(dm.face_normals(V, F)[5], [0.3, -0.2, 1.0]))
    res = ch44.trace_straightest(V, F, 5, p, d, 2 * np.pi)
    P = res["points"]
    n = ch44.unit(np.cross(p, d))
    devs.append(np.abs(P @ n).max())
    closes.append(np.linalg.norm(P[-1] - p))
    check(f"Example 44.3.11: icosphere({level}) path hits no vertex", not any(e[0] == "vertex" for e in res["events"]))
close("Example 44.3.11: max deviations 0.0862, 0.0338, 0.0052", devs, [0.0862, 0.0338, 0.0052], 5e-5)
close("Example 44.3.11: |x(2 pi) - p| = 0.0452, 0.0143, 0.0064", closes, [0.0452, 0.0143, 0.0064], 5e-5)
V3, F3 = dm.icosphere(3)
check("Example 44.3.11: icosphere(3) angle defects are about 0.02 or less", dm.angle_defect(V3, F3).max() < 0.03)

# ---------------------------------------------------------------- Exercise 44.3.1
for name, Th_, kg_ in (("octahedron", 4 * np.pi / 3, np.pi / 3), ("icosahedron", 5 * np.pi / 3, np.pi / 2)):
    close(f"Exercise 44.3.1: {name}: theta_l = theta_r = Theta/2, one-triangle kappa_g", [Th_ / 2, (Th_ - 2 * np.pi / 3) / 2],
          [Th_ / 2, kg_], 1e-15)

# ---------------------------------------------------------------- Exercise 44.3.6 (Coding)
check("Exercise 44.3.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-3-6") <= 30)
ns2, out2 = ch44.run_printed(PAGE, "code-ex-44-3-6", dict(ns))
check("Exercise 44.3.6: printed output matches the page", out2.strip() == ch44.printed_block(PAGE, "out-ex-44-3-6").strip(), out2)
summary()
