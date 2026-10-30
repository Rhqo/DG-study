"""Verification for Section 50.3 (Refinement Conservation).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-3-refinement-conservation.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import mgs50 as g  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "50-3-refinement-conservation.html"
rng = np.random.default_rng(503)


def rand_plane():
    return g.plane_proj(g.unit(rng.normal(size=3)))


def lip_tests(k=40):
    """1-Lipschitz functions for the metric |x - y| + ||P - Q||_F: distances to random finite sets of (a, B), and
    smooth ones 0.5 sin(<a, x> + c) + 0.25 <B, P>_F with |a| = ||B||_F = 1 (Lipschitz constants 1/2 and 1/4)."""
    out = []
    for _ in range(k // 2):
        A = rng.normal(size=(3, 3))
        Bs = [rand_plane() for _ in range(3)]
        out.append(lambda x, P, A=A, Bs=Bs: min(np.linalg.norm(x - a) + np.linalg.norm(P - B) for a, B in zip(A, Bs)))
        a, B, c = g.unit(rng.normal(size=3)), rng.normal(size=(3, 3)), rng.uniform(0, 2 * math.pi)
        B = (B + B.T) / np.linalg.norm(B + B.T)
        out.append(lambda x, P, a=a, B=B, c=c: 0.5 * math.sin(a @ x + c) + 0.25 * np.sum(B * P))
    return out


def V(atoms, phi):
    return sum(q * phi(mu, P) for mu, P, q in atoms)


# ---------------------------------------------------------------- Proposition 50.3.1 and Corollary 50.3.2
tests = lip_tests()
worst = 0.0
for trial in range(30):
    mu, P, q = rng.normal(size=3), rand_plane(), rng.uniform(0.5, 2)
    m = rng.integers(2, 5)
    w = rng.dirichlet(np.ones(m))
    t = rng.normal(scale=0.3, size=(m, 3))
    t -= w @ t
    kids = [(mu + t[j], rand_plane() if trial % 2 else P, w[j] * q) for j in range(m)]
    check(f"Proposition 50.3.1 (trial {trial}): (50.3.1) holds",
          abs(sum(k[2] for k in kids) - q) < 1e-12 and np.allclose(sum(k[2] * k[0] for k in kids), q * mu))
    bound = sum(k[2] * (np.linalg.norm(k[0] - mu) + np.linalg.norm(k[1] - P)) for k in kids)
    for phi in tests:
        worst = max(worst, abs(V(kids, phi) - q * phi(mu, P)) / bound)
    sharp = lambda x, Q: np.linalg.norm(x - mu) + np.linalg.norm(Q - P)       # distance to the parent atom
    close(f"Proposition 50.3.1 (trial {trial}): the distance to the parent attains (50.3.2)",
          abs(V(kids, sharp) - q * sharp(mu, P)), bound, 1e-10)
check("Proposition 50.3.1: Delta_i(phi) <= (50.3.2) for 40 Lipschitz test functions x 30 trials", worst <= 1 + 1e-12)
# affine test functions see nothing when the barycentre and the plane are kept
mu, P, q = np.zeros(3), g.plane_proj(np.array([0, 0, 1.0])), 1.0
t = rng.normal(scale=0.2, size=(3, 3))
t -= t.mean(0)
kids = [(mu + tj, P, q / 3) for tj in t]
b = rng.normal(size=3)
close("Remark 50.3.3: an affine test function does not change", V(kids, lambda x, Q: 1 + b @ x) - (1 + b @ mu), 0, 1e-12)
# tangent moment with positive masses forces inheritance: two different planes never average to a plane
for _ in range(5):
    Pa, Pb = rand_plane(), rand_plane()
    M = 0.5 * (Pa + Pb)
    check("Remark 50.3.3: the mean of two different planes is not a projection", not np.allclose(M @ M, M))
# Corollary 50.3.2: several parents refined at once, the rest unchanged
atoms = [(rng.normal(size=3), rand_plane(), rng.uniform(0.2, 1)) for _ in range(8)]
after, c = [], 0.0
for i, (mu, P, q) in enumerate(atoms):
    if i % 2:
        after.append((mu, P, q))
        continue
    t = rng.normal(scale=0.2, size=(2, 3))
    t -= t.mean(0)
    for tj in t:
        Pj = rand_plane() if i == 4 else P
        after.append((mu + tj, Pj, q / 2))
        c += q / 2 * (np.linalg.norm(tj) + np.linalg.norm(Pj - P))
check("Corollary 50.3.2: |V_after(phi) - V_before(phi)| <= c for all test functions",
      max(abs(V(after, phi) - V(atoms, phi)) for phi in tests) <= c + 1e-12)

# ---------------------------------------------------------------- Example 50.3.5
t1, t2 = np.array([0.24, 0.06, 0]), np.array([0.12, -0.09, 0])
tb = (t1 + t2) / 2
close("Example 50.3.5: the offsets are (0.8, 0.4) sigma and (0.4, -0.6) sigma", [t1[0] / 0.3, t1[1] / 0.15, t2[0] / 0.3,
                                                                                t2[1] / 0.15], [0.8, 0.4, 0.4, -0.6], 1e-12)
close("Example 50.3.5: raw barycentre (0.18, -0.015), moved by 0.181", [*tb[:2], np.linalg.norm(tb)], [0.18, -0.015, 0.181],
      5e-4)
close("Example 50.3.5: affine change along the barycentre = 0.181", (g.unit(tb) @ tb), 0.181, 5e-4)
close("Example 50.3.5: |t1| = 0.2474, raw bound 0.199", [np.linalg.norm(t1), 0.5 * (np.linalg.norm(t1) + 0.15)],
      [0.2474, 0.199], 5e-4)
close("Example 50.3.5: |t2| = 0.15", np.linalg.norm(t2), 0.15, 1e-12)
close("Example 50.3.5: recentred children +-(0.06, 0.075), bound 0.0960", [*(t1 - tb)[:2], np.linalg.norm(t1 - tb)],
      [0.06, 0.075, 0.0960], 5e-5)
close("Example 50.3.5: 3DGS split area factor 2 / 1.6^2 = 0.78", 2 / 1.6 ** 2, 0.78, 0.0013)

# ---------------------------------------------------------------- Proposition 50.3.6
for trial in range(200):
    m = rng.integers(2, 6)
    N = np.array([g.unit(rng.normal(size=3) + (3 * np.array([0, 0, 1.0]) if trial % 2 else 0)) for _ in range(m)])
    qs = rng.uniform(0.1, 1, m)
    q = qs.sum()
    Pbar = sum(qq * g.plane_proj(n) for qq, n in zip(qs, N)) / q
    lam, U = np.linalg.eigh(Pbar)
    P = U[:, 1:] @ U[:, 1:].T
    D = q * np.linalg.norm(Pbar - P)
    if trial < 40:
        check(f"Proposition 50.3.6 (a) (trial {trial}): P is the nearest plane to the mean",
              all(np.linalg.norm(Pbar - P) <= np.linalg.norm(Pbar - rand_plane()) + 1e-12 for _ in range(50)))
    check(f"Proposition 50.3.6 (trial {trial}): eigenvalues in [0, 1] with sum 2",
          lam.min() >= -1e-12 and lam.max() <= 1 + 1e-12 and abs(lam.sum() - 2) < 1e-12)
    check(f"Proposition 50.3.6 (trial {trial}): ||Pbar - P||_F <= sqrt 2 lambda_3",
          np.linalg.norm(Pbar - P) <= math.sqrt(2) * lam[0] + 1e-12)
    for n in [U[:, 0], g.unit(rng.normal(size=3)), np.array([0, 0, 1.0])]:
        s2 = 1 - (N @ n) ** 2
        check(f"Proposition 50.3.6 (b) (trial {trial}): D <= sqrt 2 sum q_j sin^2 theta_j",
              D <= math.sqrt(2) * np.sum(qs * s2) + 1e-12)
check("Proposition 50.3.6 (c): equal planes give D = 0",
      np.linalg.norm(sum(qq * g.plane_proj(np.array([0, 0.6, 0.8])) for qq in (0.3, 0.7)) - g.plane_proj(
          np.array([0, 0.6, 0.8]))) < 1e-12)

# ---------------------------------------------------------------- Example 50.3.7 and Exercise 50.3.3
th = math.radians(30)
n1, n2 = np.array([0, 0, 1.0]), np.array([math.sin(th), 0, math.cos(th)])
Pbar = 0.5 * (g.plane_proj(n1) + g.plane_proj(n2))
lam, U = np.linalg.eigh(Pbar)
close("Example 50.3.7: eigenvalues of Pbar 0.067, 0.933, 1", lam, [0.067, 0.933, 1.0], 5e-4)
mvec = U[:, 0] * np.sign(U[2, 0])
close("Example 50.3.7: the merged normal is the bisector, 15 degrees", math.degrees(math.acos(mvec[2])), 15.0, 1e-9)
P = np.eye(3) - np.outer(mvec, mvec)
D = 2 * np.linalg.norm(Pbar - P)
close("Example 50.3.7: D = 0.1895 = sqrt 2 x 2 sin^2 15 deg", [D, math.sqrt(2) * 2 * math.sin(math.radians(15)) ** 2],
      [0.1895, 0.1895], 5e-5)
close("Example 50.3.7: barycentre (0.1, 0, 0) and bound (50.3.2) = 0.932",
      [0.1, 2 * (0.1 + math.sqrt(2) * math.sin(math.radians(15)))], [0.1, 0.932], 5e-4)
close("Example 50.3.7: ||P_j - P||_F = sqrt 2 sin 15 deg = 0.366",
      [np.linalg.norm(g.plane_proj(n1) - P), np.linalg.norm(g.plane_proj(n2) - P)], [0.366, 0.366], 5e-4)
for deg, d_e, p_e in ((10, 0.0215, 0.2465), (30, 0.1895, None)):
    th = math.radians(deg)
    n2 = np.array([math.sin(th), 0, math.cos(th)])
    Pbar = 0.5 * (g.plane_proj(n1) + g.plane_proj(n2))
    U = np.linalg.eigh(Pbar)[1]
    P = U[:, 1:] @ U[:, 1:].T
    D = 2 * np.linalg.norm(Pbar - P)
    close(f"Exercise 50.3.3 ({deg} deg): D = 2 sqrt 2 sin^2(theta / 2)", D, 2 * math.sqrt(2) * math.sin(th / 2) ** 2, 1e-12)
    close(f"Exercise 50.3.3 ({deg} deg): D", D, d_e, 5e-5)
    if p_e is not None:
        plane = np.linalg.norm(g.plane_proj(n1) - P) + np.linalg.norm(g.plane_proj(n2) - P)
        close("Exercise 50.3.3: plane term 2 sqrt 2 sin 5 deg = 0.2465, about 11 times D", [plane, plane / D],
              [0.2465, 11.47], 5e-3)

# ---------------------------------------------------------------- Proposition 50.3.8 and Example 50.3.9
for trial in range(30):
    atoms = [(rng.normal(size=3), rand_plane(), rng.uniform(0.1, 1)) for _ in range(9)]
    R = list(rng.choice(9, size=3, replace=False))
    keep = [k for k in range(9) if k not in R]
    a = {i: min(keep, key=lambda k: np.linalg.norm(atoms[k][0] - atoms[i][0])) for i in R}
    q = {k: atoms[k][2] for k in keep}
    for i in R:
        q[a[i]] += atoms[i][2]
    after = [(atoms[k][0], atoms[k][1], q[k]) for k in keep]
    check(f"Proposition 50.3.8 (trial {trial}): total mass kept",
          abs(sum(x[2] for x in after) - sum(x[2] for x in atoms)) < 1e-12)
    dm = sum(x[2] * x[0] for x in after) - sum(x[2] * x[0] for x in atoms)
    close(f"Proposition 50.3.8 (trial {trial}): first-moment change (19)", dm,
          sum(atoms[i][2] * (atoms[a[i]][0] - atoms[i][0]) for i in R), 1e-12)
    cost = sum(atoms[i][2] * np.linalg.norm(atoms[i][0] - atoms[a[i]][0]) for i in R)
    check(f"Proposition 50.3.8 (trial {trial}): bound (10)", np.linalg.norm(dm) <= cost + 1e-12)
    bl = cost + sum(atoms[i][2] * np.linalg.norm(atoms[i][1] - atoms[a[i]][1]) for i in R)
    check(f"Proposition 50.3.8 (trial {trial}): (50.3.8) for all test functions",
          max(abs(V(after, phi) - V(atoms, phi)) for phi in tests) <= bl + 1e-12)
e1 = np.array([1.0, 0, 0])
dmom = 0.05 * (0 - 0.1) + 0.05 * (1 - 0.9)
close("Example 50.3.9: first-moment change 0, bound 0.01, deletion 0.1", [dmom, 0.05 * 0.1 * 2, 0.1], [0, 0.01, 0.1], 1e-15)

# ---------------------------------------------------------------- Example 50.3.10 / Table 50.3.1 / Figure 50.3.2
expect = {"split, north": (0.847, 0.0260, 0.0422), "clone, north": (3.472, 0.0114, 0.1494),
          "mixed, everywhere": (1.799, 0.0249, 0.0712)}
for name, p, reg in g.SCHEDULES:
    h, natoms = g.schedule_run(p, reg)
    a_e, l_e, o_e = expect[name]
    check(f"Table 50.3.1 ({name}): 428 atoms after 300 steps", natoms == 428)
    check(f"Table 50.3.1 ({name}): ledger mass stays 4 pi", np.allclose(h[:, 1], 1.0, atol=1e-12))
    close(f"Table 50.3.1 ({name}): opacity area / 4 pi", round(h[-1, 2], 3), a_e, 1e-9)
    close(f"Table 50.3.1 ({name}): MMD ledger, opacity", [round(h[-1, 3], 4), round(h[-1, 4], 4)], [l_e, o_e], 1e-9)
    close(f"Table 50.3.1 ({name}): start MMD 0.0114", round(h[0, 3], 4), 0.0114, 1e-9)
    if name == "clone, north":
        check("Example 50.3.10: the ledger clone leaves the varifold unchanged", np.allclose(h[:, 3], h[0, 3], atol=1e-12))
        close("Example 50.3.10: opacity / ledger after clone, about 13", h[-1, 4] / h[-1, 3], 13.1, 0.05)
    if name == "mixed, everywhere":
        close("Example 50.3.10: mixed opacity MMD peaks at 0.0790", round(h[:, 4].max(), 4), 0.0790, 1e-9)

# ---------------------------------------------------------------- Implementation 50.3.1
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-50-3-1")
check("Implementation 50.3.1: at most 30 lines", dgcode.code_lines(PAGE, "code-50-3-1") <= 30)
check("Implementation 50.3.1: printed output equals the page",
      buf.getvalue().strip() == dgcode.code_block(PAGE, "out-50-3-1").strip())
close("Implementation 50.3.1: defect and transport cost", [ns["D"], ns["cost"]], [0.1895, 0.01], 5e-5)

# ---------------------------------------------------------------- exercises
for k in (2, 3, 5):
    close(f"Exercise 50.3.1 (k = {k}): opacity area x k, ledger x 1", [k * 0.7 * 0.3 * 0.2 / (0.7 * 0.3 * 0.2), k * (1 / k)],
          [k, 1], 1e-12)
close("Exercise 50.3.2: d = 3, transport bound 0.1 against deletion 0.05", [0.05 * min(2, 3), 0.05], [0.1, 0.05], 1e-15)
d = t1 - t2
for w, e in ((0.5, 0.0960), (0.2, 0.0615)):
    tb = w * t1 + (1 - w) * t2
    direct = w * np.linalg.norm(t1 - tb) + (1 - w) * np.linalg.norm(t2 - tb)
    close(f"Exercise 50.3.4 (w = {w}): bound = 2 w (1 - w) |d|", [direct, 2 * w * (1 - w) * np.linalg.norm(d)], [e, e], 5e-5)
close("Exercise 50.3.4: |d| = 0.1921", np.linalg.norm(d), 0.1921, 5e-5)

summary()
