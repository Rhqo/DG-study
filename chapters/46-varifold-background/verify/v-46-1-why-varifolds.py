"""Verification for Section 46.1 (Why Varifolds?).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-1-why-varifolds.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Figure 46.1.1 bookkeeping
close("Figure 46.1.1: mesh edge length 2 sin(pi/18) = 0.3473", 2 * np.sin(np.pi / 18), 0.3473, 5e-5)
u = (np.arange(9) + 0.5) / 9
tp = 2 * np.pi / 3 + (2 * np.pi / 3) * (u + 0.08 * np.sin(2 * np.pi * u))
edges = np.concatenate([[2 * np.pi / 3], 0.5 * (tp[1:] + tp[:-1]), [4 * np.pi / 3]])
close("Figure 46.1.1: the point weights add up to the arc length 2 pi / 3", np.diff(edges).sum(), 2 * np.pi / 3, 1e-12)
cell = 2 * np.pi / 15
close("Figure 46.1.1: Gaussian long standard deviation (cell/2)/1.2 = 0.175", cell / 2 / 1.2, 0.175, 5e-4)
wp = np.diff(edges)
k = int(np.argmin(wp))
check("Figure 46.1.1: the smallest point weight sits at t = pi (densest sampling)", abs(tp[k] - np.pi) < 1e-12)


# ---------------------------------------------------------------- Example 46.1.1: one circle, three lists
def cos2(P):
    """phi(x, P) = cos^2(theta_P) = P_11 for a line P (projection matrix)."""
    return P[0, 0]


close("Example 46.1.1: smooth V(1) = 2 pi", v.integrate_1d(lambda t: np.ones_like(t), 0, 2 * np.pi), 2 * np.pi, 1e-12)
close("Example 46.1.1: smooth V(phi) = int sin^2 = pi",
      v.integrate_1d(lambda t: np.array([cos2(v.dir_proj(v.circle_tangent(s))) for s in np.atleast_1d(t)]), 0, 2 * np.pi),
      np.pi, 1e-10)
for N, (m1, mphi) in {6: (6.0, 3.0), 24: (6.2653, 3.1326)}.items():
    t = 2 * np.pi * np.arange(N + 1) / N
    X = v.circle(t)
    seg = np.diff(X, axis=0)
    L = np.linalg.norm(seg, axis=1)
    Vphi = sum(l * cos2(v.dir_proj(s)) for l, s in zip(L, seg))
    close(f"Example 46.1.1: {N}-gon V(1) = 2N sin(pi/N)", L.sum(), 2 * N * np.sin(np.pi / N), 1e-12)
    close(f"Example 46.1.1: {N}-gon V(phi) = N sin(pi/N)", Vphi, N * np.sin(np.pi / N), 1e-12)
    close(f"Example 46.1.1: {N}-gon V(1) = {m1}", L.sum(), m1, 5e-5)
    close(f"Example 46.1.1: {N}-gon V(phi) = {mphi}", Vphi, mphi, 5e-5)
for N in (3, 4, 7, 50):
    t = 2 * np.pi * np.arange(N) / N
    Vphi = sum(2 * np.pi / N * cos2(v.dir_proj(v.circle_tangent(s))) for s in t)
    close(f"Example 46.1.1: {N} equally spaced points give V(phi) = pi exactly", Vphi, np.pi, 1e-12)

# ---------------------------------------------------------------- Example 46.1.2: the staircase
for n in (1, 2, 4, 8, 16, 32, 64):
    S = v.staircase(n)
    close(f"Example 46.1.2: length(S_{n}) = 2", v.polyline_length(S, closed=False), 2.0, 1e-12)
for n in (1, 2, 4, 8, 16):
    S = v.staircase(n)
    close(f"Example 46.1.2: d_H(S_{n}, D) = 1/(sqrt2 n)", v.hausdorff_polyline_to_segment(S, (0, 0), (1, 1)),
          1 / (np.sqrt(2) * n), 1e-6)
close("Example 46.1.2: d_H(S_16, D) = 0.0442", 1 / (np.sqrt(2) * 16), 0.0442, 5e-5)
close("Figure 46.1.2: d_H(S_1, D) = 0.707", 1 / np.sqrt(2), 0.707, 5e-4)
close("Figure 46.1.2: d_H(S_64, D) = 0.0110", 1 / (np.sqrt(2) * 64), 0.0110, 5e-5)
close("Example 46.1.2: length(D) = sqrt2 = 1.414", np.hypot(1, 1), 1.414, 5e-4)
# points of D are closer to S_n than 1/(sqrt2 n): sample D and measure the distance to S_n
for n in (3, 8):
    S = v.staircase(n)
    tt = np.linspace(0, 1, 2001)
    dmax = 0.0
    for t in tt:
        p = np.array([t, t])
        d = min(np.linalg.norm(p - (a + np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1) * (b - a)))
                for a, b in zip(S[:-1], S[1:]))
        dmax = max(dmax, d)
    check(f"Example 46.1.2: sup over D of dist(., S_{n}) = 1/(2n) < 1/(sqrt2 n)",
          abs(dmax - 1 / (2 * n)) < 1e-3 and dmax < 1 / (np.sqrt(2) * n))


def turning_angles(Vtx):
    T = np.diff(Vtx, axis=0)
    ang = np.arctan2(T[:, 1], T[:, 0])
    d = np.diff(ang)
    return (d + np.pi) % (2 * np.pi) - np.pi


for n in (1, 3, 8, 16):
    th = turning_angles(v.staircase(n))
    check(f"Example 46.1.2: S_{n} has 2n - 1 corners, each turning by +-pi/2",
          len(th) == 2 * n - 1 and np.allclose(np.abs(th), np.pi / 2))
    close(f"Example 46.1.2: total |turning| of S_{n} = (2n - 1) pi / 2", np.sum(np.abs(th)), (2 * n - 1) * np.pi / 2, 1e-12)
    T = np.diff(v.staircase(n), axis=0)
    ang = np.degrees(np.arctan2(T[:, 1], T[:, 0])) % 180
    check(f"Example 46.1.2: tangents of S_{n} are 0 or 90 degrees (45 degrees from D)",
          np.all(np.isclose(ang, 0) | np.isclose(ang, 90)))
close("Example 46.1.2: total |turning| of S_16 = 48.69", 31 * np.pi / 2, 48.69, 5e-3)

# ---------------------------------------------------------------- Exercise 46.1.1: any monotone staircase
rng = np.random.default_rng(1)
for trial in range(5):
    a, b = rng.uniform(0.5, 2.0, 2)
    k = rng.integers(3, 30)
    xs = np.sort(np.concatenate([[0, a], rng.uniform(0, a, k - 1)]))
    ys = np.sort(np.concatenate([[0, b], rng.uniform(0, b, k - 1)]))
    pts = [(xs[0], ys[0])]
    for i in range(1, len(xs)):
        pts.append((xs[i], ys[i - 1]))
        pts.append((xs[i], ys[i]))
    L = v.polyline_length(np.array(pts), closed=False)
    close(f"Exercise 46.1.1: monotone staircase to ({a:.2f}, {b:.2f}) has length a + b", L, a + b, 1e-12)
th = np.linspace(0, np.pi / 2, 1001)
r = np.cos(th) + np.sin(th)
check("Exercise 46.1.1: cos + sin on [0, pi/2] has max sqrt2 at pi/4 and min 1 at the ends",
      abs(r.max() - np.sqrt(2)) < 1e-6 and abs(th[np.argmax(r)] - np.pi / 4) < 2e-3 and abs(r.min() - 1) < 1e-12)

# 3D version (In computer vision box): a monotone height-field staircase over [0,1]^2 approximating z = a x + b y
for a, b, n in ((0.7, 0.4, 200), (1.0, 1.0, 300)):
    c = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(c, c, indexing="ij")
    h = np.floor(n * (a * Xc + b * Yc)) / n                  # voxel columns of size 1/n
    tops = 1.0                                                # horizontal faces project onto [0,1]^2
    risers = (np.abs(np.diff(h, axis=0)).sum() + np.abs(np.diff(h, axis=1)).sum()) / n
    area_stair = tops + risers
    nrm = np.array([-a, -b, 1.0]) / np.sqrt(1 + a * a + b * b)
    area_plane = np.sqrt(1 + a * a + b * b)
    close(f"CV box: voxel staircase of z = {a} x + {b} y has area ratio |n1|+|n2|+|n3|",
          area_stair / area_plane, np.abs(nrm).sum(), 3.0 / n)
close("CV box: max of |n1|+|n2|+|n3| over unit n is sqrt3", np.abs(np.ones(3) / np.sqrt(3)).sum(), np.sqrt(3), 1e-12)

# ---------------------------------------------------------------- Exercise 46.1.2: zigzag over [0, 1]
for n in (1, 5, 20):
    xs = np.linspace(0, 1, 2 * n + 1)
    ys = np.where(np.arange(2 * n + 1) % 2 == 0, 0.0, 1.0 / (2 * n))
    Z = np.stack([xs, ys], 1)
    close(f"Exercise 46.1.2: zigzag Z_{n} has length sqrt2", v.polyline_length(Z, closed=False), np.sqrt(2), 1e-12)
    close(f"Exercise 46.1.2: d_H(Z_{n}, [0,1]) = 1/(2n)", v.hausdorff_polyline_to_segment(Z, (0, 0), (1, 0)),
          1 / (2 * n), 1e-6)
    T = np.diff(Z, axis=0)
    ang = np.degrees(np.arctan2(T[:, 1], T[:, 0]))
    check(f"Exercise 46.1.2: Z_{n} tangents alternate +45 / -45 degrees, half of the length each",
          np.allclose(ang[0::2], 45) and np.allclose(ang[1::2], -45))

# ---------------------------------------------------------------- Exercise 46.1.3: the square and the rectangle
for N in (1, 5, 40):
    s = (np.arange(N) + 0.5) / N
    items = []
    for x in s:
        items += [((x, 0), v.P_H, 1 / N), ((x, 1), v.P_H, 1 / N)]
    for y in s:
        items += [((0, y), v.P_V, 1 / N), ((1, y), v.P_V, 1 / N)]
    V1 = sum(m for _, _, m in items)
    Vp = sum(m * cos2(P) for _, P, m in items)
    close(f"Exercise 46.1.3: square, N = {N}: V(1) = 4", V1, 4.0, 1e-12)
    close(f"Exercise 46.1.3: square, N = {N}: V(phi) = 2", Vp, 2.0, 1e-12)
for a, b in ((3.0, 1.0), (1.0, 0.1)):
    close(f"Exercise 46.1.3: rectangle {a} x {b}: ratio a/(a+b)", (2 * a) / (2 * a + 2 * b), a / (a + b), 1e-15)

# ---------------------------------------------------------------- Exercise 46.1.4: the flattening ellipsoid
gx, gw = np.polynomial.legendre.leggauss(2000)


def pancake(eps):
    tot = np.zeros(5)
    for lo, hi in ((0, np.pi / 2), (np.pi / 2, np.pi)):
        uu = (hi - lo) / 2 * gx + (lo + hi) / 2
        ww = (hi - lo) / 2 * gw
        J = np.sin(uu) * np.sqrt(eps ** 2 * np.sin(uu) ** 2 + np.cos(uu) ** 2)   # |x_u x x_v|
        Nz = np.sin(uu) * np.cos(uu) / J
        Nr = eps * np.sin(uu) ** 2 / J                                         # radial component of N
        # integrate over v in [0, 2pi]: cos^2 v and sin^2 v average to 1/2, cos v averages to 0
        tot += np.array([2 * np.pi * np.sum(ww * J), 2 * np.pi * np.sum(ww * J * Nz ** 2),
                         np.pi * np.sum(ww * J * Nr ** 2), 2 * np.pi * np.sum(ww * J * Nz), 0.0])
    return tot


for eps, (A, nzz, nxx) in {0.1: (6.4722, 6.1557, 0.1582), 0.01: (6.2865, 6.2805, 0.0030)}.items():
    tot = pancake(eps)
    close(f"Exercise 46.1.4: eps = {eps}: area = {A}", tot[0], A, 5e-5)
    close(f"Exercise 46.1.4: eps = {eps}: (3,3) entry of int N N^T = {nzz}", tot[1], nzz, 5e-5)
    close(f"Exercise 46.1.4: eps = {eps}: (1,1) entry of int N N^T = {nxx}", tot[2], nxx, 5e-5)
    close(f"Exercise 46.1.4: eps = {eps}: trace of int N N^T = area", tot[1] + 2 * tot[2], tot[0], 1e-9)
    close(f"Exercise 46.1.4: eps = {eps}: z component of int N dA = 0", tot[3], 0.0, 1e-9)
close("Exercise 46.1.4: eps = 0.001: area -> 2 pi", pancake(1e-3)[0], 2 * np.pi, 1e-3)
# closed-form area of the oblate spheroid (a = 1, c = eps) as a cross-check
for eps in (0.3, 0.1):
    e = np.sqrt(1 - eps ** 2)
    A = 2 * np.pi * (1 + (1 - e ** 2) / e * np.arctanh(e))
    close(f"Exercise 46.1.4: oblate spheroid area formula, eps = {eps}", pancake(eps)[0], A, 1e-8)

summary()
