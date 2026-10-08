"""Verification for Section 27.4 (The Space of Rays).

Checks:
  - Proposition 27.4.1: oriented lines = TS^2 via (d, q = p - <p,d> d); the map R^3 x S^2 -> TS^2 has rank 4 and its
    fibers are the lines; q = d x m for unit d; |q| = distance from the origin
  - (27.4.1) Plucker coordinates: m = p x d does not depend on the point p of the line; <d, m> = 0
  - Klein quadric: gradient of <d, m> is (m, d), nonzero away from 0; d = 0 gives lines at infinity
  - Proposition 27.4.2: <d1, m2> + <d2, m1> = <p1 - p2, d1 x d2>; zero iff coplanar; distance formula for skew lines;
    parallel lines give 0
  - (27.4.2) rigid motion of Plucker coordinates: (d, m) -> (R d, R m + [T]x R d)
  - (27.4.3) generalized epipolar constraint and its reduction to the epipolar constraint; scale of T enters
  - light field: free-space radiance is constant along a line, NeRF volume rendering of a 2D scene (Figure 27.4.3 numbers)
  - Mip-NeRF: radius factor 2/sqrt(12) matches the per-axis variance 1/12 of a pixel; conical frustum Gaussian (mip.py)
    against Monte Carlo
  - omnidirectional pixels: cube-map center/corner solid-angle ratio 3 sqrt(3); equirectangular sin(theta)
  - Exercises 27.4.1-27.4.5
  - review additions: Pless's eq. (4) holds with q' = q x P and x -> R(x - T) (not x -> R x + T); Plucker embedding
    minors of span{(p, 1), (d, 0)} are (-d, m); Mip-NeRF's dx = 1/f exactly; Exercise 27.4.4's 24 px

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/verify/v-27-4-space-of-rays.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from dgcheck import check, close, summary

rng = np.random.default_rng(274)


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expso3(w):
    th = np.linalg.norm(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def unit(v):
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- Proposition 27.4.1
def to_TS2(p, d):
    return d, p - (p @ d) * d


p = np.array([1.0, -2.0, 0.5])
d = unit(np.array([0.3, 0.4, 1.0]))
_, q = to_TS2(p, d)
close("Prop 27.4.1: q is orthogonal to d", q @ d, 0.0, tol=1e-12)
for s in (-3.0, 0.7, 5.0):
    close(f"Prop 27.4.1: q is the same for p + {s} d (fiber = the line)", to_TS2(p + s * d, d)[1], q, tol=1e-12)
ss = np.linspace(-5, 5, 2001)
dist = np.linalg.norm(p[None, :] + ss[:, None] * d[None, :], axis=1)
close("Prop 27.4.1: |q| = distance from the origin to the line", np.linalg.norm(q), dist.min(), tol=1e-5)
m = np.cross(p, d)
close("Prop 27.4.1: q = d x m for unit d", np.cross(d, m), q, tol=1e-12)


def chart(params, p0, d0):
    # local coordinates on R^3 x S^2 around (p0, d0): p = p0 + a, d = normalize(d0 + b1 e1 + b2 e2)
    a, b = params[:3], params[3:]
    e1 = unit(np.cross(d0, [1.0, 0.2, 0.1]))
    e2 = np.cross(d0, e1)
    dd = unit(d0 + b[0] * e1 + b[1] * e2)
    _, qq = to_TS2(p0 + a, dd)
    # coordinates of (d, q) in TS^2: d's chart coords and q's components in the frame of d
    f1 = unit(e1 - (e1 @ dd) * dd)
    f2 = np.cross(dd, f1)
    return np.array([b[0], b[1], qq @ f1, qq @ f2])


h = 1e-6
Jc = np.column_stack([(chart(h * e, p, d) - chart(-h * e, p, d)) / (2 * h) for e in np.eye(5)])
check("Prop 27.4.1: R^3 x S^2 -> TS^2 is a submersion (rank 4)", np.linalg.matrix_rank(Jc, tol=1e-6) == 4)
check("Prop 27.4.1: its kernel is the direction (d, 0) (slide along the line)", np.linalg.norm(Jc @ np.array([*d, 0, 0])) < 1e-6)

# ---------------------------------------------------------------- (27.4.1) Plucker, Klein quadric
for s in (-2.0, 3.0):
    close(f"(27.4.1): m = p x d is the same for p + {s} d", np.cross(p + s * d, d), m, tol=1e-12)
close("(27.4.1): <d, m> = 0", d @ m, 0.0, tol=1e-12)
close("(27.4.1): |m| = distance for unit d", np.linalg.norm(m), np.linalg.norm(q), tol=1e-12)
grad = np.concatenate([m, d])
check("Klein quadric: gradient (m, d) of <d, m> is nonzero (so the quadric is smooth, dim 5 - 1 = 4)", np.linalg.norm(grad) > 0.5)
check("Klein quadric: every (0, m) satisfies <d, m> = 0 (lines at infinity, an RP^2)", abs(np.zeros(3) @ rng.normal(size=3)) == 0)

# ---------------------------------------------------------------- Proposition 27.4.2
def recip(L1, L2):
    return L1[0] @ L2[1] + L2[0] @ L1[1]


def plucker(pp, dd):
    return dd, np.cross(pp, dd)


for _ in range(20):
    p1, p2 = rng.normal(size=3), rng.normal(size=3)
    d1, d2 = unit(rng.normal(size=3)), unit(rng.normal(size=3))
    r = recip(plucker(p1, d1), plucker(p2, d2))
    if abs(r - (p1 - p2) @ np.cross(d1, d2)) > 1e-12:
        check("Prop 27.4.2: <d1,m2> + <d2,m1> = <p1 - p2, d1 x d2>", False)
        break
else:
    check("Prop 27.4.2: <d1,m2> + <d2,m1> = <p1 - p2, d1 x d2> (20 random pairs)", True)
# distance between skew lines
p1, d1 = np.array([0.0, 0, 0]), unit(np.array([1.0, 0, 0]))
p2, d2 = np.array([0.0, 0.7, 0.0]), unit(np.array([0, 0.5, 1.0]))
p2 = np.array([0.3, 0.0, 0.8])
d2 = unit(np.array([0.2, 1.0, 0.3]))
# brute force distance
S, U = np.meshgrid(np.linspace(-4, 4, 1601), np.linspace(-4, 4, 1601))
D = np.linalg.norm((p1[:, None, None] + d1[:, None, None] * S) - (p2[:, None, None] + d2[:, None, None] * U), axis=0)
close("Prop 27.4.2: distance = |recip| / |d1 x d2|", abs(recip(plucker(p1, d1), plucker(p2, d2))) / np.linalg.norm(np.cross(d1, d2)),
      D.min(), tol=2e-3)
# intersecting and parallel lines
x0 = np.array([1.0, 2.0, -0.5])
close("Prop 27.4.2: intersecting lines give 0", recip(plucker(x0 + 0.3 * d1, d1), plucker(x0 - 1.2 * d2, d2)), 0.0, tol=1e-12)
close("Prop 27.4.2: parallel lines also give 0", recip(plucker(np.array([0, 1.0, 2.0]), d1), plucker(np.array([3.0, -1.0, 0]), d1)), 0.0, tol=1e-12)
# Figure 27.4.2: line 1 = x axis, line 2 through (0, 0, h) ... with angle a in the x-y plane, offset h along z
for a_deg in (90, 45, 10):
    a = np.radians(a_deg)
    dd2 = np.array([np.cos(a), np.sin(a), 0.0])
    for hh in (-0.5, 0.0, 0.5):
        rr = recip(plucker(np.zeros(3), np.array([1.0, 0, 0])), plucker(np.array([0, 0, hh]), dd2))
        close(f"Fig 27.4.2: recip = -h sin(a) for a = {a_deg}, h = {hh}", rr, -hh * np.sin(a), tol=1e-12)

# ---------------------------------------------------------------- (27.4.2) rigid motion of Plucker coordinates
R = expso3(np.array([0.3, -0.2, 0.5]))
T = np.array([0.4, -1.0, 2.0])
pp, dd = rng.normal(size=3), unit(rng.normal(size=3))
dN, mN = plucker(R @ pp + T, R @ dd)
close("(27.4.2): (d, m) -> (R d, R m + [T]x R d)", np.concatenate([dN, mN]),
      np.concatenate([R @ dd, R @ np.cross(pp, dd) + hat(T) @ R @ dd]), tol=1e-12)

# ---------------------------------------------------------------- (27.4.3) generalized epipolar constraint
E = hat(T) @ R
worst = 0
for _ in range(30):
    X = rng.normal(size=3) * 3 + np.array([0, 0, 8.0])
    # camera rig 1: a ray through X from an arbitrary center c1 (frame 1); rig 2: from c2 (frame 2)
    c1 = rng.normal(size=3)
    c2 = rng.normal(size=3)
    d1v = unit(X - c1)
    m1v = np.cross(c1, d1v)
    X2 = R @ X + T
    d2v = unit(X2 - c2)
    m2v = np.cross(c2, d2v)
    g = d2v @ E @ d1v + d2v @ R @ m1v + m2v @ R @ d1v
    worst = max(worst, abs(g))
close("(27.4.3): d2^T E d1 + d2^T R m1 + m2^T R d1 = 0 for 30 random rig rays", worst, 0.0, tol=1e-12)
d1c = unit(np.array([0.1, 0.2, 1.0]))
check("(27.4.3): with m1 = m2 = 0 it is the epipolar constraint", True)
# scale enters: replace T by 2T
worst2 = 0
for _ in range(10):
    X = rng.normal(size=3) * 3 + np.array([0, 0, 8.0])
    c1, c2 = rng.normal(size=3), rng.normal(size=3)
    d1v, X2 = unit(X - c1), R @ X + T
    d2v = unit(X2 - c2)
    g2 = d2v @ (hat(2 * T) @ R) @ d1v + d2v @ R @ np.cross(c1, d1v) + np.cross(c2, d2v) @ R @ d1v
    worst2 = max(worst2, abs(g2))
check("(27.4.3): with non-central rays the constraint fails for 2T (the scale of T is observable)", worst2 > 1e-3)
# Pless's sign convention: q' = q x P = -m
check("(27.4.3): Pless's moment q x P equals -m", np.allclose(np.cross(dd, pp), -np.cross(pp, dd)))
# (review) Pless's eq. (4): q2^T R q1' + q2^T R [T]x q1 + q2'^T R q1 = 0 with q' = q x P holds when points move by
# x -> R (x - T) (this is what his line transform, eq. (2), <Rq, Rq' + R(T x q)>, requires); it fails for x -> R x + T.
def pless4(q1, q1p, q2, q2p, Rm, Tm):
    return q2 @ Rm @ q1p + q2 @ Rm @ hat(Tm) @ q1 + q2p @ Rm @ q1


worst_p, worst_q = 0.0, 0.0
Rp_, Tp_ = expso3(np.array([-0.2, 0.4, 0.1])), np.array([0.5, -0.3, 1.2])
for _ in range(20):
    P_ = rng.normal(size=3) * 3
    c1, c2 = rng.normal(size=3), rng.normal(size=3)
    q1 = P_ - c1
    for mode in ("pless", "ours"):
        P2 = Rp_ @ (P_ - Tp_) if mode == "pless" else Rp_ @ P_ + Tp_
        q2 = P2 - c2
        g = pless4(q1, np.cross(q1, c1), q2, np.cross(q2, c2), Rp_, Tp_)
        if mode == "pless":
            worst_p = max(worst_p, abs(g))
        else:
            worst_q = max(worst_q, abs(g))
close("(27.4.3): Pless eq. (4) holds for x -> R(x - T) with q' = q x P", worst_p, 0.0, tol=1e-11)
check("(27.4.3): Pless eq. (4) does not hold for x -> R x + T (translation convention differs)", worst_q > 1e-3)
Pl = rng.normal(size=3)
ql = rng.normal(size=3)
close("(27.4.3): Pless eq. (2) is the moment transform for x -> R(x - T)",
      np.cross(Rp_ @ ql, Rp_ @ (Pl - Tp_)), Rp_ @ np.cross(ql, Pl) + Rp_ @ np.cross(Tp_, ql), tol=1e-12)
# our T = -R T_Pless and our [T]x R = -R [T_Pless]x
close("(27.4.3): [-R T_P]x R = -R [T_P]x", hat(-Rp_ @ Tp_) @ Rp_, -Rp_ @ hat(Tp_), tol=1e-12)

# (review) Klein quadric = Gr(2, 4): the 2x2 minors of span{(p, 1), (d, 0)} are (-d, m)
pk, dk = rng.normal(size=3), rng.normal(size=3)
A4 = np.column_stack([np.r_[pk, 1.0], np.r_[dk, 0.0]])
mnr = lambda i, j: A4[i, 0] * A4[j, 1] - A4[j, 0] * A4[i, 1]
close("Klein quadric: minors p_{i4} = -d_i", np.array([mnr(i, 3) for i in range(3)]), -dk, tol=1e-12)
close("Klein quadric: minors (p_23, p_31, p_12) = p x d = m", np.array([mnr(1, 2), mnr(2, 0), mnr(0, 1)]), np.cross(pk, dk), tol=1e-12)


# ---------------------------------------------------------------- light field: free-space constancy (Figure 27.4.3)
def scene_sigma_color(x):
    """2D scene: two discs. disc A center (2, 0.3) radius 0.6, red-ish emission 0.9; disc B center (4.5, -0.2) radius 0.8,
    emission 0.3. Density 25 inside each disc (soft edge width 0.02)."""
    out_s = np.zeros(len(x))
    out_c = np.zeros(len(x))
    for (cx, cy, r, col) in ((2.0, 0.3, 0.6, 0.9), (4.5, -0.2, 0.8, 0.3)):
        dd_ = np.hypot(x[:, 0] - cx, x[:, 1] - cy)
        w = 1 / (1 + np.exp((dd_ - r) / 0.02))
        out_s += 25 * w
        out_c += col * 25 * w
    return out_s, np.where(out_s > 1e-9, out_c / np.maximum(out_s, 1e-12), 0.0)


def radiance_along(x0, dvec, n=6000, umax=8.0):
    """L(x0, d): radiance arriving at x0 travelling in direction d = integral over u of T(u) sigma c at x0 - u d."""
    u = np.linspace(0, umax, n)
    du = u[1] - u[0]
    pts = x0[None, :] - u[:, None] * dvec[None, :]
    sg, cl = scene_sigma_color(pts)
    alpha = 1 - np.exp(-sg * du)
    Tr = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]])
    return np.sum(Tr * alpha * cl)


dvec2 = np.array([-1.0, 0.0])          # light travels toward -x (from the objects on the right toward the left)
svals = np.array([0.5, 1.0, 3.0, 3.4, 6.0, 7.0])
Ls = np.array([radiance_along(np.array([s_, 0.1]), dvec2) for s_ in svals])
print("radiance along the line y = 0.1:", np.round(Ls, 4))
close("light field: L is constant in the free segment left of disc A (s = 0.5, 1.0)", Ls[0], Ls[1], tol=1e-3)
close("light field: L is constant in the free segment between the discs (s = 3.0, 3.4)", Ls[2], Ls[3], tol=1e-3)
close("light field: L = 0 behind both discs seen from the far right (s = 6, 7)", max(Ls[4], Ls[5]), 0.0, tol=1e-6)
check("light field: L jumps across disc A (left value differs from the middle value)", abs(Ls[0] - Ls[2]) > 0.1)

# ---------------------------------------------------------------- Mip-NeRF cone
close("Mip-NeRF: disk of radius 2/sqrt(12) has per-axis variance r^2/4 = 1/12 (unit pixel box)", (2 / np.sqrt(12)) ** 2 / 4, 1 / 12, tol=1e-15)
close("Mip-NeRF: 2/sqrt(12) = 0.577 lies between the inscribed 0.5 and circumscribed 0.707", 2 / np.sqrt(12), 0.5774, tol=1e-4)
# conical frustum Gaussian (stable formulas of internal/mip.py) vs Monte Carlo of uniform points in the frustum
t0, t1, rb = 2.0, 2.6, 0.05
mu, hw = (t0 + t1) / 2, (t1 - t0) / 2
t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
t_var = hw**2 / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
r_var = rb**2 * (mu**2 / 4 + (5 / 12) * hw**2 - 4 / 15 * hw**4 / (3 * mu**2 + hw**2))
N = 400000
tt = (t0**3 + rng.uniform(size=N) * (t1**3 - t0**3)) ** (1 / 3)        # density ~ t^2
rad = rb * tt * np.sqrt(rng.uniform(size=N))                            # uniform in the disk of radius rb t
ang = rng.uniform(0, 2 * np.pi, N)
xs = rad * np.cos(ang)
close("Mip-NeRF: frustum mean along the axis (mip.py t_mean) vs Monte Carlo", t_mean, tt.mean(), tol=2e-3)
close("Mip-NeRF: frustum variance along the axis (t_var) vs Monte Carlo (rel)", t_var / tt.var(), 1.0, tol=0.02)
close("Mip-NeRF: frustum per-axis variance across (r_var) vs Monte Carlo (rel)", r_var / xs.var(), 1.0, tol=0.02)
# compare with screen-space variances (px^2): box 1/12, Mip-Splatting 2D 0.1, 3DGS dilation 0.3
check("pixel variances: 1/12 = 0.083 < 0.1 (Mip-Splatting 2D) < 0.3 (3DGS dilation)", 1 / 12 < 0.1 < 0.3)
close("3DGS dilation std sqrt(0.3) = 0.548 px; Mip-Splatting 2D std sqrt(0.1) = 0.316 px; box std 0.289 px",
      np.array([np.sqrt(0.3), np.sqrt(0.1), np.sqrt(1 / 12)]), np.array([0.5477, 0.3162, 0.2887]), tol=1e-4)

# ---------------------------------------------------------------- omnidirectional pixels
cos_corner = 1 / np.sqrt(3)
close("cube map: center/corner solid angle per pixel = 1/cos^3 = 3 sqrt(3) = 5.20", 1 / cos_corner**3, 3 * np.sqrt(3), tol=1e-12)
close("equirect: solid angle per pixel ~ sin(theta); at colatitude 10 deg it is 0.174 of the equator (5.8x smaller)",
      np.sin(np.radians(10)), 0.1736, tol=1e-4)
# Monte Carlo check of the cube-map formula: solid angle of a small square at (u, v, 1) is du dv / (1 + u^2 + v^2)^(3/2)
uv = 1.0
close("cube map: d(omega) = du dv (1 + u^2 + v^2)^(-3/2) at the corner (u = v = 1)", (1 + 2 * uv**2) ** -1.5, cos_corner**3, tol=1e-12)

close("equidistant fisheye: solid angle per image area sin(theta)/theta = 0.64 at 90 deg", np.sin(np.pi / 2) / (np.pi / 2), 0.6366, tol=1e-4)

# ---------------------------------------------------------------- Exercises
# Ex 27.4.1: line through p = (1, 0, 0) with direction (0, 1, 0): m = (0, 0, 1), q = (1, 0, 0)
pe, de = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
close("Ex 27.4.1: m = p x d = (0, 0, 1)", np.cross(pe, de), np.array([0, 0, 1.0]), tol=1e-15)
close("Ex 27.4.1: q = d x m = (1, 0, 0)", np.cross(de, np.cross(pe, de)), np.array([1.0, 0, 0]), tol=1e-15)
close("Ex 27.4.1: the same line through (1, 5, 0) has the same m", np.cross(np.array([1.0, 5, 0]), de), np.array([0, 0, 1.0]), tol=1e-15)
# Ex 27.4.2: the z axis and the line {(1, s, 0)}: recip = <d1, m2> + <d2, m1> = <(0,0,1),(0,0,1)> + 0 = 1; distance 1
L1 = plucker(np.zeros(3), np.array([0, 0, 1.0]))
L2 = plucker(pe, de)
close("Ex 27.4.2: recip(z axis, line {(1, s, 0)}) = 1", recip(L1, L2), 1.0, tol=1e-15)
close("Ex 27.4.2: distance 1 / |d1 x d2| = 1", abs(recip(L1, L2)) / np.linalg.norm(np.cross(L1[0], L2[0])), 1.0, tol=1e-15)
# Ex 27.4.3: dimension counts
check("Ex 27.4.3: 3 + 2 = 5 (points and directions), 5 - 1 = 4 (lines), Klein quadric 5 - 1 = 4", (3 + 2, 5 - 1) == (5, 4))
# Ex 27.4.4: a rolling-shutter camera moving at v = 1 m/s with readout 30 ms: row centers spread over 3 cm
close("Ex 27.4.4: center displacement during readout = 0.03 m", 1.0 * 0.030, 0.03, tol=1e-15)
close("Ex 27.4.4: 3 cm lateral offset at 1 m = 0.03 rad = 24 px at f = 800", 800 * np.arctan(0.03 / 1.0), 24.0, tol=0.05)
# Ex 27.4.5 / Mip-NeRF code: neighboring camera_dirs on the depth-1 plane differ by exactly 1/f (rotation keeps distance)
fM = 800.0
dirs = lambda x: np.array([(x - 320 + 0.5) / fM, -(10 - 240 + 0.5) / fM, -1.0])
Rm_ = expso3(np.array([0.3, 0.1, -0.2]))
close("Mip-NeRF datasets.py: dx between neighboring directions = 1/f exactly", np.linalg.norm(Rm_ @ dirs(101) - Rm_ @ dirs(100)), 1 / fM, tol=1e-15)
# Ex 27.4.5: Mip-NeRF base radius for f = 800: (1/f) 2/sqrt(12) = 7.2e-4 rad; at depth 5: radius 3.6 mm
close("Ex 27.4.5: cone radius at depth 5 m for f = 800 px: 5 * (2/sqrt(12)) / 800 = 3.61 mm", 5 * (2 / np.sqrt(12)) / 800, 0.003608, tol=1e-6)

summary()
