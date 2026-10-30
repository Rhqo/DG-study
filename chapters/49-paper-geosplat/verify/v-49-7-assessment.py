"""Verification for Section 49.7 (Assessment).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-7-assessment.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import geo49  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "49-7-assessment.html"
T1, T2, T3 = geo49.TABLE1, geo49.TABLE2, geo49.TABLE3

# ---------------------------------------------------------------- Table 49.7.1: numbers quoted from Section 49.6
gm = np.array(T1["Manifold"]["PSNR"]) - np.array(T1["GeoGaussian"]["PSNR"])
im = np.array(T2["Manifold"]["PSNR"]) - np.array(T2["GeoGaussian"]["PSNR"])
check("Table 49.7.1: PSNR above GeoGaussian in all 12 scenes (both variants)",
      all(np.all(np.array(T[v]["PSNR"]) > np.array(T["GeoGaussian"]["PSNR"])) for T in (T1, T2) for v in ("Manifold", "Varifold")))
close("Table 49.7.1: mean gains +0.78 (Replica)", gm.mean(), 0.78, tol=1e-9)
close("Table 49.7.1: mean gains +0.38 (ICL)", im.mean(), 0.38, tol=1e-9)
A = np.array(T3["Manifold"])
close("Table 49.7.1 / Exercise 49.7.1: warm-up row -0.36 (R0)", A[1, 0] - A[0, 0], -0.36, tol=1e-9)
close("Table 49.7.1 / Exercise 49.7.1: warm-up row -0.18 (Office-3)", A[1, 3] - A[0, 3], -0.18, tol=1e-9)
close("Exercise 49.7.1: MAC over mean curvature +0.12 (R0)", A[0, 0] - A[3, 0], 0.12, tol=1e-9)
close("Exercise 49.7.1: MAC over mean curvature +0.03 (Office-3)", A[0, 3] - A[3, 3], 0.03, tol=1e-9)

# ---------------------------------------------------------------- Example 49.7.1 and Implementation 49.7
rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (1500, 2))
M, pairs, mult = geo49.upsample_midpoints(X, np.ones(len(X), bool), k=10)
check("Example 49.7.1: kN = 15000 new Gaussians", len(M) == 15000)
close("Example 49.7.1: 86% made by mutual pairs", (mult == 2).mean(), 0.86, tol=0.005)
nmut = (mult == 2).sum() // 2
check("Example 49.7.1: M = 6452 mutual pairs, 8548 distinct new locations (49.7.1)", nmut == 6452 and len(M) - nmut == 8548)
close("Example 49.7.1: 43% of the new Gaussians repeat another one", nmut / len(M), 0.43, tol=0.005)
# coincidence check: the distinct locations computed geometrically
Mr = np.round(M, 12)
check("Example 49.7.1: geometric count of distinct new points = 8548", len(np.unique(Mr, axis=0)) == 8548)
ns = dgcode.run_block(PAGE, "code-49-7-1")
check("Implementation 49.7: at most 30 lines", dgcode.code_lines(PAGE, "code-49-7-1") <= 30)
check("Implementation 49.7: same midpoints as the library", np.allclose(np.sort(ns["M"], axis=0), np.sort(M, axis=0)))
close("Implementation 49.7: mass ratio 11", ns["mass_ratio"], 11.0, tol=1e-12)
check("Implementation 49.7: distinct = 8548", ns["distinct"] == 8548)
close("Implementation 49.7: duplicate fraction 0.43", ns["dup_fraction"], 0.4301, tol=1e-4)
g = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 161), np.linspace(0.1, 0.9, 161)), -1).reshape(-1, 2)


def empty_radius(P):
    return np.sqrt(((g[:, None, :] - P[None]) ** 2).sum(-1).min(1))


e0, e1 = empty_radius(X), empty_radius(np.vstack([X, M]))
close("Example 49.7.1: median empty radius before 0.0123", np.median(e0), 0.0123, tol=0.00005)
close("Example 49.7.1: median empty radius after 0.0055", np.median(e1), 0.0055, tol=0.00005)
close("Example 49.7.1: largest empty radius before 0.0432", e0.max(), 0.0432, tol=0.00005)
close("Example 49.7.1: largest empty radius after 0.0429", e1.max(), 0.0429, tol=0.00005)
# Figure 49.7.1 (b): the strip
S = np.c_[rng.uniform(-1, 1, 2500), rng.uniform(0, 0.6, 2500)]
Ms, ps, ms = geo49.upsample_midpoints(S, S[:, 0] < 0, k=10)
left = (S[:, 0] < -0.3).sum()
ratio_left = (left + (Ms[:, 0] < -0.3).sum()) / left
check("Figure 49.7.1 (b): about 11 times as many Gaussians where the test passes", 10.0 < ratio_left < 12.0)
right = (S[:, 0] > 0.3).sum()
check("Figure 49.7.1 (b): unchanged where it fails (away from the boundary)", (Ms[:, 0] > 0.3).sum() < 0.05 * right)

# ---------------------------------------------------------------- Exercise 49.7.3: splitting the parent's weight over itself and k midpoints
k = 10
ok_mass, ok_moment = True, True
for _ in range(20):
    mu = rng.normal(size=3)
    nb = mu + rng.normal(size=(k, 3))
    q = rng.uniform(0.5, 2.0)
    pts = np.vstack([mu, 0.5 * (mu + nb)])
    w = np.full(k + 1, q / (k + 1))
    ok_mass &= abs(w.sum() - q) < 1e-12
    pred = q * mu + q / (k + 1) * k / 2 * (nb.mean(0) - mu)
    ok_moment &= np.allclose(w @ pts, pred)
    # moving the parent by -(k/2)(mean - mu) restores the first moment
    pts2 = pts.copy()
    pts2[0] = mu - k / 2 * (nb.mean(0) - mu)
    ok_moment &= np.allclose(w @ pts2, q * mu)
check("Exercise 49.7.3: the weights q/(k+1) conserve the mass", ok_mass)
check("Exercise 49.7.3: first moment = q mu + q k/(2(k+1)) (mean - mu); fixed by moving the parent", ok_moment)
close("Exercise 49.7.3: barycentre moves 45% of the way to the neighbours' mean (k = 10)", k / (2 * (k + 1)), 0.4545, tol=1e-4)

# ---------------------------------------------------------------- Exercise 49.7.4 / Table 49.7.2: three errata with numbers
Xs = geo49.sphere_points(20000, seed=7)
m = np.ones(len(Xs))
i0 = 0
eps = 0.2
same = np.tile(Xs[i0], (len(Xs), 1))                  # the printed V~^j uses u_3^i for every j
B_typo = geo49.geosplat_B(Xs, same, m, i0, eps)
check("Exercise 49.7.4 (a): printed V~^j = I - u3^i u3^i^T gives B = 0", np.abs(B_typo).max() < 1e-12)
B_minus = geo49.geosplat_B_fast(Xs, Xs, m, i0, eps, sign=-1.0)
B_plus = geo49.geosplat_B_fast(Xs, Xs, m, i0, eps, sign=+1.0)
rel = abs(np.trace(B_plus) - np.trace(B_minus)) / abs(np.trace(B_minus))
close("Exercise 49.7.4 (b): sign of the second term changes the result by about 2% at eps = 0.2", rel, 0.02, tol=0.012)
B_minus4 = geo49.geosplat_B_fast(Xs, Xs, m, i0, 0.4, sign=-1.0)
B_plus4 = geo49.geosplat_B_fast(Xs, Xs, m, i0, 0.4, sign=+1.0)
rel4 = abs(np.trace(B_plus4) - np.trace(B_minus4)) / abs(np.trace(B_minus4))
close("Exercise 49.7.4 (b): about 8% at eps = 0.4", rel4, 0.08, tol=0.03)
s_print = geo49.warmup_scales(-2.0, 0.5, 0.1, clamp=False)
r_print = abs(min(max(-2.0, 1e-3), 1e9) / min(max(0.5, 1e-3), 1e9))
check("Exercise 49.7.4 (c): clamping the signed tau = (-2, 0.5) gives the ratio 0.002 instead of 4",
      abs(r_print - 0.002) < 1e-12 and abs(s_print[0] / s_print[1] - 4.0) < 1e-12)
close("Table 49.7.2: missing 1/eps in (31) is a global factor eps",
      np.trace(geo49.geosplat_B_fast(Xs, Xs, m, i0, eps, with_inv_eps=False)) / np.trace(B_minus), eps, tol=1e-12)

summary()
