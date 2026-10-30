"""Verification for Section 47.7 (Numerical Experiments).

The large runs (N = 125000 and 625000) are made by the figure scripts fig-47-7-2 and fig-47-7-3; here we recompute
the values for N = 5000 and 25000 and the cheap figures (fig-47-7-1, fig-47-7-4).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-7-numerical-experiments.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import importlib.util  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var47 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import dgfig  # noqa: E402


def load(stem):
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), HERE.parents[1] / "figures" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    save = dgfig.save
    dgfig.save = lambda *a, **k: None
    spec.loader.exec_module(mod)
    dgfig.save = save
    return mod


K = v.pairs(2)
NAT = K["natural exp"]

# ---------------------------------------------------------------- (47.7.1): the constant of (8.1)/(8.2)
close("Table 47.7.2: C_xi/C_rho = d/n = 1/2 for both natural pairs (n = 2, d = 1), 1 for (rho, rho)",
      [K["natural exp"].ratio(1), K["natural tent"].ratio(1), K["exp"].ratio(1), K["tent"].ratio(1)],
      [0.5, 0.5, 1.0, 1.0], 1e-6)
N = 20000
t, X, T, Hx = v.curve_cloud(lambda s: v.circle(s, 0.5), N)
H, Hp, _ = v.amc(X, v.proj_from_tangent(T), np.ones(N), 0.01, NAT, 1)
close("Warning (8.1)/(8.2): circle r = 0.5, N = 20000, eps = 0.01: with the constant -1.99991, as printed -3.99981",
      [Hp[0, 0], Hp[0, 0] / NAT.ratio(1)], [-1.99991, -3.99981], 1e-5)

# ---------------------------------------------------------------- Example 47.7.2: the segment example of Section 8
xs = np.concatenate([np.linspace(-1, 0, 201), np.linspace(0, 1, 51)[1:]])
Xs = np.stack([xs, np.zeros_like(xs)], 1)
Ps = v.proj_from_tangent(np.tile([1.0, 0.0], (len(xs), 1)))
H, Hp, cnt = v.amc(Xs, Ps, np.ones(len(xs)), 0.1, NAT, 1, Q=Xs[200:201], PQ=Ps[200:201])
close("Example 47.7.2: segment, 200 points per unit on the left, 50 on the right, eps = 0.1: H^V = (-9.79, 0)",
      H[0], [-9.79, 0.0], 5e-3)
check("Example 47.7.2: the orthogonal version is exactly 0", np.abs(Hp).max() < 1e-12 and cnt[0] == 25)

# ---------------------------------------------------------------- Table 47.7.1: the test shapes
close("Table 47.7.1: ellipse curvature range [0.5, 4]",
      [np.linalg.norm(v.ellipse(np.array([math.pi / 2]))[2]), v.max_abs_curvature(v.ellipse)], [0.5, 4.0], 1e-6)
fl_max, fl_mean, fl_len = v.max_abs_curvature(v.flower), v.mean_abs_curvature_param(v.flower), v.curve_length(v.flower)
close("Table 47.7.1: flower max |H| = 140, mean |H(t_j)| = 9.878, length 7.044",
      [fl_max, fl_mean, fl_len], [140.0, 9.878, 7.044], 1e-3)
tip = np.linalg.norm(v.flower(np.array([0.0]))[2])
close("Table 47.7.1: flower |H| at a petal tip = 17.33", tip, 17.333, 1e-3)
close("Remark 47.7.4: max |H| / mean |H| = 14.17 on the flower", fl_max / fl_mean, 14.17, 5e-3)
close("Table 47.7.1: eight max |H| = 6.7745", v.max_abs_curvature(v.eight), 6.7745, 1e-4)
_, T0, H0 = v.eight(np.array([0.0, math.pi]))
check("Table 47.7.1: eight crosses itself at the origin at right angles with zero curvature on both branches",
      abs(T0[0] @ T0[1]) < 1e-12 and np.abs(H0).max() < 1e-12)

# ---------------------------------------------------------------- Figure 47.7.1 = [BLM17, Fig. 3]
f1 = load("fig-47-7-1-ellipse-flower")
lo, hi, err = f1.extremes("ellipse")
close("Figure 47.7.1(a): ellipse min / max of |H^{V,perp}| = 0.5 / 3.99998 (paper 0.5 / 3.99998)", [lo, hi],
      [0.5, 3.99998], 1e-5)
lo, hi, err = f1.extremes("flower")
close("Figure 47.7.1(b): flower max |H^{V,perp}| = 138.956 (paper 138.903)", hi, 138.956, 1e-3)
check("Figure 47.7.1(b): flower min |H^{V,perp}| < 1e-3 (paper 0.000241)", lo < 1e-3)
close("Section 47.7: flower, N = 1e5, eps = 0.001: mean number of neighbours 35.2", f1.RES["flower"][4].mean(), 35.2,
      0.05)

# ---------------------------------------------------------------- Figure 47.7.2 = [BLM17, Fig. 4] for N = 5000, 25000
f2 = load("fig-47-7-2-flower-convergence")
NS2 = (5000, 25000)
R2 = f2.table(NS2)
ours = {("a", "natural exp"): (0.303, 0.037), ("a", "natural tent"): (0.281, 0.0382), ("a", "exp"): (0.401, 0.0481),
        ("a", "tent"): (0.379, 0.0568), ("a", "natural exp (regression)"): (0.374, 0.0569),
        ("b", "natural exp"): (0.146, 0.0194), ("b", "natural tent"): (0.143, 0.0204), ("b", "exp"): (0.195, 0.0266),
        ("b", "tent"): (0.202, 0.0401), ("b", "natural exp (regression)"): (0.201, 0.0322)}
for (case, lab), vals in ours.items():
    got = [R2[(case, lab, N)][0] for N in NS2]
    close(f"Tables 47.7.4-5: E for ({case}) {lab} at N = 5000, 25000", np.array(got) / np.array(vals), [1, 1], 6e-3)
# agreement with the values read from the paper (exact tangents): within 30 percent
worst = max(abs(math.log(R2[(case, lab, N)][0] / f2.PAPER[case][lab][i]))
            for case in "ab" for lab in ("natural exp", "natural tent", "exp", "tent") for i, N in enumerate(NS2))
check("Tables 47.7.4-5: exact-tangent values at N = 5000, 25000 agree with the read values within 30 %"
      f" (worst log ratio {worst:.2f})", worst < math.log(1.3))
close("Section 47.7: N_neigh for eps = 100/N is 35.2 (paper: 100)", R2[("a", "exp", 25000)][1], 35.2, 0.05)
close("Section 47.7: N_neigh for eps = (10/N)^(3/4): 16.5, 24.9 (ratio 5^(1/4) = 1.495)",
      [R2[("b", "exp", 5000)][1], R2[("b", "exp", 25000)][1]], [16.5, 24.9], 0.05)
e_max = v.erel_curve(v.flower, 5000, (10 / 5000) ** 0.75, NAT, norm="max")[0]
close("Remark 47.7.4: with the printed ||H||_inf the value is 14.17 times smaller (0.0103 instead of 0.146)",
      [e_max, R2[("b", "natural exp", 5000)][0] / e_max], [0.0103, 14.17], 2e-2)
close("Section 47.7: 1/(N eps) = 10^(-3/4) N^(-1/4) = 0.0211 at N = 5000, 0.0063 at N = 625000",
      [1 / (5000 * (10 / 5000) ** 0.75), 1 / (625000 * (10 / 625000) ** 0.75)], [0.02115, 0.00632], 1e-3)
# the regression hypothesis: total least squares (PCA) instead of ordinary least squares
pca = [v.erel_curve(v.flower, N, 100 / N, NAT, tangents="pca", R=50 / N, norm="mean")[0] for N in NS2]
close("Table 47.7.4: (a) natural exp with PCA tangent lines, R = eps/2: 0.346, 0.0471", pca, [0.346, 0.0471], 1e-2)
# our explanation of the slopes: the natural pair error is the smoothing bias O(eps^2)
close("Exercise 47.7.4: exponent of eps^2 in N: -2 for eps = 100/N, -1.5 for eps = (10/N)^(3/4)",
      [2 * -1.0, 2 * -0.75], [-2.0, -1.5], 1e-12)

# ---------------------------------------------------------------- Figure 47.7.3(a) = [BLM17, Fig. 5]
f3 = load("fig-47-7-3-density-noise")
R5 = f3.table5(NS2)
ours5 = {"uniform": (0.146, 0.0194), "non uniform": (0.145, 0.0199), "non uniform (regression)": (0.202, 0.0394),
         "uniform, no projection": (0.665, 0.760), "non uniform, no projection": (2.83, 5.56)}
for lab, vals in ours5.items():
    close(f"Table 47.7.6: {lab} at N = 5000, 25000", np.array([R5[(lab, N)] for N in NS2]) / np.array(vals), [1, 1],
          6e-3)
pca5 = [v.erel_curve(v.flower, N, (10 / N) ** 0.75, NAT, tangents="pca", R=(10 / N) ** 0.75 / 2, jitter=True,
                     norm="mean")[0] for N in NS2]
close("Table 47.7.6: non uniform with PCA tangent lines: 0.181, 0.0306", pca5, [0.181, 0.0306], 1e-2)
# the plateau of H^V on the uniform-parameter sampling: the density term theta'/theta of the limit varifold
M = 2_000_000
tt = 2 * np.pi * np.arange(M) / M
r = 0.5 + 0.25 * np.cos(6 * tt)
dr, ddr = -1.5 * np.sin(6 * tt), -9.0 * np.cos(6 * tt)
sp = np.sqrt(r ** 2 + dr ** 2)
g = -((r * dr + dr * ddr) / sp) / sp ** 2            # d/ds log theta with theta = 1/|gamma'|
close("Remark 47.7.6: plateau mean|theta'/theta| / mean|H| = 0.787 (Fig. 5 'uniform, no projection': 0.78)",
      np.abs(g).mean() / fl_mean, 0.787, 1e-3)
close("Section 47.7: d/eps^2 = N^(1/2) / 10^(3/2) for d = 1/N, eps = (10/N)^(3/4)",
      (1 / 25000) / (10 / 25000) ** 1.5, math.sqrt(25000) / 10 ** 1.5, 1e-12)

# ---------------------------------------------------------------- Figure 47.7.3(b) = [BLM17, Fig. 6]
R6 = f3.table6(NS2)
ours6 = {r"$(10/N)^{3/4}$, noise": (0.15, 0.0213), r"$(100/N)^{3/4}$, noise": (0.491, 0.255),
         r"$N^{-1/2}$": (0.229, 0.0811)}
for lab, vals in ours6.items():
    close(f"Table 47.7.7: {lab} at N = 5000, 25000", np.array([R6[(lab, N)] for N in NS2]) / np.array(vals), [1, 1],
          1e-2)
close("Exercise 47.7.4: eps = N^(-1/2): slope in N between 5000 and 25000 is -0.65",
      math.log(R6[(r"$N^{-1/2}$", 25000)] / R6[(r"$N^{-1/2}$", 5000)]) / math.log(5), -0.645, 0.01)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    e_var = v.erel_curve(v.flower, 5000, (10 / 5000) ** 0.75, NAT, noise=5000 ** -0.5, norm="mean")[0]
check("Warning (Fig. 6 noise): standard deviation N^(-1/2) leaves empty balls (E is NaN) at N = 5000",
      not np.isfinite(e_var))
close("Warning (Fig. 6 noise): noise 1/N is 0.14 of the spacing L/N", 1 / fl_len, 0.142, 1e-2)
# the regression radius for noisy clouds, eps = (100/N)^(3/4): R = eps^1.1 beats R = eps^0.9 (both fits)
rows = []
for N in NS2:
    eps = (100 / N) ** 0.75
    rows.append([v.erel_curve(v.flower, N, eps, NAT, noise=1 / N, tangents=tg, R=eps ** ex, norm="mean")[0]
                 for tg in ("pca", "ols") for ex in (0.9, 1.1)])
rows = np.array(rows)
close("Table 47.7.7: noise, regression R = eps^0.9 / eps^1.1, PCA: N = 5000 0.633 / 0.625, N = 25000 0.461 / 0.319",
      rows[:, [0, 1]].ravel(), [0.633, 0.625, 0.461, 0.319], 1e-2)
check("Table 47.7.7: R = eps^1.1 gives the lower error for both fits at N = 5000, 25000",
      bool(np.all(rows[:, 1] < rows[:, 0]) and np.all(rows[:, 3] < rows[:, 2])))

# ---------------------------------------------------------------- [BLM17, Fig. 7]: the eight
N = 10000
t, X, T, Hx = v.curve_cloud(v.eight, N)
H, Hp, _ = v.amc(X, v.proj_from_tangent(T), np.ones(N), 0.01, NAT, 1)
a, b = np.linalg.norm(H, axis=1), np.linalg.norm(Hp, axis=1)
close("Section 47.7 (Fig. 7): eight, N = 10000, eps = 0.01: max |H^{V,perp}| = 6.7639, max |H^V| = 6.7686",
      [b.max(), a.max()], [6.76389, 6.76863], 1e-5)
o = np.linalg.norm(X, axis=1) < 1e-12
check("Section 47.7 (Fig. 7): the two atoms at the crossing have |H^V|, |H^perp| < 1e-10",
      o.sum() == 2 and a[o].max() < 1e-10 and b[o].max() < 1e-10)
far = np.linalg.norm(X, axis=1) > 0.02
close("Section 47.7 (Fig. 7): away from the crossing max error 2.43 for H^V, 0.0106 for H^{V,perp}",
      [np.linalg.norm(H - Hx, axis=1)[far].max(), np.linalg.norm(Hp - Hx, axis=1)[far].max()], [2.43, 0.0106], 5e-3)
j02 = int(np.argmin(np.linalg.norm(X - np.array([0.2, 0.0]), axis=1)))
close("Section 47.7 (Fig. 7, added in review): atom nearest (0.2, 0): |H^V| = 0.616 with tangential part 0.511, "
      "|H^{V,perp}| = 0.345 (exact 0.345)", [a[j02], abs(H[j02] @ T[j02]), b[j02], np.linalg.norm(Hx[j02])],
      [0.616, 0.511, 0.345, 0.345], 1e-3)

# ---------------------------------------------------------------- Figure 47.7.4 = [BLM17, Fig. 8, Fig. 9]
f4 = load("fig-47-7-4-singular-profiles")
TC, DB, DB2 = f4.TC, f4.DB, f4.DB2
close("Figure 47.7.4(a): two circles: min |H^V| 0.926446, min |H^perp| 0.917005, max 1.99991 (paper 0.926446, "
      "0.917004, 1.99991)", [TC["HV_all"].min(), TC["Hp_all"].min(), TC["HV_all"].max()],
      [0.926446, 0.917005, 1.99991], 1e-5)
k0 = int(np.argmin(np.abs(TC["s"])))
close("Proposition 47.7.7 / Figure 47.7.4(a): atom nearest the crossing: |H^V| = 1.7309 (sqrt 3 = 1.7321), "
      "|H^perp| = 1.4984 (limit 1.5)", [TC["HV"][k0], TC["Hp"][k0]], [1.7309, 1.4984], 1e-4)
p = np.array([0.0, math.sqrt(0.25 - 0.0625)])
H1 = (np.array([-0.25, 0.0]) - p) / 0.25
H2 = (np.array([0.25, 0.0]) - p) / 0.25
n1 = (np.array([-0.25, 0.0]) - p) / 0.5
close("Proposition 47.7.7: (H1 + H2)/2 = (0, -sqrt 3), its normal part on branch 1 has norm 1.5",
      [*((H1 + H2) / 2), abs(((H1 + H2) / 2) @ n1)], [0.0, -math.sqrt(3), 1.5], 1e-12)
dip = TC["s"][np.argmin(TC["HV"])]
far_tc = np.abs(TC["s"]) > 0.0117           # the ball misses the other branch once s sin 60 > eps
check(f"Figure 47.7.4(a): the dip lies inside the eps-neighbourhood (|s| = {abs(dip):.4f} < 0.01) and |H^V| = 2 "
      "once the ball misses the other branch (|s| > eps / sin 60 = 0.0115)",
      abs(dip) < 0.01 and np.all(np.abs(TC["HV"][far_tc] - 2) < 1e-3) and np.all(np.abs(TC["Hp"][far_tc] - 2) < 1e-3))
close("Table 47.7.8: 2D double bubble: junction p = (0.0688, 0.5960), mean curvature (0, -0.8389)",
      [*DB["p"], *DB["Hm"]], [0.0688, 0.5960, 0.0, -0.8389], 1e-4)
close("Table 47.7.8: N = 800: 2 eps average (0.1065, -0.7961), relative error 0.137 (paper (0.107, -0.809), 13 %)",
      [*DB["A"][DB["j"]], DB["err_avg"]], [0.1065, -0.7961, 0.137], 1e-3)
close("Table 47.7.8: N = 1600, eps = 0.075: relative error 0.068 (paper 7 %)", DB2["err_avg"], 0.068, 1e-3)
close("Table 47.7.8: without the average: relative error 0.050 (N = 800), 0.028 (N = 1600)",
      [DB["err_raw"], DB2["err_raw"]], [0.050, 0.028], 1e-3)


def bubble_offset(N, eps, off):
    """relative errors (before, after the 2 eps average) at the atom nearest the junction when the samples are
    shifted by `off` spacings (Section 47.7: the placement sensitivity, added in review)."""
    db = v.double_bubble_2d(0.6, 1.0)
    X, _, _, _ = v.sample_arcs(db["arcs"], N, offset=off)
    H, _, _ = v.amc(X, v.pca_planes(X, eps, 1), np.ones(N), eps, f4.PAIR, 1)
    A = v.average_ball(X, H, 2 * eps)
    Hm = v.junction_mean_curvature(db["centres"], db["radii"], db["p"])
    j = int(np.argmin(np.linalg.norm(X - db["p"], axis=1)))
    a, b = np.linalg.norm(H, axis=1), np.linalg.norm(A, axis=1)
    return (np.linalg.norm(H[j] - Hm) / np.linalg.norm(Hm), np.linalg.norm(A[j] - Hm) / np.linalg.norm(Hm),
            a.min(), a.max(), b.min(), b.max())


OFF = {(N, o): bubble_offset(N, e, o) for N, e in ((800, 0.15), (1600, 0.075)) for o in (0.0, 0.25, 0.5)}
close("Table 47.7.8: placements shifted by 0, 1/4, 1/2 spacing, N = 800: before the average 0.045 / 0.071 / 0.050, "
      "after 0.133 / 0.136 / 0.137", [OFF[(800, o)][i] for i in (0, 1) for o in (0.0, 0.25, 0.5)],
      [0.045, 0.071, 0.050, 0.133, 0.136, 0.137], 1.5e-3)
close("Table 47.7.8: the same for N = 1600: before 0.050 / 0.100 / 0.028, after 0.078 / 0.090 / 0.068",
      [OFF[(1600, o)][i] for i in (0, 1) for o in (0.0, 0.25, 0.5)], [0.050, 0.100, 0.028, 0.078, 0.090, 0.068],
      1.5e-3)
rng800 = np.array([OFF[(800, o)][2:] for o in (0.0, 0.25, 0.5)])
close("Section 47.7 (Fig. 9 colorbars 0.229389-1.75353 and 0.512783-1.58696): over the three placements, N = 800, "
      "raw |H^V| min 0.095-0.230, max 1.754-1.778; averaged min 0.492-0.501, max 1.592-1.595",
      [rng800[:, 0].min(), rng800[:, 0].max(), rng800[:, 1].min(), rng800[:, 1].max(),
       rng800[:, 2].min(), rng800[:, 2].max(), rng800[:, 3].min(), rng800[:, 3].max()],
      [0.095, 0.230, 1.754, 1.778, 0.492, 0.501, 1.592, 1.595], 1.5e-3)
sel = (DB["k"] == 0) & (DB["s"] < 0.6) & (DB["X"][:, 1] > 0)
hv = np.linalg.norm(DB["H"][sel], axis=1)
close("Figure 47.7.4(b): small arc: |H^V| oscillates between 0.095 and 1.754 near the junction",
      [hv.min(), hv.max()], [0.095, 1.754], 1e-3)
def last_off(arr, s_arc, r):
    bad = np.abs(arr - 1 / r) > 0.02 / r
    return s_arc[bad].max() if bad.any() else 0.0


raw_last, avg_last = [], []
for i, rr in enumerate(DB["radii"]):
    sl = (DB["k"] == i) & (DB["X"][:, 1] > 0)
    raw_last.append(last_off(np.linalg.norm(DB["H"][sl], axis=1), DB["s"][sl], rr))
    avg_last.append(last_off(np.linalg.norm(DB["A"][sl], axis=1), DB["s"][sl], rr))
check(f"Figure 47.7.4(b): raw |H^V| is within 2 % of 1/r beyond s = {max(raw_last):.3f} (about 1.7 eps)",
      max(raw_last) < 0.26)
check(f"Figure 47.7.4(b): averaged values on the arcs r = 1, 1.5 are within 2 % beyond s = {max(avg_last[1:]):.3f} "
      "(< 3 eps)", max(avg_last[1:]) < 0.45)
far_small = (DB["k"] == 0) & (np.abs(DB["X"][:, 1]) < 0.05) & (DB["X"][:, 0] < 0)
raw_far = np.linalg.norm(DB["H"][far_small], axis=1).mean()
avg_far = np.linalg.norm(DB["A"][far_small], axis=1).mean()
close("Figure 47.7.4(b): far from the junctions on the small arc: raw 1.654, averaged 1.584 (exact 1.667), "
      "ratio 0.958", [raw_far, avg_far, avg_far / raw_far], [1.654, 1.584, 0.958], 1e-3)
phi = 2 * math.asin(0.3 / 1.2)
close("Figure 47.7.4(b): averaging over 2 eps = 0.3 on the arc of radius 0.6 shrinks |H| by sin(phi)/phi = 0.958",
      math.sin(phi) / phi, 0.958, 1e-3)

# ---------------------------------------------------------------- Section 8.3: the 3D double bubbles
d3 = v.double_bubble_2d(0.7, 1.0)
Hm3 = v.junction_mean_curvature(d3["centres"], d3["radii"], d3["p"], k=2.0)
close("Table 47.7.8: 3D double bubble 0.7 / 1: middle cap radius 2.3333, |average of the three H| = 1.4662 "
      "(paper 1.46; measured 1.51)", [d3["r3"], np.linalg.norm(Hm3)], [2.33333, 1.46617], 1e-5)
c1, c2, pp = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3) / 2])
close("Table 47.7.8: 3D double bubble 1 / 1 (flat middle disk): |average| = 2/sqrt 3 = 1.1547",
      np.linalg.norm((2 * (c1 - pp) + 2 * (c2 - pp)) / 3), 2 / math.sqrt(3), 1e-12)

summary()
