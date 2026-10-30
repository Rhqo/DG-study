"""Verification for Section 49.6 (Experiments: Tables and Figures).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-6-experiments.py
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import geo49  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "49-6-experiments.html"
T1, T2, T3 = geo49.TABLE1, geo49.TABLE2, geo49.TABLE3
REP, ICL = geo49.REPLICA, geo49.ICL
html = PAGE.read_text(encoding="utf-8")

# ---------------------------------------------------------------- the printed tables of the page = lib data
def page_table(tid):
    m = re.search(r'<table id="%s">(.*?)</table>' % tid, html, re.S)
    rows = re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S)
    out = []
    for r in rows:
        cells = [re.sub(r"<[^>]+>", "", c).replace("†", "").strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        out.append(cells)
    return out


ICLAB = {"Rm-1": "Room-1", "Rm-2": "Room-2", "Of-2": "Office-2", "Of-3": "Office-3"}
ABBR = {"Man.": "Manifold", "Var.": "Varifold", "GeoG.": "GeoGaussian", "GS-SLAM": "GS-SLAM", "P-SLAM": "Point-SLAM",
        "Vox": "Vox-Fusion", "LightGS": "LightGS", "LGS": "LightGS", "3DGS": "3DGS"}


def table_cells(tid, T, scenes):
    """compare a block table (header row: metric + scenes; then one row per method) with lib data; return #cells."""
    n, ok, met, cols = 0, True, None, None
    for r in page_table(tid):
        if r[0].split()[0] in ("PSNR", "SSIM", "LPIPS"):
            met, cols = r[0].split()[0], [ICLAB.get(c, c) for c in r[1:]]
            continue
        m = ABBR[r[0]]
        for c, sc in zip(r[1:], cols):
            ok &= abs(float(c) - T[m][met][scenes.index(sc)]) < 1e-9
            n += 1
    return n if ok else -1


check("Table 49.6.2 on the page = [GeoSplat25, Table 1] in lib (144 cells)", table_cells("tab-49-6-2", T1, REP) == 144)
check("Table 49.6.3 on the page = [GeoSplat25, Table 2] in lib (60 cells)", table_cells("tab-49-6-3", T2, ICL) == 60)
rows = [r for r in page_table("tab-49-6-6") if len(r) == 4 and r[0] != "variant"]
ok = len(rows) == 28
blocks = [(v, off) for v in ("Manifold", "Varifold") for off in (0, 3)]
for b, (v, off) in enumerate(blocks):
    for k in range(7):
        ok &= np.allclose([float(c) for c in rows[7 * b + k][1:]], T3[v][k][off:off + 3])
check("Table 49.6.6 on the page = [GeoSplat25, Table 3] in lib (84 cells)", ok)
ok, n = True, 0
panel = None
for r in page_table("tab-49-6-4"):
    if r[0] == "views":
        continue
    if len(r) == 1:
        sc, met = r[0].replace(",", "").split()[1:3]
        panel = (sc, met)
        continue
    m = ABBR[r[0]]
    for k, c in enumerate(r[1:]):
        ref = geo49.FIG1_READ[panel][m][k]
        ok &= (c == "—") if ref is None else abs(float(c.replace("≈", "")) - ref) < 1e-9
        n += 1
check("Table 49.6.4 on the page = our Figure 1 readings in lib (80 cells)", ok and n == 80)

# the bold cells (as printed) are exactly the column best, always a GeoSplat variant
ok = True
for T, B, sc in ((T1, geo49.BOLD1, REP), (T2, geo49.BOLD2, ICL)):
    for met in B:
        for j, s in enumerate(sc):
            vals = {m: T[m][met][j] for m in T}
            best = (min if met == "LPIPS" else max)(vals.values())
            who = {"M": "Manifold", "V": "Varifold"}[B[met][s]]
            ok &= vals[who] == best and [m for m in vals if vals[m] == best] == [who]
check("bold cells of Tables 1-2 = unique column best (GeoSplat in every column)", ok)

# ---------------------------------------------------------------- gains, means, unbeaten cells
gm = np.array(T1["Manifold"]["PSNR"]) - np.array(T1["GeoGaussian"]["PSNR"])
gv = np.array(T1["Varifold"]["PSNR"]) - np.array(T1["GeoGaussian"]["PSNR"])
check("Replica PSNR gains over GeoGaussian (manifold)", np.allclose(gm, [1.17, 1.31, 1.22, 0.64, 0.50, 0.43, 0.40, 0.57]))
check("Replica PSNR gains over GeoGaussian (varifold)", np.allclose(gv, [1.15, 1.30, 0.91, 0.67, 0.44, 0.41, 0.42, 0.78]))
close("Replica mean gain manifold 0.78", gm.mean(), 0.78, tol=1e-9)
close("Replica mean gain varifold 0.76", gv.mean(), 0.76, tol=1e-9)
check("room scenes R0-R2 gain more than every office scene (both variants)", gm[:3].min() > gm[3:].max() and gv[:3].min() > gv[3:].max())
im = np.array(T2["Manifold"]["PSNR"]) - np.array(T2["GeoGaussian"]["PSNR"])
iv = np.array(T2["Varifold"]["PSNR"]) - np.array(T2["GeoGaussian"]["PSNR"])
check("ICL gains (manifold) 0.38, 0.51, 0.21, 0.42", np.allclose(im, [0.38, 0.51, 0.21, 0.42]))
check("ICL gains (varifold) 0.49, 0.34, 0.14, 0.41", np.allclose(iv, [0.49, 0.34, 0.14, 0.41]))
close("ICL mean gain manifold 0.38", im.mean(), 0.38, tol=1e-9)
close("ICL mean gain varifold 0.345 (printed 0.35)", iv.mean(), 0.345, tol=1e-9)
means = {m: [np.mean(T1[m][k]) for k in ("PSNR", "SSIM", "LPIPS")] for m in T1}
check("Replica column means (PSNR/SSIM/LPIPS) as printed",
      np.allclose(means["Vox-Fusion"], [24.41, 0.801, 0.236], atol=6e-3) and np.allclose(means["Point-SLAM"], [35.05, 0.975, 0.124], atol=6e-3)
      and np.allclose(means["GS-SLAM"], [37.48, 0.964, 0.073], atol=6e-3) and np.allclose(means["GeoGaussian"], [38.78, 0.969, 0.027], atol=6e-3)
      and np.allclose(means["Manifold"], [39.56, 0.979, 0.025], atol=6e-3) and np.allclose(means["Varifold"][:2], [39.54, 0.978], atol=6e-3))
means2 = {m: [np.mean(T2[m][k]) for k in ("PSNR", "SSIM", "LPIPS")] for m in T2}
check("ICL column means as printed",
      np.allclose(means2["3DGS"], [38.45, 0.971, 0.021], atol=6e-3) and np.allclose(means2["LightGS"], [38.64, 0.971, 0.020], atol=6e-3)
      and np.allclose(means2["GeoGaussian"], [38.91, 0.974, 0.017], atol=6e-3) and np.allclose(means2["Manifold"], [39.29, 0.977, 0.015], atol=6e-3)
      and np.allclose(means2["Varifold"], [39.25, 0.976, 0.015], atol=6e-3))


def unbeaten(T, sc):
    bases = [m for m in T if m not in ("Manifold", "Varifold")]
    out = []
    for v in ("Manifold", "Varifold"):
        for met in ("PSNR", "SSIM", "LPIPS"):
            for j, s in enumerate(sc):
                x = T[v][met][j]
                vals = [T[b][met][j] for b in bases]
                if (met == "LPIPS" and x >= min(vals)) or (met != "LPIPS" and x <= max(vals)):
                    out.append((v, met, s))
    return out


check("Table 1: the cells where a GeoSplat variant is not better than every baseline",
      unbeaten(T1, REP) == [("Manifold", "SSIM", "OFF4"), ("Varifold", "SSIM", "R0"), ("Varifold", "SSIM", "R1"),
                            ("Varifold", "SSIM", "R2"), ("Varifold", "SSIM", "OFF1"), ("Varifold", "LPIPS", "OFF1")])
check("Table 1: varifold SSIM R1 0.975 < GeoGaussian 0.979 and < Point-SLAM 0.977",
      T1["Varifold"]["SSIM"][1] < T1["GeoGaussian"]["SSIM"][1] and T1["Varifold"]["SSIM"][1] < T1["Point-SLAM"]["SSIM"][1])
check("Table 1: varifold OFF1 LPIPS 0.971 is out of range of the other cells (0.013-0.040)",
      T1["Varifold"]["LPIPS"][4] > 10 * max(max(T1[m]["LPIPS"]) for m in ("GeoGaussian", "Manifold")))
check("Table 2: only varifold LPIPS Office-2 (tie 0.017) is not better", unbeaten(T2, ICL) == [("Varifold", "LPIPS", "Office-2")])
check("Figure 49.6.1: exactly one negative SSIM bar (varifold R1, -0.004)",
      sum(1 for v in ("Manifold", "Varifold") for j in range(8) if T1[v]["SSIM"][j] < T1["GeoGaussian"]["SSIM"][j]) == 1)
dmv = np.array(T1["Manifold"]["PSNR"]) - np.array(T1["Varifold"]["PSNR"])
close("manifold - varifold PSNR (Replica) max +0.31 at R2", dmv.max(), 0.31, tol=1e-9)
close("manifold - varifold PSNR (Replica) min -0.21 at OFF4", dmv.min(), -0.21, tol=1e-9)

# ---------------------------------------------------------------- percentages of the paper text (Exercise 49.6.2)
pc = lambda a, b: 100 * (a - b) / b  # noqa: E731
claims = [("2.77", pc(41.92, 40.79)), ("0.93", pc(0.971, 0.962)), ("3.42", pc(39.55, 38.24)), ("2.01", pc(39.52, 38.74)),
          ("3.11", pc(40.36, 39.14)), ("1.07", 100 * (36.37 - 35.98) / 36.37), ("0.60", 100 * (36.61 - 36.39) / 36.61)]
check("the seven percentages of §5 = truncation of the table ratios", all(math.floor(v * 100) / 100 == float(c) for c, v in claims))
check("rounding would give 0.94, 3.43, 3.12 instead", [round(pc(0.971, 0.962), 2), round(pc(39.55, 38.24), 2), round(pc(40.36, 39.14), 2)] == [0.94, 3.43, 3.12])

# ---------------------------------------------------------------- (49.6.2), Exercise 49.6.1
for g, pct in ((0.78, 16.4), (0.38, 8.4), (1.13, 22.9), (1.31, 26.0), (1.0, 20.6), (0.5, 10.9)):
    close(f"(49.6.2): {g} dB -> squared error {pct}% smaller", 100 * (1 - 10 ** (-g / 10)), pct, tol=0.06)
close("PSNR from MSE: 10 log10(1/MSE) at MSE = 1e-4 is 40 dB", 10 * math.log10(1 / 1e-4), 40.0, tol=1e-12)
close("1 dB is 3.3% of 30 dB and 2.5% of 40 dB", 100 / 30, 3.33, tol=0.01)

# ---------------------------------------------------------------- Implementation 49.6 and Example 49.6.1, Exercise 49.6.5
ns = dgcode.run_block(PAGE, "code-49-6-1")
check("Implementation 49.6: at most 30 lines", dgcode.code_lines(PAGE, "code-49-6-1") <= 30)
close("Example 49.6.1: T = 5.74 (Replica, manifold vs GeoGaussian)", ns["t"], 5.74, tol=0.005)
close("Example 49.6.1: p = 7e-4", ns["p"], 7.1e-4, tol=0.2e-4)
close("Example 49.6.1: s_d = 0.385", np.std(gm, ddof=1), 0.385, tol=0.001)
close("Implementation: squared error ratio 0.84", ns["ratio"], 0.836, tol=0.001)
t, p = ns["paired_t"](T1["Varifold"]["PSNR"], T1["GeoGaussian"]["PSNR"])
close("Example 49.6.1: varifold T = 6.30", t, 6.30, tol=0.005)
close("Example 49.6.1: varifold p = 4e-4", p, 4.0e-4, tol=0.2e-4)
# the integral against a textbook value: two-sided 5% critical value of t_7 is 2.365
nu = 7
x = np.linspace(0, 2.365, 200001)
c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
pdf = c * (1 + x ** 2 / nu) ** (-(nu + 1) / 2)
close("t_7 density integral: p(2.365) = 0.050", 1 - 2 * (pdf[1:] + pdf[:-1]).sum() / 2 * (x[1] - x[0]), 0.05, tol=1e-3)
t, p = ns["paired_t"](T2["Manifold"]["PSNR"], T2["GeoGaussian"]["PSNR"])
close("Exercise 49.6.5 (a): T = 6.05", t, 6.05, tol=0.005)
close("Exercise 49.6.5 (a): p = 0.009", p, 0.0091, tol=0.0002)
close("Exercise 49.6.5 (a): sd 0.126", np.std(im, ddof=1), 0.126, tol=0.001)
t, p = ns["paired_t"](T1["Manifold"]["PSNR"], T1["Varifold"]["PSNR"])
close("Exercise 49.6.5 (b): T = 0.40", t, 0.40, tol=0.005)
close("Exercise 49.6.5 (b): p = 0.70", p, 0.70, tol=0.005)
close("Exercise 49.6.5 (b): mean difference 0.02", dmv.mean(), 0.02, tol=0.001)

# ---------------------------------------------------------------- Figure 1 readings (Table 49.6.4) and the 7.93% claim
ok = True
for (sc, met), d in geo49.FIG1_READ.items():
    T, j = (T1, REP.index(sc)) if sc in REP else (T2, ICL.index(sc))
    for m, vals in d.items():
        if vals[3] is None or m not in T:
            continue
        ok &= abs(vals[3] - T[m][met][j]) <= (0.1 if met == "PSNR" else 0.0021)
check("Table 49.6.4: the readings at 1.0 agree with Tables 1-2 (0.1 dB / 0.002)", ok)
R1, R2 = geo49.FIG1_READ[("R1", "PSNR")], geo49.FIG1_READ[("R2", "PSNR")]
close("Figure 1: R1 gap manifold - GeoGaussian at 1/6 is 2.8 dB", R1["Manifold"][1] - R1["GeoGaussian"][1], 2.8, tol=1e-9)
close("Figure 1: R1 gap at 0.1 is 1.5 dB", R1["Manifold"][0] - R1["GeoGaussian"][0], 1.5, tol=1e-9)
close("Figure 1: R1 gap at 1.0 is 1.3 to 1.4 (table 1.31)", R1["Manifold"][3] - R1["GeoGaussian"][3], 1.4, tol=0.11)
close("7.93% check: R2 manifold / GeoGaussian at 1/6 = 6.3%", 100 * (R2["Manifold"][1] / R2["GeoGaussian"][1] - 1), 6.25, tol=0.01)
r2lo, r2hi = 100 * ((39.0 / 36.9) - 1), 100 * ((39.2 / 36.7) - 1)
check("7.93% check: R2 range with +-0.1 dB is 5.7-6.8% (excludes 7.93)", abs(r2lo - 5.7) < 0.05 and abs(r2hi - 6.8) < 0.05 and not r2lo <= 7.93 <= r2hi)
r1lo, r1hi = 100 * ((37.8 / 35.2) - 1), 100 * ((38.0 / 35.0) - 1)
check("7.93% check: R1 = 8.0% with range 7.4-8.6% (includes 7.93)",
      abs(100 * (37.9 / 35.1 - 1) - 7.98) < 0.01 and abs(r1lo - 7.4) < 0.05 and abs(r1hi - 8.6) < 0.05 and r1lo <= 7.93 <= r1hi)
close("7.93% check: R2 vs 3DGS 9.2%", 100 * (39.1 / 35.8 - 1), 9.2, tol=0.05)
close("7.93% check: R2 vs LightGS 8.9%", 100 * (39.1 / 35.9 - 1), 8.9, tol=0.05)

# ---------------------------------------------------------------- ablation (Table 3, Figure 49.6.2, Exercise 49.6.3)
ok_drop, ok_order, ok_mac, ok_all = True, True, True, True
for v in T3:
    A = np.array(T3[v])
    for col in (0, 3):
        drop = A[0, col] - A[1:, col]
        ok_drop &= np.all(drop > 0)
        ok_order &= np.argmax(drop) == 4 and np.argsort(-drop)[1] == 0          # shape reg. first, warm-up second
        ok_mac &= A[0, col] > A[3, col] > A[2, col]                              # full > MAC -> mean > no upsampling
    ok_all &= np.all(A[0, [1, 4]] > A[1:, [1, 4]]) and np.all(A[0, [2, 5]] < A[1:, [2, 5]])
check("Table 3: every removal lowers PSNR (both models, both scenes)", ok_drop)
check("Table 3: shape regularization is the largest drop, warm-up the second (all four columns)", ok_order)
check("Table 3: full > (MAC -> mean curvature) > (no upsampling) in PSNR", ok_mac)
check("Table 3: SSIM and LPIPS also strictly worse in every row", ok_all)
A = np.array(T3["Manifold"])
close("Exercise 49.6.3: sum of the five drops (manifold R0) = 1.60", (A[0, 0] - A[[1, 2, 4, 5, 6], 0]).sum(), 1.60, tol=1e-9)
close("Exercise 49.6.3: additive extrapolation 34.77", A[0, 0] - (A[0, 0] - A[[1, 2, 4, 5, 6], 0]).sum(), 34.77, tol=1e-9)
close("worst single ablation R0 manifold 35.98 is 0.78 above GeoGaussian", A[1:, 0].min() - T1["GeoGaussian"]["PSNR"][0], 0.78, tol=1e-9)
B = np.array(T3["Varifold"])
close("Figure 49.6.2: varifold R0 sum 1.57", (B[0, 0] - B[[1, 2, 4, 5, 6], 0]).sum(), 1.57, tol=1e-9)
close("MAC vs mean curvature: +0.12 dB on R0", A[0, 0] - A[3, 0], 0.12, tol=1e-9)
close("MAC vs mean curvature: +0.03 dB on Office-3", A[0, 3] - A[3, 3], 0.03, tol=1e-9)
check("Table 3 SSIM drop ranges: R0 0.002-0.012, Office-3 0.001-0.005",
      np.isclose(min((np.array(T3[v])[0, 1] - np.array(T3[v])[1:, 1]).min() for v in T3), 0.002)
      and np.isclose(max((np.array(T3[v])[0, 1] - np.array(T3[v])[1:, 1]).max() for v in T3), 0.012)
      and np.isclose(min((np.array(T3[v])[0, 4] - np.array(T3[v])[1:, 4]).min() for v in T3), 0.001)
      and np.isclose(max((np.array(T3[v])[0, 4] - np.array(T3[v])[1:, 4]).max() for v in T3), 0.005))

# ---------------------------------------------------------------- Example 49.6.2: the flatness test on a synthetic room
T = 0.004
rng = np.random.default_rng(496)
bc = np.array(geo49.ROOM_BALL_C)
far = 6 * math.sqrt(T)


def groups(X0, lab):
    de = geo49.room_edge_distance(X0)
    flat = np.where((lab == 0) & (de > far) & (np.linalg.norm(X0 - bc, axis=1) > geo49.ROOM_BALL_R + far))[0]
    return de, flat, np.where((lab == 0) & (de < 0.05))[0], np.where((lab == 1) & (X0[:, 2] > 0.45))[0]


X, X0, lab = geo49.room_points(1500, 0.0, seed=2)
de, flat, edge, ball = groups(X0, lab)
m0 = [geo49.mac(geo49.manifold_estimate(X, X[i], T)["tau"]) for i in rng.choice(flat, 10, replace=False)]
check("Example 49.6.2: without noise the flat MAC is 0 to rounding", max(m0) < 1e-10)
med = {}
for sig in (0.001, 0.003, 0.01):
    X, X0, lab = geo49.room_points(1500, sig, seed=2)
    de, flat, edge, ball = groups(X0, lab)
    med[sig] = np.array([geo49.mac(geo49.manifold_estimate(X, X[i], T)["tau"]) for i in rng.choice(flat, 60, replace=False)])
close("Example 49.6.2: flat MAC median at sigma 1 mm ~ 0.027", np.median(med[0.001]), 0.027, tol=0.006)
close("Example 49.6.2: flat MAC median at sigma 3 mm ~ 0.079", np.median(med[0.003]), 0.079, tol=0.015)
close("Example 49.6.2: flat MAC median at sigma 10 mm ~ 0.23", np.median(med[0.01]), 0.23, tol=0.04)
check("Example 49.6.2: roughly proportional to sigma (ratios 2-4.5 for 3x sigma)",
      2.0 < np.median(med[0.003]) / np.median(med[0.001]) < 4.5 and 2.0 < np.median(med[0.01]) / np.median(med[0.003]) < 4.5)
X, X0, lab = geo49.room_points(1500, 0.003, seed=2)
de, flat, edge, ball = groups(X0, lab)
te = np.array([geo49.manifold_estimate(X, X[i], T)["tau"] for i in rng.choice(edge, 60, replace=False)])
me = np.abs(te).mean(1)
mb = np.array([geo49.mac(geo49.manifold_estimate(X, X[i], T)["tau"]) for i in rng.choice(ball, 60, replace=False)])
mf = med[0.003]
close("Example 49.6.2: edge MAC median ~ 1.8", np.median(me), 1.8, tol=0.3)
close("Example 49.6.2: edge |tau_1| median ~ 3.1 (cylinder of radius ~0.3)", np.median(np.abs(te[:, 0])), 3.1, tol=0.5)
close("Example 49.6.2: edge |tau_2| median ~ 0.35", np.median(np.abs(te[:, 1])), 0.35, tol=0.2)
close("Example 49.6.2: ball MAC median ~ 2.05 (true 2.5, about 20% low)", np.median(mb), 2.05, tol=0.2)
check("Example 49.6.2: xi_min = 0.001 per metre passes no point", (mf < 1e-3).sum() + (me < 1e-3).sum() + (mb < 1e-3).sum() == 0)
check("Example 49.6.2: xi = 0.3 passes all flat at sigma 1 mm, at least 93% at 3 mm (about 99%), < 5% edge, no ball",
      (med[0.001] < 0.3).mean() == 1.0 and (mf < 0.3).mean() >= 0.93 and (me < 0.3).mean() < 0.05 and (mb < 0.3).mean() == 0)
check("Example 49.6.2: xi = 0.3 passes 50-90% of flat at sigma 10 mm", 0.5 < (med[0.01] < 0.3).mean() < 0.9)
check("Exercise 49.6.4 (cm): 0.1 per metre passes 40-80% of flat, no edge, no ball",
      0.4 < (mf < 0.1).mean() < 0.8 and (me < 0.1).mean() == 0 and (mb < 0.1).mean() == 0)
check("Exercise 49.6.4 (mm): 1 per metre passes all flat, 3-25% of edge, no ball",
      (mf < 1.0).mean() == 1.0 and 0.03 < (me < 1.0).mean() < 0.25 and (mb < 1.0).mean() == 0)
q = X[flat[0]]
a = geo49.mac(geo49.manifold_estimate(X, q, T)["tau"])
b = geo49.mac(geo49.manifold_estimate(100 * X, 100 * q, T * 1e4)["tau"])
close("Exercise 49.6.4: scaling lengths by 100 (t by 1e4) divides MAC by 100", 100 * b, a, tol=1e-12 * max(1, a))
r_t = []
for tt in (0.002, 0.008):
    r_t.append(np.median([geo49.mac(geo49.manifold_estimate(X, X[i], tt)["tau"]) for i in rng.choice(flat, 30, replace=False)]))
check("Example 49.6.2: t four times larger -> flat MAC about 1/8 (t^-3/2): ratio in (5, 12)", 5 < r_t[0] / r_t[1] < 12)
# Figure 49.6.3 (b): the failing band along the walls
floor = np.where((lab == 0) & (X0[:, 2] == 0) & (np.linalg.norm(X0[:, :2] - bc[:2], axis=1) > 1.0))[0]
fl = rng.choice(floor, 300, replace=False)
mfl = np.array([geo49.mac(geo49.manifold_estimate(X, X[i], T)["tau"]) for i in fl])
dfl = de[fl]
check("Figure 49.6.3 (b): pass rate < 15% within 0.15 m of a wall, > 90% beyond 0.2 m",
      (mfl[dfl < 0.15] < 0.3).mean() < 0.15 and (mfl[dfl > 0.2] < 0.3).mean() > 0.9)

summary()
