"""Verification for Section 50.5 (Implementation and Experiments).

Every number quoted from [ManifoldGS26] is checked against the transcription in lib/mgs50.py, and every number we
derived (relative changes, IoU, F1, shares, random-normal statistics) is recomputed.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-5-experiments.py
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

PAGE = HERE.parents[1] / "50-5-experiments.html"
html = PAGE.read_text()

# ---------------------------------------------------------------- the transcribed tables appear on the page
for name, row in g.TABLE1.items():
    check(f"Table 50.5.4: row {name} present", all(f"{v}" in html for v in row))
for s, (a, b, gain, dp) in g.TABLE2.items():
    check(f"Table 50.5.5: scan {s} present", f"{a:.4f}" in html and f"{b:.4f}" in html and f"+{gain:.2f}%" in html
          and f"+{dp:.3f}" in html)
for s, row in g.TABLE3.items():
    check(f"Table 50.5.6: scan {s} present", f"{row[0]}%" in html and f"{row[1]:.2f}%" in html and f"{row[3]:.2f}" in html)
for s, rows in g.TABLE4.items():
    for m, (f, c) in rows.items():
        cell = f"<td>{s}</td><td>{m}</td>"
        check(f"Table 50.5.8: scan {s} {m} present", cell in html and f"{f}" in html and f"{c}" in html)
for m, vals in g.SIMPLIFIED_FLOATER.items():
    check(f"Table 50.5.9: {m} present", f"<tr><td>{m}</td>" + "".join(f"<td>{v:.2f}</td>" for v in vals) + "</tr>" in html)
for s, (sup, weak, ins, tot) in g.FISHER.items():
    check(f"Table 50.5.10: scan {s} present", f"<tr><td>{s}</td><td>{sup}</td><td>{weak}</td><td>{ins}</td><td>{tot}</td></tr>" in html)
    check(f"Table 50.5.10: scan {s} the three classes add up to the stated total", sup + weak + ins == tot)

# ---------------------------------------------------------------- §7.1
close("Table 50.5.2: normal angle / 23 and normal curl / 59", [21.84 / 0.96, 1.410 / 0.024], [23, 59], 0.3)
close("Table 50.5.3: coverage +8.49 points, Chamfer -36.9 % and -42.2 %",
      [63.44 - 54.95, 100 * (1 - 0.15211 / 0.24090), 100 * (1 - 0.16346 / 0.28287)], [8.49, 36.9, 42.2], 0.05)
close("Table 50.5.3: PSNR -0.142 dB, SSIM -0.0144", [21.3542 - 21.4959, 0.5395 - 0.5539], [-0.142, -0.0144], 5e-4)

# ---------------------------------------------------------------- Table 1 (§7.2)
c0, n0, v0, p0 = g.TABLE1["RGB"]
c1, n1, v1, p1 = g.TABLE1["RGB + comp."]
c2, n2, v2, p2 = g.TABLE1["+ adapt. comp."]
c3, n3, v3, p3 = g.TABLE1["+ depth oracle"]
close("Section 50.5 (Table 1): comp. Chamfer -15.7 %, Varif. -21.6 %, PSNR -0.355",
      [100 * (1 - c1 / c0), 100 * (1 - v1 / v0), p1 - p0], [15.7, 21.6, -0.355], 0.05)
close("Section 50.5 (Table 1): adapt. comp. Chamfer -23.7 %, Varif. -24.4 %, PSNR +2.247",
      [100 * (1 - c2 / c0), 100 * (1 - v2 / v0), p2 - p0], [23.7, 24.4, 2.247], 0.05)
close("Section 50.5 (Table 1): oracle Chamfer -68.3 % (and -58.5 % against adapt.), Varif. -68.7 %, PSNR +2.846",
      [100 * (1 - c3 / c0), 100 * (1 - c3 / c2), 100 * (1 - v3 / v0), p3 - p0], [68.3, 58.5, 68.7, 2.846], 0.05)
check("Section 50.5 (Table 1): adapt. comp. normal angle is worse than comp.", n2 > n1)
check("Section 50.5 (Table 1): comp. lowers PSNR", p1 < p0)
ang = g.random_normal_angles()
close("Figure 50.5.1: random unoriented normal, mean 57.3 and median 60 degrees", [ang.mean(), np.median(ang)],
      [math.degrees(1), 60.0], 0.2)
close("Exercise 50.5.3: mean of theta sin theta on [0, pi / 2] = 1 rad",
      np.trapezoid(np.linspace(0, math.pi / 2, 20001) * np.sin(np.linspace(0, math.pi / 2, 20001)),
                   np.linspace(0, math.pi / 2, 20001)), 1.0, 1e-8)
probs = [1 - math.cos(math.radians(a)) for a in (n0, n1, n2, n3)]
close("Section 50.5: P(random angle <= Norm.) = 0.58, 0.43, 0.44, 0.012", probs, [0.576, 0.429, 0.440, 0.0119], 6e-4)
emp = [np.mean(ang <= a) for a in (n0, n3)]
close("Section 50.5: the empirical shares agree with 1 - cos", emp, [probs[0], probs[3]], 3e-3)
check("Section 50.5: RGB-only normal angles lie within 10 degrees of the random mean", all(abs(a - 57.3) < 10 for a in (n0, n1, n2)))

# ---------------------------------------------------------------- Table 2 and the DTU rule (§7.3)
gains = {s: 100 * (a - b) / a for s, (a, b, _, _) in g.TABLE2.items()}
close("Table 50.5.5: gains 1.620 % and 4.326 %, mean 2.973 %", [gains[24], gains[65], np.mean(list(gains.values()))],
      [1.620, 4.326, 2.973], 5e-4)
check("Exercise 50.5.1: 4.326 rounds to 4.33, not to the printed 4.32", round(gains[65], 2) == 4.33)
w = (0.271 + 0.38) / (1.98 + 0.38)
close("Section 50.5 (§7.3): accuracy weight 0.276, arithmetic mean 0.80 %", [w, (1.98 - 0.38) / 2], [0.276, 0.80], 5e-4)
close("Section 50.5 (§7.3): the weighted mean reproduces 0.271 %", w * 1.98 - (1 - w) * 0.38, 0.271, 1e-12)

# ---------------------------------------------------------------- Table 3 and the external targets (§7.5)
for s, (nv, iou, p, r) in g.EXTERNAL.items():
    close(f"Table 50.5.7 (scan {s}): IoU = 1 / (1/p + 1/r - 1)", g.iou_from_pr(p, r), iou, 6e-4)
close("Exercise 50.5.2: scan24 1/p + 1/r - 1 = 3.655, IoU 0.2736",
      [1 / 0.619 + 1 / 0.329 - 1, g.iou_from_pr(0.619, 0.329)], [3.655, 0.2736], 5e-4)
for s in (24, 65, 105):
    check(f"Table 50.5.6 (scan {s}): floater in Table 3 rounds (half up) to Table 4",
          abs(g.TABLE3[s][1] - g.TABLE4[s]["ours"][0]) <= 0.05 + 1e-9)
close("Section 50.5 (§7.5): total patches 164 / 0.58 to 234 / 0.42 = about 280 to 560", [164 / 0.58, 234 / 0.42],
      [283, 557], 1.0)

# ---------------------------------------------------------------- Table 4 and F1 (§7.6)
F1 = {s: {m: g.f1(f, c) for m, (f, c) in rows.items()} for s, rows in g.TABLE4.items()}
expect = {24: (0.510, 0.741, 0.326, 0.029), 65: (0.416, 0.623, 0.593, 0.049), 105: (0.579, 0.654, 0.664, 0.572)}
for s, vals in expect.items():
    close(f"Table 50.5.8 (scan {s}): F1* ours, 2DGS, SuGaR, Poisson",
          [round(F1[s][m], 3) for m in ("ours", "2DGS", "SuGaR", "Poisson")], vals, 1e-9)
    check(f"Table 50.5.8 (scan {s}): ours has the lowest floater", min(g.TABLE4[s], key=lambda m: g.TABLE4[s][m][0]) == "ours")
    check(f"Table 50.5.8 (scan {s}): ours has the lowest coverage", min(g.TABLE4[s], key=lambda m: g.TABLE4[s][m][1]) == "ours")
    check(f"Table 50.5.8 (scan {s}): 2DGS has a higher F1 than ours", F1[s]["2DGS"] > F1[s]["ours"])
check("Table 50.5.8: SuGaR beats ours in F1 on scan65 and scan105", F1[65]["SuGaR"] > F1[65]["ours"] and F1[105]["SuGaR"] > F1[105]["ours"])
close("Section 50.5: scan24 floater gaps 2.8 and 0.32 points", [21.1 - 18.3, 18.87 - 18.55], [2.8, 0.32], 1e-9)
check("Section 50.5: Poisson has the largest coverage on scan105", max(g.TABLE4[105], key=lambda m: g.TABLE4[105][m][1]) == "Poisson")
close("Figure 50.5.2: precision range of ours 81.7 to 99.1, coverage 26.3 to 41.0",
      [min(100 - g.TABLE4[s]["ours"][0] for s in g.TABLE4), max(100 - g.TABLE4[s]["ours"][0] for s in g.TABLE4),
       min(g.TABLE4[s]["ours"][1] for s in g.TABLE4), max(g.TABLE4[s]["ours"][1] for s in g.TABLE4)],
      [81.7, 99.1, 26.3, 41.0], 1e-9)
# Exercise 50.5.4: the two extremes of cutting the 2DGS mesh of scan65 to coverage 26.3
f2, cov2 = g.TABLE4[65]["2DGS"]
check("Exercise 50.5.4: uniform cut keeps 13.7 % (> 0.9 %), dropping floaters first gives 0 % (< 0.9 %)",
      f2 > g.TABLE4[65]["ours"][0] > 0.0 and cov2 > 26.3)

# ---------------------------------------------------------------- §7.7
for s_i, s in enumerate((24, 65, 105)):
    check(f"Table 50.5.9 (scan {s}): 2DGS improves after simplification",
          g.SIMPLIFIED_FLOATER["2DGS"][s_i] < g.TABLE4[s]["2DGS"][0])
close("Section 50.5 (§7.7): Figure 2 denominator 3847 / 0.1972 = 19508", 3847 / 0.1972, 19508, 1.0)
close("Section 50.5 (§7.7): 3847 / 19508 = 19.72 %", 100 * 3847 / 19508, 19.72, 0.005)

# ---------------------------------------------------------------- §7.8
close("Table 50.5.10: supported shares 89.7, 89.0, 89.7 %",
      [100 * g.FISHER[s][0] / g.FISHER[s][3] for s in (24, 65, 105)], [89.7, 89.0, 89.7], 0.05)

# ---------------------------------------------------------------- Implementation 50.5.1
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-50-5-1")
check("Implementation 50.5.1: at most 30 lines", dgcode.code_lines(PAGE, "code-50-5-1") <= 30)
check("Implementation 50.5.1: printed output equals the page",
      buf.getvalue().strip() == dgcode.code_block(PAGE, "out-50-5-1").strip())
check("Implementation 50.5.1: the tables in the code equal the library transcription",
      all(ns["table4"][s] == g.TABLE4[s] for s in g.TABLE4)
      and all(ns["table2"][s] == g.TABLE2[s][:2] for s in g.TABLE2)
      and all(ns["fisher"][s] == g.FISHER[s][:3] for s in g.FISHER)
      and all(ns["ext"][s] == g.EXTERNAL[s][2:] for s in g.EXTERNAL))

summary()
