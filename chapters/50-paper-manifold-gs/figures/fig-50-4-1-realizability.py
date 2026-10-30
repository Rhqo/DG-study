"""Figure 50.4.1: compatibility residuals see only the realizability part of the error (Example 50.4.5, Example 50.4.7).

Support: the graph chart x(u, v) = (u, v, sqrt(1 - u^2 - v^2)) of the unit sphere over [-1/2, 1/2]^2 (the ground truth V_*).
Normal fields: A = the true normal, B = the true normal twisted by 0.3 (-v, u, 0), C = the normal of a more curved surface
(N + (u, v, 0), normalised), D = the plane x = (u, v, 1) with its own normal (0, 0, 1).
(a) RMS of the residuals of Definition 50.4.2 (our formalisation; normal curl equals antisymmetry, Proposition 50.4.3).
(b) the two terms of the decomposition (11) measured by the kernel varifold distance (22) with sigma = 0.1, tau = 0.5
(our choice): Pi_Z V = the sphere cap for B and C (replace the normals by the support normals), Pi_Z V = V for D.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-4-1-realizability.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import mgs50 as g  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
u, v, cases = g.residual_cases()
R = {k: g.residuals(X, n, u, v) for k, (X, n) in cases.items()}
at = g.decomposition_atoms()
SIG, TAU = 0.1, 0.5
XS, NS, WS = at["S"]
first = {"B": g.mmd(*at["B"], XS, NS, WS, SIG, TAU), "C": g.mmd(*at["C"], XS, NS, WS, SIG, TAU), "D": 0.0}
second = {"B": 0.0, "C": 0.0, "D": g.mmd(*at["D"], XS, NS, WS, SIG, TAU)}
# self-checks: A and D have zero residuals; curl = antisymmetry; B is the only one with antisymmetry; C has a large
# Gauss residual; D is far from the truth although it is compatible
for k in ("A", "D"):
    assert max(R[k].values()) < 1e-10
for k in "ABCD":
    assert abs(R[k]["curl"] - R[k]["antisymmetry"]) < 1e-10
assert R["B"]["antisymmetry"] > 0.3 and R["C"]["antisymmetry"] < 1e-10 and R["C"]["gauss"] > 1.0
assert second["D"] > first["C"] > first["B"] > 0.03

keys = ("disagreement", "antisymmetry", "gauss", "codazzi")
labels = ("disagree", "antisym.\n(= curl)", "Gauss", "Codazzi")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(len(keys))
w = 0.26
for j, (k, c) in enumerate(zip("BCD", (C["tangent"], C["normal"], C["third"]))):
    vals = [R[k][key] for key in keys]
    ax.bar(x + (j - 1) * w, vals, w, color=c, label=f"case {k}")
for xi in x:
    ax.text(xi + w, 0.03, "0", ha="center", fontsize=8, color=C["third"])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("RMS residual")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.set_title("(a) compatibility residuals", fontsize=11)
ks = ("B", "C", "D")
f1 = [first[k] for k in ks]
f2 = [second[k] for k in ks]
bx.bar(ks, f1, 0.55, color=C["tangent"], label=r"$d_V(V, \Pi_Z V)$")
bx.bar(ks, f2, 0.55, bottom=f1, color=C["accent"], label=r"$d_V(\Pi_Z V, V_*)$")
for i, k in enumerate(ks):
    bx.text(i, f1[i] + f2[i] + 0.006, f"{f1[i] + f2[i]:.3f}", ha="center", fontsize=8.5)
bx.set_ylim(0, 0.3)
bx.set_xlabel("case")
bx.set_ylabel("kernel distance")
bx.legend(loc="upper left", fontsize=8.5, frameon=False)
bx.set_title("(b) the two terms of (11)", fontsize=11)
fig.tight_layout()
dgfig.save(fig, __file__)
for k in "ABCD":
    print(k, {key: round(val, 4) for key, val in R[k].items()})
print("first", {k: round(val, 4) for k, val in first.items()}, "second", {k: round(val, 4) for k, val in second.items()})
