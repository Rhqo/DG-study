"""Figure 32.4.2: TOTAL and DRIFT for the four discrete flows (Definition 32.4.5) on the jittered flower polygon
(48 vertices gamma(t_i), t_i = 2 pi (i + u_i)/48, u_i uniform in [-0.25, 0.25], seed 11).
Time steps are matched: h = dt for D (pointwise) and h = dt / mean(lbar) for A, B, C (integrated), dt = 5e-5,
up to t = 0.06 (1200 steps). (a) total curvature measured by each definition itself; (b) distance of the vertex
average from its initial position (log scale).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-4-2-total-drift.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402

C = dgfig.COLORS
S = ddg32.KIND_STYLE
F = ddg32.sample("flower", 48, 0.25, 11)[0]
lb = dd.curve_dual_lengths(F, True)
dt, T, every = 5e-5, 0.06, 10
steps = int(round(T / dt))
tt = dt * every * np.arange(steps // every + 1)
tot, drift = {}, {}
for k in "ABCD":
    h = dt if k == "D" else dt / lb.mean()
    tr = ddg32.flow(F, k, h, steps, every=every)
    assert len(tr) == len(tt)
    me = min(dd.curve_edge_lengths(X, True).min() for X in tr) / dd.curve_edge_lengths(F, True).min()
    assert me > 0.1, (k, me)                                        # no edge collapses in this time range
    tot[k] = np.array([ddg32.own_total(X, k) for X in tr])
    drift[k] = np.array([np.linalg.norm(X.mean(0) - F.mean(0)) for X in tr])
assert np.allclose(tot["A"], 2 * np.pi, atol=1e-12)                 # Proposition 32.4.8
assert np.all(drift["B"] < 1e-14)                                    # Proposition 32.4.9
assert np.ptp(tot["B"]) > 0.05 and np.ptp(tot["C"]) > 0.05 and np.ptp(tot["D"]) > 0.05
assert all(drift[k][-1] > 1e-6 for k in "ACD")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.1))
for k in "ABCD":
    ax.plot(tt, tot[k], color=S[k]["color"], ls=S[k]["ls"], lw=1.6, label=k)
ax.axhline(2 * np.pi, color=C["aux"], lw=dgfig.LW["aux"], ls=":", zorder=0)
ax.text(0.0605, 2 * np.pi + 0.006, r"$2\pi$", fontsize=11, color=C["aux"], va="bottom")
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"total curvature")
ax.legend(frameon=False, ncol=2, fontsize=9.5, loc="upper right")
ax.set_ylim(6.1, 6.6)
ax.set_title("(a)", fontsize=12)

floor = 1e-17
for k in "ABCD":
    bx.semilogy(tt[1:], np.maximum(drift[k][1:], floor), color=S[k]["color"], ls=S[k]["ls"], lw=1.6, label=k)
bx.set_xlabel(r"time $t$")
bx.set_ylabel("drift of the vertex average")
bx.set_ylim(1e-17, 1e-1)
bx.text(0.03, 3e-15, "B: round-off only", fontsize=9.5, color=S["B"]["color"], ha="center")
bx.legend(frameon=False, ncol=2, fontsize=9.5, loc="center right")
bx.set_title("(b)", fontsize=12)
for a in (ax, bx):
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
for k in "ABCD":
    print(k, "total: start %.4f min %.4f max %.4f end %.4f" % (tot[k][0], tot[k].min(), tot[k].max(), tot[k][-1]),
          " drift end %.1e" % drift[k][-1])
