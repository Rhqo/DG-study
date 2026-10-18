"""Figure 37.6.3: explicit versus semi-implicit steps for the discrete length flow (lbar-weighted inner product).

The flower curve sampled with 64 vertices. h0 = 2 / lambda_max(M^{-1} K) at t = 0 (the stability bound of the frozen
linear problem, Proposition 37.6.5). (a) the curves at t = 54 * 0.9 h0 = 0.201 for an explicit and a semi-implicit
run with h = 0.9 h0, and a fine explicit reference (h = h0 / 40); (b) the largest |turning angle| (a measure of
zigzag) against time for explicit h = 0.9 h0 and h0/40, and semi-implicit h = 0.9 h0 and 10 h0.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-6-3-explicit-implicit.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
X0, _ = ddg37.sample("flower", 64)
h0 = ddg37.explicit_critical_step(X0)
assert abs(h0 - 0.0041372) < 1e-6
T_END = 54 * 0.9 * h0


def run(step, h, T=T_END):
    X, t = X0.copy(), 0.0
    hist = [(0.0, np.abs(dd.turning_angles(X, True)).max(), dd.curve_length(X, True))]
    while t < T - 1e-12:
        X = step(X, h)
        t += h
        hist.append((t, np.abs(dd.turning_angles(X, True)).max(), dd.curve_length(X, True)))
    return X, np.array(hist)


Xe, He = run(ddg37.explicit_step, 0.9 * h0)
Xr, Hr = run(ddg37.explicit_step, h0 / 40)
Xs, Hs = run(ddg37.semi_implicit_step, 0.9 * h0)
_, Hs10 = run(ddg37.semi_implicit_step, 10 * h0)
assert He[:, 1].max() > 2.0 and np.any(np.diff(He[:, 2]) > 0)        # zigzag; the length sometimes grows
assert np.all(np.diff(Hs[:, 2]) <= 1e-14) and np.all(np.diff(Hs10[:, 2]) <= 1e-14)   # Proposition 37.6.7
assert Hs[1:, 1].max() < 0.65 and Hr[1:, 1].max() < 0.9

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.2]})
for X, col, ls, lab in ((Xr, C["aux"], "--", "reference"), (Xe, C["normal"], "-", "explicit"),
                        (Xs, C["tangent"], "-", "semi-implicit")):
    Xc = np.vstack([X, X[:1]])
    ax.plot(Xc[:, 0], Xc[:, 1], color=col, ls=ls, lw=1.3, label=lab)
ax.set_aspect("equal")
ax.set_axis_off()
ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=1.4)
ax.set_title("(a)", fontsize=12, loc="left")
bx.plot(He[:, 0], He[:, 1], color=C["normal"], lw=1.3, label=r"explicit $0.9h_0$")
bx.plot(Hr[:, 0], Hr[:, 1], color=C["aux"], lw=1.3, ls="--", label=r"explicit $h_0/40$")
bx.plot(Hs[:, 0], Hs[:, 1], color=C["tangent"], lw=1.5, label=r"semi-implicit $0.9h_0$")
bx.plot(Hs10[:, 0], Hs10[:, 1], "o-", color=C["third"], lw=1.2, ms=3.5, label=r"semi-implicit $10h_0$")
bx.set_xlabel(r"time $t$")
bx.set_ylabel(r"$\max_i |\theta_i|$")
bx.set_ylim(0.0, 3.05)                       # headroom so the legend does not cover the explicit curve
bx.set_yticks(np.arange(0.0, 2.51, 0.5))
bx.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
bx.set_title("(b)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
print(f"h0 = {h0:.6f}; explicit max|theta| {He[:, 1].max():.3f}, length increases {int(np.sum(np.diff(He[:, 2]) > 0))}")
