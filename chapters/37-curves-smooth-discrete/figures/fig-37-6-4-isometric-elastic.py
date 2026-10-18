"""Figure 37.6.4: isometric elastic flow of the flower polygon (64 vertices) with h = 0.2.

Each step shrinks the turning angles (theta <- theta / (1 + h)), projects them back onto the closing conditions
(37.4.4) with the edge lengths fixed, and rebuilds the polygon by (37.4.3). (a) the polygons after 0, 2, 5, 10, 40
steps (centered at their vertex average); (b) the discrete elastic energy 1/2 sum theta_i^2 / lbar_i against the step
number, with the smooth limit 2 pi^2 / L of a circle of the same length.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-6-4-isometric-elastic.py
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
ell = dd.curve_edge_lengths(X0, True)
lbar = dd.curve_dual_lengths(X0, True)
th = dd.turning_angles(X0, True)
L = ell.sum()
E = [ddg37.elastic_energy(th, lbar)]
snaps = {0: X0}
for k in range(1, 41):
    th = ddg37.isometric_elastic_step(th, ell, 0.2)
    E.append(ddg37.elastic_energy(th, lbar))
    if k in (2, 5, 10, 40):
        X = ddg37.rebuild_closed(ell, th)
        assert np.abs(dd.curve_edge_lengths(X, True) - ell).max() < 1e-12          # isometric
        assert np.abs(dd.turning_angles(X, True) - th).max() < 1e-12               # closes with these angles
        snaps[k] = X
E = np.array(E)
assert np.all(np.diff(E) < 0) and E[0] > 48 and E[-1] < 2.3

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
cols = {0: C["main"], 2: C["normal"], 5: C["accent"], 10: C["third"], 40: C["tangent"]}
for k, X in snaps.items():
    Xc = X - X.mean(0)
    Xc = np.vstack([Xc, Xc[:1]])
    ax.plot(Xc[:, 0], Xc[:, 1], color=cols[k], lw=1.6 if k in (0, 40) else 1.1, label=r"$%d$" % k)
ax.set_aspect("equal")
ax.set_axis_off()
ax.legend(title="step", fontsize=9, title_fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(0.98, 0.5))
ax.set_title("(a)", fontsize=12, loc="left")
bx.semilogy(np.arange(41), E, "o-", color=C["main"], ms=3, lw=1.2, label=r"$\frac{1}{2}\sum_i\theta_i^2/\bar\ell_i$")
bx.axhline(2 * np.pi ** 2 / L, color=C["aux"], ls="--", lw=1.0, label=r"circle: $2\pi^2/L$")
bx.set_xlabel("step")
bx.legend(fontsize=9.5, frameon=False)
bx.set_title("(b)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
print(f"E: {E[0]:.3f} -> {E[-1]:.4f}; circle {2 * np.pi ** 2 / L:.4f}; L = {L:.4f}")
