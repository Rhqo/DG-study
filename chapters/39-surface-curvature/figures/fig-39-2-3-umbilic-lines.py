"""Figure 39.2.3: lines of curvature near an umbilical point.

Monge-form graphs z = f(x, y) = (x^2 + y^2)/2 + (x^3 + 3 c x y^2)/6 with an umbilic at the origin (k1 = k2 = 1,
upward normal). (a) c = -1: index -1/2, three separatrices for each family. (b) c = 1/5: index +1/2, one separatrix
for each family. Drawn in the xy-plane over [-0.3, 0.3]^2 (projection of the graph). Blue: lines of the k1 family,
green: lines of the k2 family; thick: separatrices (lines of curvature that end at the umbilic), traced outward from
radius 0.004 along the radial directions found by ddg39.separatrix_directions.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-2-3-umbilic-lines.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg39  # noqa: E402

C = dgfig.COLORS
BOX = 0.3
CASES = [(-1.0, "(a)", -0.5, 3), (0.2, "(b)", 0.5, 1)]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
for ax, (c, title, idx_expect, nsep) in zip(axes, CASES):
    derivs = ddg39.monge(1.0, 0.0, c, 0.0)
    k1, k2, _, _ = ddg39.graph_principal(derivs, 0.0, 0.0)
    assert abs(k1 - 1) < 1e-12 and abs(k2 - 1) < 1e-12                      # umbilic at the origin
    for which in (0, 1):
        assert abs(ddg39.line_field_index(derivs, which=which) - idx_expect) < 1e-6
    gx = np.linspace(-BOX, BOX, 61)
    gaps = [ddg39.graph_principal(derivs, x, y)[0] - ddg39.graph_principal(derivs, x, y)[1]
            for x in gx for y in gx if x * x + y * y > 0.05 ** 2]
    assert min(gaps) > 0.01                                               # no other umbilic in the box
    cols = {0: C["tangent"], 1: C["third"]}
    for which in (0, 1):
        seps = ddg39.separatrix_directions(derivs, which=which, rad=0.004)
        assert len(seps) == nsep, (c, which, len(seps))
        seeds = [(r * np.cos(a), r * np.sin(a)) for r in (0.07, 0.15, 0.24) for a in np.linspace(0, 2 * np.pi, 7)[:-1] + 0.3]
        for sd in seeds:
            for dd in (1, -1):
                L = ddg39.trace_line(derivs, sd, which, step=0.004, nmax=260, box=BOX, stop_rad=0.006, direction=dd)
                ax.plot(L[:, 0], L[:, 1], color=cols[which], lw=0.55, alpha=0.7)
        for t0 in seps:
            p0 = 0.004 * np.array([np.cos(t0), np.sin(t0)])
            L = ddg39.trace_line(derivs, p0, which, step=0.004, nmax=300, box=BOX,
                                 direction=1.0 if (ddg39.graph_principal(derivs, *p0)[2 + which] @ p0) > 0 else -1.0)
            ax.plot(L[:, 0], L[:, 1], color=cols[which], lw=2.0)
    ax.plot(0, 0, "o", color=C["main"], ms=5, zorder=10)
    ax.set_xlim(-BOX, BOX)
    ax.set_ylim(-BOX, BOX)
    ax.set_aspect("equal")
    ax.set_xticks([-0.3, 0, 0.3])
    ax.set_yticks([-0.3, 0, 0.3])
    ax.set_xlabel(r"$x$")
    ax.set_title(title + rf"  $c = {c:g}$", loc="left", fontsize=12)
axes[0].set_ylabel(r"$y$", rotation=0)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.92, wspace=0.18)
dgfig.save(fig, __file__)
