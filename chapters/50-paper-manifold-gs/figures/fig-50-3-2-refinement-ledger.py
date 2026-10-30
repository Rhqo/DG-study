"""Figure 50.3.2: the same refinement schedules under a geometric mass ledger and under "opacity as area".

Start: 128 flat Gaussians tiling the unit sphere (the n_theta = 8 tiling of Section 50.2), scales s1 = s2 = sqrt(A(C_i)/pi),
s3 = 0.01, opacity 0.7, masses q_i = A(C_i). 300 random steps of three schedules: split only on the northern half
(z > 0), clone only on the northern half, and split or clone with probability 1/2 everywhere (seed 0). Split and clone
follow the 3DGS rules ([Kerbl23]: split = two children drawn from the parent Gaussian, scales / 1.6, same opacity; clone =
a copy); the ledger recentres the children and gives each child q/2 with the parent's plane (Proposition 50.3.1).
(a) total mass relative to 4 pi: the ledger sum q_i and the opacity area kappa sum alpha_i s_i1 s_i2 (kappa makes it
4 pi at the start). (b) the kernel varifold distance (eq. (22), sigma = 0.25, tau = 1, our choice) between the
normalised varifold and the sphere (reference: the n_theta = 32 tiling).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-3-2-refinement-ledger.py
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
cols = (C["tangent"], C["normal"], C["third"])
runs = [(name, *g.schedule_run(p, reg)) for name, p, reg in g.SCHEDULES]
# self-checks: the ledger keeps the total mass exactly; clone-only leaves the ledger varifold unchanged; the opacity
# area drifts and its kernel distance ends larger than the ledger's in every schedule
for name, h, n in runs:
    assert np.allclose(h[:, 1], 1.0, atol=1e-12)
    assert abs(h[-1, 2] - 1.0) > 0.1
    assert h[-1, 4] > 1.5 * h[-1, 3]
assert np.allclose(runs[1][1][:, 3], runs[1][1][0, 3], atol=1e-12)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.2))
ax.axhline(1.0, color=C["main"], lw=1.8, label=r"ledger $\sum q_i$ (all)")
for (name, h, n), c in zip(runs, cols):
    ax.plot(h[:, 0], h[:, 2], "--", color=c, lw=1.3, label=name)
ax.set_xlabel("refinement step")
ax.set_ylabel(r"total mass / $4\pi$")
ax.set_ylim(0, 3.8)
ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.6)
ax.set_title("(a) total mass", fontsize=11)
for (name, h, n), c in zip(runs, cols):
    bx.plot(h[:, 0], h[:, 3], "-", color=c, lw=1.4)
    bx.plot(h[:, 0], h[:, 4], "--", color=c, lw=1.3)
bx.plot([], [], "-", color=C["main"], label="ledger")
bx.plot([], [], "--", color=C["main"], label="opacity as area")
bx.set_xlabel("refinement step")
bx.set_ylabel(r"MMD to the sphere")
bx.set_ylim(0, 0.16)
bx.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.8)
bx.set_title("(b) kernel varifold distance", fontsize=11)
fig.tight_layout()
dgfig.save(fig, __file__)
for name, h, n in runs:
    print(f"{name:18s} atoms {n}: opacity area {h[-1, 2]:.4f}, MMD ledger {h[0, 3]:.4f} -> {h[-1, 3]:.4f} "
          f"(max {h[:, 3].max():.4f}), opacity {h[-1, 4]:.4f} (max {h[:, 4].max():.4f})")
