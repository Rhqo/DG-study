"""Figure 49.7.1: the upsampling (6) of [GeoSplat25] as a refinement without a mass ledger (Section 49.7).

(a) 1500 uniform random points on the unit square (a flat patch, every point passes the flatness test); each adds
    the midpoints with its 10 nearest neighbours (paper (6) with KNN = 10, App. F.1). Shown: the window
    [0.35, 0.65]^2; originals (black), new points made by two directed pairs (i chose j and j chose i: the same
    location twice, orange) and by one pair (blue); dashed circles: the two largest empty discs in the window after
    the upsampling (probes on a grid), which were empty before as well.
(b) the strip [-1, 1] x [0, 0.6] with 2500 uniform points; points with x < 0 pass the test, points with x > 0 do not.
    Number of Gaussians per unit area, relative to the original density, against x (bins of width 0.1): before,
    after counting every new Gaussian (weight 1 each, as in the paper), and after merging coincident ones. The line
    at 1 is what a conservative refinement (Proposition 50.3.1: children carry the parent's mass) keeps.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-7-1-upsampling-ledger.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (1500, 2))
M, pairs, mult = geo49.upsample_midpoints(X, np.ones(len(X), bool), k=10)
dup = (mult == 2).mean()
print("added", len(M), "fraction made twice", round(dup, 3), "distinct new locations", len(M) - (mult == 2).sum() // 2)
assert len(M) == 10 * len(X) and 0.8 < dup < 0.92
# empty-space radius (distance from a probe to the nearest point) on a grid of probes, before and after
g = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 161), np.linspace(0.1, 0.9, 161)), -1).reshape(-1, 2)


def empty_radius(P):
    return np.sqrt(((g[:, None, :] - P[None]) ** 2).sum(-1).min(1))


e0, e1 = empty_radius(X), empty_radius(np.vstack([X, M]))
print("empty radius before: median %.4f max %.4f; after: median %.4f max %.4f" % (np.median(e0), e0.max(), np.median(e1), e1.max()))
assert np.median(e1) < 0.6 * np.median(e0) and e1.max() > 0.95 * e0.max()

S = np.c_[rng.uniform(-1, 1, 2500), rng.uniform(0, 0.6, 2500)]
Ms, ps, ms = geo49.upsample_midpoints(S, S[:, 0] < 0, k=10)
bins = np.linspace(-1, 1, 21)
area = 0.1 * 0.6
dens0 = len(S) / (2 * 0.6)
before = np.histogram(S[:, 0], bins)[0] / area / dens0
after = (np.histogram(S[:, 0], bins)[0] + np.histogram(Ms[:, 0], bins)[0]) / area / dens0
keep = np.r_[True, np.any(np.abs(np.diff(Ms[np.lexsort(Ms.T[::-1])], axis=0)) > 1e-12, axis=1)]
Mu = Ms[np.lexsort(Ms.T[::-1])][keep]
after_u = (np.histogram(S[:, 0], bins)[0] + np.histogram(Mu[:, 0], bins)[0]) / area / dens0
xc = 0.5 * (bins[1:] + bins[:-1])
print("left mean after", round(after[xc < -0.3].mean(), 2), "unique", round(after_u[xc < -0.3].mean(), 2), "right", round(after[xc > 0.3].mean(), 2))
assert abs(after[xc < -0.3].mean() - 11) < 1.0 and abs(after[xc > 0.3].mean() - 1) < 0.3
assert 5.0 < after_u[xc < -0.3].mean() < 8.0

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
w = (0.35, 0.65)
inw = lambda P: np.all((P > w[0]) & (P < w[1]), axis=1)  # noqa: E731
P1, P2 = M[(mult == 1) & inw(M)], M[(mult == 2) & inw(M)]
ax.scatter(P2[:, 0], P2[:, 1], s=9, color=C["normal"], lw=0, label="new, made twice")
ax.scatter(P1[:, 0], P1[:, 1], s=9, color=C["tangent"], lw=0, label="new, made once")
ax.scatter(X[inw(X), 0], X[inw(X), 1], s=16, color=C["main"], lw=0, label="original")
inwin = np.all((g > w[0] + 0.03) & (g < w[1] - 0.03), axis=1)
order = np.argsort(-np.where(inwin, e1, 0))
centres = []
for c in order:
    if len(centres) == 2:
        break
    if all(np.linalg.norm(g[c] - g[d]) > e1[c] + e1[d] for d in centres):
        centres.append(c)
for c in centres:
    ax.add_patch(plt.Circle(g[c], e1[c], fill=False, ls="--", color=C["aux"], lw=1.0))
ax.set_xlim(w)
ax.set_ylim(w)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("(a) midpoints with 10 nearest neighbours", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7.5, frameon=False, markerscale=1.5,
          handletextpad=0.2, columnspacing=0.8)
ax = axes[1]
ax.step(xc, before, where="mid", color=C["aux"], lw=1.2, ls="--", label="before")
ax.step(xc, after, where="mid", color=C["main"], lw=1.6, label="after (paper)")
ax.step(xc, after_u, where="mid", color=C["tangent"], lw=1.4, ls="-.", label="after, duplicates merged")
ax.axhline(1, color=C["third"], lw=1.0, ls=":")
ax.text(0.98, 2.0, "conservative: stays 1", ha="right", fontsize=7.5, color=C["third"])
ax.axvline(0, color=C["aux"], lw=0.8)
ax.text(-0.5, 12.3, "passes (flat)", ha="center", fontsize=8.5)
ax.text(0.5, 12.3, "fails", ha="center", fontsize=8.5)
ax.set_xlim(-1, 1)
ax.set_ylim(0, 13.5)
ax.set_xlabel("x")
ax.set_ylabel("Gaussians per area / before")
ax.set_title("(b) mass across the boundary of the flat set", fontsize=10)
ax.legend(loc="upper right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.86))
fig.tight_layout()
dgfig.save(fig, __file__)
