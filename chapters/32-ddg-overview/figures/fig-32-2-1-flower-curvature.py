"""Figure 32.2.1: the flower curve r(t) = 1 + 0.3 cos 5t and its signed curvature (Example 32.2.1(c)).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-2-1-flower-curvature.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg32  # noqa: E402

C = dgfig.COLORS
J = ddg32.J

t = np.linspace(0, 2 * np.pi, 4001)
P = ddg32.point("flower", t)
k = ddg32.kappa("flower", t)
# inflection points: zeros of kappa_s (10 of them); locate by sign changes
idx = np.where(np.sign(k[:-1]) != np.sign(k[1:]))[0]
assert len(idx) == 10, len(idx)
assert abs(ddg32.kappa("flower", np.array([0.0]))[0] - 5.207100591715976) < 1e-9
assert abs(ddg32.kappa("flower", np.array([np.pi / 5]))[0] + 13.877551020408163) < 1e-9

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) the curve: kappa_s > 0 black, kappa_s < 0 orange; t and n_s at t = 0 and t = pi/5
pos = k >= 0
Pm = P.copy()
Pm[~pos] = np.nan
Pn = P.copy()
Pn[pos] = np.nan
ax.plot(Pm[:, 0], Pm[:, 1], color=C["main"], lw=dgfig.LW["main"])
ax.plot(Pn[:, 0], Pn[:, 1], color=C["accent"], lw=2.6)
ti = 0.5 * (t[idx] + t[idx + 1])
Q = ddg32.point("flower", ti)
ax.plot(Q[:, 0], Q[:, 1], "o", ms=3.5, mfc="white", mec=C["main"], mew=0.8, zorder=5)
for t0, lab, loff in ((0.0, "tip", (0.10, -0.30)), (np.pi / 5, "waist", (-0.04, -0.19))):
    p = ddg32.point("flower", np.array([t0]))[0]
    d = ddg32.point("flower", np.array([t0 + 1e-6]))[0] - ddg32.point("flower", np.array([t0 - 1e-6]))[0]
    tt = d / np.linalg.norm(d)
    nn = J @ tt
    assert abs(np.dot(tt, nn)) < 1e-12 and abs(np.linalg.norm(nn) - 1) < 1e-12
    s = 0.36
    for vec, col, name in ((tt, C["tangent"], r"$\mathbf{t}$"), (nn, C["normal"], r"$\mathbf{n}_s$")):
        ax.annotate("", xy=p + s * vec, xytext=p,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, mutation_scale=10))
        q = p + (s + 0.13) * vec
        ax.text(q[0], q[1], name, color=col, fontsize=12, ha="center", va="center")
    ax.plot(*p, "o", color=C["main"], ms=3.5, zorder=6)
    ax.text(p[0] + loff[0], p[1] + loff[1], lab, fontsize=10, ha="left", va="center")
ax.set_aspect("equal")
ax.set_xlim(-1.45, 1.85)
ax.set_ylim(-1.45, 1.45)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

# (b) kappa_s(t)
bx.axhline(0, color=C["aux"], lw=dgfig.LW["aux"])
kp = np.where(pos, k, np.nan)
kn = np.where(~pos, k, np.nan)
bx.plot(t, kp, color=C["main"], lw=1.6)
bx.plot(t, kn, color=C["accent"], lw=2.2)
bx.plot([0, np.pi / 5], [5.2071, -13.8776], "o", color=C["main"], ms=4)
bx.text(0.15, 5.6, r"$5.21$", fontsize=10)
bx.text(np.pi / 5 + 0.12, -14.2, r"$-13.88$", fontsize=10)
bx.plot(ti, np.zeros_like(ti), "o", ms=3.5, mfc="white", mec=C["main"], mew=0.8)
bx.set_xlim(0, 2 * np.pi)
bx.set_xticks([0, np.pi / 5, np.pi, 2 * np.pi])
bx.set_xticklabels(["0", r"$\pi/5$", r"$\pi$", r"$2\pi$"])
bx.set_xlabel(r"$t$")
bx.set_ylabel(r"$\kappa_s(t)$")
bx.set_title("(b)", fontsize=12)
for sp_ in ("top", "right"):
    bx.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
