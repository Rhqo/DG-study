"""Figure 32.3.3: the four discrete curvatures on samplings gamma_i = gamma(2 pi i/n) of the ellipse (2 cos t, sin t).
(a) pointwise values kappa^X_i / lbar_i (X = A, B, C) and kappa^D_i for n = 16 against the smooth kappa_s(t);
(b) the largest error max_i |pointwise value - kappa_s(t_i)| against n (log-log); all four decay like n^-2.

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-3-3-convergence.py
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
NAMES = {"A": "turning angle", "B": "length variation", "C": "Steiner", "D": "osculating circle"}


def pointwise(X):
    lb = dd.curve_dual_lengths(X, True)
    return {k: f(X, True) / (1.0 if k == "D" else lb) for k, f in dd.DISCRETE_CURVATURES.items()}


fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.1))

# (a) n = 16
n = 16
X, tv = ddg32.sample("ellipse", n)
pw = pointwise(X)
ts = np.linspace(0, np.pi, 400)
ax.plot(ts, ddg32.kappa("ellipse", ts), color=C["main"], lw=1.4, label=r"$\kappa_s(t)$")
sel = tv <= np.pi + 1e-12
offs = {"A": -0.035, "B": -0.012, "C": 0.012, "D": 0.035}
for k in "ABCD":
    ax.plot(tv[sel] + offs[k], pw[k][sel], ls="none", marker=S[k]["marker"], color=S[k]["color"], ms=4.5,
            mfc="white" if k in "BD" else S[k]["color"], label=k)
ax.set_xticks([0, np.pi / 2, np.pi])
ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
ax.set_xlabel(r"$t$")
ax.set_ylabel("curvature")
ax.legend(frameon=False, fontsize=9, ncol=1, loc="upper center")
ax.set_title(r"(a)  $n = 16$", fontsize=12)

# (b) errors
ns = np.array([8, 16, 32, 64, 128, 256, 512])
err = {k: [] for k in "ABCD"}
for n in ns:
    X, tv = ddg32.sample("ellipse", n)
    pw = pointwise(X)
    ks = ddg32.kappa("ellipse", tv)
    for k in "ABCD":
        err[k].append(np.max(np.abs(pw[k] - ks)))
for k in "ABCD":
    e = np.array(err[k])
    order = np.log2(e[-2] / e[-1])
    assert abs(order - 2) < 0.02, (k, order)                          # observed order 2
    bx.loglog(ns, e, color=S[k]["color"], ls=S[k]["ls"], marker=S[k]["marker"], ms=4, lw=1.3,
              mfc="white" if k in "BD" else S[k]["color"], label=f"{k}  {NAMES[k]}")
ref = 2.0 * (ns / 8.0) ** -2.0 * err["A"][0] / 2.0 * 1.6
bx.loglog(ns, ref, color=C["aux"], lw=dgfig.LW["aux"], ls="--")
bx.text(ns[3], ref[3] * 1.5, r"slope $-2$", color=C["aux"], fontsize=10)
bx.set_xlabel(r"number of vertices $n$")
bx.set_ylabel("max error")
bx.legend(frameon=False, fontsize=8.5, loc="lower left")
bx.set_title("(b)", fontsize=12)
for a in (ax, bx):
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
for k in "ABCD":
    print(k, " ".join(f"{e:.2e}" for e in err[k]))
