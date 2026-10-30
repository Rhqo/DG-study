"""Figure 47.7.1: our recomputation of [BLM17, Fig. 3], the orthogonal approximate mean curvature (8.2) on the
ellipse and on the flower.

Setting of [BLM17] Section 8.1.2: N = 10^5 points at uniform parameter t_j = 2 pi j / N, exact tangent lines, equal
masses, natural exp pair, eps = 0.001, formula (8.2) with the constant C_xi/C_rho = d/n = 1/2 restored.
(a) ellipse (cos t, 0.5 sin t), exact curvature in [0.5, 4]; (b) flower r = 0.5 + 0.25 cos 6 theta (polar angle
theta = t), exact |H| in [0, 140]. The curve is coloured by |H^{V,perp}| (viridis; square-root colour scale in (b)),
arrows show H^{V,perp} at
36 points evenly spaced in t (length proportional to |H|, scaled per panel).
The extreme values are printed and compared with the colorbar ends of [BLM17, Fig. 3]
(ellipse 0.5 / 3.99998, flower 0.000241203 / 138.903).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-7-1-ellipse-flower.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize, PowerNorm  # noqa: E402

import var47 as v  # noqa: E402

C = dgfig.COLORS
N, EPS = 100_000, 0.001
PAIR = v.pairs(2)["natural exp"]


def run(shape):
    t, X, T, Hx = v.curve_cloud(shape, N)
    P = v.proj_from_tangent(T)
    H, Hp, cnt = v.amc(X, P, np.ones(N), EPS, PAIR, 1)
    return t, X, Hx, Hp, cnt


RES = {}
for name, shape in (("ellipse", v.ellipse), ("flower", v.flower)):
    RES[name] = run(shape)


def extremes(name):
    _, _, Hx, Hp, _ = RES[name]
    a = np.linalg.norm(Hp, axis=1)
    return float(a.min()), float(a.max()), float(np.linalg.norm(Hp - Hx, axis=1).max())


# checks quoted in Section 47.7
lo, hi, err = extremes("ellipse")
assert abs(hi - 4.0) < 1e-4 and abs(lo - 0.5) < 1e-5 and err < 1e-3        # paper: 0.5 / 3.99998
lo, hi, err = extremes("flower")
assert abs(hi - 138.956) < 0.01 and lo < 1e-3                                # paper: 0.000241 / 138.903
assert abs(v.max_abs_curvature(v.flower) - 140.0) < 1e-3

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.25})
    for ax, name, ttl in ((axes[0], "ellipse", "(a) ellipse"), (axes[1], "flower", "(b) flower")):
        t, X, Hx, Hp, cnt = RES[name]
        a = np.linalg.norm(Hp, axis=1)
        k = np.arange(0, N + 1, 50) % N                       # draw 2000 segments (the SVG stays small)
        seg = np.stack([X[k[:-1]], X[k[1:]]], 1)
        norm = Normalize(0, a.max()) if name == "ellipse" else PowerNorm(0.5, 0, a.max())
        lc = LineCollection(seg, cmap="viridis", norm=norm, lw=2.2)
        lc.set_array(a[k[:-1]])
        ax.add_collection(lc)
        sel = (np.arange(36) * (N // 36)).astype(int)
        scale = (0.35 if name == "ellipse" else 0.3) / a.max()
        for j in sel:
            ax.annotate("", xy=X[j] + scale * Hp[j], xytext=X[j],
                        arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9, mutation_scale=6))
        cb = fig.colorbar(lc, ax=ax, fraction=0.05, pad=0.02, shrink=0.8)
        ticks = [0, 1, 2, 3, a.max()] if name == "ellipse" else [0, 5, 20, 50, 100, a.max()]
        cb.set_ticks(ticks)
        cb.set_ticklabels(["%g" % x for x in ticks[:-1]] + ["%.6g" % a.max()])
        cb.ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(ttl + r", $|\vec H^{V,\perp}|$", fontsize=10)
    axes[0].set_xlim(-1.15, 1.15)
    axes[0].set_ylim(-0.75, 0.75)
    axes[1].set_xlim(-0.8, 0.8)
    axes[1].set_ylim(-0.8, 0.8)
    dgfig.save(fig, __file__)
    for name in RES:
        print(name, "min %.6g max %.6g max|err| %.3g  mean neighbours %.1f" % (extremes(name) + (RES[name][4].mean(),)))
