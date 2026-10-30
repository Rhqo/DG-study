"""Figure 47.7.2: our reproduction of [BLM17, Fig. 4], the convergence of the orthogonal approximate mean curvature
(8.2) on the flower for four kernel pairs.

Flower r = 0.5 + 0.25 cos 6 theta, N points at uniform polar angle t_j = 2 pi j / N, equal masses, formula (8.2) with
the constant C_xi/C_rho restored. Tangent lines: exact, or (label "regression") the ordinary least squares line
y = a + b x through the points of the open ball of radius R = eps/2 (our reading of "linear regression"; the paper
does not specify the fit).
Error: E = sum_j |H^{V,perp}(x_j) - H(t_j)| / sum_j |H(t_j)|, i.e. (8.5) with ||H||_inf replaced by the mean of |H|
(our reading: with the printed ||H||_inf = 140 our values are 14.17 times smaller than the plotted ones, while with
the mean they agree to two digits; see Section 47.7).
(a) eps = 100/N, (b) eps = (10/N)^(3/4); N = 5000, 25000, 125000, 625000.
Open markers: values read by eye from [BLM17, Fig. 4] (two significant digits at best).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-7-2-flower-convergence.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import var47 as v  # noqa: E402

K = v.pairs(2)
NS = (5000, 25000, 125000, 625000)
EPS = {"a": lambda N: 100.0 / N, "b": lambda N: (10.0 / N) ** 0.75}
# (label, pair, tangents)
CURVES = (("natural exp", "natural exp", "exact"), ("natural tent", "natural tent", "exact"),
          ("exp", "exp", "exact"), ("tent", "tent", "exact"),
          ("natural exp (regression)", "natural exp", "ols"))
# values read from [BLM17, Fig. 4] at N = 5e3, 2.5e4, 1.25e5, 6.25e5 (Table 47.7.4 and 47.7.5)
PAPER = {
    "a": {"exp": (0.40, 0.048, 2.6e-3, 6e-4), "natural exp": (0.3, 0.037, 1.6e-3, 6.5e-5),
          "natural exp (regression)": (0.4, 0.06, 8.5e-3, 1.7e-3), "tent": (0.35, 0.058, 1.5e-2, 1.5e-2),
          "natural tent": (0.28, 0.038, 1.7e-3, 8e-5)},
    "b": {"exp": (0.2, 0.027, 2.7e-3, 2.7e-4), "natural exp": (0.15, 0.019, 1.8e-3, 1.6e-4),
          "natural exp (regression)": (0.2, 0.037, 9e-3, 2.6e-3), "tent": (0.2, 0.04, 1.3e-2, 9e-3),
          "natural tent": (0.145, 0.02, 1.9e-3, 1.75e-4)},
}


def erel(case, label, N):
    """(E, mean number of neighbours) for one curve of Figure 47.7.2."""
    pair, tangents = {c[0]: c[1:] for c in CURVES}[label]
    eps = EPS[case](N)
    return v.erel_curve(v.flower, N, eps, K[pair], tangents=tangents, R=eps / 2, norm="mean")


def table(Ns=NS):
    return {(case, lab, N): erel(case, lab, N) for case in "ab" for lab, _, _ in CURVES for N in Ns}


def slope(y, Ns=NS):
    return float(np.polyfit(np.log(Ns), np.log(y), 1)[0])


if __name__ == "__main__":
    import dgfig

    dgfig.setup()
    import matplotlib.pyplot as plt

    C = dgfig.COLORS
    RES = table()
    style = {"natural exp": (C["normal"], "o", "-"), "natural tent": (C["third"], "s", "-"),
             "exp": (C["tangent"], "o", "--"), "tent": (C["aux"], "s", "--"),
             "natural exp (regression)": (C["accent"], "D", ":")}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, case, ttl in ((axes[0], "a", r"(a) $\varepsilon = 100/N$"),
                          (axes[1], "b", r"(b) $\varepsilon = (10/N)^{3/4}$")):
        for lab, _, _ in CURVES:
            col, mk, ls = style[lab]
            y = [RES[(case, lab, N)][0] for N in NS]
            ax.loglog(NS, y, ls, marker=mk, color=col, ms=3.5, lw=1.2, label=lab)
            ax.loglog(NS, PAPER[case][lab], mk, color=col, ms=6.5, mfc="none", mew=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(r"number of points $N$")
        ax.set_xlim(3e3, 1e6)
        ax.set_ylim(2e-5, 1.0)
    axes[0].set_ylabel(r"relative error $E$")
    axes[0].plot([], [], "o", color=C["aux"], ms=6.5, mfc="none", mew=0.8, label="read from [BLM17]")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    dgfig.save(fig, __file__)
    for key in sorted(RES):
        print(key, "E = %.3g  neighbours %.1f" % RES[key])
    for case in "ab":
        for lab, _, _ in CURVES:
            print(case, lab, "slope over the last two N: %.2f" % slope([RES[(case, lab, N)][0] for N in NS[2:]], NS[2:]))
