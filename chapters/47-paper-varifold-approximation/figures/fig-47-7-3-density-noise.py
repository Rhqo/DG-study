"""Figure 47.7.3: our reproduction of [BLM17, Fig. 5] (uneven sampling, with and without the normal projection) and
of part of [BLM17, Fig. 6] (error against 1/(N eps), with and without positional noise).

Flower r = 0.5 + 0.25 cos 6 theta, equal masses, natural exp pair, formula (8.2) (constant restored); error E as in
Figure 47.7.2 (normalised by the mean of |H|).
(a) eps = (10/N)^(3/4), N = 5000, 25000, 125000, 625000. "uniform": t_j = 2 pi j / N; "non uniform":
t_j = 2 pi (j + n_j) / N with n_j standard normal (seed 0); "regression": ordinary least squares tangent lines in the
ball of radius eps/2; "no projection": H^V instead of H^{V,perp}.
(b) E against 1/(N eps), exact tangent lines: eps = (10/N)^(3/4) without noise and with i.i.d. Gaussian noise of
standard deviation 1/N on each coordinate (our reading of "variance sigma = 1/N"), eps = (100/N)^(3/4) with the same
noise, and eps = N^(-1/2) without noise; N = 5000, ..., 625000 (right to left).
Open markers: values read by eye from [BLM17, Fig. 5] and [BLM17, Fig. 6].

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-7-3-density-noise.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import var47 as v  # noqa: E402

PAIR = v.pairs(2)["natural exp"]
NS = (5000, 25000, 125000, 625000)
E10 = lambda N: (10.0 / N) ** 0.75
E100 = lambda N: (100.0 / N) ** 0.75
EHALF = lambda N: N ** -0.5
# (label, eps rule, keyword arguments of var47.erel_curve)
FIG5 = (("uniform", E10, dict()),
        ("non uniform", E10, dict(jitter=True)),
        ("non uniform (regression)", E10, dict(jitter=True, tangents="ols", R="half")),
        ("uniform, no projection", E10, dict(project=False)),
        ("non uniform, no projection", E10, dict(jitter=True, project=False)))
FIG6 = ((r"$(10/N)^{3/4}$", E10, dict()),
        (r"$(10/N)^{3/4}$, noise", E10, dict(noise="1/N")),
        (r"$(100/N)^{3/4}$, noise", E100, dict(noise="1/N")),
        (r"$N^{-1/2}$", EHALF, dict()))
# values read from [BLM17, Fig. 5] and [BLM17, Fig. 6] at N = 5e3, 2.5e4, 1.25e5, 6.25e5 (Tables 47.7.5, 47.7.6)
PAPER5 = {"uniform": (0.145, 0.019, 1.8e-3, 1.6e-4), "non uniform": (0.15, 0.02, 1.9e-3, 1.9e-4),
          "non uniform (regression)": (0.165, 0.026, 4.7e-3, 1.6e-3),
          "uniform, no projection": (0.66, 0.76, 0.78, 0.78),
          "non uniform, no projection": (2.9, 5.5, 11.0, 21.0)}
PAPER6 = {r"$(10/N)^{3/4}$": (0.15, 0.02, 1.85e-3, 1.65e-4),
          r"$(10/N)^{3/4}$, noise": (0.15, 0.02, 3e-3, 1.1e-3),
          r"$(100/N)^{3/4}$, noise": (0.5, 0.26, 0.05, 5e-3),
          r"$N^{-1/2}$": (0.24, 0.08, 0.02, 4e-3)}


def erel(rule, kw, N):
    kw = dict(kw)
    eps = rule(N)
    if kw.get("R") == "half":
        kw["R"] = eps / 2
    if kw.get("noise") == "1/N":
        kw["noise"] = 1.0 / N
    return v.erel_curve(v.flower, N, eps, PAIR, norm="mean", **kw)[0]


def table5(Ns=NS):
    return {(lab, N): erel(rule, kw, N) for lab, rule, kw in FIG5 for N in Ns}


def table6(Ns=NS):
    return {(lab, N): erel(rule, kw, N) for lab, rule, kw in FIG6 for N in Ns}


if __name__ == "__main__":
    import dgfig

    dgfig.setup()
    import matplotlib.pyplot as plt

    C = dgfig.COLORS
    R5, R6 = table5(), table6()
    st5 = {"uniform": (C["tangent"], "o", "-"), "non uniform": (C["normal"], "s", "-"),
           "non uniform (regression)": (C["accent"], "D", ":"),
           "uniform, no projection": (C["tangent"], "o", "--"),
           "non uniform, no projection": (C["normal"], "s", "--")}
    st6 = {r"$(10/N)^{3/4}$": (C["tangent"], "o", "-"), r"$(10/N)^{3/4}$, noise": (C["third"], "^", "-"),
           r"$(100/N)^{3/4}$, noise": (C["covector"], "v", "-"), r"$N^{-1/2}$": (C["aux"], "s", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for lab, _, _ in FIG5:
        col, mk, ls = st5[lab]
        ax.loglog(NS, [R5[(lab, N)] for N in NS], ls, marker=mk, color=col, ms=3.5, lw=1.2, label=lab)
        ax.loglog(NS, PAPER5[lab], mk, color=col, ms=6.5, mfc="none", mew=0.8)
    ax.set_xlabel(r"number of points $N$")
    ax.set_ylabel(r"relative error $E$")
    ax.set_title(r"(a) $\varepsilon = (10/N)^{3/4}$", fontsize=10)
    ax.set_xlim(3e3, 1e6)
    ax.set_ylim(5e-5, 60)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[1]
    for lab, rule, _ in FIG6:
        col, mk, ls = st6[lab]
        x = [1.0 / (N * rule(N)) for N in NS]
        ax.loglog(x, [R6[(lab, N)] for N in NS], ls, marker=mk, color=col, ms=3.5, lw=1.2,
                  label=r"$\varepsilon = $" + lab)
        xp = [xx for xx, p in zip(x, PAPER6[lab]) if p is not None]
        ax.loglog(xp, [p for p in PAPER6[lab] if p is not None], mk, color=col, ms=6.5, mfc="none", mew=0.8)
    xx = np.array([1.1e-3, 4.4e-3])
    ax.loglog(xx, 1e-4 * (xx / 1.1e-3), ":", color=C["aux"], lw=0.9)
    ax.text(1.15e-3, 2.1e-4, "slope 1", fontsize=8, color=C["aux"])
    ax.set_xlabel(r"$1/(N\varepsilon)$")
    ax.set_title(r"(b) noise of size $1/N$", fontsize=10)
    ax.set_ylim(5e-5, 30)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    dgfig.save(fig, __file__)
    for key in sorted(R5):
        print("fig5", key, "%.3g" % R5[key])
    for key in sorted(R6):
        print("fig6", key, "%.3g" % R6[key])
