"""Figure 47.5.1: why natural kernel pairs work ([BLM17] §5, (5.11), (5.12); panel (b) is our experiment).

(a) the integrand (s rho'(s) + d C_{rho,xi} xi(s)) s^(d-1) of (5.11) for d = 1, n = 2: for the pair (rho, rho) it is
not zero but integrates to zero (C_{rho,rho} = 1); for the natural pair xi = -s rho'/n it vanishes identically
(C_{rho,xi} = n/d). exp profile (black) and tent profile (grey).
(b) random neighbourhoods: K neighbours drawn uniformly (in arc length or in the tangent disc) inside the ball
B_eps(x) of a point x on a circle of radius 0.5 (d = 1, n = 2), a sphere of radius 0.5 and a cylinder of radius 0.5
(d = 2, n = 3), eps = 0.05, exact planes, equal masses, x itself included; 400 trials per K (seed 5).
Plotted: root mean square relative error of the normal component of H^{V,perp}(x) against K, natural exp pair
(solid) and exp pair (rho, rho) (dashed). Exact values |H| = 2, 4, 2.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-5-1-natural-kernel-pairs.py
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

import var47 as v  # noqa: E402

C = dgfig.COLORS
R, EPS, TRIALS = 0.5, 0.05, 400
KS = np.array([4, 8, 16, 32, 64, 128, 256])


def rms_errors(seed=5):
    rng = np.random.default_rng(seed)
    K2, K3 = v.pairs(2), v.pairs(3)
    out = {}
    for kind, exact in (("circle", 2.0), ("sphere", 4.0), ("cylinder", 2.0)):
        for nm in ("natural exp", "exp"):
            pair = (K2 if kind == "circle" else K3)[nm]
            errs = []
            for K in KS:
                e = np.empty(TRIALS)
                for k in range(TRIALS):
                    e[k] = sample_estimate(rng, kind, K, pair) - exact
                errs.append(np.sqrt(np.mean(e ** 2)) / exact)
            out[(kind, nm)] = np.array(errs)
    return out


def sample_estimate(rng, kind, K, pair):
    if kind == "circle":
        phimax = 2 * np.arcsin(EPS / (2 * R))
        phi = np.concatenate([[0.0], rng.uniform(-phimax, phimax, K)])
        X = R * np.stack([np.cos(phi), np.sin(phi)], 1)
        P = v.proj_from_tangent(np.stack([-np.sin(phi), np.cos(phi)], 1))
        H, Hp = v.amc_single(X[0], P[0], X, P, np.ones(len(X)), EPS, pair, 1)
        return -Hp[0]
    rr, th = EPS * np.sqrt(rng.uniform(0, 1, K)), rng.uniform(0, 2 * np.pi, K)
    a, b = np.concatenate([[0.0], rr * np.cos(th)]), np.concatenate([[0.0], rr * np.sin(th)])
    if kind == "cylinder":                      # axis e3, x = (R, 0, 0); a = arc length, b = height
        ph = a / R
        X = np.stack([R * np.cos(ph), R * np.sin(ph), b], 1)
        Nr = np.stack([np.cos(ph), np.sin(ph), 0 * ph], 1)
    else:                                       # sphere, geodesic polar coordinates around x = (R, 0, 0)
        rho = np.hypot(a, b)
        dirv = np.zeros((len(a), 3))
        nz = rho > 0
        dirv[nz, 1], dirv[nz, 2] = a[nz] / rho[nz], b[nz] / rho[nz]
        X = R * (np.cos(rho / R)[:, None] * np.array([1.0, 0, 0]) + np.sin(rho / R)[:, None] * dirv)
        Nr = X / R
    P = v.proj_from_normal(Nr)
    H, Hp = v.amc_single(X[0], P[0], X, P, np.ones(len(X)), EPS, pair, 2)
    return -Hp[0]


if __name__ == "__main__":
    s = np.linspace(0, 1, 801)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"wspace": 0.32, "width_ratios": [1, 1.15]})
    ax = axes[0]
    for prof, dprof, col, lab in ((v.rho_exp, v.drho_exp, C["main"], "exp"), (v.rho_tent, v.drho_tent, C["aux"], "tent")):
        c = v.normaliser(prof, 2)
        f = c * (s * dprof(s) + 1.0 * prof(s))          # (rho, rho): C_{rho,rho} = 1, d = 1
        assert abs(np.trapezoid(f, s)) < 2e-3
        ax.plot(s, f, color=col, lw=1.4, label=r"$(\rho, \rho)$, " + lab)
        ax.fill_between(s, 0, f, color=col, alpha=0.12, lw=0)
    ax.axhline(0, color=C["normal"], lw=1.6, label="natural pairs")
    ax.set_xlabel(r"$s$")
    ax.set_xlim(0, 1)
    ax.set_title(r"(a) $s\rho'(s) + d\,C_{\rho,\xi}\,\xi(s)$, $d = 1$", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    err = rms_errors()
    ax = axes[1]
    for kind, col in (("circle", C["main"]), ("sphere", C["tangent"]), ("cylinder", C["normal"])):
        ax.loglog(KS, err[(kind, "natural exp")], "o-", color=col, ms=3, lw=1.2, label=kind)
        ax.loglog(KS, err[(kind, "exp")], "s--", color=col, ms=3, lw=1.0)
    ax.plot([], [], "k-", label="natural exp")
    ax.plot([], [], "k--", label=r"exp $(\rho, \rho)$")
    ax.set_xlabel(r"number of neighbours $K$")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.minorticks_off()
    ax.set_title(r"(b) relative RMS error, $\varepsilon = 0.05$", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    ax.set_ylim(1e-5, 3)
    assert err[("circle", "natural exp")].max() < 2e-3 and err[("sphere", "natural exp")].max() < 3e-3
    assert err[("circle", "exp")][0] > 0.3 and err[("cylinder", "natural exp")][0] > 0.2
    assert err[("cylinder", "natural exp")][-1] < err[("cylinder", "natural exp")][0] / 4
    dgfig.save(fig, __file__)
    for key, val in err.items():
        print(key, np.round(val, 4))
