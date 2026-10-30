"""Figure 47.4.2: how the errors of H^V and H^{V,perp} scale with eps/h and N ([BLM17] Thm 4.5, Thm 4.8).

Ellipse (2 cos t, sin t) (length L = 9.6884), N points t_j = 2 pi (j + n_j)/N with n_j standard normal (seed 12),
exact tangent lines, masses m_j = |gamma'(t_j)| 2 pi / N (so the smooth density is 1 and only the local jitter
remains). h = L/N. The error is the mean over all points of |H(x_j) - H_exact(x_j)|.
(a) H^V with the exp pair (rho, rho), N = 1000, 4000, 16000: at fixed eps/h the error grows like N (curves shift
up by about 4 for each factor 4 in N), as the bound (d_i + |x - z_i|)/eps_i^2 of Thm 4.5 with d_i ~ h allows.
(b) H^{V,perp}: with the exp pair the three curves collapse onto one curve of eps/h (slope about -1), as the bound
(d_1 + d_2 + |x - z|)/eps of Thm 4.8 with d_1 ~ h, d_2 = 0 suggests; with the natural exp pair (Section 47.5) the
error is far smaller and flat until the smoothing bias O(eps^2) takes over.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-4-2-convergence-rates.py
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
K = v.pairs(2)


def shape(t):
    return v.ellipse(t, 2.0, 1.0)


L = v.curve_length(shape)
Ns = (1000, 4000, 16000)
F = np.array([4, 8, 16, 32, 64, 128])
res = {}
for N in Ns:
    rng = np.random.default_rng(12)
    t = 2 * np.pi * (np.arange(N) + rng.standard_normal(N)) / N
    X, T, Hx = shape(t)
    P = v.proj_from_tangent(T)
    m = np.sqrt(4 * np.sin(t) ** 2 + np.cos(t) ** 2) * 2 * np.pi / N
    h = L / N
    for f in F:
        eps = f * h
        if eps > 0.4:
            continue
        H, Hp, _ = v.amc(X, P, m, eps, K["exp"], 1)
        Hn, Hpn, _ = v.amc(X, P, m, eps, K["natural exp"], 1)
        err = lambda A: float(np.linalg.norm(A - Hx, axis=1).mean())
        res[(N, f)] = (err(H), err(Hp), err(Hpn))

cols = {1000: C["accent"], 4000: C["tangent"], 16000: C["main"]}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"wspace": 0.3})
for N in Ns:
    ff = [f for f in F if (N, f) in res]
    e = np.array([res[(N, f)] for f in ff])
    axes[0].loglog(ff, e[:, 0], "o-", color=cols[N], ms=3, lw=1.1, label=f"$N = {N}$")
    axes[1].loglog(ff, e[:, 1], "o-", color=cols[N], ms=3, lw=1.1)
    axes[1].loglog(ff, e[:, 2], "s:", color=cols[N], ms=3, lw=1.1)
x0 = np.array([8.0, 64.0])
axes[0].loglog(x0, 1.0 * (x0 / 8) ** -2, "--", color=C["aux"], lw=0.9)
axes[0].text(9, 0.06, "slope $-2$", fontsize=9, color=C["aux"])
axes[0].set_title(r"(a) $|\vec H^V - \vec H|$, exp pair", fontsize=10)
axes[1].set_title(r"(b) $|\vec H^{V,\perp} - \vec H|$", fontsize=10)
for ax in axes:
    ax.set_xlabel(r"$\varepsilon / h$")
    ax.set_xticks(list(F))
    ax.set_xticklabels([str(f) for f in F])
    ax.minorticks_off()
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
axes[1].plot([], [], "o-", color=C["main"], ms=3, lw=1.1, label="exp")
axes[1].plot([], [], "s:", color=C["main"], ms=3, lw=1.1, label="natural exp")
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
axes[1].set_ylim(5e-5, 1.0)

# checks quoted in Section 47.4
r = lambda N, f, k: res[(N, f)][k]
assert 3.0 < r(4000, 16, 0) / r(1000, 16, 0) < 5.0 and 3.0 < r(16000, 16, 0) / r(4000, 16, 0) < 5.0
assert all(abs(np.log(r(N, 16, 1) / r(4000, 16, 1))) < 0.15 for N in Ns)          # collapse in eps/h
assert r(16000, 16, 2) < 5e-4 and r(16000, 16, 1) > 40 * r(16000, 16, 2)
dgfig.save(fig, __file__)
for key in sorted(res):
    print(key, ["%.3g" % x for x in res[key]])
