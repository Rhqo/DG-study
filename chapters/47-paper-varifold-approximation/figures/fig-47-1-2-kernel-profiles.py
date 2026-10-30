"""Figure 47.1.2: the kernel profiles of [BLM17] Section 8.1.1, normalised in R^2 (n = 2) as in [BLM17] Section 1.

(a) the exp profile rho(r) = c exp(-1/(1 - r^2)) and its natural partner xi(r) = -r rho'(r) / n (Section 47.5);
(b) the tent profile rho(r) = c' (1 - r) and its natural partner xi(r) = c' r / n, which jumps at r = 1.
Both rho and xi are normalised so that 2 pi int_0^1 f(r) r dr = 1 (the integral of f(|x|) over R^2 is 1);
the natural partner of a normalised rho is automatically normalised (Proposition 47.5.2).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-1-2-kernel-profiles.py
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
r = np.linspace(0, 1.15, 1200)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=False, gridspec_kw={"wspace": 0.3})
for ax, (key, title) in zip(axes, (("natural exp", r"(a) exp profile"), ("natural tent", r"(b) tent profile"))):
    p = K[key]
    c_rho, c_xi = v.normaliser(p.rho, 2), v.normaliser(p.xi, 2)
    assert abs(c_rho - c_xi) < 1e-9          # the natural partner is normalised with the same constant
    rr = r[r < 1]
    ax.plot(r, c_rho * p.rho(r), color=C["main"], lw=1.6, label=r"$\rho$")
    ax.plot(rr, c_xi * p.xi(rr), color=C["normal"], lw=1.6, label=r"$\xi = -r\rho'/2$")
    ax.plot(r[r >= 1], 0 * r[r >= 1], color=C["normal"], lw=1.6)
    ax.set_xlabel(r"$r = |x|/\varepsilon$")
    ax.set_xlim(0, 1.15)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.axvline(1.0, color=C["aux"], lw=0.6, ls="--")
axes[0].set_ylim(0, 0.9)
axes[1].set_ylim(0, 1.05)
# the jump of the natural tent partner at r = 1: c' / 2
c_t = v.normaliser(v.rho_tent, 2)
axes[1].plot([1, 1], [0, c_t / 2], color=C["normal"], lw=0.8, ls=":")
axes[1].annotate(r"jump $c'/2$", (1.0, c_t / 2), xytext=(-62, 18), textcoords="offset points", fontsize=10,
                 arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
assert abs(c_t - 3 / np.pi) < 1e-9
assert abs(v.normaliser(v.rho_exp, 2) - 2.1436) < 1e-4
dgfig.save(fig, __file__)
