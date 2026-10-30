"""Figure 48.4.1: the G-linear variation of four junctions of half-lines ([BLM22] (4.4)).

V = sum_l v(R_l, 1), R_l = {t u_l : t > 0}, u_l = (cos a_l, sin a_l). By (4.4) delta_ijk V = -T_ijk delta_0 with the
symmetric 3-tensor T = sum_l u_l (x) u_l (x) u_l. We draw T through its cubic form f(w) = sum_l <u_l, w>^3 on unit
vectors w = (cos p, sin p): the polar curve r = 0.9 |f(p)|, red where f > 0 and blue where f < 0 (f(-w) = -f(w)).
(a) Y junction 90, 210, 330 degrees: f = -(3/4) sin 3p.   (b) T junction 0, 90, 180: f = sin^3 p.
(c) X junction 30, 150, 210, 330: f = 0.                   (d) regular pentagon 90 + 72 l: f = 0.
Below each panel: whether the junction is stationary (sum_l u_l = 0) and whether T = 0.

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-4-1-junction-variations.py
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

C = dgfig.COLORS
cases = [
    (r"(a) Y", [90, 210, 330]),
    (r"(b) T", [0, 90, 180]),
    (r"(c) X", [30, 150, 210, 330]),
    (r"(d) pentagon", [90 + 72 * l for l in range(5)]),
]
p = np.linspace(0, 2 * np.pi, 721)
W = np.stack([np.cos(p), np.sin(p)], 1)


def tensor(deg):
    U = np.stack([np.cos(np.radians(deg)), np.sin(np.radians(deg))], 1)
    return U, np.einsum("li,lj,lk->ijk", U, U, U)


# self-checks
U, T = tensor(cases[0][1])
assert np.allclose(np.einsum("ijk,pi,pj,pk->p", T, W, W, W), -0.75 * np.sin(3 * p))
U, T = tensor(cases[1][1])
assert np.allclose(np.einsum("ijk,pi,pj,pk->p", T, W, W, W), np.sin(p) ** 3)
assert np.allclose(U.sum(0), [0, 1])
for _, deg in cases[2:]:
    U, T = tensor(deg)
    assert np.abs(T).max() < 1e-12 and np.abs(U.sum(0)).max() < 1e-12
assert np.abs(tensor(cases[0][1])[0].sum(0)).max() < 1e-12

fig, axes = plt.subplots(2, 2, figsize=(5.0, 5.6), gridspec_kw={"wspace": 0.05, "hspace": 0.12})  # 2 x 2: legible at phone width
for ax, (title, deg) in zip(axes.ravel(), cases):
    U, T = tensor(deg)
    f = np.einsum("ijk,pi,pj,pk->p", T, W, W, W)
    r = 0.9 * np.abs(f)
    for sgn, col in ((1, C["normal"]), (-1, C["tangent"])):
        mask = sgn * f > 1e-9
        if mask.any():
            rr = np.where(mask, r, 0.0)
            ax.fill(rr * np.cos(p), rr * np.sin(p), color=col, alpha=0.35, lw=0)
            ax.plot(rr * np.cos(p), rr * np.sin(p), color=col, lw=0.8)
    for u in U:
        ax.plot([0, u[0]], [0, u[1]], color=C["main"], lw=1.6, solid_capstyle="round")
    ax.plot(0, 0, "o", color=C["main"], ms=3.5)
    stat = np.abs(U.sum(0)).max() < 1e-12
    zero = np.abs(T).max() < 1e-12
    ax.text(0, -1.32, r"$\Sigma_l\, u_l %s 0$" % ("=" if stat else r"\neq"), ha="center", va="center", fontsize=12)
    ax.text(0, -1.68, r"$\Sigma_l\, u_l^{\otimes 3} %s 0$" % ("=" if zero else r"\neq"), ha="center", va="center",
            fontsize=12)
    ax.set_title(title, fontsize=12.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.95, 1.15)
    ax.set_axis_off()

print("Y max |f| = 0.75, T max |f| = 1; X and pentagon: T = 0")
dgfig.save(fig, __file__)
