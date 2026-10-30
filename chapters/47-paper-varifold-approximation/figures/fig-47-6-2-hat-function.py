"""Figure 47.6.2: the first variation of a polyhedral varifold tested on a hat function ([BLM17] Prop 7.1, Remark 7.2).

One-ring of the vertex x = (0, 0, 1) (north pole of the unit sphere) with six neighbours on the sphere at polar
angle 0.35 and azimuths 0, 55, 118, 180, 235, 300 degrees. On each face F = (x, w, v) the hat function phi_x is
affine with tangential gradient (x - y_F)/h_F^2 (y_F = foot of the perpendicular from x on the line wv,
h_F = |x - y_F|), so int_F grad^{P_F} phi_x = Area(F) (x - y_F)/h_F^2 (grey arrows, drawn at the face centroids).
Their sum is delta V_T(phi_x) = -(L x)_x (black), and H_{V_T}(x) = -delta V_T(phi_x) / (Area/3) (vermilion).
(a) top view (x_1, x_2); (b) side view (x_1, x_3). Arrows: face terms and the sum 2 times, H 0.1 times.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-6-2-hat-function.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS


def one_ring():
    az = np.radians([0, 55, 118, 180, 235, 300])
    th = 0.35
    V = np.stack([np.sin(th) * np.cos(az), np.sin(th) * np.sin(az), np.cos(th) * np.ones(6)], 1)
    return np.array([0.0, 0.0, 1.0]), V


def cot(p, q, r):
    a, b = q - p, r - p
    return (a @ b) / np.linalg.norm(np.cross(a, b))


def hat_terms(x, V):
    terms, cents, area, Lx = [], [], 0.0, np.zeros(3)
    for k in range(len(V)):
        w, v = V[k], V[(k + 1) % len(V)]
        t = (v - w) / np.linalg.norm(v - w)
        y = w + ((x - w) @ t) * t
        h = np.linalg.norm(x - y)
        A = 0.5 * np.linalg.norm(np.cross(w - x, v - x))
        terms.append(A * (x - y) / h ** 2)
        cents.append((x + w + v) / 3)
        area += A
        Lx += 0.5 * (cot(w, x, v) * (v - x) + cot(v, x, w) * (w - x))   # face part of (L x)_x, (40.3.3)
    return np.array(terms), np.array(cents), area, Lx


if __name__ == "__main__":
    x, V = one_ring()
    terms, cents, area, Lx = hat_terms(x, V)
    dV = terms.sum(0)
    H = -dV / (area / 3)
    assert np.allclose(dV, -Lx, atol=1e-14)                       # Prop 7.1 = Track 2 sign convention
    assert abs(np.linalg.norm(H) - 1.987) < 1e-3 and H[2] < 0

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"wspace": 0.15})
    for ax, (i, j), ttl in ((axes[0], (0, 1), r"(a) top view $(x_1, x_2)$"), (axes[1], (0, 2), r"(b) side view $(x_1, x_3)$")):
        for k in range(len(V)):
            tri = np.array([x, V[k], V[(k + 1) % len(V)]])[:, [i, j]]
            ax.add_patch(Polygon(tri, closed=True, facecolor=C["surface"], alpha=0.35 if i == 0 and j == 1 else 0.15,
                                 edgecolor=C["main"], lw=0.7))
        S = 2.0
        for c, tvec in zip(cents, terms):
            ax.annotate("", xy=c[[i, j]] + S * tvec[[i, j]], xytext=c[[i, j]],
                        arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=1.0, mutation_scale=7))
        ax.annotate("", xy=x[[i, j]] + S * dV[[i, j]], xytext=x[[i, j]],
                    arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.8, mutation_scale=9))
        ax.annotate("", xy=x[[i, j]] + 0.1 * H[[i, j]], xytext=x[[i, j]],
                    arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.6, mutation_scale=9))
        ax.plot(*x[[i, j]], "o", color=C["main"], ms=3.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(ttl, fontsize=11)
    axes[0].set_xlim(-0.42, 0.42)
    axes[0].set_ylim(-0.42, 0.42)
    axes[1].set_xlim(-0.42, 0.42)
    axes[1].set_ylim(0.72, 1.48)
    axes[1].text(0.03, 1.0 + 2 * dV[2] - 0.02, r"$\delta V_{\mathcal{T}}(\hat\varphi_x)$", fontsize=10)
    axes[1].text(0.03, 1.0 + 0.1 * H[2] + 0.02, r"$\vec H_{V_{\mathcal{T}}}(x)$", fontsize=10, color=C["normal"])
    dgfig.save(fig, __file__)
    print("delta V =", dV, " -(Lx) =", -Lx, " area/3 =", area / 3, " H =", H, " |H| =", np.linalg.norm(H))
