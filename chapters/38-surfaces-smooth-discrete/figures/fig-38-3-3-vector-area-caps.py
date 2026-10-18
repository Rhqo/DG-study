"""Figure 38.3.3: surfaces with the same boundary have the same vector area.

Spherical caps spanning the unit circle {x^2 + y^2 = 1, z = 0}, with apex height h in {0 (flat disk), 0.5, 1
(hemisphere), 1.8}: the sphere through the circle with apex (0, 0, h) has radius R_h = (1 + h^2) / (2h) and center
(0, 0, h - R_h). All caps are oriented by the upward normal at the apex, so their boundaries are the same counterclockwise
circle. Area A(h) = 2 pi R_h h = pi (1 + h^2) (Archimedes), vector area = (0, 0, pi) for every h.
(a) the sections in the xz-plane with unit normals; (b) area and |vector area| against h.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-3-3-vector-area-caps.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
HS = [0.5, 1.0, 1.8]


def cap(h):
    """(x(th, ph), th_max): the cap parametrized by the polar angle th from the apex, outward (upward) normal."""
    Rh = (1 + h * h) / (2 * h)
    c = h - Rh
    thmax = np.arccos(-c / Rh)

    def x(th, ph):
        return np.array([Rh * np.sin(th) * np.cos(ph), Rh * np.sin(th) * np.sin(ph), c + Rh * np.cos(th) + 0 * ph])
    return x, thmax


def patch_integrals(x, a, b, n=600, hh=1e-6):
    th = (np.arange(n) + 0.5) * (b - a) / n + a
    ph = (np.arange(n) + 0.5) * 2 * np.pi / n
    T, P = np.meshgrid(th, ph, indexing="ij")
    xu = (x(T + hh, P) - x(T - hh, P)) / (2 * hh)
    xv = (x(T, P + hh) - x(T, P - hh)) / (2 * hh)
    cr = np.cross(xu, xv, axis=0)
    w = (b - a) / n * 2 * np.pi / n
    return cr.sum(axis=(1, 2)) * w, np.linalg.norm(cr, axis=0).sum() * w


# self-checks: boundary is the unit circle, upward at the apex, area pi (1 + h^2), vector area (0, 0, pi)
for h in HS:
    x, tm = cap(h)
    assert np.allclose(x(tm, np.linspace(0, 6, 7)), np.array([np.cos(np.linspace(0, 6, 7)), np.sin(np.linspace(0, 6, 7)),
                                                               np.zeros(7)]), atol=1e-12)
    VA, A = patch_integrals(x, 0, tm)
    assert abs(A - np.pi * (1 + h * h)) < 1e-4 * A
    assert np.allclose(VA, [0, 0, np.pi], atol=2e-4)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax.plot([-1, 1], [0, 0], color=C["main"], lw=1.6)
for k in np.linspace(-0.8, 0.8, 5):
    ax.annotate("", xy=(k, 0.2), xytext=(k, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9,
                                                                  mutation_scale=7))
for h in HS:
    x, tm = cap(h)
    th = np.linspace(-tm, tm, 300)
    p = x(th, 0 * th)
    ax.plot(p[0], p[2], color=C["main"], lw=1.3)
    for t0 in np.linspace(-tm, tm, 9)[1:-1]:
        q = x(t0, 0.0)
        n = q - np.array([0, 0, h - (1 + h * h) / (2 * h)])
        n = 0.2 * n / np.linalg.norm(n)
        ax.annotate("", xy=(q[0] + n[0], q[2] + n[2]), xytext=(q[0], q[2]),
                    arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9, mutation_scale=7))
    ax.text(0.06, h + 0.04, rf"$h = {h:g}$", fontsize=10)
ax.plot([-1, 1], [0, 0], "o", color=C["accent"], ms=6, zorder=5)
ax.annotate("", xy=(1.42, np.pi / 4), xytext=(1.42, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=2.4, mutation_scale=14))
ax.text(1.48, 0.32, r"$\mathbf{A}/4$", fontsize=12, color=C["normal"])
ax.set_aspect("equal")
ax.set_xlim(-1.55, 1.85)
ax.set_ylim(-0.25, 2.05)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$z$", rotation=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", loc="left", fontsize=12)

hh = np.linspace(0, 2, 200)
ax2.plot(hh, np.pi * (1 + hh ** 2), color=C["main"], lw=1.6, label=r"area $A$")
ax2.plot(hh, np.pi + 0 * hh, color=C["normal"], lw=1.8, label=r"$|\mathbf{A}|$")
for h in [0.0] + HS:
    ax2.plot([h], [np.pi * (1 + h * h)], "o", color=C["main"], ms=4)
ax2.set_xlabel(r"$h$")
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 16.5)
ax2.legend(frameon=False, loc="upper left")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.set_title("(b)", loc="left", fontsize=12)
fig.tight_layout(w_pad=1.5)
dgfig.save(fig, __file__)
