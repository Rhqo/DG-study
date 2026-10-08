"""Figure 27.4.2: the reciprocal product of Plucker coordinates detects when two lines meet (Section 27.4).

Line 1: the x axis, (d1, m1) = (e_x, 0). Line 2: direction d2 = (cos a, sin a, 0) through (0, 0, h), i.e. the line 1 rotated
by the angle a about the z axis and lifted by h.
(a) 3D, a = 60 deg: line 1 (black) and line 2 for h = -0.6 (blue, passes below), h = 0 (orange, meets line 1 at the
    origin), h = +0.6 (green, passes above). Gray dotted segment: the common perpendicular (the z axis).
(b) <d1, m2> + <d2, m1> as a function of h for a = 90, 45, 10 deg and a = 0 (parallel lines). It equals -h sin a.
Self-checks: the reciprocal product equals <p1 - p2, d1 x d2> = -h sin a; it is 0 for all h when a = 0.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-4-2-plucker-meet.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def plucker(p, d):
    return d, np.cross(p, d)


def recip(L1, L2):
    return L1[0] @ L2[1] + L2[0] @ L1[1]


L1 = plucker(np.zeros(3), np.array([1.0, 0, 0]))
hs = np.linspace(-1, 1, 41)
curves = {}
for a_deg in (90, 45, 10, 0):
    a = np.radians(a_deg)
    d2 = np.array([np.cos(a), np.sin(a), 0.0])
    vals = np.array([recip(L1, plucker(np.array([0, 0, h]), d2)) for h in hs])
    assert np.allclose(vals, -hs * np.sin(a), atol=1e-12)
    curves[a_deg] = vals
assert np.allclose(curves[0], 0)

fig = plt.figure(figsize=(8.6, 4.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)
ax = dgfig.axes3d(fig, pos=gs[0], elev=18, azim=-100)
s = np.linspace(-1.6, 1.6, 2)
ax.plot(s, 0 * s, 0 * s, color="k", lw=2.2, zorder=4)
ax.text(1.05, 0, -0.22, "line 1", fontsize=12)
a = np.radians(60)
d2 = np.array([np.cos(a), np.sin(a), 0.0])
s2 = np.linspace(-0.55, 0.55, 2)          # short segments: they do not cross line 1 on the screen unless h = 0
for h, col, lab in ((-0.6, C["tangent"], r"$h=-0.6$: below"), (0.0, C["accent"], r"$h=0$: meets"),
                    (0.6, C["third"], r"$h=+0.6$: above")):
    P = np.array([0, 0, h]) + s2[:, None] * d2[None, :]
    ax.plot(*P.T, color=col, lw=2.0, zorder=6 if h >= 0 else 3)
    ax.scatter(0, 0, h, color=col, s=22, zorder=7)
    ax.text(*(P[1] + np.array([0.08, 0.0, 0.02])), lab, fontsize=11.5, color=col)
ax.plot([0, 0], [0, 0], [-0.6, 0.6], color=C["aux"], lw=1.0, ls=":", zorder=2)
ax.text(-0.12, 0, 0.3, r"$|h|$", fontsize=11, color=C["aux"], ha="right")
ax.text(-0.12, 0, -0.3, r"$|h|$", fontsize=11, color=C["aux"], ha="right")
dgfig.equal_aspect(ax, np.array([[-1.6, -0.8, -0.7], [1.6, 0.8, 0.7]]), zoom=1.35)
ax.set_title(r"(a) line 2 at angle $a=60^\circ$, lifted by $h$", fontsize=12)

bx = fig.add_subplot(gs[1])
styles = {90: (C["normal"], "-"), 45: (C["accent"], "-"), 10: (C["tangent"], "-"), 0: ("k", "--")}
for a_deg, vals in curves.items():
    col, ls = styles[a_deg]
    bx.plot(hs, vals, color=col, ls=ls, lw=1.8, label=rf"$a={a_deg}^\circ$" + (" (parallel)" if a_deg == 0 else ""))
bx.axhline(0, color=C["aux"], lw=0.6)
bx.axvline(0, color=C["aux"], lw=0.6)
bx.plot([0], [0], "o", color=C["accent"], ms=7)
bx.text(0.05, 0.08, "lines meet", fontsize=11, color=C["accent"])
bx.set_xlabel(r"lift $h$ of line 2")
bx.set_ylabel(r"$\langle d_1,m_2\rangle+\langle d_2,m_1\rangle$")
bx.legend(loc="upper right", fontsize=10.5, frameon=False)
bx.set_title(r"(b) reciprocal product $=-h\sin a$", fontsize=12)
dgfig.save(fig, __file__)
