"""Figure 38.2.2: a straight line in Mercator coordinates is a loxodrome (unit sphere).

The line t = phi cot(beta), beta = 60 degrees (the angle with the meridians), through (phi, t) = (0, 0).
(a) the Mercator plane (phi, t) with the meridians phi = k pi / 2 (gray dashed), the segment t in [0, 3] (orange)
and its continuation up to t = 8 (black). (b) its image z(phi, t) = (sech t cos phi, sech t sin phi, tanh t) on the
unit sphere: it crosses every meridian at 60 degrees and spirals into the north pole. The length of the orange
part is gd(3) / cos(beta) = 2.94261 and the length of the whole curve from the equator to the pole is
(pi / 2) / cos(beta) = pi.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-2-2-loxodrome.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg38  # noqa: E402

C = dgfig.COLORS
beta = np.pi / 3


def curve(t):
    return ddg38.mercator_inv(t * np.tan(beta), t)


def length(t0, t1, n=200000):
    t = np.linspace(t0, t1, n + 1)
    P = curve(t)
    return np.sum(np.linalg.norm(np.diff(P, axis=1), axis=0))


# self-checks: the angle with the meridian is beta; lengths
for t0 in (0.0, 0.7, 2.5):
    h = 1e-6
    vel = (curve(t0 + h) - curve(t0 - h)) / (2 * h)
    mer = (ddg38.mercator_inv(t0 * np.tan(beta), t0 + h) - ddg38.mercator_inv(t0 * np.tan(beta), t0 - h)) / (2 * h)
    cosang = vel @ mer / (np.linalg.norm(vel) * np.linalg.norm(mer))
    assert abs(cosang - np.cos(beta)) < 1e-6
L3 = length(0, 3)
assert abs(L3 - np.arctan(np.sinh(3.0)) / np.cos(beta)) < 1e-6 and abs(L3 - 2.94261) < 1e-5
assert abs(length(0, 40) - np.pi) < 1e-6

fig = plt.figure(figsize=(7.0, 3.6))
ax = fig.add_axes([0.06, 0.13, 0.42, 0.80])
for k in range(0, 9):
    ax.axvline(k * np.pi / 2, color=C["aux"], lw=0.6, ls=(0, (3, 2.5)))
ax.axhline(0, color=C["aux"], lw=0.6)
ts = np.linspace(0, 3, 50)
ax.plot(ts * np.tan(beta), ts, color=C["accent"], lw=2.0)
ts2 = np.linspace(3, 8, 50)
ax.plot(ts2 * np.tan(beta), ts2, color=C["main"], lw=1.2)
# angle beta between the line and the meridian at (0, 0)
arc = np.linspace(0, beta, 30)
ax.plot(0.9 * np.sin(arc), 0.9 * np.cos(arc), color=C["main"], lw=0.8)
ax.text(0.55, 1.15, r"$\beta$", fontsize=12)
ax.set_xlim(-0.3, 4 * np.pi + 0.3)
ax.set_ylim(-0.5, 8.3)
ax.set_aspect("equal")
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
ax.set_xlabel(r"$\varphi$")
ax.set_ylabel(r"$t$", rotation=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", loc="left", fontsize=12)

ax3 = dgfig.axes3d(fig, 111, elev=35, azim=70)
ax3.set_position([0.48, 0.0, 0.52, 1.0])
TH, PH = np.meshgrid(np.linspace(0, np.pi, 37), np.linspace(0, 2 * np.pi, 73), indexing="ij")
X, Y, Z = ddg38.sphere(TH, PH)
dgfig.surface(ax3, X, Y, Z, alpha=0.25, grid_every=6)
P1 = curve(np.linspace(0, 3, 300))
P2 = curve(np.linspace(3, 12, 600))
view = dgfig.view_vector(ax3)
dgfig.curve3d(ax3, P1.T, role="accent", lw=2.0, visible=P1.T @ view > -0.05)
dgfig.curve3d(ax3, P2.T, role="main", lw=1.0, visible=P2.T @ view > -0.05)
dgfig.point3d(ax3, [0, 0, 1], None, size=10)
dgfig.point3d(ax3, curve(0.0), None, size=10)
dgfig.equal_aspect(ax3, np.stack([X, Y, Z], -1), zoom=1.3)
ax3.set_title("(b)", loc="left", fontsize=12, y=0.92)
dgfig.save(fig, __file__)
