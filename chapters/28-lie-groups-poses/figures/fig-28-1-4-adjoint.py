"""Figure 28.1.4: one rigid velocity field, two twist coordinates related by Ad (Section 28.1).

A body frame B sits at c = (2, 1, 0) in the world, rotated by 30 degrees about z (T_wb = (R, c)). It spins about its
own z axis with angular velocity 0.5 rad/s: in body coordinates the twist is xi_b = (v_b, w_b) = (0, (0, 0, 0.5)).
The same motion in world coordinates is xi_w = Ad_{T_wb} xi_b = (c x w_w, w_w) with w_w = R w_b = (0, 0, 0.5),
so v_w = (0.5, -1, 0) != 0 although the motion is a pure rotation.
(a) world coordinates (x, y): gray arrows = the velocity field xdot = w_w x x + v_w on a grid (drawn at 0.6 times the
    true length); the black dot is c (velocity 0), the orange arrow at the world origin O is v_w (true length).
(b) body coordinates (x_b, y_b) of the same points: the field is xdot_b = w_b x x_b, the body origin has velocity 0,
    and the world origin O appears at -R^T c with velocity R^T v_w (orange, true length).
Both panels use the same scale. Self-check: xi_w = Ad xi_b and T_wb Exp(s xi_b) T_wb^{-1} = Exp(s xi_w); the field
in (b) is R^T times the field in (a) at corresponding points.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-1-4-adjoint.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
c = np.array([2.0, 1.0, 0.0])
R = L.expSO3(np.radians(30) * np.array([0, 0, 1.0]))
T_wb = L.make_T(R, c)
w_b = np.array([0.0, 0.0, 0.5])
xi_b = np.concatenate([np.zeros(3), w_b])
xi_w = L.Ad_SE3(T_wb) @ xi_b
v_w, w_w = xi_w[:3], xi_w[3:]

assert np.allclose(w_w, R @ w_b)
assert np.allclose(v_w, np.cross(c, w_w))
assert np.allclose(v_w, [0.5, -1.0, 0.0])
for s in [0.3, 1.0, 2.5]:
    assert np.allclose(T_wb @ L.expSE3(s * xi_b) @ L.inv_T(T_wb), L.expSE3(s * xi_w), atol=1e-12)


def field_w(x):
    return np.cross(w_w, x) + v_w


def field_b(xb):
    return np.cross(w_b, xb)


for x in np.random.default_rng(1).normal(size=(5, 3)):
    x[2] = 0
    xb = R.T @ (x - c)
    assert np.allclose(field_b(xb), R.T @ field_w(x))

SPAN = 4.6
SCALE = 0.6
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 4.3), gridspec_kw=dict(wspace=0.22))
akw = dict(arrowstyle="-|>", lw=2.2, mutation_scale=14, shrinkA=0, shrinkB=0)
gkw = dict(arrowstyle="-|>", lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0, color="#9A9A9A")


def frame(axh, origin, Rm, lab, length=0.7):
    for k, col, nm in [(0, C["tangent"], "x"), (1, C["third"], "y")]:
        d = Rm[:2, k] * length
        axh.annotate("", xy=origin + d, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.6, color=col,
                                                                       mutation_scale=11, shrinkA=0, shrinkB=0))
        axh.text(*(origin + 1.25 * d), r"$%s_%s$" % (nm, lab), fontsize=12, color=col, ha="center", va="center")


# (a) world coordinates
xs = np.linspace(-0.6, 3.6, 8)
ys = np.linspace(-1.0, 3.0, 8)
for x in xs:
    for y in ys:
        p = np.array([x, y, 0.0])
        d = SCALE * field_w(p)[:2]
        ax.annotate("", xy=p[:2] + d, xytext=p[:2], arrowprops=gkw)
ax.plot(*c[:2], "o", color=C["main"], ms=7, zorder=5)
frame(ax, c[:2], R, "b")
ax.text(c[0] - 0.3, c[1] - 0.45, r"$c$: velocity $0$", fontsize=11, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
ax.plot(0, 0, "o", color=C["accent"], ms=6, zorder=5)
ax.annotate("", xy=v_w[:2], xytext=(0, 0), arrowprops=dict(color=C["accent"], **akw), zorder=6)
ax.text(-0.4, 0.12, r"$O$", fontsize=12)
ax.text(0.45, -1.35, r"$v_w = c \times \omega_w$", fontsize=12, color="#B07400")
fig.text(0.29, -0.03, r"$\xi_w = (v_w, \omega_w) = ((0.5, -1, 0), (0, 0, 0.5))$", fontsize=11.5, ha="center")
ax.set_xlim(-1.0, -1.0 + SPAN)
ax.set_ylim(-1.6, -1.6 + SPAN * 1.0)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$ (world)")
ax.set_ylabel(r"$y$ (world)")
ax.set_title("(a) world coordinates", fontsize=12)

# (b) body coordinates
O_b = R.T @ (-c)
vO_b = R.T @ v_w
cx, cy = O_b[0] / 2, O_b[1] / 2
xs2 = np.linspace(cx - 2.1, cx + 2.1, 8)
ys2 = np.linspace(cy - 2.0, cy + 2.0, 8)
for x in xs2:
    for y in ys2:
        p = np.array([x, y, 0.0])
        d = SCALE * field_b(p)[:2]
        bx.annotate("", xy=p[:2] + d, xytext=p[:2], arrowprops=gkw)
bx.plot(0, 0, "o", color=C["main"], ms=7, zorder=5)
frame(bx, np.zeros(2), np.eye(3), "b")
bx.text(-0.95, -0.45, r"body origin: velocity $0$", fontsize=11, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
bx.plot(*O_b[:2], "o", color=C["accent"], ms=6, zorder=5)
bx.annotate("", xy=O_b[:2] + vO_b[:2], xytext=O_b[:2], arrowprops=dict(color=C["accent"], **akw), zorder=6)
bx.text(O_b[0] - 0.55, O_b[1] - 0.15, r"$O$", fontsize=12)
bx.text(O_b[0] - 1.05, O_b[1] - 0.8, r"$R^{\mathsf{T}} v_w$", fontsize=12, color="#B07400")
fig.text(0.73, -0.03, r"$\xi_b = (v_b, \omega_b) = ((0, 0, 0), (0, 0, 0.5))$", fontsize=11.5, ha="center")
bx.set_xlim(cx - SPAN / 2, cx + SPAN / 2)
bx.set_ylim(cy - SPAN / 2 - 0.2, cy + SPAN / 2 - 0.2)
bx.set_aspect("equal")
bx.set_xlabel(r"$x_b$ (body)")
bx.set_ylabel(r"$y_b$ (body)")
bx.set_title("(b) body coordinates", fontsize=12)
fig.text(0.51, -0.1, r"$\xi_w = \mathrm{Ad}_{T_{wb}}\,\xi_b$" + "  (same motion, two coordinate descriptions)",
         ha="center", fontsize=12)

dgfig.save(fig, __file__)
