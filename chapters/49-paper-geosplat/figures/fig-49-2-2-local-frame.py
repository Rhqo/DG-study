"""Figure 49.2.2: the first and second order quantities that GeoSplat attaches to one Gaussian (Section 2.2 of [GeoSplat25]).

The surface is the torus T_{2,0.8}: x(u, v) = ((2 + 0.8 cos v) cos u, (2 + 0.8 cos v) sin u, 0.8 sin v), and the Gaussian
centre is q = x(0, 0) = (2.8, 0, 0) on the outer equator. With the outward unit normal u3 = e_x and W = -dN, the
principal curvatures are tau_1 = -1/0.8 = -1.25 (meridian direction w1 = e_z) and tau_2 = -1/2.8 = -0.357 (parallel
direction w2 = e_y); tau_1 has the larger absolute value, as in the paper's ordering.
(a) the torus patch, its tangent plane at q, the normal u3, the principal directions w1, w2 and an arbitrary orthonormal
    tangent basis u1, u2 (rotated by 35 degrees from w1, w2). Vectors have true length 0.6 (scale 0.6 of unit vectors).
(b) the two normal sections through q: height h along u3 against the arc parameter s along w1 (meridian circle of radius
    0.8) and along w2 (parallel circle of radius 2.8), with the second order approximations h = tau s^2 / 2 (dotted).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-2-2-local-frame.py
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8


def X(u, v):
    return np.array([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])


def N(u, v):  # outward unit normal
    return np.array([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)])


# self-check: shape operator W = -dN at q by finite differences, in the basis (x_u, x_v)
h = 1e-5
xu = (X(h, 0) - X(-h, 0)) / (2 * h)
xv = (X(0, h) - X(0, -h)) / (2 * h)
Nu = (N(h, 0) - N(-h, 0)) / (2 * h)
Nv = (N(0, h) - N(0, -h)) / (2 * h)
B = np.stack([xu, xv], axis=1)
Wmat = np.linalg.lstsq(B, -np.stack([Nu, Nv], axis=1), rcond=None)[0]
ev, evec = np.linalg.eig(Wmat)
order = np.argsort(-np.abs(ev))
tau = ev[order].real
assert np.allclose(tau, [-1.25, -1 / 2.8], atol=1e-6), tau
w1 = B @ evec[:, order[0]].real
w1 /= np.linalg.norm(w1)
w2 = B @ evec[:, order[1]].real
w2 /= np.linalg.norm(w2)
assert abs(abs(w1 @ np.array([0, 0, 1])) - 1) < 1e-8 and abs(abs(w2 @ np.array([0, 1, 0])) - 1) < 1e-8
q = X(0, 0)
u3 = N(0, 0)
w1, w2 = np.array([0.0, 0, 1]), np.array([0.0, 1, 0])
a = np.radians(35)
u1 = np.cos(a) * w1 + np.sin(a) * w2
u2 = -np.sin(a) * w1 + np.cos(a) * w2
assert abs(u1 @ u2) < 1e-12 and abs(u1 @ u3) < 1e-12 and abs(u2 @ u3) < 1e-12

fig = plt.figure(figsize=(7.8, 3.7))
ax = dgfig.axes3d(fig, pos=121, elev=20, azim=-50)
uu, vv = np.meshgrid(np.linspace(-0.42, 0.42, 25), np.linspace(-1.0, 1.0, 33))
P = X(uu, vv)
dgfig.surface(ax, P[0], P[1], P[2], grid_every=4)
dgfig.tangent_plane(ax, q, w2, w1, size=0.42, zorder=4)
L = 0.6
dgfig.arrow3d(ax, q, u3, role="normal", scale=L, label=r"$\tilde{\mathbf{u}}_3$", label_offset=(0.05, 0, 0.06))
dgfig.arrow3d(ax, q, w1, role="tangent", scale=L, label=r"$\tilde{\mathbf{w}}_1$", label_offset=(0, 0, 0.05))
dgfig.arrow3d(ax, q, w2, role="tangent", scale=L, label=r"$\tilde{\mathbf{w}}_2$", label_offset=(0, 0.06, 0))
dgfig.arrow3d(ax, q, u1, role="aux", scale=L, lw=1.0, head=8, label=r"$\mathbf{u}_1$", label_offset=(0, 0.05, 0.03),
              fontsize=10)
dgfig.arrow3d(ax, q, u2, role="aux", scale=L, lw=1.0, head=8, label=r"$\mathbf{u}_2$", label_offset=(0, 0.04, -0.06),
              fontsize=10)
dgfig.point3d(ax, q, label=r"$\mathbf{q}$", label_offset=(0.05, -0.12, -0.12))
dgfig.equal_aspect(ax, P.reshape(3, -1).T, np.array([q + L * u3]), zoom=1.45)
ax.set_title("(a) frame at a Gaussian", fontsize=11)

ax2 = fig.add_subplot(122)
s = np.linspace(-0.79, 0.79, 300)
for rad, tv, lab, ls in ((0.8, tau[0], r"along $\tilde{\mathbf{w}}_1$: $\tau_1=-1.25$", "-"),
                         (2.8, tau[1], r"along $\tilde{\mathbf{w}}_2$: $\tau_2=-0.357$", "--")):
    hh = -(rad - np.sqrt(rad ** 2 - s ** 2))
    ax2.plot(s, hh, color=C["tangent"], ls=ls, lw=1.8, label=lab)
    ax2.plot(s, 0.5 * tv * s ** 2, color=C["aux"], ls=":", lw=1.1)
ax2.axhline(0, color=C["region"], lw=3, alpha=0.6)
ax2.annotate("", xy=(0, 0.3), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.6))
ax2.text(0.04, 0.27, r"$\tilde{\mathbf{u}}_3$", color=C["normal"], fontsize=12)
ax2.plot(0, 0, "o", color=C["main"], ms=4)
ax2.text(0.55, 0.05, "tangent plane", color=C["tangent"], fontsize=9, ha="center")
ax2.set_xlabel(r"$s$ (distance along the direction)")
ax2.set_ylabel(r"height $h$ along $\tilde{\mathbf{u}}_3$")
ax2.set_xlim(-0.82, 0.82)
ax2.set_ylim(-0.75, 0.38)
ax2.legend(loc="lower center", fontsize=8.5, frameon=False)
ax2.set_title("(b) two normal sections", fontsize=11)
# the curves at s = 0.5: meridian -0.1755, parallel -0.0450; quadratic approximations -0.15625, -0.0446
assert abs(-(0.8 - np.sqrt(0.64 - 0.25)) + 0.17550) < 1e-4
assert abs(-(2.8 - np.sqrt(7.84 - 0.25)) + 0.04500) < 1e-4
fig.subplots_adjust(left=0.0, right=0.98, wspace=0.18, bottom=0.14)
dgfig.save(fig, __file__)
