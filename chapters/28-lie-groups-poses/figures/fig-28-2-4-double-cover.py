"""Figure 28.2.4: the double cover S^3 -> SO(3) restricted to rotations about one axis n (Section 28.2).

(a) The great circle {(cos a, sin a n)} of S^3 drawn in its own plane: horizontal axis q_0, vertical axis
    <q_{1:3}, n>. The unit quaternion of the rotation by theta about n is q(theta) = (cos(theta/2), sin(theta/2) n), so
    a = theta/2. Blue: 0 <= theta < 2 pi (first full turn, a in [0, pi)); orange dashed: 2 pi <= theta < 4 pi (second
    turn). Black dots: q for theta = 80 degrees and -q (theta = 440 degrees).
(b) The rotations themselves: R(theta) acting on a unit vector u perpendicular to n, drawn in the plane
    perpendicular to n. Both turns cover the whole circle (the orange copy is drawn at radius 1.08 to stay visible),
    and q and -q give the same rotation (one black dot).
Self-check: R(q) = R(-q) via (29.2.5); q(2 pi) = -q(0); the angle between q(theta1) and q(theta2) on S^3 is
|theta1 - theta2|/2 (for |theta1 - theta2| <= 2 pi).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-2-4-double-cover.py``
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
n = np.array([0.0, 0.6, 0.8])
u = np.array([1.0, 0.0, 0.0])
v = np.cross(n, u)


def q_of(theta):
    return np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * n])


th0 = np.radians(80)
q = q_of(th0)
assert np.allclose(L.q_to_R(q), L.q_to_R(-q))
assert np.allclose(L.q_to_R(q), L.expSO3(th0 * n))
assert np.allclose(q_of(2 * np.pi), -q_of(0))
assert np.allclose(q_of(th0 + 2 * np.pi), -q)
for t1, t2 in [(0.3, 2.0), (1.0, 5.5)]:
    assert np.isclose(np.arccos(np.clip(q_of(t1) @ q_of(t2), -1, 1)), abs(t1 - t2) / 2)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.2), gridspec_kw=dict(wspace=0.5))
a1 = np.linspace(0, np.pi, 200)
a2 = np.linspace(np.pi, 2 * np.pi, 200)
ax.plot(np.cos(a1), np.sin(a1), color=C["tangent"], lw=2.4)
ax.plot(np.cos(a2), np.sin(a2), color=C["accent"], lw=2.4, ls=(0, (5, 2)))
for th, lab, off in [(0, r"$\theta = 0$", (0.08, 0.06)), (np.pi, r"$\theta = \pi$", (0.06, 0.06)),
                     (2 * np.pi, r"$\theta = 2\pi$", (-0.42, 0.1)), (3 * np.pi, r"$\theta = 3\pi$", (0.06, -0.2))]:
    p = q_of(th)
    ax.plot(p[0], p[1:] @ n, "o", color=C["aux"], ms=5)
    ax.text(p[0] + off[0], p[1:] @ n + off[1], lab, fontsize=10.5)
for pp, lab, off in [(q, r"$q$", (0.07, 0.05)), (-q, r"$-q$", (-0.25, -0.15))]:
    ax.plot(pp[0], pp[1:] @ n, "o", color=C["main"], ms=8, zorder=5)
    ax.text(pp[0] + off[0], pp[1:] @ n + off[1], lab, fontsize=13)
ax.plot([q[0], -q[0]], [q[1:] @ n, -q[1:] @ n], color=C["main"], lw=0.7, ls=":")
ax.axhline(0, color="#CCCCCC", lw=0.6)
ax.axvline(0, color="#CCCCCC", lw=0.6)
ax.set_aspect("equal")
ax.set_xlim(-1.45, 1.45)
ax.set_ylim(-1.45, 1.35)
ax.set_xlabel(r"$q_0$")
ax.set_ylabel(r"$\langle q_{1:3}, n\rangle$")
ax.set_title(r"(a) $q(\theta) = (\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\,n)$ on $S^3$", fontsize=12)
ax.text(-1.4, -1.3, "blue: first turn\norange: second turn", fontsize=10)

t1 = np.linspace(0, 2 * np.pi, 300)
bx.plot(np.cos(t1), np.sin(t1), color=C["tangent"], lw=2.4)
bx.plot(1.08 * np.cos(t1), 1.08 * np.sin(t1), color=C["accent"], lw=2.0, ls=(0, (5, 2)))
p = L.q_to_R(q) @ u
assert np.allclose([p @ u, p @ v], [np.cos(th0), np.sin(th0)])
bx.plot(np.cos(th0), np.sin(th0), "o", color=C["main"], ms=8, zorder=5)
bx.annotate(r"$R(q) = R(-q)$", xy=(np.cos(th0), np.sin(th0)), xytext=(0.35, 1.25), fontsize=12,
            arrowprops=dict(arrowstyle="->", lw=0.9))
bx.annotate("", xy=(0.7 * np.cos(th0), 0.7 * np.sin(th0)), xytext=(0.7, 0),
            arrowprops=dict(arrowstyle="->", lw=1.0, connectionstyle="arc3,rad=0.35"))
bx.text(0.55, 0.3, r"$80^\circ$", fontsize=11)
bx.plot(1, 0, "o", color=C["aux"], ms=5)
bx.text(1.12, -0.12, r"$\theta = 0 \equiv 2\pi$", fontsize=10.5)
bx.arrow(0, 0, 0.92, 0, color=C["aux"], width=0.004, head_width=0.05, length_includes_head=True)
bx.text(0.35, -0.16, r"$u$", fontsize=11, color=C["aux"])
bx.axhline(0, color="#CCCCCC", lw=0.6)
bx.axvline(0, color="#CCCCCC", lw=0.6)
bx.set_aspect("equal")
bx.set_xlim(-1.45, 1.9)
bx.set_ylim(-1.35, 1.45)
bx.set_xlabel(r"$\langle R u, u\rangle$")
bx.set_ylabel(r"$\langle R u, n \times u\rangle$")
bx.set_title(r"(b) the rotation $R(\theta)$ about $n$", fontsize=12)
fig.text(0.468, 0.56, r"$2:1$", fontsize=13, ha="center")
fig.text(0.468, 0.5, r"$\longrightarrow$", fontsize=16, ha="center")

dgfig.save(fig, __file__)
