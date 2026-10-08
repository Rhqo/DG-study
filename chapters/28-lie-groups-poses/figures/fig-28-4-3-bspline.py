"""Figure 28.4.3: a cumulative cubic B-spline on SE(3) versus piecewise geodesic interpolation (Section 28.4).

Eight control poses T_k (k = 0, ..., 7; body-to-world, body x forward and z up, world z up) with centers on a wavy path and yaw angles
(0, 0.6, 1.4, 1.0, 0.2, -0.4, 0.3, 1.0) rad, placed at times tau = k.
  gray : piecewise interpolation through the control poses: on [k, k+1], SLERP for the rotation and a straight line
         for the center (the SO(3) x R^3 geodesic of Section 28.4).
  blue : the uniform cumulative cubic B-spline (lib/dglie.bspline_SE3, [Sommer20 (23)-(24)], [Lovegrove13]),
         T(tau) = T_i prod_{j=1}^{3} Exp(lambda_j(u) Log(T_{i+j-1}^{-1} T_{i+j})), drawn for tau in [1, 6] with
         i = floor(tau) - 1, u = tau - floor(tau). It does not pass through the control poses.
(a) top view of the centers (dots = control centers, small arrows = body x axes (heading) of the control poses);
(b) the z component omega^3 of the body angular velocity (finite differences) of both curves; dotted lines = knots.
Self-check: the B-spline is continuous with continuous first and second derivatives at the knots (finite differences
from both sides agree), and only 4 control poses affect each segment (moving T_7 does not change T(tau) for
tau <= 4).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-4-3-bspline.py``
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
yaws = np.array([0, 0.6, 1.4, 1.0, 0.2, -0.4, 0.3, 1.0])
cent = np.array([[0, 0, 0], [1.0, 0.6, 0.1], [1.7, 1.8, 0.2], [2.4, 2.9, 0.1], [3.6, 3.1, 0.0],
                 [4.8, 2.5, 0.1], [5.7, 2.9, 0.2], [6.3, 4.0, 0.1]])


def ctrl(c, yaw):
    return L.make_T(L.expSO3([0, 0, yaw]) @ L.expSO3([0.05 * np.sin(3 * yaw), 0.0, 0.0]), c)


Ts = [ctrl(c, y) for c, y in zip(cent, yaws)]
NC = len(Ts)


def spline(tau):
    return L.bspline_SE3(Ts, tau - 1.0)


def piecewise(tau):
    k = min(int(np.floor(tau)), NC - 2)
    s = tau - k
    A, Bm = Ts[k], Ts[k + 1]
    Rm = A[:3, :3] @ L.expSO3(s * L.logSO3(A[:3, :3].T @ Bm[:3, :3]))
    return L.make_T(Rm, (1 - s) * A[:3, 3] + s * Bm[:3, 3])


h = 1e-4
taus_s = np.linspace(1.0, 6.0 - h, 1001)
taus_p = np.linspace(0.0, 7.0 - h, 1401)


def yaw_rate(f, taus):
    """z component of the body angular velocity Log(R(t)^T R(t + h))/h (the camera turns mostly about z)."""
    return np.array([L.logSO3(f(t)[:3, :3].T @ f(t + h)[:3, :3])[2] / h for t in taus])


ws = yaw_rate(spline, taus_s)
seg_rates = np.array([yaw_rate(piecewise, [k + 0.5])[0] for k in range(NC - 1)])

# self-checks: continuity of the spline and its derivatives at the knots
for knot in [2.0, 3.0, 4.0, 5.0]:
    e = 1e-3
    left = [spline(knot - k * e) for k in range(3)]
    right = [spline(knot + k * e) for k in range(3)]
    assert np.allclose(spline(knot - 1e-9), spline(knot + 1e-9), atol=1e-7)
    vL = L.logSE3(L.inv_T(left[1]) @ left[0]) / e
    vR = L.logSE3(L.inv_T(right[0]) @ right[1]) / e
    assert np.allclose(vL, vR, atol=5e-3)
    aL = (L.logSE3(L.inv_T(left[1]) @ left[0]) - L.logSE3(L.inv_T(left[2]) @ left[1])) / e ** 2
    aR = (L.logSE3(L.inv_T(right[1]) @ right[2]) - L.logSE3(L.inv_T(right[0]) @ right[1])) / e ** 2
    assert np.allclose(aL, aR, atol=2e-2)
Ts_mod = list(Ts)
Ts_mod[7] = ctrl(cent[7] + np.array([2.0, -3.0, 1.0]), 2.5)
for t in [1.2, 2.5, 3.9]:
    assert np.allclose(L.bspline_SE3(Ts_mod, t - 1.0), spline(t), atol=1e-12)
assert not np.allclose(L.bspline_SE3(Ts_mod, 4.5), spline(5.5))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.9), gridspec_kw=dict(wspace=0.28, width_ratios=[1.05, 1]))
cp = np.array([piecewise(t)[:3, 3] for t in taus_p])
cs = np.array([spline(t)[:3, 3] for t in taus_s])
ax.plot(cp[:, 0], cp[:, 1], color=C["aux"], lw=1.6, label="piecewise geodesic")
ax.plot(cs[:, 0], cs[:, 1], color=C["tangent"], lw=2.2, label="cumulative cubic B-spline")
for k, T in enumerate(Ts):
    ax.plot(*T[:2, 3], "o", color=C["main"], ms=5, zorder=5)
    f = T[:3, 0]
    ax.annotate("", xy=T[:2, 3] + 0.5 * f[:2], xytext=T[:2, 3],
                arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.text(T[0, 3] - 0.15, T[1, 3] - 0.45, r"$T_%d$" % k, fontsize=10.5)
ax.set_aspect("equal")
ax.set_xlim(-0.7, 7.2)
ax.set_ylim(-0.9, 4.8)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.plot([], [], color=C["main"], marker=r"$\rightarrow$", ms=12, ls="none", label="heading (body $x$)")
ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95)
ax.set_title("(a) centers (top view)", fontsize=12)

for k in range(NC - 1):
    bx.plot([k, k + 1], [seg_rates[k]] * 2, color=C["aux"], lw=1.8, label="piecewise geodesic" if k == 0 else None)
    if k < NC - 2:
        bx.plot([k + 1, k + 1], [seg_rates[k], seg_rates[k + 1]], color=C["aux"], lw=0.8, ls=":")
bx.plot(taus_s, ws, color=C["tangent"], lw=2.2, label="B-spline")
for k in range(NC):
    bx.axvline(k, color="#CCCCCC", lw=0.7, ls=":")
bx.set_xlabel(r"time $\tau$ (knots at integers)")
bx.set_ylabel(r"yaw rate $\omega^3(\tau)$ (rad/s)")
bx.set_xlim(0, 7)
bx.set_ylim(-1.0, 1.15)
bx.axhline(0, color="#BBBBBB", lw=0.6)
bx.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.95)
bx.set_title("(b) angular velocity about $z$", fontsize=12)
bx.annotate("jump at a knot", xy=(2.0, -0.25), xytext=(0.15, -0.75), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))

assert np.all(np.abs(np.diff(ws)) < 1e-2)  # samples 0.005 apart: no jumps
print("piecewise yaw rates per segment:", np.round(seg_rates, 3), " spline range:", np.round([ws.min(), ws.max()], 3))
dgfig.save(fig, __file__)
