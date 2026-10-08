"""Figure 28.2.2: Gauss-Newton for camera pose (PnP) with consistent and with mixed perturbations (Section 28.2).

40 known world points, observed without noise in normalized image coordinates by a camera T_cw. Unknown: T_cw.
Start: the true pose multiplied on the left by Exp((0.15, -0.1, 0.2, 10 deg about a fixed axis)). Each GN step solves
J d = -r in the least-squares sense and updates T. Four variants:
  black  : Jacobian for the left perturbation  Exp(d) T, update Exp(d) T       (consistent)
  blue   : Jacobian for the right perturbation T Exp(d), update T Exp(d)       (consistent)
  orange : Jacobian for the right perturbation, update Exp(d) T                (mixed)
  red    : Jacobian for the left perturbation,  update T Exp(d)                (mixed)
(a) the true camera is rotated by 3 degrees from the world frame and sits at the world origin (Ad_T close to I);
(b) the true camera is rotated by 120 degrees and sits at (4, -2, 3).
Vertical axis: RMS reprojection error (normalized coordinates, log scale); errors below 1e-16 are drawn at 1e-16.
A cross marks the last iterate before some point fell behind the camera (z <= 0), after which the run stops.
Self-check: the consistent variants reach 1e-15 within 5 iterations in both cases; in (a) the mixed variants still
converge (linearly); in (b) both mixed variants put points behind the camera.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-2-2-mixed-perturbation.py``
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
ITERS = 20
FLOOR = 1e-16


def make_problem(angle, center, seed=0):
    r = np.random.default_rng(seed)
    Xc = np.c_[r.uniform(-2, 2, 40), r.uniform(-1.5, 1.5, 40), r.uniform(4, 8, 40)]
    ax = np.array([0.3, 1.0, 0.2])
    ax /= np.linalg.norm(ax)
    Rwc = L.expSO3(angle * ax)
    Xw = (Rwc @ Xc.T).T + center
    return Xw, Xc[:, :2] / Xc[:, 2:], L.inv_T(L.make_T(Rwc, center))


def residual(T, Xw, u):
    xc = (T[:3, :3] @ Xw.T).T + T[:3, 3]
    return (xc[:, :2] / xc[:, 2:] - u).ravel(), xc


def jacobian(T, Xw, xc, mode):
    rows = []
    for X, x in zip(Xw, xc):
        z = x[2]
        Dphi = np.array([[1 / z, 0, -x[0] / z ** 2], [0, 1 / z, -x[1] / z ** 2]])
        if mode == "L":
            rows.append(Dphi @ np.hstack([np.eye(3), -L.hat3(x)]))
        else:
            rows.append(Dphi @ T[:3, :3] @ np.hstack([np.eye(3), -L.hat3(X)]))
    return np.vstack(rows)


def run(Xw, u, T0, jmode, umode):
    T = T0.copy()
    hist = []
    for _ in range(ITERS + 1):
        r, xc = residual(T, Xw, u)
        if np.any(xc[:, 2] <= 0):
            return np.array(hist), True
        hist.append(np.sqrt(np.mean(r ** 2)))
        d = -np.linalg.lstsq(jacobian(T, Xw, xc, jmode), r, rcond=None)[0]
        T = L.expSE3(d) @ T if umode == "L" else T @ L.expSE3(d)
    return np.array(hist), False


cases = [(np.radians(3), np.zeros(3), r"(a) camera $3^\circ$ from world, at origin"),
         (np.radians(120), np.array([4.0, -2.0, 3.0]), r"(b) camera $120^\circ$, at $(4, -2, 3)$")]
variants = [("L", "L", C["main"], "-", "left Jacobian, left update"),
            ("R", "R", C["tangent"], (0, (5, 2)), "right Jacobian, right update"),
            ("R", "L", C["accent"], "-", "right Jacobian, left update (mixed)"),
            ("L", "R", C["normal"], (0, (1.5, 1.5)), "left Jacobian, right update (mixed)")]
dT = L.expSE3(np.r_[0.15, -0.1, 0.2, np.radians(10) * np.array([0.6, -0.5, 0.62])])

fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, gridspec_kw=dict(wspace=0.07))
results = {}
for ax, (ang, cen, ttl) in zip(axs, cases):
    Xw, u, Tcw = make_problem(ang, cen)
    T0 = dT @ Tcw
    for jm, um, col, ls, lab in variants:
        h, behind = run(Xw, u, T0, jm, um)
        results[(ttl[:3], jm, um)] = (h, behind)
        k = np.arange(len(h))
        ax.semilogy(k, np.maximum(h, FLOOR), color=col, ls=ls, lw=1.9, label=lab, marker="o", ms=3)
        if behind:
            ax.plot(k[-1], h[-1], "x", color=col, ms=11, mew=2.2)
    ax.set_xlabel("Gauss–Newton iteration")
    ax.set_title(ttl, fontsize=12)
    ax.set_xlim(-0.5, ITERS + 0.5)
    ax.set_ylim(3e-17, 1e2)
    ax.grid(True, which="major", color="#E5E5E5", lw=0.5)
axs[0].set_ylabel("RMS reprojection error")
axs[1].text(6.0, 0.6, r"$\times$: points fell" + "\n" + "behind the camera", fontsize=10.5)
axs[1].legend(loc="center right", bbox_to_anchor=(0.99, 0.45), fontsize=9.5, frameon=True, framealpha=0.95)

# self-checks
for key in ["(a)", "(b)"]:
    for jm, um in [("L", "L"), ("R", "R")]:
        h, behind = results[(key, jm, um)]
        assert not behind and h[5] < 1e-15, (key, jm, um, h[:6])
for jm, um in [("R", "L"), ("L", "R")]:
    h, behind = results[("(a)", jm, um)]
    assert not behind and h[-1] < 1e-14 and h[3] > 1e-8
    h, behind = results[("(b)", jm, um)]
    assert behind

dgfig.save(fig, __file__)
