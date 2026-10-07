"""Figure 29.4.3: interpolating two flat Gaussians with four metrics; the swelling effect (Section 29.4).

Sigma_0 = diag(1, 0.15^2) (s = (1, 0.15), major axis along u^1) and Sigma_1 = the same rotated by 70 degrees.
(a) Rows: geodesics Sigma(t), t = 0, 0.25, 0.5, 0.75, 1, for the Euclidean, log-Euclidean, affine-invariant and
    Bures-Wasserstein metrics; each glyph is the 1-sigma ellipse (same scale everywhere).
(b) det Sigma(t) along each geodesic (log-Euclidean and affine-invariant coincide: det0^(1-t) det1^t = 0.0225).
Self-checks: midpoint dets 0.2334, 0.0225, 0.0225, 0.0604 and axis ratios 1.40, 1.91, 1.40, 2.61 (as in the verify
script and the caption).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-4-3-interpolation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def fun(S, f):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(f(w)) @ V.T


def geo(S0, S1, s, kind):
    if kind == "E":
        return (1 - s) * S0 + s * S1
    if kind == "LE":
        return fun((1 - s) * fun(S0, np.log) + s * fun(S1, np.log), np.exp)
    h = fun(S0, np.sqrt); hi = np.linalg.inv(h)
    if kind == "AI":
        return h @ fun(hi @ S1 @ hi, lambda w: w ** s) @ h
    T = hi @ fun(h @ S1 @ h, np.sqrt) @ hi
    M = (1 - s) * np.eye(2) + s * T
    return M @ S0 @ M


def R2(d):
    a = np.radians(d)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


F0 = np.diag([1.0, 0.15 ** 2])
F1 = R2(70) @ F0 @ R2(70).T
kinds = [("E", "Euclidean", C["main"], "-"), ("LE", "log-Euclidean", C["tangent"], "-"),
         ("AI", "affine-invariant", C["accent"], (0, (4, 2))), ("BW", "Bures–Wasserstein", C["third"], "-")]
for k, val in zip(("E", "LE", "AI", "BW"), (0.2334, 0.0225, 0.0225, 0.0604)):
    assert abs(np.linalg.det(geo(F0, F1, 0.5, k)) - val) < 5e-5

fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 7.6), gridspec_kw=dict(height_ratios=[1.75, 1], hspace=0.25))
ts = [0, 0.25, 0.5, 0.75, 1.0]
t = np.linspace(0, 2 * np.pi, 120)
dx, dy = 2.25, 3.05
for row, (k, name, col, _) in enumerate(kinds):
    y0 = -row * dy
    axa.text(-1.3, y0 + 0.7, name, fontsize=11, ha="left", va="bottom", color=col)
    for j, s in enumerate(ts):
        S = geo(F0, F1, s, k)
        w, V = np.linalg.eigh(S)
        E = np.array([j * dx, y0]) + (np.stack([np.cos(t), np.sin(t)], 1) * np.sqrt(w)) @ V.T
        axa.fill(*E.T, color=col, alpha=0.25, lw=0)
        axa.plot(*E.T, color=col, lw=1.4)
        if j == 2:
            ratio = np.sqrt(w[1] / w[0])
            assert abs(ratio - [1.40, 1.91, 1.40, 2.61][row]) < 0.006
            axa.text(j * dx, y0 - 0.95, r"$\det = %.4f$" % np.linalg.det(S) + "\n" + "ratio %.2f" % ratio, fontsize=9,
                     ha="center", va="top", color=col, linespacing=1.15)
for j, s in enumerate(ts):
    axa.text(j * dx, 1.35, r"$t = %g$" % s, fontsize=11, ha="center")
dgfig.schematic_axes(axa, (-1.35, 4 * dx + 1.15), (-3 * dy - 1.15, 1.7))
axa.set_title(r"(a) geodesics $\Sigma(t)$ from $\Sigma_0$ (left) to $\Sigma_1$ (right, rotated $70^\circ$)", fontsize=11)

tt = np.linspace(0, 1, 201)
for k, name, col, ls in kinds:
    axb.plot(tt, [np.linalg.det(geo(F0, F1, s, k)) for s in tt], color=col, ls=ls, lw=1.9, label=name)
axb.axhline(0.0225, color="#BBBBBB", lw=0.6, zorder=0)
axb.text(1.01, 0.0225, r"$\det\Sigma_0 = \det\Sigma_1$", fontsize=9, va="center")
axb.set_xlim(0, 1)
axb.set_ylim(0, 0.31)
axb.set_xlabel(r"$t$")
axb.set_ylabel(r"$\det\Sigma(t)$")
axb.legend(fontsize=9.5, frameon=False, loc="upper center", ncol=2)
axb.set_title(r"(b) volume $\det\Sigma(t)$ along each geodesic", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
