"""Figure 44.4.1: the Karcher mean on the unit sphere (Section 44.4, Example 44.4.8, Non-example 44.4.9).

Data: y_i = x(theta_i, phi_i) (x of GUIDELINES 7, r = 1): six points in a cap, (theta, phi) = (0.4, 0.2),
(0.8, 0), (0.6, 0.9), (1.0, 0.55), (0.35, 0.95), (0.9, 1.2), and one far point y_7 = x(2.0, -0.6).
Iteration x <- exp_x(mean_i log_x(y_i)) from x_0 = e_3 (the north pole).
(a) view from the direction of the Karcher mean: data (black dots), iterates (gray polyline with small dots), the
    Karcher mean (orange), the normalised Euclidean mean (blue cross), and the seven vectors log_x(y_i) at the
    Karcher mean drawn in its tangent plane scaled by 0.5 (blue arrows; their sum is zero).
(b) two antipodal points y and -y: f(x) = (1/2)(d(x, y)^2 + d(x, -y)^2) = (1/2)(theta^2 + (pi - theta)^2) as a
    function of the angle theta = d(x, y); the minimum value pi^2/4 is attained on the whole great circle theta = pi/2.
Self-checks: |sum log_x(y_i)| < 1e-10 at the result; f decreases monotonically; distance to the normalised Euclidean
mean 0.0742; the Karcher mean is closer to y_7 than the normalised Euclidean mean; every drawn point is on the visible
hemisphere.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-4-1-karcher-mean.py
"""

import pathlib
import sys

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
DATA = ((0.4, 0.2), (0.8, 0.0), (0.6, 0.9), (1.0, 0.55), (0.35, 0.95), (0.9, 1.2), (2.0, -0.6))
Y = [ch44.sphere_point(t, p) for t, p in DATA]
x = np.array([0.0, 0.0, 1.0])
iterates = [x.copy()]
fvals = []
for _ in range(60):
    v = np.mean([ch44.sphere_log(x, y) for y in Y], axis=0)
    fvals.append(0.5 * sum(np.arccos(np.clip(x @ y, -1, 1)) ** 2 for y in Y))
    if np.linalg.norm(v) < 1e-13:
        break
    x = ch44.sphere_exp(x, v)
    iterates.append(x.copy())
xb = x
assert np.linalg.norm(sum(ch44.sphere_log(xb, y) for y in Y)) < 1e-10
assert all(a >= b - 1e-15 for a, b in zip(fvals, fvals[1:]))
eu = ch44.unit(np.mean(Y, axis=0))
assert abs(np.arccos(xb @ eu) - 0.0742) < 5e-5
assert np.arccos(xb @ Y[6]) < np.arccos(eu @ Y[6])
th_b, ph_b = np.arccos(xb[2]), np.arctan2(xb[1], xb[0])
print(f"Karcher mean at (theta, phi) = ({th_b:.4f}, {ph_b:.4f}), f = {fvals[-1]:.6f}, {len(iterates) - 1} steps")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.0, 1.1]})
view = xb.copy()
right = ch44.unit(np.cross([0, 0, 1.0], view))
up = np.cross(view, right)
proj = lambda X: np.stack([np.asarray(X) @ right, np.asarray(X) @ up], -1)  # noqa: E731
tt = np.linspace(0, 2 * np.pi, 400)
axa.plot(np.cos(tt), np.sin(tt), color=col["main"], lw=0.9)
for th in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):
    C = np.stack([np.sin(th) * np.cos(tt), np.sin(th) * np.sin(tt), np.cos(th) + 0 * tt], 1)
    vis = C @ view > 0
    Q = proj(C)
    axa.plot(np.where(vis, Q[:, 0], np.nan), np.where(vis, Q[:, 1], np.nan), color=col["aux"], lw=0.4)
for ph in np.linspace(0, np.pi, 6, endpoint=False):
    C = np.stack([np.sin(tt) * np.cos(ph), np.sin(tt) * np.sin(ph), np.cos(tt)], 1)
    vis = C @ view > 0
    Q = proj(C)
    axa.plot(np.where(vis, Q[:, 0], np.nan), np.where(vis, Q[:, 1], np.nan), color=col["aux"], lw=0.4)
for y in list(Y) + iterates:
    assert y @ view > 0
for y in Y:
    axa.plot(*proj(y), "o", color=col["main"], ms=5, zorder=5)
It = np.array(iterates)
axa.plot(*proj(It).T, color=col["aux"], lw=1.0, zorder=4)
axa.plot(*proj(It).T, ".", color=col["aux"], ms=4, zorder=4)
for y in Y:
    lg = 0.5 * ch44.sphere_log(xb, y)
    a0, a1 = proj(xb), proj(xb + lg)
    axa.annotate("", xy=a1, xytext=a0, arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=1.0), zorder=6)
axa.plot(*proj(xb), "*", color=col["accent"], ms=13, zorder=7, mec=col["main"], mew=0.5)
axa.plot(*proj(eu), "X", color=col["tangent"], ms=8, mec="white", mew=0.6, zorder=8)
axa.text(*(proj(iterates[0]) + [0.05, 0.03]), r"$x_0 = e_3$", fontsize=10)
axa.text(*(proj(Y[6]) + [0.05, -0.1]), r"$y_7$", fontsize=10)
axa.text(*(proj(xb) + [-0.2, -0.13]), r"$\bar x$", fontsize=12)
axa.set_aspect("equal")
axa.set_xlim(-1.1, 1.1)
axa.set_ylim(-1.1, 1.1)
axa.set_axis_off()
axa.set_title("(a) a cap of six points and a far point", fontsize=11)
th = np.linspace(0, np.pi, 300)
axb.plot(th, 0.5 * (th ** 2 + (np.pi - th) ** 2), color=col["accent"], lw=2.0)
axb.axvline(np.pi / 2, color=col["aux"], lw=0.7, ls="--")
axb.plot(np.pi / 2, np.pi ** 2 / 4, "o", color=col["main"], ms=4)
axb.text(np.pi / 2 + 0.06, np.pi ** 2 / 4 + 0.25, r"min $\pi^2/4$:" + "\n" + r"whole equator $\theta = \pi/2$", fontsize=10)
axb.set_xlabel(r"$\theta = d(x, y)$")
axb.set_ylabel(r"$\frac{1}{2}\left(d(x,y)^2 + d(x,-y)^2\right)$")
axb.set_xticks([0, np.pi / 2, np.pi])
axb.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
axb.spines[["top", "right"]].set_visible(False)
axb.set_title(r"(b) antipodal pair $y, -y$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.17, wspace=0.28)
dgfig.save(fig, __file__)
