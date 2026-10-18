"""Figure 37.2.2: a regular homotopy and a homotopy that is not regular.

(a) the averaging homotopy of the Proof sketch of Theorem 37.2.11 from the limacon r = 1/2 + cos t (rotation index 2,
rescaled to length 2 pi) to the circle of radius 1/2 traversed twice: lambda = 0, 1/4, 1/2, 3/4, 1, all regular
(minimum speed >= 0.94);
(b) the limacons r = a + cos t for a = 0.5, 0.8, 1.0, 1.2, 1.5: closed curves throughout, but at a = 1 (cardioid) the
speed vanishes at t = pi and the rotation index jumps from 2 to 1.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-2-2-regular-homotopy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg37  # noqa: E402

C = dgfig.COLORS
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.0, 3.9))

# (a) averaging homotopy, limacon (k = 2) -> doubled circle
s, phiA, _ = ddg37.unit_length_tangent_angle("limacon")
phiB = phiA[0] + 2 * s                                   # circle of radius 1/2 traversed twice, same start direction
assert abs(phiA[-1] - phiA[0] - 4 * np.pi) < 1e-6
lams = [0, 0.25, 0.5, 0.75, 1.0]
vmins = []
for j, lam in enumerate(lams):
    X, vmin, mean = ddg37.averaged_curve(s, phiA, phiB, lam)
    vmins.append(vmin)
    assert np.linalg.norm(X[-1] - X[0]) < 1e-9 and vmin > 0.9
    X = X - X.mean(0) + np.array([2.6 * j, 0.0])
    ax.plot(X[:, 0], X[:, 1], color=C["main"] if lam in (0, 1) else C["tangent"], lw=1.5)
    ax.text(2.6 * j, -1.45, r"$\lambda = %s$" % ("0", r"\frac{1}{4}", r"\frac{1}{2}", r"\frac{3}{4}", "1")[j],
            ha="center", fontsize=12)
assert min(vmins) > 0.94
ax.text(-1.3, 0.95, r"$\mathrm{rot} = 2$ for every $\lambda$", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.4, 2.6 * 4 + 1.3)
ax.set_ylim(-1.7, 1.2)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")

# (b) the limacon family through the cardioid
tt = np.linspace(0, 2 * np.pi, 2001)
for j, a in enumerate([0.5, 0.8, 1.0, 1.2, 1.5]):
    r = a + np.cos(tt)
    X = np.stack([r * np.cos(tt), r * np.sin(tt)], 1)
    vel = np.stack([-np.sin(tt) * np.cos(tt) - r * np.sin(tt), -np.sin(tt) ** 2 + r * np.cos(tt)], 1)
    vmin = np.linalg.norm(vel, axis=1).min()
    assert abs(vmin - abs(1 - a)) < 1e-6                 # |gamma'|^2 = 1 + a^2 + 2 a cos t
    off = np.array([2.6 * j - 0.3, 0.0])
    col = C["normal"] if a == 1.0 else C["main"]
    X = X / (1 + a)                                      # drawn scaled to maximal radius 1
    bx.plot(X[:, 0] + off[0], X[:, 1], color=col, lw=1.5)
    if a == 1.0:
        bx.plot([off[0]], [0], "o", ms=5, mfc="white", mec=C["normal"], zorder=5)
        lab = r"$a = 1$: cusp"
    else:
        V = vel / np.linalg.norm(vel, axis=1, keepdims=True)
        k = round(float(ddg37.angle_lift(V)[-1] - ddg37.angle_lift(V)[0]) / (2 * np.pi))
        assert k == (2 if a < 1 else 1)
        lab = r"$a = %.1f$, rot $= %d$" % (a, k)
    bx.text(off[0] + 0.3, -1.4, lab, ha="center", fontsize=11, color=col)
bx.set_aspect("equal")
bx.set_xlim(-1.4, 2.6 * 4 + 1.3)
bx.set_ylim(-1.7, 1.2)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")

fig.tight_layout()
dgfig.save(fig, __file__)
