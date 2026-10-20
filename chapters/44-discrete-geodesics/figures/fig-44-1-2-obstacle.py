"""Figure 44.1.2: the shortest path in the plane around a disk-shaped obstacle (Section 44.1, Example 44.1.4).

Domain: the plane minus the open unit disk. p = (-2, 0), q = (1, 1). The shortest path from p to q (solid orange)
consists of the tangent segment from p to t1 = (cos 2pi/3, sin 2pi/3) (length sqrt 3), the arc of the unit circle from
t1 to t2 = (0, 1) (central angle pi/6), and the tangent segment from t2 to q (length 1): total sqrt 3 + pi/6 + 1.
The curve of the same kind around the other side (dashed orange; tangent points at angles 4 pi/3 and 0, arc 2 pi/3)
is longer: sqrt 3 + 2 pi/3 + 1.
(b) signed curvature kappa_s (counterclockwise positive, GUIDELINES 6.2) of the shortest path against arc length s.
Self-checks: the segment pq meets the open disk; the tangent points satisfy <t - p, t> = 0; the lengths; the polygon
p - m - q through the intersection m of the two tangent lines (Exercise 44.1.4(a)) avoids the disk and is longer.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-1-2-obstacle.py
"""

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
p, q = np.array([-2.0, 0.0]), np.array([1.0, 1.0])
dpq = q - p
foot = p + (-(p @ dpq) / (dpq @ dpq)) * dpq
assert 0 < -(p @ dpq) / (dpq @ dpq) < 1 and np.linalg.norm(foot) < 1          # the segment pq is blocked
a1, a2 = 2 * np.pi / 3, np.pi / 2                                           # upper tangent points t1, t2
b1, b2 = 4 * np.pi / 3, 2 * np.pi                                           # lower tangent points
t1, t2 = np.array([np.cos(a1), np.sin(a1)]), np.array([np.cos(a2), np.sin(a2)])
s1, s2 = np.array([np.cos(b1), np.sin(b1)]), np.array([np.cos(b2), np.sin(b2)])
for T_, P_ in ((t1, p), (t2, q), (s1, p), (s2, q)):
    assert abs((T_ - P_) @ T_) < 1e-12
seg1, seg2 = np.linalg.norm(t1 - p), np.linalg.norm(t2 - q)
assert abs(seg1 - np.sqrt(3)) < 1e-12 and abs(seg2 - 1) < 1e-12
total = seg1 + (a1 - a2) + seg2
lower = np.linalg.norm(s1 - p) + (b2 - b1) + np.linalg.norm(s2 - q)
assert abs(total - (np.sqrt(3) + np.pi / 6 + 1)) < 1e-12 and lower > total
# Exercise 44.1.4(a): m = intersection of the tangent lines at t1 and t2; p - m - q avoids the open disk, is longer
m = np.array([np.sqrt(3) - 2, 1.0])
assert abs(m @ t1 - 1) < 1e-12 and abs(m @ t2 - 1) < 1e-12
for A_, B_ in ((p, m), (m, q)):
    d_ = B_ - A_
    tt_ = np.clip(-(A_ @ d_) / (d_ @ d_), 0, 1)
    assert np.linalg.norm(A_ + tt_ * d_) >= 1 - 1e-12
poly = np.linalg.norm(m - p) + np.linalg.norm(q - m)
assert abs(poly - (5 - np.sqrt(3))) < 1e-12 and poly > total
print(f"shortest length sqrt(3) + pi/6 + 1 = {total:.4f}, other side {lower:.4f}, polygon p-m-q {poly:.4f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.25, 1.0]})
tt = np.linspace(0, 2 * np.pi, 300)
axa.fill(np.cos(tt), np.sin(tt), color=col["surface"], alpha=0.6, lw=0)
axa.plot(np.cos(tt), np.sin(tt), color=col["main"], lw=0.8)
arc = np.linspace(a1, a2, 60)
up = np.vstack([p, np.stack([np.cos(arc), np.sin(arc)], 1), q])
axa.plot(up[:, 0], up[:, 1], color=col["accent"], lw=2.4)
arc2 = np.linspace(b1, b2, 120)
low = np.vstack([p, np.stack([np.cos(arc2), np.sin(arc2)], 1), q])
axa.plot(low[:, 0], low[:, 1], color=col["accent"], lw=1.6, ls="--")
axa.plot([p[0], q[0]], [p[1], q[1]], color=col["aux"], lw=0.8, ls=":")
for P_, lab, off in ((p, r"$p$", (-0.25, -0.05)), (q, r"$q$", (0.08, 0.02))):
    axa.plot(*P_, "o", color=col["main"], ms=4)
    axa.text(P_[0] + off[0], P_[1] + off[1], lab, fontsize=12)
for T_, lab, off in ((t1, r"$t_1$", (-0.28, 0.06)), (t2, r"$t_2$", (-0.05, 0.1))):
    axa.plot(*T_, "o", color=col["accent"], ms=3.5)
    axa.text(T_[0] + off[0], T_[1] + off[1], lab, fontsize=10)
axa.text(-0.36, -0.12, "obstacle", fontsize=10)
mid = np.array([np.cos((a1 + a2) / 2), np.sin((a1 + a2) / 2)])
axa.annotate("arc of the boundary", xy=tuple(mid), xytext=(-1.75, 1.45), fontsize=10,
             arrowprops=dict(arrowstyle="->", color=col["main"], lw=0.8))
axa.text(0.55, -1.2, "other side: longer", fontsize=10, color=col["accent"])
axa.set_aspect("equal")
axa.set_xlim(-2.4, 2.0)
axa.set_ylim(-1.3, 1.7)
axa.set_axis_off()
axa.set_title("(a) the shortest path from $p$ to $q$", fontsize=11)

s = np.array([0, seg1, seg1, seg1 + (a1 - a2), seg1 + (a1 - a2), total])
k = np.array([0, 0, -1, -1, 0, 0])
axb.plot(s, k, color=col["accent"], lw=2.0)
axb.axhline(0, color=col["aux"], lw=0.6)
axb.set_xlabel("arc length $s$ along the shortest path")
axb.set_ylabel(r"$\kappa_s$")
axb.set_yticks([-1, 0, 1])
axb.set_ylim(-1.4, 1.0)
axb.text(seg1 - 0.6, -1.25, r"arc: $\kappa_s = -1$", fontsize=10)
axb.text(0.1, 0.25, "segments: 0", fontsize=10)
axb.spines[["top", "right"]].set_visible(False)
axb.set_title(r"(b) signed curvature of the path", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.88, bottom=0.18, wspace=0.25)
dgfig.save(fig, __file__)
