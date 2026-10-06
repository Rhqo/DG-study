"""Figure 0.6.4: SO(3) seen through the exponential map: the ball of rotation vectors |ω| ≤ π (Section 0.6).

Cross-section of the ball {ω ∈ R³ : |ω| ≤ π} by the plane ω² = 0 (horizontal axis ω¹, vertical axis ω³).
A rotation vector ω (here ω = 2·ω̂ with ω̂ = (cos 35°, 0, sin 35°)) is sent by exp to the rotation about the axis ω̂
by the angle |ω| (Rodrigues formula). The origin goes to I_3. The grey dashed circle is |ω| = π/2.
The two boundary points ±π ω̂ give the same rotation (rotation by π), so the orange segment from −π ω̂ to π ω̂,
i.e. t ↦ exp(t[ω̂]_×) for −π ≤ t ≤ π, is a closed loop in SO(3).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-6-4-so3-ball.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w)
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def expm_series(A, terms=40):
    out, term = np.eye(3), np.eye(3)
    for k in range(1, terms):
        term = term @ A / k
        out = out + term
    return out


ang = np.deg2rad(35)
wh = np.array([np.cos(ang), 0.0, np.sin(ang)])
w = 2.0 * wh

# self-checks (§12.1)
R = rodrigues(w)
assert np.allclose(R, expm_series(hat(w)))
assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
assert np.allclose(R @ wh, wh)                                    # the axis is fixed
assert np.isclose(np.arccos((np.trace(R) - 1) / 2), 2.0)          # the angle is |ω|
assert np.allclose(rodrigues(np.pi * wh), rodrigues(-np.pi * wh))  # antipodal boundary points agree
assert np.allclose(rodrigues(2 * np.pi * wh), np.eye(3))           # the loop closes
rng = np.random.default_rng(1)
samples = []
while len(samples) < 60:
    v = rng.uniform(-np.pi, np.pi, 3)
    if np.linalg.norm(v) < 0.98 * np.pi:
        samples.append(v)
Rs = [rodrigues(v) for v in samples]
for i in range(len(Rs)):
    for j in range(i):
        assert np.linalg.norm(Rs[i] - Rs[j]) > 1e-6   # exp is injective on the open ball (samples)

fig, ax = plt.subplots(figsize=(4.6, 4.3))
dgfig.schematic_axes(ax, (-4.1, 4.1), (-3.9, 3.9))
t = np.linspace(0, 2 * np.pi, 400)
ax.fill(np.pi * np.cos(t), np.pi * np.sin(t), color=C["region"], alpha=0.22, lw=0)
ax.plot(np.pi * np.cos(t), np.pi * np.sin(t), color=C["main"], lw=LW["main"])
ax.plot(np.pi / 2 * np.cos(t), np.pi / 2 * np.sin(t), color=C["aux"], lw=0.8, ls=(0, (4, 3)))
for d in ((1, 0), (0, 1)):
    ax.annotate("", xy=(3.9 * d[0], 3.9 * d[1]), xytext=(-3.9 * d[0], -3.9 * d[1]),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(3.9, -0.3, r"$\omega^1$", fontsize=12, ha="right", va="top")
ax.text(0.15, 3.85, r"$\omega^3$", fontsize=12, va="top")

e = np.array([wh[0], wh[2]])
ax.plot([-np.pi * e[0], np.pi * e[0]], [-np.pi * e[1], np.pi * e[1]], color=C["accent"], lw=1.5, zorder=4)
ax.annotate("", xy=2.0 * e, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=2.2, mutation_scale=14), zorder=6)
for s in (1, -1):
    ax.plot(*(s * np.pi * e), "s", ms=6, color=C["accent"], mec=C["main"], zorder=7)
ax.plot(0, 0, "o", ms=4.5, color=C["main"], zorder=7)

box = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85)
ax.text(2.0 * e[0] + 0.05, 2.0 * e[1] - 0.5, r"$\omega$", fontsize=13, color=C["tangent"], bbox=box, zorder=8)
ax.text(np.pi * e[0] + 0.12, np.pi * e[1] + 0.12, r"$\pi\hat\omega$", fontsize=12, bbox=box, zorder=8)
ax.text(-np.pi * e[0] - 0.12, -np.pi * e[1] - 0.15, r"$-\pi\hat\omega$", fontsize=12, ha="right", va="top",
        bbox=box, zorder=8)
ax.text(0.15, -0.5, r"$\exp(0) = I_3$", fontsize=11, bbox=box, zorder=8)
ax.text(-2.75, 2.3, r"$|\omega| \leq \pi$", fontsize=12, ha="center")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
