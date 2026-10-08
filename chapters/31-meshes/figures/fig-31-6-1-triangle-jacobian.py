"""Figure 31.6.1: what the UV map does to one triangle (Section 31.6).

Surface triangle (coordinates in its own plane): x_0 = (0, 0), x_1 = (2, 0), x_2 = (0.6, 1.4).
UV triangle: u_0 = (0.1, 0.1), u_1 = (0.6, 0.35), u_2 = (0.15, 0.9).
The affine map UV -> surface triangle has the 2 x 2 Jacobian J = [x_1 - x_0, x_2 - x_0] [u_1 - u_0, u_2 - u_0]^{-1}
(a 3 x 2 matrix in R^3; here written in the plane of the triangle). Its singular values sigma_1 >= sigma_2 are the
semi-axes of the image of a unit circle.
(a) UV square [0, 1]^2 with an 8 x 8 texel grid, the UV triangle, and a circle of radius eps = 0.06 at its centroid.
(b) The surface triangle with the images of the texel lines inside the triangle and of the circle (an ellipse with
    semi-axes sigma_1 eps, sigma_2 eps along the left singular vectors, drawn as arrows).
Self-check: J maps the UV edges to the surface edges; the ellipse semi-axes are sigma_1 eps, sigma_2 eps; the area ratio
sigma_1 sigma_2 equals (surface area)/(UV area).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-6-1-triangle-jacobian.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
X = np.array([[0.0, 0.0], [2.0, 0.0], [0.6, 1.4]])
U = np.array([[0.1, 0.1], [0.6, 0.35], [0.15, 0.9]])
P = np.stack([X[1] - X[0], X[2] - X[0]], 1)
Q = np.stack([U[1] - U[0], U[2] - U[0]], 1)
J = P @ np.linalg.inv(Q)
Ul, S, Vt = np.linalg.svd(J)
s1, s2 = S
assert np.allclose(J @ Q, P)
area_x = 0.5 * abs(np.linalg.det(P))
area_u = 0.5 * abs(np.linalg.det(Q))
assert np.isclose(s1 * s2, area_x / area_u)


def phi(uv):
    return X[0] + (np.asarray(uv) - U[0]) @ J.T


def inside(uv, tri, eps=1e-12):
    a, b, c = tri
    M = np.stack([b - a, c - a], 1)
    l = np.linalg.solve(M, (np.asarray(uv) - a).T).T
    return (l[..., 0] >= -eps) & (l[..., 1] >= -eps) & (l.sum(-1) <= 1 + eps)


K = 8
eps = 0.06
centres = [U.mean(0)]
t = np.linspace(0, 2 * np.pi, 120)
circ = np.stack([np.cos(t), np.sin(t)], 1)
# the image of a circle is an ellipse with semi-axes s_i eps
E = circ @ (J * eps).T
r = np.linalg.norm(E, axis=1)
assert np.isclose(r.max(), s1 * eps, rtol=1e-3) and np.isclose(r.min(), s2 * eps, rtol=1e-3)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.08))
# (a) UV
for k in range(K + 1):
    axa.plot([k / K, k / K], [0, 1], color="#BBBBBB", lw=0.6, zorder=0)
    axa.plot([0, 1], [k / K, k / K], color="#BBBBBB", lw=0.6, zorder=0)
axa.add_patch(Polygon(U, closed=True, fc=(0.34, 0.71, 0.91, 0.25), ec=C["tangent"], lw=1.6, zorder=1))
for c in centres:
    axa.plot(*(c + eps * circ).T, color=C["normal"], lw=1.4, zorder=3)
for i, (u, off) in enumerate(zip(U, [(-0.07, -0.07), (0.02, -0.06), (-0.05, 0.03)])):
    axa.plot(*u, "o", color=C["tangent"], ms=4, zorder=4)
    axa.text(u[0] + off[0], u[1] + off[1], rf"$u_{i}$", fontsize=12, color=C["tangent"])
axa.set_xlim(-0.02, 1.02)
axa.set_ylim(-0.02, 1.02)
axa.set_aspect("equal")
axa.set_xticks([0, 0.5, 1])
axa.set_yticks([0, 0.5, 1])
axa.tick_params(labelsize=10)
axa.set_title(f"(a) UV square, {K} x {K} texels", fontsize=12)
axa.text(*(U.mean(0) + np.array([0.0, -0.11])), r"$\varepsilon$-circle", fontsize=10.5, ha="center", color=C["normal"])

# (b) surface triangle
axb.add_patch(Polygon(X, closed=True, fc=C["surface"], alpha=0.5, ec=C["main"], lw=1.6, zorder=1))
uu = np.linspace(0, 1, 2001)
for k in range(K + 1):
    for line in (np.stack([np.full_like(uu, k / K), uu], 1), np.stack([uu, np.full_like(uu, k / K)], 1)):
        m = inside(line, U)
        if m.sum() > 1:
            seg = phi(line[m][[0, -1]])
            axb.plot(*seg.T, color="#888888", lw=0.7, zorder=2)
for c in centres:
    axb.plot(*(phi(c) + E).T, color=C["normal"], lw=1.4, zorder=3)
# singular directions at the centroid
c0 = phi(U.mean(0))
for s, vec, lab in ((s1, Ul[:, 0], r"$\sigma_1\varepsilon$"), (s2, Ul[:, 1], r"$\sigma_2\varepsilon$")):
    vec = vec * np.sign(vec[0] if abs(vec[0]) > 1e-9 else 1)
    tip = c0 + s * eps * vec
    axb.annotate("", xy=tip, xytext=c0, arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.2, mutation_scale=10),
                 zorder=5)
    lab_at = c0 + (s * eps + 0.12) * vec
    axb.text(*lab_at, lab, fontsize=12, ha="center", va="center", zorder=6,
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
for i, (x, off) in enumerate(zip(X, [(-0.13, -0.1), (0.03, -0.1), (0.04, 0.03)])):
    axb.plot(*x, "o", color=C["main"], ms=4, zorder=4)
    axb.text(x[0] + off[0], x[1] + off[1], rf"$x_{i}$", fontsize=12)
axb.set_xlim(-0.2, 2.15)
axb.set_ylim(-0.2, 1.55)
axb.set_aspect("equal")
axb.set_axis_off()
axb.set_title(rf"(b) surface triangle: $\sigma_1 = {s1:.2f}$, $\sigma_2 = {s2:.2f}$", fontsize=12)
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.08)
dgfig.save(fig, __file__)
print(f"s1 = {s1:.4f}, s2 = {s2:.4f}, s1/s2 = {s1 / s2:.4f}, s1 s2 = {s1 * s2:.4f}, area ratio = {area_x / area_u:.4f}")
