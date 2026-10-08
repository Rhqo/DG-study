"""Figure 31.5.4: a Gaussian bound to a mesh face, before and after the face deforms (Section 31.5).

Rest face (z = 0): x_0 = (0, 0), x_1 = (2, 0), x_2 = (0.6, 1.6). A Gaussian with centre at barycentric coordinates
(0.3, 0.45, 0.25) and in-plane covariance R(25 deg) diag(0.35^2, 0.12^2) R(25 deg)^T (thin normal axis 0.02).
Deformed face: x_0' = (0, 0), x_1' = (2.4, 0.3), x_2' = (1.5, 1.2) (stretch and shear).
Two ways to carry the Gaussian:
  affine (blue):    mu' = barycentric combination of x_i', Sigma' = A Sigma A^T with A the affine map of the face
                    (Proposition 29.1.3);
  similarity (orange, GaussianAvatars [Qian24]): local frame of the face R = [unit(x_1 - x_0), n, cross], origin T =
                    centroid, scalar k = (|x_1 - x_0| + height of x_2 over that edge)/2; the local mean, rotation and
                    scale are kept, mu' = k R mu_loc + T, Sigma' = k^2 R Sigma_loc R^T (paper eqs. (4)-(6) and the
                    official code compute_face_orientation).
Grey lines: lines of constant barycentric coordinate (spacing 0.2), i.e. material lines painted on the face.
Ellipses: 1-sigma level sets in the face plane. Self-check: the affine Gaussian follows the painted lines exactly (its
1-sigma ellipse is the image of the rest ellipse under A); the two agree when A is a similarity.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-5-4-gaussian-on-face.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
X0 = np.array([[0, 0, 0], [2, 0, 0], [0.6, 1.6, 0]], float)
X1 = np.array([[0, 0, 0], [2.4, 0.3, 0], [1.5, 1.2, 0]], float)
BARY = np.array([0.3, 0.45, 0.25])
a = np.radians(25)
R2 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
SIG0 = R2 @ np.diag([0.35 ** 2, 0.12 ** 2, 0.02 ** 2]) @ R2.T


def unit(v):
    return v / np.linalg.norm(v)


def face_frame(X):
    """GaussianAvatars utils/graphics_utils.py compute_face_orientation (one face)."""
    a0 = unit(X[1] - X[0])
    a1 = unit(np.cross(a0, X[2] - X[0]))
    a2 = -unit(np.cross(a1, a0))
    Rf = np.stack([a0, a1, a2], 1)
    s0 = np.linalg.norm(X[1] - X[0])
    s1 = abs(a2 @ (X[2] - X[0]))
    return Rf, (s0 + s1) / 2, X.mean(0)


def affine_of(Xa, Xb):
    """Linear part A with A (x_i - x_0) = x_i' - x_0' and A n = n' (unit normals)."""
    Ea = np.stack([Xa[1] - Xa[0], Xa[2] - Xa[0], unit(np.cross(Xa[1] - Xa[0], Xa[2] - Xa[0]))], 1)
    Eb = np.stack([Xb[1] - Xb[0], Xb[2] - Xb[0], unit(np.cross(Xb[1] - Xb[0], Xb[2] - Xb[0]))], 1)
    return Eb @ np.linalg.inv(Ea)


mu0 = BARY @ X0
A = affine_of(X0, X1)
mu_aff = BARY @ X1
sig_aff = A @ SIG0 @ A.T
assert np.allclose(A @ (mu0 - X0[0]) + X1[0], mu_aff)
R0, k0, T0 = face_frame(X0)
R1, k1, T1 = face_frame(X1)
assert np.isclose(np.linalg.det(R0), 1) and np.isclose(np.linalg.det(R1), 1)
mu_loc = R0.T @ (mu0 - T0) / k0
sig_loc = R0.T @ SIG0 @ R0 / k0 ** 2
mu_sim = k1 * R1 @ mu_loc + T1
sig_sim = k1 ** 2 * R1 @ sig_loc @ R1.T
assert np.allclose(k0 * R0 @ mu_loc + T0, mu0) and np.allclose(k0 ** 2 * R0 @ sig_loc @ R0.T, SIG0)
# a similarity deformation is followed exactly by both
Rs = np.array([[np.cos(0.4), -np.sin(0.4), 0], [np.sin(0.4), np.cos(0.4), 0], [0, 0, 1]]) * 1.3
Xs = X0 @ Rs.T + np.array([0.5, -0.2, 0])
Rss, kss, Tss = face_frame(Xs)
As = affine_of(X0, Xs)
assert np.allclose((kss ** 2 * Rss @ sig_loc @ Rss.T)[:2, :2], (As @ SIG0 @ As.T)[:2, :2])  # in-plane part
assert np.allclose(kss * Rss @ mu_loc + Tss, BARY @ Xs)


def ellipse(mu, S, n=200):
    w, U = np.linalg.eigh(S[:2, :2])
    t = np.linspace(0, 2 * np.pi, n)
    return mu[:2, None] + (U @ (np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)])))


def bary_lines(ax, X):
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for c in np.arange(0.2, 1.0, 0.2):
            p = c * X[i] + (1 - c) * X[j]
            q = c * X[i] + (1 - c) * X[k]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#AAAAAA", lw=0.7, zorder=1)


def stats(S):
    w = np.linalg.eigvalsh(S[:2, :2])
    return np.sqrt(w[1] / w[0])


asp = (stats(SIG0), stats(sig_aff), stats(sig_sim))
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw=dict(wspace=0.05))
for ax, (title, X) in zip(axs, (("(a) rest face", X0), ("(b) deformed face", X1))):
    ax.add_patch(Polygon(X[:, :2], closed=True, fc=C["surface"], ec=C["main"], lw=1.4, alpha=0.5, zorder=0))
    bary_lines(ax, X)
    for i, p in enumerate(X):
        ax.plot(*p[:2], "o", color=C["main"], ms=4)
        ax.text(p[0] + (0.06 if i else -0.22), p[1] + (0.06 if i == 2 else -0.15), rf"$x_{i}{chr(39) if X is X1 else ''}$",
                fontsize=12)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.35, 2.6)
    ax.set_ylim(-0.35, 1.75)
    ax.set_title(title, fontsize=11.5)
E0 = ellipse(mu0, SIG0)
axs[0].fill(*E0, color=C["region"], alpha=0.6, lw=1.5, ec=C["tangent"], zorder=3)
axs[0].plot(*mu0[:2], "o", color=C["tangent"], ms=4, zorder=4)
Ea = ellipse(mu_aff, sig_aff)
Es = ellipse(mu_sim, sig_sim)
axs[1].fill(*Ea, color=C["region"], alpha=0.6, lw=1.5, ec=C["tangent"], zorder=3, label="affine (follows the face)")
axs[1].plot(*Es, color=C["accent"], lw=2.0, ls=(0, (4, 2)), zorder=4, label="similarity (GaussianAvatars)")
axs[1].plot(*mu_aff[:2], "o", color=C["tangent"], ms=4, zorder=5)
axs[1].plot(*mu_sim[:2], "o", color=C["accent"], ms=4, zorder=5)
axs[1].legend(fontsize=9.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.02))
axs[0].text(1.2, -0.3, f"axis ratio {asp[0]:.2f}", ha="center", fontsize=10, color=C["tangent"])
axs[1].text(1.2, -0.3, f"axis ratio: affine {asp[1]:.2f}, similarity {asp[2]:.2f}", ha="center", fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.03)
dgfig.save(fig, __file__)
print("axis ratios", np.round(asp, 3), "centre offset", np.round(np.linalg.norm(mu_aff - mu_sim), 3))
