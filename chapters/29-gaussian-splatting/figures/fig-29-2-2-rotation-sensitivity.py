"""Figure 29.2.2: a rotation changes Sigma in proportion to s_i^2 - s_j^2 (Section 29.2).

(a) Three 2D Gaussians with s = (3, 1), (1.5, 1), (1.1, 1) (1-sigma ellipses, gray), each rotated by 15 degrees (blue).
    The number under each pair is the relative change |Delta Sigma|_F / |Sigma|_F = sqrt2 |s1^2 - s2^2| sin 15 / |Sigma|_F.
(b) The six singular values of dPhi at (R, s) for s = (1, s_2, 0.25), as functions of s_2 (3D), computed numerically
    from the 6 x 6 Jacobian of (omega, log s) -> Sigma (Frobenius-orthonormal basis of Sym(3)).
    Blue: the three scale directions, 2 lambda_i. Orange/green/purple (dash-dot): the three rotation directions,
    sqrt2 |lambda_i - lambda_j|; they vanish at s_2 = 0.25 (= s_3) and s_2 = 1 (= s_1), where rank dPhi = 5.
Self-check: numerical singular values agree with the closed form to 1e-6.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-2-2-rotation-sensitivity.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def R2(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expso3(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-14:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


BASIS = []
for i in range(3):
    E = np.zeros((3, 3)); E[i, i] = 1; BASIS.append(E)
for i, j in ((0, 1), (0, 2), (1, 2)):
    E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1 / np.sqrt(2); BASIS.append(E)


def jac(R, s, h=1e-6):
    def f(p):
        S = (R @ expso3(p[:3])) @ np.diag((s * np.exp(p[3:])) ** 2) @ (R @ expso3(p[:3])).T
        return np.array([np.sum(S * E) for E in BASIS])
    Jm = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6); e[k] = h
        Jm[:, k] = (f(e) - f(-e)) / (2 * h)
    return Jm


rng = np.random.default_rng(5)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
if np.linalg.det(Q) < 0:
    Q[:, 0] *= -1

fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 5.6), gridspec_kw=dict(height_ratios=[0.55, 1], hspace=0.25))

# (a)
t = np.linspace(0, 2 * np.pi, 300)
circ = np.stack([np.cos(t), np.sin(t)], 1)
for k, s1 in enumerate((3.0, 1.5, 1.1)):
    cx = 6.6 * k
    s = np.array([s1, 1.0])
    E0 = circ * s
    E1 = E0 @ R2(15).T
    axa.fill(*(E0 + [cx, 0]).T, color="#BBBBBB", alpha=0.35, lw=0)
    axa.plot(*(E0 + [cx, 0]).T, color=C["aux"], lw=1.2)
    axa.plot(*(E1 + [cx, 0]).T, color=C["tangent"], lw=1.6)
    lam = s ** 2
    rel = np.sqrt(2) * abs(lam[0] - lam[1]) * np.sin(np.radians(15)) / np.linalg.norm(lam)
    dS = R2(15) @ np.diag(lam) @ R2(15).T - np.diag(lam)
    assert abs(np.linalg.norm(dS) / np.linalg.norm(lam) - rel) < 1e-12
    axa.text(cx, -1.75, r"$s_1/s_2 = %.1f$" % s1, ha="center", fontsize=11.5)
    axa.text(cx, -2.45, r"$\|\Delta\Sigma\| / \|\Sigma\| = %.3f$" % rel, ha="center", fontsize=11, color=C["tangent"])
axa.text(-3.2, 1.75, "gray: before", color=C["aux"], fontsize=10.5)
axa.text(-3.2, 1.3, r"blue: rotated by $15^\circ$", color=C["tangent"], fontsize=10.5)
dgfig.schematic_axes(axa, (-3.3, 14.6), (-2.8, 2.1))
axa.set_title(r"(a) rotate by $15^\circ$", fontsize=11)

# (b)
s2s = np.linspace(0.1, 1.3, 121)
svs = []
for s2 in s2s:
    s = np.array([1.0, s2, 0.25])
    sv = np.sort(np.linalg.svd(jac(Q, s), compute_uv=False))
    lam = s ** 2
    closed = np.sort([2 * lam[0], 2 * lam[1], 2 * lam[2], np.sqrt(2) * abs(lam[0] - lam[1]),
                      np.sqrt(2) * abs(lam[0] - lam[2]), np.sqrt(2) * abs(lam[1] - lam[2])])
    assert np.allclose(sv, closed, atol=1e-6)
    svs.append(lam)
lam = np.array(svs)
axb.plot(s2s, 2 * lam[:, 0], color=C["tangent"], lw=1.4)
axb.plot(s2s, 2 * lam[:, 1], color=C["tangent"], lw=1.4)
axb.plot(s2s, 2 * lam[:, 2], color=C["tangent"], lw=1.4, label=r"scale: $2\lambda_i$")
axb.plot(s2s, np.sqrt(2) * abs(lam[:, 0] - lam[:, 1]), color=C["accent"], lw=2.0,
         label=r"about $r_3$: $\sqrt{2}|\lambda_1-\lambda_2|$")
# reddish purple + dash-dot so that it is not confused with the orange curve (about r_3)
axb.plot(s2s, np.sqrt(2) * abs(lam[:, 1] - lam[:, 2]), color=C["covector"], lw=2.0, ls=(0, (6, 2, 1.5, 2)),
         label=r"about $r_1$: $\sqrt{2}|\lambda_2-\lambda_3|$")
axb.plot(s2s, np.sqrt(2) * abs(lam[:, 0] - lam[:, 2]), color=C["third"], lw=2.0, ls=(0, (4, 2)),
         label=r"about $r_2$: $\sqrt{2}|\lambda_1-\lambda_3|$")
for x0, lab in ((0.25, r"$s_2 = s_3$"), (1.0, r"$s_2 = s_1$")):
    axb.axvline(x0, color=C["aux"], lw=0.8, ls=(0, (3, 3)))
    axb.plot(x0, 0, "o", color=C["main"], ms=5, zorder=5)
    axb.annotate(lab + "\nrank 5", xy=(x0, 0), xytext=(x0 + 0.04, 0.85), fontsize=9.5, ha="left",
                 arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
axb.set_xlim(0.1, 1.3)
axb.set_ylim(-0.05, 3.5)
axb.set_xlabel(r"$s_2$  ($s_1 = 1$, $s_3 = 0.25$)")
axb.set_ylabel(r"singular value of $d\Phi$")
axb.legend(fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, handlelength=2.2)
axb.set_title(r"(b) singular values of $d\Phi$", fontsize=11)
axb.tick_params(labelsize=9)

dgfig.save(fig, __file__)
