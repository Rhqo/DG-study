"""Figure 28.2.3: adding to a rotation vector is not composing rotations; the right Jacobian J_r measures the difference
(Section 28.2).

(a) For omega = theta * n (n fixed, theta = 0.5, 1.5, 2.5) and small delta = eps * m (m a fixed unit vector,
    1e-6 <= eps <= 0.3), the error of two first-order approximations of Log(Exp(omega) Exp(delta)):
      naive      : omega + delta                      (solid)
      BCH, first : omega + J_r(omega)^{-1} delta       (dashed)
    The naive error grows like eps (slope 1); the corrected error like eps^2 (slope 2).
(b) The singular values of J_r(theta n) as functions of theta in [0, 2 pi]: 1 (along the axis, gray) and
    2 |sin(theta/2)| / theta (twice, on the plane perpendicular to the axis, blue). They vanish at theta = 2 pi,
    where Exp stops being a local diffeomorphism.
Self-check: fitted slopes ~1 and ~2; singular values from SVD equal the closed form; J_r agrees with a central
finite difference of Log(Exp(omega)^T Exp(omega + h e_k)) / h.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-2-3-exp-jacobian.py``
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
n = np.array([0.36, 0.48, 0.8])
n /= np.linalg.norm(n)
m = np.array([0.9, -0.3, 0.1])
m -= 0.3 * (m @ n) * n
m /= np.linalg.norm(m)
eps = np.logspace(-6, np.log10(0.3), 40)
thetas = [0.5, 1.5, 2.5]
cols = [C["third"], C["tangent"], C["accent"]]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.8), gridspec_kw=dict(wspace=0.3))
for th, col in zip(thetas, cols):
    w = th * n
    true = np.array([L.logSO3(L.expSO3(w) @ L.expSO3(e * m)) for e in eps])
    naive = np.linalg.norm(true - (w + eps[:, None] * m), axis=1)
    corr = np.linalg.norm(true - (w + (L.Jr_inv(w) @ (eps[:, None] * m).T).T), axis=1)
    ax.loglog(eps, naive, color=col, lw=1.9, label=r"$\theta = %.1f$" % th)
    ax.loglog(eps, np.maximum(corr, 1e-17), color=col, lw=1.6, ls=(0, (4, 2)))
    sel = (eps > 1e-4) & (eps < 1e-2)
    s1 = np.polyfit(np.log(eps[sel]), np.log(naive[sel]), 1)[0]
    s2 = np.polyfit(np.log(eps[sel]), np.log(corr[sel]), 1)[0]
    assert abs(s1 - 1) < 0.02 and abs(s2 - 2) < 0.05, (s1, s2)
    # finite-difference check of J_r
    h = 1e-6
    J = np.column_stack([(L.logSO3(L.expSO3(w).T @ L.expSO3(w + h * e))
                          - L.logSO3(L.expSO3(w).T @ L.expSO3(w - h * e))) / (2 * h) for e in np.eye(3)])
    assert np.allclose(J, L.Jr(w), atol=1e-8)
ax.plot([], [], color=C["aux"], lw=1.9, label=r"naive: $\omega + \delta$")
ax.plot([], [], color=C["aux"], lw=1.6, ls=(0, (4, 2)), label=r"$\omega + J_r^{-1}\delta$")
ax.text(2e-5, 3e-3, "slope 1", fontsize=10.5)
ax.text(3e-5, 1e-13, "slope 2", fontsize=10.5)
ax.set_xlabel(r"$|\delta|$")
ax.set_ylabel(r"error vs. $\mathrm{Log}(\mathrm{Exp}(\omega)\,\mathrm{Exp}(\delta))$")
ax.set_title("(a) first-order approximations", fontsize=12)
ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95)
ax.set_ylim(1e-17, 1)
ax.grid(True, which="major", color="#E5E5E5", lw=0.5)

ths = np.linspace(1e-3, 2 * np.pi, 400)
sv_perp = 2 * np.abs(np.sin(ths / 2)) / ths
for th in [0.3, 1.0, np.pi, 5.0]:
    sv = np.linalg.svd(L.Jr(th * n), compute_uv=False)
    assert np.allclose(np.sort(sv), np.sort([1.0, 2 * abs(np.sin(th / 2)) / th, 2 * abs(np.sin(th / 2)) / th]))
bx.plot(ths, np.ones_like(ths), color=C["aux"], lw=1.6, label="along the axis")
bx.plot(ths, sv_perp, color=C["tangent"], lw=2.0, label=r"$\perp$ axis: $2|\sin(\theta/2)|/\theta$ (twice)")
bx.axvline(np.pi, color=C["aux"], lw=0.8, ls=":")
bx.plot(np.pi, 2 / np.pi, "o", color=C["tangent"], ms=5)
bx.annotate(r"$2/\pi \approx 0.64$ at $\theta = \pi$", xy=(np.pi, 2 / np.pi), xytext=(3.5, 0.85), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))
bx.annotate(r"$0$ at $\theta = 2\pi$", xy=(2 * np.pi, 0), xytext=(3.3, 0.12), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))
bx.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
bx.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
bx.set_xlim(0, 2 * np.pi)
bx.set_ylim(-0.05, 1.45)
bx.set_xlabel(r"$\theta = |\omega|$")
bx.set_ylabel(r"singular value of $J_r(\omega)$")
bx.set_title(r"(b) how far $J_r$ is from $I_3$", fontsize=12)
bx.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.95)

dgfig.save(fig, __file__)
