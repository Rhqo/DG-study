"""Figure 29.2.3: plain gradient descent from an isotropic start (Section 29.2).

2D Gaussian with parameters (theta, log s_1, log s_2), Sigma = R(theta) diag(s_1^2, s_2^2) R(theta)^T,
loss L = |Sigma - Sigma*|_F^2 / 2, learning rate 0.01, start theta = 0, s = (1 + eps, 1).
Target Sigma*: s* = (2, 0.5) with major axis at 45 degrees (entries 2.125, 1.875 written exactly) for
eps = 0, 1e-8, 1e-4, 1e-1, and at 30 degrees for eps = 0.
(a) theta (degrees) vs iteration; (b) anisotropy s_1/s_2 vs iteration.
Self-checks (same numbers as verify/v-29-2-...): eps = 0 with the 45-degree target never moves theta and stalls
at Sigma = 2.125 I; the other runs reach within 1 degree after 148, 83, 34, 24 iterations.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-2-3-isotropic-start.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def R2(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def run(target, eps, eta=0.01, n=300):
    th, u = 0.0, np.log([1.0 + eps, 1.0])
    out = []
    for _ in range(n):
        lam = np.exp(2 * u)
        Gm = R2(th) @ np.diag(lam) @ R2(th).T - target
        M = R2(th).T @ Gm @ R2(th)
        out.append((np.degrees(th), np.exp(u[0] - u[1]), 0.5 * np.sum(Gm ** 2)))
        th -= eta * 2 * (lam[0] - lam[1]) * M[0, 1]
        u = u - eta * 2 * lam * np.diag(M)
    return np.array(out)


T45 = np.array([[2.125, 1.875], [1.875, 2.125]])
T30 = R2(np.radians(30)) @ np.diag([4.0, 0.25]) @ R2(np.radians(30)).T
runs = [
    (T45, 0.0, 45, C["main"], "-", r"target $45^\circ$, $\varepsilon = 0$"),
    (T45, 1e-8, 45, C["normal"], "-", r"target $45^\circ$, $\varepsilon = 10^{-8}$"),
    (T45, 1e-4, 45, C["accent"], "-", r"target $45^\circ$, $\varepsilon = 10^{-4}$"),
    (T45, 1e-1, 45, C["tangent"], "-", r"target $45^\circ$, $\varepsilon = 10^{-1}$"),
    (T30, 0.0, 30, C["third"], (0, (4, 2)), r"target $30^\circ$, $\varepsilon = 0$"),
]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True, gridspec_kw=dict(wspace=0.3))
reach = []
for T, eps, tgt, col, ls, lab in runs:
    h = run(T, eps)
    if eps == 0 and tgt == 45:
        assert np.all(h[:, 0] == 0) and np.all(h[:, 1] == 1)
        hl = run(T, eps, n=1500)
        assert abs(hl[-1, 2] - 0.5 * 2 * 1.875 ** 2) < 1e-9
    else:
        reach.append(int(np.argmax(np.abs(run(T, eps, n=1500)[:, 0] - tgt) < 1.0)))
    k = np.arange(len(h))
    axa.plot(k, h[:, 0], color=col, ls=ls, lw=1.7, label=lab)
    axb.plot(k, h[:, 1], color=col, ls=ls, lw=1.7)
assert reach == [148, 83, 34, 24], reach
axa.set_ylabel(r"angle $\theta$ (degrees)")
axb.set_ylabel(r"anisotropy $s_1/s_2$")
for ax in (axa, axb):
    ax.set_xlabel("iteration")
    ax.set_xlim(0, 300)
    ax.tick_params(labelsize=9)
axa.set_ylim(-3, 52)
axb.set_ylim(0.9, 4.1)
axa.annotate(r"stuck: $s_1 = s_2$," + "\n" + r"$\partial L/\partial\theta = 0$", xy=(250, 0), xytext=(170, 12), fontsize=9.5,
             arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=0.7))
axa.set_title(r"(a) rotation angle $\theta$", fontsize=11)
axb.set_title(r"(b) anisotropy $s_1/s_2$", fontsize=11)
fig.legend(*axa.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.2))

dgfig.save(fig, __file__)
