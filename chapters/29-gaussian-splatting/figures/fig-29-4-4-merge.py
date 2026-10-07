"""Figure 29.4.4: merging two Gaussians: moment matching vs. a Riemannian mean (Section 29.4).

Two flat 2D Gaussians with s = (0.8, 0.15), major axes at +20 and -30 degrees, equal weights 1/2.
(a) Different centers mu_1 = (-1.2, 0), mu_2 = (1.2, 0.3). Gray: the two inputs (1-sigma filled, 2-sigma outline).
    Orange: moment matching (mean and covariance of the mixture), 1 and 2 sigma.
    Blue: mean position with the affine-invariant mean of the two covariances, 1 and 2 sigma.
(b) Same center for both inputs (0, 0): moment matching becomes the Euclidean mean of the covariances.
Self-checks: KL(p || q) = 0.86 (moment) and 4.0 (AI) in (a); dets 0.0144 (inputs, AI) and 0.0703 (moment) in (b).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-4-4-merge.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def R2(d):
    a = np.radians(d)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def fun(S, f):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(f(w)) @ V.T


Sg = [R2(20) @ np.diag([0.64, 0.0225]) @ R2(20).T, R2(-30) @ np.diag([0.64, 0.0225]) @ R2(-30).T]
h = fun(Sg[0], np.sqrt); hi = np.linalg.inv(h)
S_ai = h @ fun(hi @ Sg[1] @ hi, np.sqrt) @ h
t = np.linspace(0, 2 * np.pi, 200)


def ell(mu, S, k):
    w, V = np.linalg.eigh(S)
    return mu + k * (np.stack([np.cos(t), np.sin(t)], 1) * np.sqrt(w)) @ V.T


gx = np.linspace(-6, 6, 601)
GX, GY = np.meshgrid(gx, gx, indexing="ij")
P = np.stack([GX, GY], -1)


def npdf(m_, S_):
    d = P - m_
    return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, np.linalg.inv(S_), d)) / (2 * np.pi * np.sqrt(np.linalg.det(S_)))


fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.05), sharey=True, gridspec_kw=dict(wspace=0.06))
for ax, mus, lab in ((axes[0], [np.array([-1.2, 0.0]), np.array([1.2, 0.3])], "(a) different centers"),
                     (axes[1], [np.zeros(2), np.zeros(2)], "(b) same center")):
    mu = 0.5 * (mus[0] + mus[1])
    S_mm = sum(0.5 * (S_ + np.outer(m_ - mu, m_ - mu)) for m_, S_ in zip(mus, Sg))
    for m_, S_ in zip(mus, Sg):
        ax.fill(*ell(m_, S_, 1).T, color="#999999", alpha=0.45, lw=0)
        ax.plot(*ell(m_, S_, 2).T, color="#777777", lw=0.8)
    for S_, col, ls in ((S_mm, C["accent"], "-"), (S_ai, C["tangent"], "-")):
        ax.plot(*ell(mu, S_, 1).T, color=col, lw=1.9, ls=ls)
        ax.plot(*ell(mu, S_, 2).T, color=col, lw=1.0, ls=(0, (4, 2.5)))
    ax.plot(*mu, "+", color=C["main"], ms=9, mew=1.5)
    if lab.startswith("(a)"):
        pmix = 0.5 * npdf(mus[0], Sg[0]) + 0.5 * npdf(mus[1], Sg[1])
        dA = (gx[1] - gx[0]) ** 2
        KL = lambda q: np.sum(pmix * (np.log(pmix + 1e-300) - np.log(q + 1e-300))) * dA
        kmm, kai = KL(npdf(mu, S_mm)), KL(npdf(mu, S_ai))
        assert abs(kmm - 0.86) < 0.006 and abs(kai - 4.0) < 0.03
        ax.text(-3.0, -1.72, r"$\mathrm{KL}(p\,\|\,q)$:" + "\n" + "moment %.2f,  AI mean %.1f" % (kmm, kai), fontsize=9.5,
                va="bottom")
    else:
        dets = [np.linalg.det(Sg[0]), np.linalg.det(S_mm), np.linalg.det(S_ai)]
        assert np.allclose(np.round(dets, 4), [0.0144, 0.0703, 0.0144])
        ax.text(-3.0, -1.72, r"$\det$:  inputs %.4f" % dets[0] + "\n" + "moment %.4f,  AI mean %.4f" % (dets[1], dets[2]),
                fontsize=9.5, va="bottom")
    ax.set_title(lab, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-1.8, 1.8)
    ax.tick_params(labelsize=9)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Patch(color="#999999", alpha=0.45, label="inputs"),
           Line2D([], [], color=C["accent"], lw=1.9, label="moment matching"),
           Line2D([], [], color=C["tangent"], lw=1.9, label="affine-invariant mean"),
           Line2D([], [], color="#555555", lw=1.0, ls=(0, (4, 2.5)), label=r"$2\sigma$")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

dgfig.save(fig, __file__)
