"""Figure 30.1.4: turning an SDF into a density: VolSDF and NeuS (Section 30.1).

(a) density as a function of the signed distance f (f > 0 outside):
    blue   = VolSDF sigma = (1/beta) Psi_beta(-f), beta = 0.1 (Psi_beta = Laplace CDF),
    orange = NeuS opaque density for a plane hit head-on, rho = s Phi_s(-f), s = 10 (Phi_s = logistic CDF),
    gray dashed = the "naive" choice sigma = phi_s(f) (logistic density, s = 10).
(b) a ray hits the plane head-on at t* = 1 (f(t) = t* - t); rendering weight w(t) = T(t) sigma(t) for the three
    densities, computed by numerical integration of T = exp(-int sigma). Vertical ticks mark the maxima:
    naive t* - ln(golden ratio)/s = t* - 0.048, NeuS exactly t*, VolSDF t* + 0.2693 beta = t* + 0.027.
Self-checks: peak positions against the closed forms; total weights ~ 1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-1-4-sdf-to-density.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
BETA, S = 0.1, 10.0
TSTAR = 1.0
GOLD = (1 + np.sqrt(5)) / 2


def psi(sv):
    return np.where(sv <= 0, 0.5 * np.exp(np.minimum(sv, 0) / BETA), 1 - 0.5 * np.exp(-np.maximum(sv, 0) / BETA))


def sig_volsdf(f):
    return psi(-f) / BETA


def Phi(f):
    return 1 / (1 + np.exp(-S * f))


def rho_neus(f):
    return S * Phi(-f)


def phi_naive(f):
    return S * Phi(f) * (1 - Phi(f))


t = np.linspace(0.0, 2.0, 200001)
dt = t[1] - t[0]
fr = TSTAR - t


def weights(sig):
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * dt)])
    return np.exp(-tau) * sig


W = {"VolSDF": weights(sig_volsdf(fr)), "NeuS": weights(rho_neus(fr)), "naive": weights(phi_naive(fr))}
peaks = {k: t[np.argmax(v)] for k, v in W.items()}
assert abs(peaks["NeuS"] - TSTAR) < 2 * dt
assert abs(peaks["naive"] - (TSTAR - np.log(GOLD) / S)) < 2 * dt
assert abs(peaks["VolSDF"] - (TSTAR - np.log(3 - np.sqrt(5)) * BETA)) < 2 * dt
for k, v in W.items():
    print(f"{k}: peak at t* {peaks[k] - TSTAR:+.4f}, total weight {v.sum() * dt:.4f}")

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.6, 7.0), gridspec_kw=dict(hspace=0.42))

# (a) density vs signed distance
f = np.linspace(-0.5, 0.5, 2001)
axa.axvspan(-0.5, 0.0, color=C["region"], alpha=0.15, lw=0)
axa.axvline(0, color=C["aux"], lw=0.9)
axa.plot(f, sig_volsdf(f), color=C["tangent"], lw=2.2, label=r"VolSDF $\frac{1}{\beta}\Psi_\beta(-f)$")
axa.plot(f, rho_neus(f), color=C["accent"], lw=2.2, label=r"NeuS $s\,\Phi_s(-f)$")
axa.plot(f, phi_naive(f), color=C["aux"], lw=1.8, ls=(0, (4, 3)), label=r"naive $\varphi_s(f)$")
axa.text(-0.25, 11.0, "inside ($f < 0$)", ha="center", fontsize=11.5)
axa.text(0.25, 11.0, "outside ($f > 0$)", ha="center", fontsize=11.5)
axa.set_xlim(-0.5, 0.5)
axa.set_ylim(0, 12.5)
axa.set_xlabel(r"signed distance $f$")
axa.set_ylabel("density")
axa.legend(loc="center right", frameon=False, fontsize=11.5, bbox_to_anchor=(1.0, 0.55))
axa.set_title(r"(a) density from SDF ($\beta = 0.1$, $s = 10$)", fontsize=13.5)

# (b) weights along a head-on ray
cols = {"VolSDF": C["tangent"], "NeuS": C["accent"], "naive": C["aux"]}
styles = {"VolSDF": "-", "NeuS": "-", "naive": (0, (4, 3))}
# same reading as (a): the shaded part is inside the object (f < 0), i.e. behind the surface as seen from the camera
axb.axvspan(TSTAR, 1.4, color=C["region"], alpha=0.15, lw=0)
axb.text(1.3, 4.35, "inside ($f < 0$)", ha="center", fontsize=11.5)
axb.annotate("", xy=(0.95, 4.05), xytext=(0.75, 4.05),
             arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C["main"], mutation_scale=10))
axb.text(0.85, 3.7, "ray from camera", ha="center", fontsize=10.5)
for k in ("naive", "VolSDF", "NeuS"):
    axb.plot(t, W[k], color=cols[k], lw=2.2 if k != "naive" else 1.8, ls=styles[k], label=k)
    axb.plot([peaks[k]] * 2, [0, W[k].max()], color=cols[k], lw=1.0, ls=":")
axb.axvline(TSTAR, color=C["main"], lw=1.0)
axb.text(TSTAR + 0.004, 4.35, r"surface $t^*$", fontsize=11.5)
axb.annotate(r"$-0.048$", xy=(peaks["naive"], 1.15), xytext=(0.70, 0.95), fontsize=11.5, color=C["main"],
             arrowprops=dict(arrowstyle="->", lw=0.8, color=C["aux"]), ha="center")
axb.annotate(r"$+0.027$", xy=(peaks["VolSDF"], 3.0), xytext=(1.24, 3.3), fontsize=11.5, color=C["main"],
             arrowprops=dict(arrowstyle="->", lw=0.8, color=C["tangent"]), ha="center")
axb.set_xlim(0.6, 1.4)
axb.set_ylim(0, 4.8)
axb.set_xlabel(r"ray parameter $t$")
axb.set_ylabel(r"weight $w(t) = T(t)\,\sigma(t)$")
axb.legend(loc="upper left", frameon=False, fontsize=11.5, bbox_to_anchor=(0.0, 0.78))
axb.set_title(r"(b) weights on a ray hitting a plane at $t^* = 1$", fontsize=13.5)

dgfig.save(fig, __file__)
