"""Figure 37.5.2: discrete curvature, torsion, and Bishop holonomy converge at second order.

(a) the helix (a, b) = (1, 0.3) sampled with n vertices per turn: errors |theta/lbar - kappa| and |psi/l - tau| from the
closed forms of Example 37.5.6; (b) the trefoil torus knot sampled with n vertices: |holonomy_n - 3.0270| where 3.0270
is -int tau ds mod 2 pi (Example 37.3.14).

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-5-2-convergence.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
kap, tau = ddg37.helix_kappa_tau()
ns = np.array([8, 16, 32, 64, 128, 256])
ek, et = [], []
for n in ns:
    ell, th, psi = ddg37.helix_discrete_angles(2 * np.pi / n)
    ek.append(abs(th / ell - kap))
    et.append(abs(psi / ell - tau))
    X = ddg37.helix_point(2 * np.pi * np.arange(-2, 5) / n)
    k_i, t_i = dd.space_curve_curvature_torsion(X, False)
    assert abs(k_i[3] - th / ell) < 1e-9 and abs(t_i[3] - psi / ell) < 1e-9
ek, et = np.array(ek), np.array(et)
sk = np.polyfit(np.log(ns), np.log(ek), 1)[0]
st = np.polyfit(np.log(ns), np.log(et), 1)[0]
assert abs(sk + 2) < 0.02 and abs(st + 2) < 0.05


def trefoil(n):
    t = 2 * np.pi * np.arange(n) / n
    return np.stack([(2 + np.cos(3 * t)) * np.cos(2 * t), (2 + np.cos(3 * t)) * np.sin(2 * t), np.sin(3 * t)], 1)


HOL = 3.026992185112181                 # -int tau ds mod 2 pi (Example 37.3.14, verify/v-37-3)
nt = np.array([60, 120, 240, 480, 960])
eh = []
for n in nt:
    X = trefoil(n)
    h = dd.bishop_holonomy(X)
    assert abs(np.angle(np.exp(1j * (h + dd.torsion_angles(X, True).sum())))) < 1e-9   # = -sum psi (mod 2 pi)
    eh.append(abs(h - HOL))
eh = np.array(eh)
sh = np.polyfit(np.log(nt), np.log(eh), 1)[0]
assert abs(sh + 2) < 0.1

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0))
ax.loglog(ns, ek, "o-", color=C["normal"], lw=1.4, ms=4, label=r"$|\theta_i/\bar\ell_i - \kappa|$")
ax.loglog(ns, et, "s--", color=C["third"], lw=1.4, ms=4, label=r"$|\psi_i/\ell_i - \tau|$")
ax.loglog(ns, 1.1 * ek[0] * (ns / ns[0]) ** -2.0, color=C["aux"], lw=0.8, ls=":", label="slope $-2$")
ax.set_xlabel(r"vertices per turn $n$")
ax.legend(fontsize=9.5, frameon=False)
ax.set_title("(a)", fontsize=12, loc="left")
bx.loglog(nt, eh, "D-", color=C["accent"], lw=1.4, ms=4, label=r"$|\Delta_n - 3.0270|$")
bx.loglog(nt, 1.1 * eh[0] * (nt / nt[0]) ** -2.0, color=C["aux"], lw=0.8, ls=":", label="slope $-2$")
bx.set_xlabel(r"vertices $n$")
bx.legend(fontsize=9.5, frameon=False)
bx.set_title("(b)", fontsize=12, loc="left")
for a_ in (ax, bx):
    for sp2 in ("top", "right"):
        a_.spines[sp2].set_visible(False)
fig.tight_layout()
dgfig.save(fig, __file__)
print(f"orders: kappa {-sk:.2f}, tau {-st:.2f}, holonomy {-sh:.2f}; holonomy errors {eh}")
