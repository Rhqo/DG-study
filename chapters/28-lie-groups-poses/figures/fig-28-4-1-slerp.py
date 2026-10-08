"""Figure 28.4.1: SLERP moves at constant angular speed along the geodesic; NLERP and the wrong sign do not
(Section 28.4).

Two rotations R0 = I and R1 = Exp(150 deg * n) (n a fixed unit axis), quaternions q0 = (1, 0, 0, 0) and q1 with
<q0, q1> = cos 75 deg > 0. For 0 <= s <= 1:
  blue   : SLERP(q0, q1; s)                         [Shoemake85]
  black  : NLERP, (1 - s) q0 + s q1 normalized       (same path, different speed)
  orange : SLERP(q0, -q1; s), i.e. without the sign check (same end rotation, the long way round)
(a) the angle travelled along the path, int_0^s |omega(u)| du in degrees, where omega is the angular velocity of
    R(s) (computed by finite differences of Log(R(s)^T R(s + h)));
(b) the angular speed |omega(s)| in degrees per unit s.
Self-check: SLERP equals R0 Exp(s Log(R0^T R1)) (the bi-invariant geodesic of Tour Remark 0.8.4) to 1e-12; its speed is
150 deg everywhere; NLERP has the same end points and passes through the same rotations; the wrong-sign path has
speed 210 deg and ends at R1.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-4-1-slerp.py``
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
n = np.array([0.3, -0.5, 0.81])
n /= np.linalg.norm(n)
TH = np.radians(150)
q0 = np.array([1.0, 0, 0, 0])
q1 = L.expq(TH * n)
R1 = L.q_to_R(q1)
assert q0 @ q1 > 0

ss = np.linspace(0, 1, 401)
paths = {
    "slerp": [L.q_to_R(L.slerp(q0, q1, s)) for s in ss],
    "nlerp": [L.q_to_R(((1 - s) * q0 + s * q1) / np.linalg.norm((1 - s) * q0 + s * q1)) for s in ss],
    "wrong": [L.q_to_R(L.slerp(q0, -q1, s, shortest=False)) for s in ss],
}
for s, Rs in zip(ss[::40], paths["slerp"][::40]):
    assert np.allclose(Rs, L.expSO3(s * L.logSO3(R1)), atol=1e-12)
for k in paths:
    assert np.allclose(paths[k][-1], R1, atol=1e-12)
# NLERP passes through the same rotations (all on the one-parameter subgroup of n)
for Rn in paths["nlerp"][::50]:
    w = L.logSO3(Rn)
    assert np.linalg.norm(np.cross(w, n)) < 1e-12


def speed(Rs):
    ds = ss[1] - ss[0]
    return np.array([np.linalg.norm(L.logSO3(Rs[k].T @ Rs[k + 1])) / ds for k in range(len(Rs) - 1)])


sp = {k: np.degrees(speed(v)) for k, v in paths.items()}
assert np.allclose(sp["slerp"], 150, atol=1e-6)
assert np.allclose(sp["wrong"], 210, atol=1e-6)
mid = 0.5 * (ss[1:] + ss[:-1])
trav = {k: np.concatenate([[0], np.cumsum(v) * (ss[1] - ss[0])]) for k, v in sp.items()}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 3.8), gridspec_kw=dict(wspace=0.3))
sty = [("slerp", C["tangent"], "-", "SLERP"), ("nlerp", C["main"], (0, (5, 2)), "NLERP (normalized lerp)"),
       ("wrong", C["accent"], "-", r"SLERP to $-q_1$ (no sign check)")]
for k, col, ls, lab in sty:
    ax.plot(ss, trav[k], color=col, ls=ls, lw=2.0, label=lab)
    bx.plot(mid, sp[k], color=col, ls=ls, lw=2.0, label=lab)
ax.axhline(150, color=C["aux"], lw=0.7, ls=":")
ax.text(0.02, 156, r"$d(R_0, R_1) = 150^\circ$", fontsize=10.5, color=C["aux"])
ax.set_xlabel(r"$s$")
ax.set_ylabel("angle travelled (deg)")
ax.set_title("(a) distance along the path", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 225)
ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95)
bx.set_xlabel(r"$s$")
bx.set_ylabel(r"angular speed $|\omega(s)|$ (deg per unit $s$)")
bx.set_title("(b) speed", fontsize=12)
bx.set_xlim(0, 1)
bx.set_ylim(80, 230)
bx.annotate(r"max %.0f at $s = 1/2$" % sp["nlerp"].max(), xy=(0.5, sp["nlerp"].max()), xytext=(0.55, 185),
            fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=0.9))
bx.annotate(r"%.0f at the ends" % sp["nlerp"][0], xy=(0.01, sp["nlerp"][0]), xytext=(0.1, 95), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))

print(f"NLERP speed: ends {sp['nlerp'][0]:.1f}, middle {sp['nlerp'].max():.1f} deg")
dgfig.save(fig, __file__)
