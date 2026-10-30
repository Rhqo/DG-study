"""Figure 47.7.4: the approximate mean curvature near a crossing and near a triple junction ([BLM17] Section 8.2,
Fig. 8 and Fig. 9), as profiles along the curve.

(a) Two circles of radius 0.5 centred at (-0.25, 0) and (0.25, 0), 5000 points each at uniform angle (N = 10000),
exact tangent lines, equal masses, natural exp pair, eps = 0.01 (the setting of [BLM17, Fig. 8]). For the points of
the left circle near the upper crossing p = (0, 0.4330): |H^V| and |H^{V,perp}| against the signed arc length s from
p. Exact |H| = 2 on each circle; the average (H_1 + H_2)/2 at p has norm sqrt 3.
(b) Planar standard double bubble with outer radii 0.6 and 1 (middle arc radius 1.5), N = 800 points at equal arc
length, equal masses, tangent lines by PCA in the ball of radius eps, natural exp pair, eps = 0.15 (the setting of
[BLM17, Fig. 9]). For the points within arc length 0.6 of the upper junction p: |H^V| (small markers) and its average
over the ball of radius 2 eps (lines), against the arc length from p, one colour per arc. Dashed: the exact
curvatures 1/0.6, 1, 1/1.5; star: the norm 0.839 of the average of the three curvature vectors at p.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-7-4-singular-profiles.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import var47 as v  # noqa: E402

PAIR = v.pairs(2)["natural exp"]


def two_circles_profile(N=10000, eps=0.01):
    X, T, Hx = v.two_circles(N)
    P = v.proj_from_tangent(T)
    H, Hp, _ = v.amc(X, P, np.ones(N), eps, PAIR, 1)
    p = np.array([0.0, math.sqrt(0.25 - 0.0625)])
    c1 = np.array([-0.25, 0.0])
    left = np.arange(N // 2)
    ang = np.arctan2(X[left, 1] - c1[1], X[left, 0] - c1[0]) - math.pi / 3
    s = 0.5 * ang
    near = np.abs(s) < 4 * eps
    j = left[near]
    o = np.argsort(s[near])
    return dict(s=s[near][o], HV=np.linalg.norm(H[j[o]], axis=1), Hp=np.linalg.norm(Hp[j[o]], axis=1), p=p,
                HV_all=np.linalg.norm(H, axis=1), Hp_all=np.linalg.norm(Hp, axis=1), X=X, H=H, Hp_vec=Hp)


def double_bubble(N=800, eps=0.15, tangents="pca"):
    db = v.double_bubble_2d(0.6, 1.0)
    X, T, Hx, k = v.sample_arcs(db["arcs"], N)
    P = v.proj_from_tangent(T) if tangents == "exact" else v.pca_planes(X, eps, 1)
    H, _, _ = v.amc(X, P, np.ones(N), eps, PAIR, 1)
    A = v.average_ball(X, H, 2 * eps)
    p = db["p"]
    Hm = v.junction_mean_curvature(db["centres"], db["radii"], p)
    j = int(np.argmin(np.linalg.norm(X - p, axis=1)))
    s = np.zeros(N)
    for i, (c, r) in enumerate(zip(db["centres"], db["radii"])):
        sel = k == i
        a = np.arctan2(X[sel, 1] - c[1], X[sel, 0] - c[0])
        ap = math.atan2(p[1] - c[1], p[0] - c[0])
        s[sel] = r * np.abs(np.angle(np.exp(1j * (a - ap))))
    return dict(X=X, k=k, H=H, A=A, s=s, p=p, Hm=Hm, j=j, radii=db["radii"],
                err_avg=float(np.linalg.norm(A[j] - Hm) / np.linalg.norm(Hm)),
                err_raw=float(np.linalg.norm(H[j] - Hm) / np.linalg.norm(Hm)))


TC = two_circles_profile()
DB = double_bubble()
DB2 = double_bubble(1600, 0.075)

# checks quoted in Section 47.7
assert abs(TC["HV_all"].min() - 0.926446) < 1e-5 and abs(TC["Hp_all"].min() - 0.917005) < 1e-5   # paper 0.926446, 0.917004
assert abs(TC["HV_all"].max() - 1.99991) < 1e-5
k0 = int(np.argmin(np.abs(TC["s"])))
assert abs(TC["HV"][k0] - 1.7309) < 1e-3 and abs(TC["s"][k0]) < 3.2e-4                            # sqrt 3 = 1.7321
assert np.allclose(DB["Hm"], [0.0, -0.838870], atol=1e-6)
assert np.allclose(DB["A"][DB["j"]], [0.106, -0.796], atol=1e-3)                                   # paper (0.107, -0.809)
assert abs(DB["err_avg"] - 0.137) < 2e-3 and abs(DB2["err_avg"] - 0.068) < 2e-3                   # paper 13 %, 7 %
assert DB["err_raw"] < 0.06 and DB2["err_raw"] < 0.03

if __name__ == "__main__":
    import dgfig

    dgfig.setup()
    import matplotlib.pyplot as plt

    C = dgfig.COLORS
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.axhline(2.0, color=C["aux"], lw=0.8, ls="--")
    ax.axhline(math.sqrt(3), color=C["aux"], lw=0.8, ls=":")
    ax.axvspan(-0.01, 0.01, color=C["region"], alpha=0.2, lw=0)
    ax.plot(TC["s"], TC["HV"], color=C["normal"], lw=1.3, label=r"$|\vec H^V|$")
    ax.plot(TC["s"], TC["Hp"], color=C["tangent"], lw=1.3, label=r"$|\vec H^{V,\perp}|$")
    ax.text(0.03, math.sqrt(3) + 0.02, r"$\sqrt{3}$", fontsize=10, color=C["aux"], ha="center", va="bottom")
    ax.set_xlim(-0.04, 0.04)
    ax.set_ylim(0.85, 2.1)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_xlabel(r"arc length $s$ from the crossing")
    ax.set_title("(a) two circles", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    cols = (C["normal"], C["tangent"], C["third"])
    names = (r"$r = 0.6$", r"$r = 1$", r"$r = 1.5$")
    for i in range(3):
        sel = (DB["k"] == i) & (DB["s"] < 0.6) & (DB["X"][:, 1] > 0)
        o = np.argsort(DB["s"][sel])
        s = DB["s"][sel][o]
        ax.plot(s, np.linalg.norm(DB["H"][sel], axis=1)[o], ".", color=cols[i], ms=3)
        ax.plot(s, np.linalg.norm(DB["A"][sel], axis=1)[o], "-", color=cols[i], lw=1.3, label=names[i])
        ax.axhline(1.0 / DB["radii"][i], color=cols[i], lw=0.7, ls="--")
    ax.plot([0], [np.linalg.norm(DB["Hm"])], "*", color=C["main"], ms=9)
    ax.axvspan(0, 0.15, color=C["region"], alpha=0.2, lw=0)
    ax.set_xlim(-0.01, 0.6)
    ax.set_ylim(0.0, 1.85)
    ax.set_xlabel(r"arc length from the junction")
    ax.set_title("(b) double bubble", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    dgfig.save(fig, __file__)
    print("two circles: min |H^V| %.6f  min |H^perp| %.6f  max |H^V| %.6f  nearest to p: s = %.2e |H^V| = %.4f "
          "|H^perp| = %.4f" % (TC["HV_all"].min(), TC["Hp_all"].min(), TC["HV_all"].max(), TC["s"][k0],
                               TC["HV"][k0], TC["Hp"][k0]))
    for nm, D in (("N=800", DB), ("N=1600", DB2)):
        print("double bubble %s: nearest point at %.4f from p; H^V %s; averaged %s; mean %s; rel err raw %.3f avg %.3f"
              % (nm, np.linalg.norm(D["X"][D["j"]] - D["p"]), np.round(D["H"][D["j"]], 4),
                 np.round(D["A"][D["j"]], 4), np.round(D["Hm"], 4), D["err_raw"], D["err_avg"]))
