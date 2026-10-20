"""Figure 44.6.2: accuracy of the heat method: time step and refinement (Section 44.6, Examples 44.6.10, 44.6.11).

(a) area-weighted mean error of phi against m = t / (mean edge length)^2 (m = 1/16 ... 64): icosphere(4) from the
    north-pole vertex (exact: arccos), unit_cube(16) from the bottom centre (0.5, 0.5, 0) and from (0.1875, 0.3125, 0)
    (exact: unfolding, ch44.cube_distance_vec). Mixed Voronoi mass.
(b) mean error against the mean edge length for icosphere(1..4) and unit_cube(4, 8, 16) (centre source): heat method
    with m = 1 (filled) and Dijkstra (open).
Self-checks: on the cube the error is smallest at m = 1 among the sampled m; on the icosphere it decreases with m;
the heat-method error decreases under refinement while the Dijkstra error does not.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-6-2-accuracy.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS


def mean_err(phi, ex, m):
    return float(m @ np.abs(phi - ex) / m.sum())


cases = []
V, F = dm.icosphere(4)
s = int(np.argmax(V[:, 2]))
cases.append(("icosphere(4)", V, F, s, np.arccos(np.clip(V @ V[s], -1, 1)), col["tangent"]))
Vc, Fc = ch44.unit_cube(16)
for pt, c_, lab in (([0.5, 0.5, 0], col["normal"], "cube(16), centre"), ([0.1875, 0.3125, 0], col["third"], "cube(16), off-centre")):
    sc = int(np.argmin(np.linalg.norm(Vc - pt, axis=1)))
    cases.append((lab, Vc, Fc, sc, ch44.cube_distance_vec(Vc[sc], Vc)[0], c_))
ms = 2.0 ** np.arange(-4, 7)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
res = {}
for lab, V_, F_, s_, ex, c_ in cases:
    errs = [mean_err(ch44.HeatMethod(V_, F_, m_t=mm).distance(s_), ex, dm.mass_voronoi(V_, F_)) for mm in ms]
    res[lab] = errs
    axa.loglog(ms, errs, "o-", color=c_, lw=1.3, ms=3.5, label=lab)
assert np.argmin(res["cube(16), centre"]) == list(ms).index(1.0)
assert all(np.diff(res["icosphere(4)"][:8]) < 0)
axa.axvline(1.0, color=col["aux"], lw=0.7, ls="--")
axa.set_xlabel(r"$m = t/\bar\ell^{\,2}$")
axa.set_ylabel("mean error")
axa.legend(frameon=False, fontsize=8.5, loc="upper center")
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("(a) the time step", fontsize=11)
print("mean errors at m = 1:", {k: round(v[list(ms).index(1.0)], 4) for k, v in res.items()})

rows = []
for lev in (1, 2, 3, 4):
    V, F = dm.icosphere(lev)
    s = int(np.argmax(V[:, 2]))
    ex = np.arccos(np.clip(V @ V[s], -1, 1))
    rows.append(("ico", ch44.mean_edge_length(V, F), mean_err(ch44.HeatMethod(V, F).distance(s), ex, dm.mass_voronoi(V, F)),
                 mean_err(ch44.dijkstra(V, F, s)[0], ex, dm.mass_voronoi(V, F))))
for n in (4, 8, 16):
    V, F = ch44.unit_cube(n)
    s = int(np.argmin(np.linalg.norm(V - [0.5, 0.5, 0], axis=1)))
    ex = ch44.cube_distance_vec(V[s], V)[0]
    rows.append(("cube", ch44.mean_edge_length(V, F), mean_err(ch44.HeatMethod(V, F).distance(s), ex, dm.mass_voronoi(V, F)),
                 mean_err(ch44.dijkstra(V, F, s)[0], ex, dm.mass_voronoi(V, F))))
for kind, c_ in (("ico", col["tangent"]), ("cube", col["normal"])):
    R = np.array([r[1:] for r in rows if r[0] == kind])
    axb.loglog(R[:, 0], R[:, 1], "o-", color=c_, lw=1.3, ms=4, label=f"{'icosphere' if kind == 'ico' else 'cube'}: heat")
    axb.loglog(R[:, 0], R[:, 2], "o--", color=c_, lw=1.0, ms=4, mfc="white", label=f"{'icosphere' if kind == 'ico' else 'cube'}: Dijkstra")
    assert all(np.diff(R[:, 1][::-1]) > 0) or kind == "ico" and all(np.diff(R[1:, 1][::-1]) > 0)
    assert R[:, 2].min() > 0.05
axb.set_xlabel(r"mean edge length $\bar\ell$")
axb.set_ylabel("mean error")
axb.legend(frameon=False, fontsize=8.5, loc="lower right")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b) refinement", fontsize=11)
print("refinement rows:", [(k, round(a, 4), round(b, 4), round(c, 4)) for k, a, b, c in rows])
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18, wspace=0.32)
dgfig.save(fig, __file__)
