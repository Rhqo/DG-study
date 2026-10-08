"""Verification for Section 31.6 (Parametrization and Texture Mapping).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-6-parametrization-texture.py``
"""

import pathlib
import runpy
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from dgcheck import check, close, summary, sym_equal  # noqa: E402

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Definition 31.6.1, Proposition 31.6.2
ns1 = runpy.run_path(str(FIG / "fig-31-6-1-triangle-jacobian.py"))
J, s1, s2 = ns1["J"], ns1["s1"], ns1["s2"]
close("Def 31.6.1: J maps the UV edges onto the surface edges", J @ ns1["Q"], ns1["P"], tol=1e-12)
check(f"Fig 31.6.1: sigma_1 = {s1:.2f}, sigma_2 = {s2:.2f}, ratio {s1 / s2:.2f}, product {s1 * s2:.2f} = area ratio",
      (round(s1, 2), round(s2, 2), round(s1 / s2, 2), round(s1 * s2, 2)) == (3.85, 1.88, 2.05, 7.23))
G = J.T @ J
close("Def 31.6.1: the eigenvalues of the first fundamental form J^T J are sigma_i^2",
      np.sort(np.linalg.eigvalsh(G)), np.sort([s2 ** 2, s1 ** 2]), tol=1e-10)
check(f"Text: an 8 x 8 texture gives {8 / s1:.2f} and {8 / s2:.2f} texels per unit surface length along the two "
      "singular directions", (round(8 / s1, 2), round(8 / s2, 2)) == (2.08, 4.26))
R = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
for name, M, want in (("isometric", R, (1, 1)), ("conformal", 2.5 * R, (2.5, 2.5)),
                      ("equiareal", R @ np.diag([2.0, 0.5]), (2, 0.5))):
    check(f"Prop 31.6.2: a {name} example has singular values {want}",
          np.allclose(np.linalg.svd(M, compute_uv=False), want))

# ---------------------------------------------------------------- Proposition 31.6.3: E_C = E_D - det = (1/2)(t1 - t2)^2
ux, uy, vx, vy = sp.symbols("u_x u_y v_x v_y", real=True)
EC = sp.Rational(1, 2) * ((ux - vy) ** 2 + (uy + vx) ** 2)
ED = sp.Rational(1, 2) * (ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2)
sym_equal("Prop 31.6.3: E_C = E_D - det (pointwise)", EC, ED - (ux * vy - uy * vx))
ok = True
for _ in range(200):
    M = rng.normal(size=(2, 2))
    if np.linalg.det(M) < 0:
        M[:, 0] *= -1
    t1, t2 = np.linalg.svd(M, compute_uv=False)
    ec = 0.5 * ((M[0, 0] - M[1, 1]) ** 2 + (M[0, 1] + M[1, 0]) ** 2)
    ok &= np.isclose(ec, 0.5 * (t1 - t2) ** 2) and np.isclose(0.5 * (M ** 2).sum(), 0.5 * (t1 ** 2 + t2 ** 2))
check("Prop 31.6.3: E_C = (1/2)(t_1 - t_2)^2 and E_D = (1/2)(t_1^2 + t_2^2) for orientation-preserving 2 x 2 maps "
      "(200 random)", ok)


def energies(V, F, UV):
    """E_C, E_D and the signed UV area of the piecewise-linear map surface -> UV, computed per face from the
    Jacobian in the face frame (independent of dm.lscm)."""
    X = dm.face_frames(V, F)
    EC_, ED_, A_ = 0.0, 0.0, 0.0
    for tri, x in zip(F, X):
        P = np.stack([x[1] - x[0], x[2] - x[0]], 1)
        Q = np.stack([UV[tri[1]] - UV[tri[0]], UV[tri[2]] - UV[tri[0]]], 1)
        Df = Q @ np.linalg.inv(P)        # Jacobian of surface -> UV
        area = 0.5 * abs(np.linalg.det(P))
        EC_ += area * 0.5 * ((Df[0, 0] - Df[1, 1]) ** 2 + (Df[0, 1] + Df[1, 0]) ** 2)
        ED_ += area * 0.5 * (Df ** 2).sum()
        A_ += 0.5 * np.linalg.det(Q)
    return EC_, ED_, A_


Vh, Fh = dm.icosphere(2)
Vc, Fc, _ = dm.submesh(Vh, Fh, Vh[Fh].mean(1)[:, 2] > 0.1)
UVr = rng.normal(size=(len(Vc), 2))
ec, ed, a = energies(Vc, Fc, UVr)
close("Prop 31.6.3: on a mesh, E_C(f) = E_D(f) - A(f) for an arbitrary UV assignment", ec, ed - a, tol=1e-10)

# ---------------------------------------------------------------- LSCM implementation checks
xs = np.linspace(0, 1, 6)
Vp, Fp = dm.grid_mesh(xs, xs, pattern="alternate")
Vp = Vp.copy()
inner = [i for i in range(len(Vp)) if 0 < Vp[i, 0] < 1 and 0 < Vp[i, 1] < 1]
Vp[inner, :2] += rng.uniform(-0.3, 0.3, size=(len(inner), 2)) / 5
UVp = dm.lscm(Vp, Fp, (0, 35), [Vp[0, :2], Vp[35, :2]])
close("LSCM: a planar mesh with pins at their true positions is reproduced exactly", UVp, Vp[:, :2], tol=1e-9)
# a developable piecewise-linear surface: the (unjittered) grid folded by 60 degrees along the grid line x = 0.4
Vg, Fg = dm.grid_mesh(xs, xs, pattern="alternate")
Vf = Vg.copy()
m_ = Vg[:, 0] > 0.4 + 1e-9
d_ = Vg[m_, 0] - 0.4
Vf[m_, 0] = 0.4 + d_ * np.cos(np.radians(60))
Vf[m_, 2] = d_ * np.sin(np.radians(60))
inner_f = [i for i in range(len(Vf)) if i not in set(dm.boundary_vertices(Fg))]
close("LSCM: the folded grid is developable (all interior defects are 0)", dm.angle_defect(Vf, Fg)[inner_f],
      np.zeros(len(inner_f)), tol=1e-12)
UVf = dm.lscm(Vf, Fg, (0, 35), [Vg[0, :2], Vg[35, :2]])
close("LSCM: the folded (developable) grid is unfolded isometrically", UVf, Vg[:, :2], tol=1e-9)
pins_uv = np.array([[0.0, 0.0], [1.0, 0.0]])
UV1 = dm.lscm(Vc, Fc, (0, 5), pins_uv)
Rm = np.array([[np.cos(1.1), -np.sin(1.1)], [np.sin(1.1), np.cos(1.1)]]) * 2.3
UV2 = dm.lscm(Vc, Fc, (0, 5), pins_uv @ Rm.T + np.array([0.4, -1.0]))
close("LSCM: moving the pins by a similarity moves the whole solution by the same similarity", UV2,
      UV1 @ Rm.T + np.array([0.4, -1.0]), tol=1e-9)
# stationarity of the independently computed E_C at the LSCM solution (finite differences)
ns3 = runpy.run_path(str(FIG / "fig-31-6-3-bijectivity.py"))
V3, F3 = ns3["V"], ns3["F"]
UVl = ns3["UV_l"]
free = [i for i in range(len(V3)) if i not in (0, 15)]
g = []
for i in free:
    for c in range(2):
        e = np.zeros_like(UVl)
        e[i, c] = 1e-6
        g.append((energies(V3, F3, UVl + e)[0] - energies(V3, F3, UVl - e)[0]) / 2e-6)
check(f"LSCM: the gradient of E_C at the solution vanishes (max |dE/du| = {np.abs(g).max():.1e})",
      np.abs(g).max() < 1e-6)
E0 = energies(V3, F3, UVl)[0]
better = all(energies(V3, F3, UVl + np.vstack([np.zeros(2) if i in (0, 15) else rng.normal(scale=1e-2, size=2)
                                              for i in range(len(V3))]))[0] > E0 for _ in range(20))
check("LSCM: random perturbations of the free vertices increase E_C", better)
UVa = dm.lscm(V3, F3, (0, 15), pins_uv)
UVb = dm.lscm(V3, F3, (0, 12), pins_uv)

# compare the two solutions after the best complex-linear fit (a similarity)
za = (UVa - UVa.mean(0)) @ np.array([1, 1j])
zb = (UVb - UVb.mean(0)) @ np.array([1, 1j])
c = (np.conj(zb) @ za) / (np.conj(zb) @ zb)
check(f"Text: different pins give solutions that differ by more than a similarity (residual "
      f"{np.linalg.norm(za - c * zb) / np.linalg.norm(za):.3f})", np.linalg.norm(za - c * zb) / np.linalg.norm(za) > 0.01)

# ---------------------------------------------------------------- Theorem 31.6.5 (Tutte, convex combination maps)
nflip = 0
for seed in range(40):
    r = np.random.default_rng(seed)
    n = int(r.integers(4, 8))
    xs = np.linspace(0, 1, n)
    Vq, Fq = dm.grid_mesh(xs, xs, pattern="alternate")
    Vq = Vq.copy()
    inn = [i for i in range(len(Vq)) if 0 < Vq[i, 0] < 1 and 0 < Vq[i, 1] < 1]
    Vq[inn, :2] += r.uniform(-0.45, 0.45, size=(len(inn), 2)) / (n - 1)
    Vq[:, 2] = r.uniform(0, 1.2) * np.sin(r.uniform(1, 5) * Vq[:, 0]) * np.sin(r.uniform(1, 5) * Vq[:, 1])
    _, det = dm.uv_distortion(Vq, Fq, dm.fixed_boundary_map(Vq, Fq, "uniform"))
    nflip += (det <= 0).sum()
check("Thm 31.6.5: Tutte maps of 40 random crumpled grids (boundary on a circle) have no flipped face", nflip == 0)
for h in (0.2, 0.1):
    xs = np.arange(-1.3, 1.3 + 1e-9, h) + 0.0137
    Gr = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
    Vm, Fm = dm.marching_tetrahedra(np.linalg.norm(Gr, axis=-1) - 1, xs, xs, xs)
    Vs, Fs, _ = dm.submesh(Vm, Fm, Vm[Fm].mean(1)[:, 2] > 0.3)
    w = dm.cotan_weights(Vs, Fs)
    neg = sum(v < 0 for v in w.values())
    flips = [(dm.uv_distortion(Vs, Fs, dm.fixed_boundary_map(Vs, Fs, k))[1] <= 0).sum() for k in ("uniform", "cotan")]
    check(f"Text: marching-tetrahedra cap (h = {h}, {len(Fs)} faces, {neg} negative cotan weights): "
          f"flips Tutte {flips[0]}, cotan {flips[1]}", flips == [0, 0] and neg > 100)

# ---------------------------------------------------------------- Figure 31.6.3
w3 = dm.cotan_weights(V3, F3)
check("Fig 31.6.3: the crumpled patch has 16 vertices, 18 faces, smallest angle 9 degrees, w_{9,13} = -1.75",
      len(V3) == 16 and len(F3) == 18 and round(np.degrees(dm.corner_angles(V3, F3)).min()) == 9
      and round(w3[(9, 13)], 2) == -1.75)
fl = ns3["flips"]
check("Fig 31.6.3: Tutte 0 flipped, cotan 2 flipped ((5, 8, 9), (5, 9, 10)), LSCM with pins x_0, x_15 1 flipped",
      len(fl["tutte"]) == 0 and len(fl["cotan"]) == 2 and len(fl["lscm"]) == 1)
check("Fig 31.6.3: over all 66 pairs of boundary pins LSCM flips 1 face (37 pairs) or 2 faces (29 pairs)",
      list(np.bincount(ns3["worst"])) == [0, 37, 29])
UVc = ns3["UV_c"]
nb9 = sorted(dm.neighbors(F3)[9])
# the image of x_9 is not a convex combination of its neighbours: the oriented angles around it sum to 0
ring = [f for f in F3 if 9 in f]
tot = 0.0
for f in ring:
    k = list(f).index(9)
    a_, b_ = UVc[f[(k + 1) % 3]] - UVc[9], UVc[f[(k + 2) % 3]] - UVc[9]
    tot += np.arctan2(a_[0] * b_[1] - a_[1] * b_[0], a_ @ b_)
check(f"Text: in the cotan map the oriented angles around x_9 sum to {tot:.3f} (not 2 pi): x_9 left its 1-ring",
      abs(tot) < 1e-9)
tot_t = 0.0
for f in ring:
    k = list(f).index(9)
    a_, b_ = ns3["UV_t"][f[(k + 1) % 3]] - ns3["UV_t"][9], ns3["UV_t"][f[(k + 2) % 3]] - ns3["UV_t"][9]
    tot_t += np.arctan2(a_[0] * b_[1] - a_[1] * b_[0], a_ @ b_)
close("Text: in the Tutte map they sum to 2 pi", tot_t, 2 * np.pi, tol=1e-9)
lam = np.array([w3[(min(9, j), max(9, j))] for j in nb9])
check(f"Text: the normalised cotan weights of x_9 include a negative one ({(lam / lam.sum()).min():.2f})",
      (lam / lam.sum()).min() < 0)
# how extreme the patch is: x_9 is a sharp cone (angle sum 116 deg, defect 244 deg), adjacent normals bend up to 154 deg
def9 = np.degrees(dm.angle_defect(V3, F3)[9])
check(f"Fig 31.6.3: the angle sum at x_9 is {360 - def9:.0f} deg (defect {def9:.0f} deg)",
      round(360 - def9) == 116 and round(def9) == 244)
bend3 = max(abs(th_) for th_, _ in dm.edge_bending_angles(V3, F3).values())
check(f"Fig 31.6.3: adjacent face normals of the patch differ by up to {np.degrees(bend3):.0f} deg",
      round(np.degrees(bend3)) == 154)
check("Fig 31.6.3: the flipped LSCM face has x_9 as a vertex (it lies in the star of the sharp cone)",
      all(9 in F3[f] for f in fl["lscm"]))
# an independent LSCM in the complex form of [Levy02]: C(U) = sum_T |sum_j W_j U_j|^2 / d_T, solved by lstsq
def lscm_levy_complex(V, F, pins, pin_uv):
    rows = []
    for tri in F:
        P = V[tri]
        e1 = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        e2 = np.cross(nrm / np.linalg.norm(nrm), e1)
        z = np.array([(p_ - P[0]) @ e1 + 1j * ((p_ - P[0]) @ e2) for p_ in P])
        dT = abs((np.conj(z[1] - z[0]) * (z[2] - z[0])).imag)
        row = np.zeros(len(V), complex)
        row[tri] = np.array([z[2] - z[1], z[0] - z[2], z[1] - z[0]]) / np.sqrt(dT)
        rows.append(row)
    Mc = np.array(rows)
    free = [i for i in range(len(V)) if i not in pins]
    Up = np.array([complex(*q) for q in pin_uv])
    Mf, rhs = Mc[:, free], -Mc[:, list(pins)] @ Up
    sol = np.linalg.lstsq(np.block([[Mf.real, -Mf.imag], [Mf.imag, Mf.real]]), np.r_[rhs.real, rhs.imag], rcond=None)[0]
    U = np.zeros(len(V), complex)
    U[free] = sol[:len(free)] + 1j * sol[len(free):]
    U[list(pins)] = Up
    return np.c_[U.real, U.imag]


def signed_uv(UV, F):
    a_, b_ = UV[F[:, 1]] - UV[F[:, 0]], UV[F[:, 2]] - UV[F[:, 0]]
    return a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]


UVx = lscm_levy_complex(V3, F3, (0, 15), [(0.0, 0.0), (1.0, 0.0)])
close("Fig 31.6.3: the complex-form LSCM of [Levy02] gives the same map as dgmesh.lscm", UVx, ns3["UV_l"], tol=1e-8)
loop3 = dm.boundary_loop(F3)
hist = np.bincount([(signed_uv(lscm_levy_complex(V3, F3, (loop3[a_], loop3[b_]), [(0.0, 0.0), (1.0, 0.0)]), F3) < 0).sum()
                    for a_ in range(len(loop3)) for b_ in range(a_ + 1, len(loop3))])
check("Fig 31.6.3: the complex-form LSCM flips 1 face for 37 pin pairs and 2 faces for 29", list(hist) == [0, 37, 29])
# how rare are LSCM flips? random crumpled grid patches that are graphs over the xy-plane with smallest angle >= 15 deg,
# two opposite corners pinned
def mild_patch(rng):
    n_ = int(rng.integers(4, 7))
    amp_, jit_ = rng.uniform(0.1, 0.6), rng.uniform(0, 0.3)
    xs_ = np.linspace(0, 1, n_)
    Vm, Fm = dm.grid_mesh(xs_, xs_, pattern="alternate")
    Vm = Vm.copy()
    inn = [i for i in range(len(Vm)) if 0 < Vm[i, 0] < 1 and 0 < Vm[i, 1] < 1]
    Vm[inn, :2] += rng.uniform(-jit_, jit_, size=(len(inn), 2)) / (n_ - 1)
    fx_, fy_ = rng.uniform(1, 5, 2)
    ph_ = rng.uniform(0, 6.28, 2)
    Vm[:, 2] = amp_ * np.sin(fx_ * Vm[:, 0] + ph_[0]) * np.sin(fy_ * Vm[:, 1] + ph_[1])
    return n_, Vm, Fm, inn


rng_m = np.random.default_rng(0)
valid_m, flipped_m = 0, []
for _ in range(6000):
    n_, Vm, Fm, inn = mild_patch(rng_m)
    if (signed_uv(Vm[:, :2], Fm) <= 0).any() or np.degrees(dm.corner_angles(Vm, Fm)).min() < 15:
        continue
    valid_m += 1
    if (signed_uv(lscm_levy_complex(Vm, Fm, (0, n_ * n_ - 1), [(0.0, 0.0), (1.0, 0.0)]), Fm) < 0).any():
        dd = np.degrees(dm.angle_defect(Vm, Fm)[inn])
        flipped_m.append(dd[np.argmax(np.abs(dd))])
check(f"Text: {valid_m} milder random patches, LSCM flips in {len(flipped_m)} (largest |defect| there: "
      f"{flipped_m[0] if flipped_m else 0:.0f} deg, a saddle)", valid_m == 5892 and len(flipped_m) == 1 and round(flipped_m[0]) == -149)
# Proposition 31.6.4 needs one-to-one, not only 'no flipped face': a star wound twice has positive triangles
# and angle sum 4 pi
ring2 = np.array([[np.cos(4 * np.pi * k / 8), np.sin(4 * np.pi * k / 8)] for k in range(8)])
areas2 = [0.5 * (ring2[k, 0] * ring2[(k + 1) % 8, 1] - ring2[k, 1] * ring2[(k + 1) % 8, 0]) for k in range(8)]
angs2 = [np.arctan2(ring2[k, 0] * ring2[(k + 1) % 8, 1] - ring2[k, 1] * ring2[(k + 1) % 8, 0],
                    ring2[k] @ ring2[(k + 1) % 8]) for k in range(8)]
check("Prop 31.6.4: a vertex star wound twice has no flipped triangle but its angles sum to 4 pi",
      min(areas2) > 0 and abs(sum(angs2) - 4 * np.pi) < 1e-12)

# ---------------------------------------------------------------- Proposition 31.6.4 (angle sums) and the caps
ns2 = runpy.run_path(str(FIG / "fig-31-6-2-cap-distortion.py"))
Vc90, Fc90b, UVb90, _, _ = ns2["flatten_cap"](90)
bd = set(dm.boundary_vertices(Fc90b))
inner90 = [i for i in range(len(Vc90)) if i not in bd]
Vuv = np.c_[UVb90, np.zeros(len(UVb90))]
uv_sums = dm.angle_sums(Vuv, Fc90b)[inner90]
mesh_def = dm.angle_defect(Vc90, Fc90b)[inner90]
close("Prop 31.6.4: in the flattened 90-degree cap the angles around every interior vertex sum to 2 pi",
      uv_sums, np.full(len(inner90), 2 * np.pi), tol=1e-9)
close("Prop 31.6.4: so the angle changes around vertex i add up to its defect delta_i",
      uv_sums - dm.angle_sums(Vc90, Fc90b)[inner90], mesh_def, tol=1e-9)
th, T = sp.symbols("theta Theta", positive=True)
sym_equal("Text: total curvature of a cap of the unit sphere: int_0^Theta sin = 1 - cos Theta (times 2 pi)",
          sp.integrate(2 * sp.pi * sp.sin(th), (th, 0, T)), 2 * sp.pi * (1 - sp.cos(T)))
rr = sp.tan(th / 2)
lam_rad = sp.diff(rr, th)
lam_circ = rr / sp.sin(th)
sym_equal("Text: stereographic projection r = tan(theta/2) is conformal (radial and circumferential scales agree)",
          sp.simplify(lam_rad - lam_circ), 0)
sym_equal("Text: its area factor is (1/4) sec^4(theta/2)", sp.simplify(lam_rad ** 2 - sp.sec(th / 2) ** 4 / 4), 0)
ratio = ns2["ratio"]
thetas = ns2["thetas"]
sec4 = 1 / np.cos(np.radians(np.array(thetas)) / 2) ** 4
check(f"Fig 31.6.2: LSCM area ratios {np.round(ratio[[2, 4, 6]], 1)} at 60, 90, 120 deg; sec^4 = "
      f"{np.round(sec4[[2, 4, 6]], 2)}", np.allclose(np.round(ratio[[2, 4, 6]], 1), [1.9, 4.6, 20.4]))
check("Fig 31.6.2: all within a factor 1.5 of sec^4(Theta/2)", np.all(np.abs(np.log(ratio / sec4)) < np.log(1.5)))
check(f"Fig 31.6.2: conformal distortion median sigma_1/sigma_2 <= {max(ns2['conf']):.3f}", max(ns2["conf"]) < 1.08)

# ---------------------------------------------------------------- Figure 31.6.4
ns4 = runpy.run_path(str(FIG / "fig-31-6-4-charts-seams.py"))
res = ns4["results"]
check("Fig 31.6.4: worst area ratios 14470, 4.84, 1.47, 1.10 for 1, 2, 6, 20 charts",
      [round(res[k][0], 2) for k in (1, 2, 6, 20)] == [14470.06, 4.84, 1.47, 1.10])
check("Fig 31.6.4: seam lengths 1.52, 7.06, 17.74, 33.21",
      [round(res[k][1], 2) for k in (1, 2, 6, 20)] == [1.52, 7.06, 17.74, 33.21])
check("Text: the 1-chart case is a cap with Theta = 170 deg: sec^4(85 deg) = 17331",
      round(1 / np.cos(np.radians(85)) ** 4) == 17331)
check("Text: the equator has length 2 pi = 6.28 < 7.06 (the seam follows triangle edges)", 2 * np.pi < res[2][1])

# ---------------------------------------------------------------- exercises
# 31.6.1: sigma = (2, 0.5): equiareal, not conformal; a 1024 texture: 512 and 2048 texels per unit length
check("Exercise 31.6.1: sigma = (2, 0.5) is equiareal (product 1), not conformal; 1024/sigma = 512 and 2048",
      2 * 0.5 == 1 and (1024 / 2, 1024 / 0.5) == (512, 2048))
# 31.6.2: cube: every vertex has defect pi/2, total 4 pi; a net must cut through all 8 vertices
Vb = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
Fq = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
Fb = np.array([t for a_, b_, c_, d_ in Fq for t in ((a_, b_, c_), (a_, c_, d_))])
Fb, ok_, _ = dm.orient(Fb)
dcube = dm.angle_defect(Vb, Fb)
close("Exercise 31.6.2: each cube vertex has defect pi/2, total 4 pi", dcube, np.full(8, np.pi / 2), tol=1e-12)
check("Exercise 31.6.2: the cross-shaped net has 14 boundary vertices from 8 cube vertices; a spanning tree on 8 "
      "vertices has 7 edges and the cut boundary has 2 x 7 = 14 edges", 2 * (8 - 1) == 14)
# 31.6.3: E_C is zero on constant maps and invariant under similarities -> 4 parameters -> two pins
check("Exercise 31.6.3: a similarity of the plane has 4 real parameters (complex a z + b), two pinned points fix it",
      len(sp.symbols("a_re a_im b_re b_im")) == 4)
# 31.6.4: hemisphere stereographic ratio 4; total curvature 2 pi
close("Exercise 31.6.4: hemisphere: sec^4(45 deg) = 4, total curvature 2 pi", [1 / np.cos(np.pi / 4) ** 4,
      2 * np.pi * (1 - np.cos(np.pi / 2))], [4.0, 2 * np.pi], tol=1e-12)
# 31.6.5: positive weights -> inside the convex hull; flipped -> angle sum 0 (checked above)
lam_t = np.ones(len(nb9)) / len(nb9)
close("Exercise 31.6.5: Tutte image of x_9 is the average of its neighbours", ns3["UV_t"][9],
      lam_t @ ns3["UV_t"][nb9], tol=1e-10)

# ---------------------------------------------------------------- text: angle distortion of fixed-boundary maps
Vh3, Fh3 = dm.icosphere(3)
Vhh, Fhh, _ = dm.submesh(Vh3, Fh3, Vh3[Fh3].mean(1)[:, 2] > 0.0)
loop_h = dm.boundary_loop(Fhh)
q95 = []
for UVh in (dm.fixed_boundary_map(Vhh, Fhh, "uniform"), dm.fixed_boundary_map(Vhh, Fhh, "cotan"),
            dm.lscm(Vhh, Fhh, (loop_h[0], loop_h[len(loop_h) // 2]), [(0.0, 0.0), (1.0, 0.0)])):
    sh, _ = dm.uv_distortion(Vhh, Fhh, UVh)
    q95.append(np.percentile(sh[:, 0] / sh[:, 1], 95))
check(f"Text: hemisphere (level 3), 95th percentile of sigma_1/sigma_2: Tutte {q95[0]:.2f}, cotan {q95[1]:.2f}, "
      f"LSCM {q95[2]:.2f}", [round(v, 2) for v in q95] == [1.72, 1.6, 1.09])

summary()
