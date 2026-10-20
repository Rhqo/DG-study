"""Verification for Section 41.5 (Curvature Flows on Surfaces).

Run from the project root: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/verify/v-41-5-surface-flows.py
(about two minutes; the torus collapse of Remark 41.5.18 and the finite-difference Willmore gradients dominate).
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "41-5-surface-flows.html"
R, r = 2.0, 0.8
area = lambda V, F: dm.face_areas(V, F).sum()  # noqa: E731


def close_all(name, a, value, tol):
    """every entry of the array a equals the scalar value within tol"""
    a = np.asarray(a, float)
    return close(name, a, np.full(a.shape, value), tol)


# ---------------------------------------------------------------- Implementation 41.5 and the coding exercises
for block in ("code-41-5-1", "code-ex-41-5-5", "code-ex-41-5-6"):
    check(f"{block}: at most 30 lines", dgcode.code_lines(PAGE, block) <= 30)
ns, out = ddg41.run_printed(PAGE, "code-41-5-1")
check("Implementation 41.5: printed output matches the page", out.strip() == ddg41.printed_block(PAGE, "out-41-5-1").strip(), out)
V0, F0 = dm.icosphere(3)
close("Implementation 41.5: its Gauss curvature normal = ddg41.gauss_curvature_normal", ns["gauss_curvature_normal"](V0, F0),
      ddg41.gauss_curvature_normal(V0, F0), 1e-13)
close("Implementation 41.5: its velocities = ddg41.mcf_velocity, gcf_velocity",
      [ns["velocity"](V0, F0, "mean"), ns["velocity"](V0, F0, "gauss")], [ddg41.mcf_velocity(V0, F0), ddg41.gcf_velocity(V0, F0)], 1e-12)
_, out5 = ddg41.run_printed(PAGE, "code-ex-41-5-5")
check("Exercise 41.5.5: printed output matches the page", out5.strip() == ddg41.printed_block(PAGE, "out-ex-41-5-5").strip(), out5)
ns6, out6 = ddg41.run_printed(PAGE, "code-ex-41-5-6")
check("Exercise 41.5.6: printed output matches the page", out6.strip() == ddg41.printed_block(PAGE, "out-ex-41-5-6").strip(), out6)
close("Exercise 41.5.6: global finite differences = ddg41.willmore_gradient_fd", ns6["grad"], ddg41.willmore_gradient_fd(ns6["V"], ns6["F"]), 1e-7)

# ---------------------------------------------------------------- the gradient-flow framework (41.5.1), (41.5.2)
lam = {lev: ddg41.largest_eigenvalue(*dm.icosphere(lev)) for lev in (2, 3, 4)}
ell = {lev: ddg41.mean_edge_length(*dm.icosphere(lev)) for lev in (2, 3, 4)}
check("Section text: lambda_max ell^2 is about 7 on icospheres", all(6.5 < lam[k] * ell[k] ** 2 < 7.3 for k in lam),
      [round(lam[k] * ell[k] ** 2, 3) for k in lam])
close("Example 41.5.5: lambda_max of icosphere(3) = 307.6, h_c = 0.0065", [lam[3], 2 / lam[3]], [307.6, 0.0065], 0.05)
close("Example 41.5.5: h = 6.5e-4 is 0.1 h_c", 6.5e-4 / (2 / lam[3]), 0.1, 0.001)

# ---------------------------------------------------------------- Proposition 41.5.1, Example 41.5.2 (normal flow)
t, r0, R_, rr = sp.symbols("t r0 R r", positive=True)
A_s, V_s = 4 * sp.pi * r0 ** 2, sp.Rational(4, 3) * sp.pi * r0 ** 3
intH, intK = -4 * sp.pi * r0, 4 * sp.pi
check("(41.5.3) on the sphere: A(t) = 4 pi (r0 - t)^2", sp.simplify(A_s + 2 * t * intH + t ** 2 * intK - 4 * sp.pi * (r0 - t) ** 2) == 0)
check("(41.5.3) on the sphere: V(t) = 4/3 pi (r0 - t)^3",
      sp.simplify(V_s - t * A_s - t ** 2 * intH - t ** 3 / 3 * intK - sp.Rational(4, 3) * sp.pi * (r0 - t) ** 3) == 0)
A_T, V_T, H_T = 4 * sp.pi ** 2 * R_ * rr, 2 * sp.pi ** 2 * R_ * rr ** 2, -2 * sp.pi ** 2 * R_
check("(41.5.3) on the torus (r -> r - t, int K = 0)", sp.simplify(A_T + 2 * t * H_T - 4 * sp.pi ** 2 * R_ * (rr - t)) == 0
      and sp.simplify(V_T - t * A_T - t ** 2 * H_T - 2 * sp.pi ** 2 * R_ * (rr - t) ** 2) == 0)
u = np.linspace(0, 2 * np.pi, 20001)[:-1]
du = 2 * np.pi / 20000
rho = R + r * np.cos(u)
dA = r * rho * du * 2 * np.pi
Hu, Ku = -(R + 2 * r * np.cos(u)) / (2 * r * rho), np.cos(u) / (r * rho)
close("torus: int H dA = -2 pi^2 R", np.sum(Hu * dA), -2 * np.pi ** 2 * R, 1e-9)


def normal_velocity(V, F):
    return -ddg41.vector_area(V, F) / dm.mass_voronoi(V, F)[:, None]


Vn = ddg41.explicit_flow(V0, F0, normal_velocity, 1e-3, 300)
close("Example 41.5.2: normal flow on icosphere(3), mean radius at t = 0.3 is 0.7017", np.linalg.norm(Vn, axis=1).mean(), 0.7017, 1e-4)
Vc, Fc = ddg41.cube_mesh(8, center=(0, 0, 0), side=2.0)
kind = np.isclose(np.abs(Vc), 1).sum(1)
vel = normal_velocity(Vc, Fc)
speed = np.linalg.norm(vel, axis=1)
close_all("Example 41.5.2: cube face vertices move with speed 1", speed[kind == 1], 1.0, 1e-12)
close_all("Example 41.5.2: cube edge vertices move with speed 0.7071", speed[kind == 2], np.sqrt(0.5), 1e-12)
ne = np.where(kind == 2)[0]
nsum = np.sign(Vc[ne]) * np.isclose(np.abs(Vc[ne]), 1)          # n_1 + n_2 for an edge vertex
close("Example 41.5.2: edge velocity = -(n_1 + n_2) / 2", vel[ne], -nsum / 2, 1e-12)
check("Example 41.5.2: corner speeds are 0.5443 or 0.7698", set(np.round(speed[kind == 3], 4)) == {0.5443, 0.7698},
      set(np.round(speed[kind == 3], 4)))


def cube_random_diagonals(n, seed):
    """[-1, 1]^3 with n x n cells per face and a random diagonal in every cell (review check of Example 41.5.2)"""
    rng_c, idx, P, T = np.random.default_rng(seed), {}, [], []
    g = np.linspace(-1, 1, n + 1)

    def vid(p):
        key = tuple(np.round(p, 9))
        if key not in idx:
            idx[key] = len(P)
            P.append(np.array(p, float))
        return idx[key]
    for ax in range(3):
        o = [c for c in range(3) if c != ax]
        for s in (-1, 1):
            for a in range(n):
                for b in range(n):
                    q = []
                    for (da, db) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[ax], p[o[0]], p[o[1]] = s, g[a + da], g[b + db]
                        q.append(vid(p))
                    tris = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if rng_c.random() < 0.5 else [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
                    for t_ in tris:
                        nn = np.cross(P[t_[1]] - P[t_[0]], P[t_[2]] - P[t_[0]])
                        T.append(t_ if nn[ax] * s > 0 else t_[::-1])
    return np.array(P), np.array(T)


Vq, Fq = cube_random_diagonals(8, 8)
sq = np.linalg.norm(ddg41.vector_area(Vq, Fq) / dm.mass_voronoi(Vq, Fq)[:, None], axis=1)
eq = np.isclose(np.abs(Vq), 1).sum(1) == 2
close("Example 41.5.2: random diagonals, edge speeds range over 0.47..0.94", [sq[eq].min(), sq[eq].max()], [np.sqrt(2) / 3, np.sqrt(8) / 3], 1e-12)
sl = np.zeros_like(V0)
for f in F0:
    for a in range(3):
        i, j, k = f[a], f[(a + 1) % 3], f[(a + 2) % 3]
        sl[i] += np.cross(V0[j], V0[k]) / 6
close("slide 35: (1/6) sum x_j x x_k = vector area = grad V (so its update has the right sign)", sl, ddg41.vector_area(V0, F0), 1e-13)

# ---------------------------------------------------------------- Definition 41.5.3, Proposition 41.5.4
Vr, Fr = ddg41.noisy_sphere(3, 0.05, 5)
Lx, m = dm.cotan_apply(Vr, Fr, Vr), dm.mass_voronoi(Vr, Fr)
v = Lx / m[:, None]
eps = 1e-6
rate = (area(Vr + eps * v, Fr) - area(Vr - eps * v, Fr)) / (2 * eps)
close("Proposition 41.5.4(c): dA/dt = -sum |Lx|^2 / A_i = -4 W_h (noisy sphere)", [rate, rate], [-np.sum(Lx * Lx, 1).dot(1 / m), -4 * ddg41.willmore_energy(Vr, Fr)], 1e-5)
rate_I = (area(Vr + eps * Lx, Fr) - area(Vr - eps * Lx, Fr)) / (2 * eps)
close("Exercise 41.5.3(b): without the mass matrix dA/dt = -sum |Lx|^2", rate_I, -np.sum(Lx * Lx), 1e-6)
Ff = F0[:, ::-1]
close("orientation: (L x) unchanged, (K N) flips", [dm.cotan_apply(V0, Ff, V0), ddg41.gauss_curvature_normal(V0, Ff)],
      [dm.cotan_apply(V0, F0, V0), -ddg41.gauss_curvature_normal(V0, F0)], 1e-13)
rs = sp.Function("r")(t)
check("Example 41.5.5: r' = -2/r gives r^2 = r0^2 - 4t", sp.simplify(sp.diff(r0 ** 2 - 4 * t, t) - 2 * rs * (-2 / rs)) == 0)
check("Example 41.5.5: dA/dt = -16 pi = -int 4 H^2 dA", sp.simplify(sp.diff(4 * sp.pi * (r0 ** 2 - 4 * t), t) + 4 / r0 ** 2 * 4 * sp.pi * r0 ** 2) == 0)
check("Example 41.5.5: cylinder rho' = 2H = -1/rho gives rho^2 = rho0^2 - 2t", sp.simplify(sp.diff(r0 ** 2 - 2 * t, t) - 2 * rs * (-1 / rs)) == 0)
close("Exercise 41.5.1: extinction times 0.25, 0.0625, 0.125 and the neck bound 1/sqrt 2",
      [1 / 4, 0.5 ** 2 / 4, 0.5 ** 2 / 2, np.sqrt(2 * 0.25)], [0.25, 0.0625, 0.125, 1 / np.sqrt(2)], 1e-15)

# ---------------------------------------------------------------- Figure 41.5.1 (mean and Gauss curvature flow on icosphere(3))
h = 6.5e-4
steps = int(round(0.2 / h))
res = {}
for name, velf in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
    rec = []
    ddg41.explicit_flow(V0, F0, velf, h, steps, record=lambda k, V: rec.append((k * h, np.linalg.norm(V - V.mean(0), axis=1).mean())))
    res[name] = np.array(rec)
tm, rm = res["mean"].T
tg, rg = res["gauss"].T
em, eg = np.abs(rm ** 2 - (1 - 4 * tm)), np.abs(rg ** 3 - (1 - 3 * tg))
check("Figure 41.5.1: 308 steps, last t = 0.2002", steps == 308 and abs(tm[-1] - 0.2002) < 1e-12)
close("Figure 41.5.1: max deviations 1.04e-3 and 1.32e-3", [em.max(), eg.max()], [1.04e-3, 1.32e-3], 0.006e-3)
check("Figure 41.5.1: the deviations are largest at the last point", em.argmax() == len(em) - 1 and eg.argmax() == len(eg) - 1)
close("Example 41.5.5: last point 0.20024 vs 0.19920", [rm[-1] ** 2, 1 - 4 * tm[-1]], [0.20024, 0.19920], 6e-6)

# ---------------------------------------------------------------- Examples 41.5.6, 41.5.9 and Figure 41.5.2 (the cube)
ellc = 2 / 8
Lxc, mc = dm.cotan_apply(Vc, Fc, Vc), dm.mass_voronoi(Vc, Fc)
sp0 = np.linalg.norm(Lxc / mc[:, None], axis=1)
close_all("Example 41.5.6: at t = 0 face vertices do not move", sp0[kind == 1], 0.0, 1e-11)
close_all("Example 41.5.6: edge speed sqrt 2 / l", sp0[kind == 2], np.sqrt(2) / ellc, 1e-10)
close_all("Example 41.5.6: corner speed 4 / (sqrt 3 l)", sp0[kind == 3], 4 / (np.sqrt(3) * ellc), 1e-10)
for lab, vals, val in (("edge |grad A|^2 = 2 l^2", np.sum(Lxc[kind == 2] ** 2, 1), 2 * ellc ** 2), ("edge A_i = l^2", mc[kind == 2], ellc ** 2),
                       ("corner |grad A|^2 = 3 l^2", np.sum(Lxc[kind == 3] ** 2, 1), 3 * ellc ** 2), ("corner A_i = 3 l^2 / 4", mc[kind == 3], 0.75 * ellc ** 2)):
    close_all(f"Exercise 41.5.5: {lab}", vals, val, 1e-12)
KNc = ddg41.gauss_curvature_normal(Vc, Fc)
close_all("Example 41.5.9: (K N)_i = 0 except at the 8 corners", KNc[kind < 3], 0.0, 1e-13)
close("Example 41.5.9: corner (K N)_i = (pi/4)(sign of the corner)", KNc[kind == 3], np.pi / 4 * np.sign(Vc[kind == 3]), 1e-13)
hc_cube = 2 / ddg41.largest_eigenvalue(Vc, Fc)
check("Example 41.5.6: h = 0.1/32 is about 0.2 h_c", abs((0.1 / 32) / hc_cube - 0.2) < 0.01, (0.1 / 32) / hc_cube)
cube = {}
for name, velf in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
    Vt = ddg41.explicit_flow(Vc, Fc, velf, 0.1 / 32, 32)
    d = np.linalg.norm(Vt - Vc, axis=1)
    center = int(np.argmin(np.linalg.norm(Vc - np.array([0, 0, 1.0]), axis=1)))
    cube[name] = [d[kind == 3].max(), d[kind == 2].max(), d[center], area(Vt, Fc), dm.signed_volume(Vt, Fc)]
close("Example 41.5.6: MCF at t = 0.1: corner 0.705, edge 0.478, face center 0.017, area 14.76, volume 5.14", cube["mean"],
      [0.705, 0.478, 0.017, 14.76, 5.14], 0.0051)
close("Example 41.5.9: GCF at t = 0.1: corner 0.416, edge 0.290, area 18.66, volume 6.91",
      [cube["gauss"][k] for k in (0, 1, 3, 4)], [0.416, 0.290, 18.66, 6.91], 0.0051)
check("Example 41.5.9: GCF face center moves about 5e-9", 3e-9 < cube["gauss"][2] < 7e-9, cube["gauss"][2])
close("Figure 41.5.2 caption: 768 triangles", len(Fc), 768, 0)

# ---------------------------------------------------------------- Definition 41.5.7, Proposition 41.5.8
Vrev = ddg41.explicit_flow(V0, Ff, ddg41.gcf_velocity, 6.5e-4, 50)
close("text: reversed orientation, GCF radius 1.0316 at t = 0.0325 (exact 1.0315)",
      [np.linalg.norm(Vrev, axis=1).mean(), (1 + 3 * 0.0325) ** (1 / 3)], [1.0316, 1.0315], 6e-5)
check("text: reversed orientation, roughness 1.8e-4 at t = 0.0325", 1.75e-4 < ddg41.roughness(Vrev) < 1.85e-4, ddg41.roughness(Vrev))
Vrev = ddg41.explicit_flow(Vrev, Ff, ddg41.gcf_velocity, 6.5e-4, 50)
close("text: reversed orientation, roughness 0.33 at t = 0.065", ddg41.roughness(Vrev), 0.33, 5e-3)
check("Proposition 41.5.8: rho' = -1/rho^2 gives rho^3 = rho0^3 - 3t", sp.simplify(sp.diff(r0 ** 3 - 3 * t, t) - 3 * rs ** 2 * (-1 / rs ** 2)) == 0)
check("text: d/dt (4/3 pi rho^3) = -4 pi", sp.simplify(sp.diff(sp.Rational(4, 3) * sp.pi * (r0 ** 3 - 3 * t), t) + 4 * sp.pi) == 0)
close("text: discrete dV/dt on icosphere(3) is -12.498 (-4 pi = -12.566)", np.sum(ddg41.vector_area(V0, F0) * ddg41.gcf_velocity(V0, F0)), -12.498, 6e-4)
close("torus: int K dA = 0 (Gauss-Bonnet, chi = 0)", np.sum(Ku * dA), 0.0, 1e-9)

# ---------------------------------------------------------------- Non-example 41.5.10, Remark 41.5.11, Figure 41.5.3
d0 = dm.angle_defect(Vr, Fr)
check("Non-example 41.5.10: 373 of 642 vertices have negative angle defect", len(Vr) == 642 and np.sum(d0 < 0) == 373)
close("Non-example 41.5.10: roughness 0.0497", ddg41.roughness(Vr), 0.0497, 5e-5)
hr = 0.1 * 2 / ddg41.largest_eigenvalue(Vr, Fr)
nst = int(np.ceil(0.033 / hr))
rough = {}
for name, velf in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
    rec = []
    ddg41.explicit_flow(Vr, Fr, velf, hr, nst, record=lambda k, V: rec.append(ddg41.roughness(V)))
    rough[name] = rec
close("Non-example 41.5.10: t at 10 steps = 0.0066", 10 * hr, 0.0066, 5e-5)
close("Non-example 41.5.10: MCF roughness 0.0250 (10 steps), 0.0101 (t = 0.033)", [rough["mean"][10], rough["mean"][-1]], [0.0250, 0.0101], 5e-5)
close("Non-example 41.5.10: GCF roughness 0.164 (10 steps), 0.329 (t = 0.033)", [rough["gauss"][10], rough["gauss"][-1]], [0.164, 0.329], 5e-4)
check("Figure 41.5.3: MCF roughness decreases at every step", np.all(np.diff(rough["mean"]) < 0))


def rough_at(V, F, frac, T):
    hh = frac * 2 / ddg41.largest_eigenvalue(V, F)
    tt = 0.0
    while tt < T - 1e-12:
        V = V + hh * ddg41.gcf_velocity(V, F)
        tt += hh
    return ddg41.roughness(V)


close("Non-example 41.5.10: at t = 0.0164 GCF roughness 0.293 (0.1 h_c) and 0.313 (0.02 h_c)",
      [rough_at(Vr, Fr, 0.1, 0.0164), rough_at(Vr, Fr, 0.02, 0.0164)], [0.293, 0.313], 5e-4)
E, Ar = [ddg41.half_sum_ltheta(Vr, Fr)], area(Vr, Fr)
V10 = ddg41.explicit_flow(Vr, Fr, ddg41.gcf_velocity, hr, 10)
close("Non-example 41.5.10: (1/2) sum l theta from 14.39 to -70.4 in 10 steps", [E[0], ddg41.half_sum_ltheta(V10, Fr)], [14.39, -70.4], 0.051)
close("Non-example 41.5.10: area 14.80 -> 26.59 in 10 steps", [Ar, area(V10, Fr)], [14.80, 26.59], 0.0051)
Vs2, Fs2 = ddg41.noisy_sphere(3, 0.002, 5)
check("Non-example 41.5.10: noise 0.002 has every angle defect positive", np.all(dm.angle_defect(Vs2, Fs2) > 0))
hs2 = 0.1 * 2 / ddg41.largest_eigenvalue(Vs2, Fs2)
Vs2t = ddg41.explicit_flow(Vs2, Fs2, ddg41.gcf_velocity, hs2, int(np.ceil(0.033 / hs2)))
close("Non-example 41.5.10: noise 0.002, GCF roughness 0.0020 -> 0.0004", [ddg41.roughness(Vs2), ddg41.roughness(Vs2t)], [0.0020, 0.0004], 5e-5)
H0 = dd.mean_curvature(Vr, Fr)
pit, sad = (d0 > 0) & (H0 > 0), d0 < 0
rad = lambda V: np.linalg.norm(V - V.mean(0), axis=1)  # noqa: E731
dr = rad(V10) - rad(Vr)
check("Remark 41.5.11: 96 pits and 373 saddles", pit.sum() == 96 and sad.sum() == 373)
close("Remark 41.5.11: in 10 steps pits move 0.275 inward, saddles 0.049 outward on average", [dr[pit].mean(), dr[sad].mean()], [-0.275, 0.049], 5e-4)
tail = rad(V10) < 0.85
check("Figure 41.5.3 text: 94 vertices below 0.85, 84 of them pits", tail.sum() == 94 and (tail & pit).sum() == 84)
Vm10 = ddg41.explicit_flow(Vr, Fr, ddg41.mcf_velocity, hr, 10)
close("Figure 41.5.3: distance ranges GCF 0.38-1.14, MCF 0.91-1.08", [rad(V10).min(), rad(V10).max(), rad(Vm10).min(), rad(Vm10).max()],
      [0.38, 1.14, 0.91, 1.08], 0.0051)
x, y, a_, c_, eps_, k_ = sp.symbols("x y a c epsilon k", real=True)
eta = sp.Function("eta")(x, y)
uu = (a_ * x ** 2 + c_ * y ** 2) / 2 + eps_ * eta
Kexpr = sp.diff(uu, x, 2) * sp.diff(uu, y, 2) - sp.diff(uu, x, y) ** 2
first = sp.diff(Kexpr, eps_).subs(eps_, 0)
check("(41.5.7): first-order part of K is c eta_xx + a eta_yy", sp.simplify(first - (c_ * sp.diff(eta, x, 2) + a_ * sp.diff(eta, y, 2))) == 0)
mode = sp.cos(k_ * x)
check("Exercise 41.5.4: the mode cos(kx) grows at rate c k^2", sp.simplify((-c_ * sp.diff(mode, x, 2) - a_ * sp.diff(mode, y, 2)) / mode - c_ * k_ ** 2) == 0)

# ---------------------------------------------------------------- Definition 41.5.12, Proposition 41.5.13, Example 41.5.14
k1, k2 = sp.symbols("kappa1 kappa2", real=True)
check("Proposition 41.5.13(b): H^2 - K = (k1 - k2)^2 / 4", sp.expand(((k1 + k2) / 2) ** 2 - k1 * k2 - (k1 - k2) ** 2 / 4) == 0)
close("Proposition 41.5.13(a): W_h is scale invariant and orientation independent",
      [ddg41.willmore_energy(3 * V0, F0), ddg41.willmore_energy(V0, Ff)], [ddg41.willmore_energy(V0, F0)] * 2, 1e-10)
W_exact = lambda a: np.pi ** 2 * a * a / np.sqrt(a * a - 1)  # noqa: E731
close("Example 41.5.14(a): quadrature of int H^2 dA on T_{2,0.8} = pi^2 a^2 / sqrt(a^2 - 1) = 26.922", [np.sum(Hu ** 2 * dA), W_exact(2.5)], [26.922] * 2, 6e-4)
aa = sp.symbols("a", positive=True)
Ws = sp.pi ** 2 * aa ** 2 / sp.sqrt(aa ** 2 - 1)
check("Example 41.5.14: dW/da = pi^2 a (a^2 - 2) / (a^2 - 1)^(3/2), zero only at sqrt 2",
      sp.simplify(sp.diff(Ws, aa) - sp.pi ** 2 * aa * (aa ** 2 - 2) / (aa ** 2 - 1) ** sp.Rational(3, 2)) == 0
      and sp.solve(sp.Eq(sp.diff(Ws, aa), 0), aa) == [sp.sqrt(2)])
close("Example 41.5.14: minimum 2 pi^2 = 19.739", [float(Ws.subs(aa, sp.sqrt(2))), 2 * np.pi ** 2], [19.739] * 2, 6e-4)
Wt = [ddg41.willmore_energy(*dm.torus_mesh(R, r, n, 2 * n)) for n in (12, 24, 48, 96)]
close("Example 41.5.14(b): W_h of torus_mesh(2, 0.8, n, 2n)", Wt, [26.398, 26.786, 26.887, 26.913], 6e-4)
err = W_exact(2.5) - np.array(Wt)
check("Example 41.5.14(b): from below, error ratio about 1/4", np.all(err > 0) and np.all(np.abs(err[1:] / err[:-1] - 0.25) < 0.02), err[1:] / err[:-1])
Wi = [ddg41.willmore_energy(*dm.icosphere(lev)) for lev in range(6)]
close("Example 41.5.14(c): icosphere W_h", Wi, [9.575, 11.666, 12.331, 12.507, 12.551, 12.563], 6e-4)
check("Example 41.5.14(c): all below 4 pi", max(Wi) < 4 * np.pi)
close("Example 41.5.14(d): cube W_h = 6n + 2", [ddg41.willmore_energy(*ddg41.cube_mesh(n, center=(0, 0, 0), side=2.0)) for n in (2, 4, 8, 16)],
      [14, 26, 50, 98], 1e-9)
avals = [1.15, 1.25, 1.414, 1.6, 2.0, 2.5, 3.0]
Wa = np.array([ddg41.willmore_energy(*dm.torus_mesh(a, 1.0, 48, 96)) for a in avals])
check("Figure 41.5.4: all W_h below the curve by less than 0.2", np.all(Wa < W_exact(np.array(avals))) and np.all(W_exact(np.array(avals)) - Wa < 0.2))
close("Figure 41.5.4: W_h at a = 1.414 is 19.681", Wa[2], 19.681, 6e-4)
check("Figure 41.5.4: every torus value above 4 pi", np.all(Wa > 4 * np.pi))

# ---------------------------------------------------------------- Definition 41.5.15, Example 41.5.16
Hf = lambda u: -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))  # noqa: E731
du_ = 1e-4
q = lambda u: (R + r * np.cos(u)) / r * (Hf(u + du_) - Hf(u - du_)) / (2 * du_)  # noqa: E731
lapH = (q(u + du_) - q(u - du_)) / (2 * du_) / (r * rho)
G = ddg41.torus_willmore_gradient(u)
close("Exercise 41.5.6: closed form of Delta H", G - 2 * Hu * (Hu ** 2 - Ku), lapH, 1e-6)
close("Example 41.5.16: G at u = 0, pi/2, pi = -0.121, -0.332, -1.537", ddg41.torus_willmore_gradient(np.array([0, np.pi / 2, np.pi])),
      [-0.121, -0.332, -1.537], 6e-4)
check("Example 41.5.16: G < 0 everywhere", G.max() < 0)
a0 = R / r
Wp = np.pi ** 2 * a0 * (a0 ** 2 - 2) / (a0 ** 2 - 1) ** 1.5
close("Example 41.5.16(a): int G cos u dA = d W / dR = 10.8968, int G dA = d W / dr = -27.2420",
      [np.sum(G * np.cos(u) * dA), Wp / r, np.sum(G * dA), -Wp * R / r ** 2], [10.8968, 10.8968, -27.2420, -27.2420], 6e-5)
errs, tans = [], []
for n in (12, 24, 48):
    V, F = dm.torus_mesh(R, r, n, 2 * n)
    gW = ddg41.willmore_gradient_fd(V, F)
    A = dm.mass_voronoi(V, F)
    ui, *_ = ddg41.torus_exact(V, R, r)
    rr_ = np.hypot(V[:, 0], V[:, 1])
    N = np.c_[np.cos(ui) * V[:, 0] / rr_, np.cos(ui) * V[:, 1] / rr_, np.sin(ui)]
    gn = np.sum(gW * N, 1)
    errs.append(np.abs(gn / A - ddg41.torus_willmore_gradient(ui, R, r)).max())
    tans.append((np.linalg.norm(gW - gn[:, None] * N, axis=1) / A).max())
close("Example 41.5.16(b): normal part errors 0.122, 0.033, 0.0084", errs, [0.122, 0.033, 0.0084], 6e-4)
close("Example 41.5.16(b): tangent parts 0.056, 0.025, 0.012", tans, [0.056, 0.025, 0.012], 6e-4)
vel_in, vel_out = -ddg41.torus_willmore_gradient(np.pi) * -1, -ddg41.torus_willmore_gradient(0.0) * 1   # radial components (N = -e_rho, +e_rho)
close("text: inner equator toward the axis at 1.54, outer equator outward at 0.12", [vel_in, vel_out], [-1.54, 0.12], 0.0051)

# ---------------------------------------------------------------- Remark 41.5.17 (stiffness)
close("Remark 41.5.17: MCF limits 0.027, 0.0065, 0.0016", [2 / lam[k] for k in (2, 3, 4)], [0.027, 0.0065, 0.0016], 0.0005)
close("Remark 41.5.17: Willmore limits 7.4e-4, 4.2e-5, 2.5e-6", [4 / lam[k] ** 2 / s for k, s in ((2, 1e-4), (3, 1e-5), (4, 1e-6))],
      [7.4, 4.2, 2.5], 0.051)
check("Remark 41.5.17: 4 / lambda^2 is about 0.08 l^4", all(0.075 < 4 / (lam[k] * ell[k] ** 2) ** 2 < 0.095 for k in lam))
check("Remark 41.5.17: one refinement divides the limit by about 16", all(14 < (lam[k + 1] / lam[k]) ** 2 < 18 for k in (2, 3)))

# ---------------------------------------------------------------- Conventions: slides 36 and 38
V2, F2 = dm.icosphere(2)
Lx2, KN2 = dm.cotan_apply(V2, F2, V2), ddg41.gauss_curvature_normal(V2, F2)
tau = 1e-3
close("Conventions: slide 36 update raises the area 12.32985 -> 12.33361, correct sign 12.32608",
      [area(V2, F2), area(V2 - tau * Lx2, F2), area(V2 + tau * Lx2, F2)], [12.32985, 12.33361, 12.32608], 6e-6)
close("Conventions: slide 38 update raises (1/2) sum l theta 12.48719 -> 12.48816, correct sign 12.48623",
      [ddg41.half_sum_ltheta(V2, F2), ddg41.half_sum_ltheta(V2 + tau * KN2, F2), ddg41.half_sum_ltheta(V2 - tau * KN2, F2)],
      [12.48719, 12.48816, 12.48623], 6e-6)
check("Conventions: (L x)_i = (1/2) sum (cot + cot)(x_j - x_i) points inward on the sphere", np.all(np.sum(Lx2 * V2, 1) < 0))
check("Conventions: (K N)_i points outward on the sphere", np.all(np.sum(KN2 * V2, 1) > 0))

# ---------------------------------------------------------------- Remark 41.5.18 (the torus collapses to a circle)
Vt, Ft = dm.torus_mesh(R, r, 24, 48)
tt, k, recs = 0.0, 0, []
hh = 0.2 * 2 / ddg41.largest_eigenvalue(Vt, Ft)
while True:
    if k % 50 == 0:
        rho_ = np.hypot(Vt[:, 0], Vt[:, 1])
        Rc = rho_.mean()
        rt = np.hypot(rho_ - Rc, Vt[:, 2]).mean()
        recs.append((tt, Rc, rt, np.degrees(dm.corner_angles(Vt, Ft).min())))
        if rt < 0.08:
            break
        if k > 0:
            hh = 0.2 * 2 / ddg41.largest_eigenvalue(Vt, Ft, iters=60)
    Vt = Vt + hh * ddg41.mcf_velocity(Vt, Ft)
    tt += hh
    k += 1
recs = np.array(recs)
close("Remark 41.5.18: t = 0.083, 0.165, 0.243 and r^2 = 0.487, 0.331, 0.182", [recs[1:4, 0], recs[1:4, 2] ** 2],
      [[0.083, 0.165, 0.243], [0.487, 0.331, 0.182]], 6e-4)
close("Remark 41.5.18: cylinder estimate 0.474, 0.310, 0.155", 0.64 - 2 * recs[1:4, 0], [0.474, 0.310, 0.155], 1.1e-3)
close("Remark 41.5.18: at t = 0.333 the tube radius is 0.079 and the core radius 1.83", [recs[-1, 0], recs[-1, 2], recs[-1, 1]], [0.333, 0.079, 1.83], 6e-3)
close("Remark 41.5.18: smallest angle 29.8 -> 2.9 degrees", [recs[0, 3], recs[-1, 3]], [29.8, 2.9], 0.051)
summary()
