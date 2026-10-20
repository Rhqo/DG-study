"""Verification for Section 44.6 (Geodesic Distance via the Heat Method).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-6-heat-method.py
"""
import sys
from pathlib import Path

import dgmesh as dm
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-6-heat-method.html"
rng = np.random.default_rng(4)


def errs(phi, ex, m):
    e = np.abs(phi - ex)
    return float(e.max()), float(m @ e / m.sum())


# ---------------------------------------------------------------- Implementation 44.6
for bid in ("code-44-6-1", "code-44-6-2"):
    check(f"Implementation 44.6: {bid} at most 30 lines", dgcode.code_lines(PAGE, bid) <= 30)
ns, _ = ch44.run_printed(PAGE, "code-44-6-1")
ns, out = ch44.run_printed(PAGE, "code-44-6-2", ns)
check("Implementation 44.6: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-44-6-2").strip(), out)
for lev in (2, 3):
    V, F = dm.icosphere(lev)
    s = int(np.argmax(V[:, 2]))
    close(f"Implementation 44.6: agrees with the prefactored lib HeatMethod (icosphere({lev}))",
          ns["heat_method"](V, F, [s]), ch44.HeatMethod(V, F).distance(s), 1e-10)

# ---------------------------------------------------------------- Proposition 44.6.1, Non-example 44.6.2
p0 = np.array([0, 0, 1.0])
ok = True
for _ in range(20):
    x = ch44.unit(rng.normal(size=3))
    if x[2] > 0.95 or x[2] < -0.95:
        continue
    g = -ch44.sphere_log(x, p0) / np.linalg.norm(ch44.sphere_log(x, p0))   # grad of d(., p0) = -log/|log|
    ok &= abs(np.linalg.norm(g) - 1) < 1e-12
    w = ch44.unit(np.cross(x, rng.normal(size=3)))
    h = 1e-6
    fd = (np.arccos(np.clip(ch44.unit(x + h * w) @ p0, -1, 1)) - np.arccos(np.clip(ch44.unit(x - h * w) @ p0, -1, 1))) / (2 * h)
    ok &= abs(fd - g @ w) < 1e-6
check("Prop 44.6.1: on the unit sphere |grad d(., p)| = 1 and d(d(., p))(w) = <grad, w>", ok)
xs = np.linspace(0, 1, 100001)
psi = np.minimum(np.minimum(xs, np.abs(xs - 0.5)), 1 - xs)
dpsi = np.abs(np.diff(psi) / np.diff(xs))
check("Non-example 44.6.2: |psi'| = 1 away from the kinks and psi(0) = psi(1) = 0, but psi != min(x, 1 - x)",
      np.mean(np.abs(dpsi - 1) < 1e-6) > 0.999 and psi[0] == 0 and psi[-1] == 0
      and np.abs(psi - np.minimum(xs, 1 - xs)).max() > 0.2)

# ---------------------------------------------------------------- Example 44.6.3, Exercise 44.6.2
t = 0.01
x = np.array([0.6, 0.8])
k = lambda y: np.exp(-(y @ y) / (4 * t)) / (4 * np.pi * t)  # noqa: E731
gk = np.array([(k(x + 1e-7 * e) - k(x - 1e-7 * e)) / 2e-7 for e in np.eye(2)])
close("Example 44.6.3: -grad k_t/|grad k_t| = x/|x| (plane)", -gk / np.linalg.norm(gk), x / np.linalg.norm(x), 1e-6)
close("Exercise 44.6.2: Varadhan estimate 0.9576 at t = 0.01, |x| = 1 (error 0.0424)",
      [np.sqrt(-4 * t * np.log(k(x))), 1 - np.sqrt(-4 * t * np.log(k(x)))], [0.9576, 0.0424], 5e-5)
ts = [1e-2, 1e-3, 1e-4]
logk = lambda tt: -1.0 / (4 * tt) - np.log(4 * np.pi * tt)  # noqa: E731  (|x| = 1, log of the kernel in closed form)
check("Example 44.6.3: -4t log k_t -> |x|^2 = 1 as t -> 0", np.all(np.diff([abs(-4 * tt * logk(tt) - 1) for tt in ts]) < 0)
      and abs(-4 * 1e-4 * logk(1e-4) - 1) < 1e-2)

# ---------------------------------------------------------------- Proposition 44.6.5
ok_a = True
for _ in range(200):
    P = rng.normal(size=(3, 3))
    Xf = rng.normal(size=3)
    n = ch44.unit(np.cross(P[1] - P[0], P[2] - P[0]))
    Xf -= (Xf @ n) * n
    G = dd.barycentric_gradients(P)
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    ang = dm.corner_angles(P, np.array([[0, 1, 2]]))[0]
    for c in range(3):
        i, j, kk = c, (c + 1) % 3, (c + 2) % 3
        lhs = -A * (Xf @ G[i])
        rhs = 0.5 * (Xf @ (P[j] - P[i]) / np.tan(ang[kk]) + Xf @ (P[kk] - P[i]) / np.tan(ang[j]))
        ok_a &= abs(lhs - rhs) < 1e-9 * max(1, abs(lhs))
check("Prop 44.6.5(a): -A <X, grad lambda_i> = (1/2)(cot theta_k <e_ij, X> + cot theta_j <e_ik, X>) (200 random triangles)", ok_a)
# (b) flux through the barycentric and the circumcentric dual pieces (planar acute and obtuse triangles)
for P2 in (np.array([[0, 0], [1.0, 0], [0.4, 0.8]]), np.array([[0, 0], [1.0, 0], [0.2, 0.25]])):
    P = np.c_[P2, np.zeros(3)]
    Xf = np.array([0.3, -0.7, 0.0])
    G = dd.barycentric_gradients(P)
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    m_ij, m_ik = (P[0] + P[1]) / 2, (P[0] + P[2]) / 2
    for name, cpt in (("barycentric", P.mean(0)), ("circumcentric", dd.circumcenters(P, np.array([[0, 1, 2]]))[0])):
        flux = 0.0
        for a, b in ((m_ij, cpt), (cpt, m_ik)):              # polyline m_ij -> c -> m_ik, normal away from x_0
            e = b - a
            nrm = -np.array([-e[1], e[0], 0.0])               # right normal of the direction (x_0 is on the left)
            flux += Xf @ nrm
        close(f"Prop 44.6.5(b): flux through the {name} dual piece = -A <X, grad lambda_i> "
              f"({'acute' if P2[2, 1] > 0.5 else 'obtuse'})", flux, -A * (Xf @ G[0]), 1e-12)
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus_mesh(2, 0.8, 8, 12)", dm.torus_mesh(2, 0.8, 8, 12))):
    X = rng.normal(size=(len(F), 3))
    Nf = dm.face_normals(V, F)
    X -= np.einsum("ij,ij->i", X, Nf)[:, None] * Nf
    b = ch44.integrated_divergence(V, F, X)
    close(f"Prop 44.6.5: cot formula = weak divergence ({name})", b, ch44.weak_divergence(V, F, X), 1e-12)
    close(f"Prop 44.6.5(c): sum_i b_i = 0 ({name})", b.sum(), 0, 1e-11)
    w = rng.normal(size=len(V))
    close(f"Prop 44.6.5(d): X = grad w gives b = L w ({name})",
          ch44.integrated_divergence(V, F, ch44.face_gradients(V, F, w)), dm.cotan_laplacian(V, F) @ w, 1e-11)
# Proposition 44.6.6: the least-squares fit
V, F = dm.icosphere(2)
X = rng.normal(size=(len(F), 3))
Nf = dm.face_normals(V, F)
X -= np.einsum("ij,ij->i", X, Nf)[:, None] * Nf
A_f = dm.face_areas(V, F)
G_all = np.zeros((3 * len(F), len(V)))
for f, tri in enumerate(F):
    Gt = dd.barycentric_gradients(V[tri])
    for c in range(3):
        G_all[3 * f:3 * f + 3, tri[c]] += np.sqrt(A_f[f]) * Gt[c]
rhs = (np.sqrt(A_f)[:, None] * X).ravel()
phi_ls = np.linalg.lstsq(G_all, rhs, rcond=None)[0]
L = dm.cotan_laplacian(V, F)
b = ch44.integrated_divergence(V, F, X)
phi_lb = np.linalg.lstsq(L, b, rcond=None)[0]
close("Prop 44.6.6: argmin int |grad phi - X|^2 solves L phi = b (up to constants)",
      (phi_ls - phi_ls.mean()), (phi_lb - phi_lb.mean()), 1e-9)
close("Prop 44.6.6: the A-weighted Gram matrix is -L", G_all.T @ G_all, -L, 1e-12)
# Exercise 44.6.1
P = np.array([[0, 0, 0], [1.0, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
Xf = np.array([1.0, 0, 0])
ang = dm.corner_angles(P, np.array([[0, 1, 2]]))[0]
vals = [0.5 * (Xf @ (P[(c + 1) % 3] - P[c]) / np.tan(ang[(c + 2) % 3]) + Xf @ (P[(c + 2) % 3] - P[c]) / np.tan(ang[(c + 1) % 3]))
        for c in range(3)]
close("Exercise 44.6.1: corner contributions sqrt3/4, -sqrt3/4, 0", vals, [np.sqrt(3) / 4, -np.sqrt(3) / 4, 0], 1e-12)

# ---------------------------------------------------------------- Examples 44.6.7, 44.6.8, 44.6.11
ico_heat, ico_dij, ico_ell = [], [], []
for lev in (1, 2, 3, 4):
    V, F = dm.icosphere(lev)
    s = int(np.argmax(V[:, 2]))
    ex = np.arccos(np.clip(V @ V[s], -1, 1))
    H = ch44.HeatMethod(V, F)
    ico_heat.append(errs(H.distance(s), ex, H.m))
    ico_dij.append(errs(ch44.dijkstra(V, F, s)[0], ex, H.m))
    ico_ell.append(H.ell)
close("Example 44.6.7: mean edge lengths 0.582, 0.299, 0.151, 0.075", ico_ell, [0.582, 0.299, 0.151, 0.075], 6e-4)
close("Example 44.6.7: heat mean errors 0.083, 0.038, 0.021, 0.014", [e[1] for e in ico_heat], [0.083, 0.038, 0.021, 0.014], 6e-4)
close("Example 44.6.7: heat max errors 0.187, 0.090, 0.050, 0.033", [e[0] for e in ico_heat], [0.187, 0.090, 0.050, 0.033], 6e-4)
close("Example 44.6.7: Dijkstra mean errors 0.064, 0.078, 0.085, 0.087", [e[1] for e in ico_dij], [0.064, 0.078, 0.085, 0.087], 6e-4)
cube_heat, cube_dij, cube_ell = [], [], []
corners = np.array([[a, b_, c] for a in (0, 1) for b_ in (0, 1) for c in (0, 1)], float)
for n in (4, 8, 16):
    V, F = ch44.unit_cube(n)
    s = int(np.argmin(np.linalg.norm(V - [0.5, 0.5, 0], axis=1)))
    ex = ch44.cube_distance_vec(V[s], V)[0]
    H = ch44.HeatMethod(V, F)
    phi = H.distance(s)
    cube_heat.append(errs(phi, ex, H.m))
    cube_dij.append(errs(ch44.dijkstra(V, F, s)[0], ex, H.m))
    cube_ell.append(H.ell)
    imax = int(np.argmax(np.abs(phi - ex)))
    check(f"Example 44.6.8: cube({n}): the max error sits at a top corner (exact distance 1.581)",
          np.linalg.norm(corners - V[imax], axis=1).min() < 1e-12 and V[imax][2] == 1 and abs(ex[imax] - np.sqrt(2.5)) < 1e-12)
close("Example 44.6.8: heat mean 0.032, 0.020, 0.012", [e[1] for e in cube_heat], [0.032, 0.020, 0.012], 6e-4)
close("Example 44.6.8: heat max 0.127, 0.090, 0.062", [e[0] for e in cube_heat], [0.127, 0.090, 0.062], 6e-4)
close("Example 44.6.8: Dijkstra mean 0.076, 0.082, 0.084", [e[1] for e in cube_dij], [0.076, 0.082, 0.084], 6e-4)
close("Example 44.6.8: Dijkstra max 0.293, 0.298, 0.310", [e[0] for e in cube_dij], [0.293, 0.298, 0.310], 6e-4)
sl_i = np.polyfit(np.log(ico_ell[1:]), np.log([e[1] for e in ico_heat[1:]]), 1)[0]
sl_c = np.polyfit(np.log(cube_ell), np.log([e[1] for e in cube_heat]), 1)[0]
close("Example 44.6.11: fitted slopes about 0.73 (icosphere(2)-(4), cube)", [sl_i, sl_c], [0.73, 0.73], 0.02)

# ---------------------------------------------------------------- Example 44.6.9 (torus) and Exercise 44.6.3
nu, nv = 32, 80
V, F = dm.torus_mesh(2, 0.8, nu, nv)
js = np.arange(1, nv // 8 + 1)
vv = 2 * np.pi * js / nv
out_err = {}
for mt in (1, 4):
    H = ch44.HeatMethod(V, F, m_t=mt)
    out_err[mt] = np.abs(H.distance(0)[js] - 2.8 * vv).max()
    if mt == 1:
        in_err = np.abs(H.distance((nu // 2) * nv)[(nu // 2) * nv + js] - 1.2 * vv).max()
        tval = H.t
close("Example 44.6.9: t = 0.0322; outer-equator errors 0.121 (m = 1), 0.071 (m = 4); inner 0.026",
      [tval, out_err[1], out_err[4], in_err], [0.0322, 0.121, 0.071, 0.026], 6e-4)
check("Example 44.6.9: the heat method underestimates on the outer equator", np.all(ch44.HeatMethod(V, F).distance(0)[js] < 2.8 * vv))
V48, F48 = dm.torus_mesh(2, 0.8, 48, 120)
js48 = np.arange(1, 120 // 8 + 1)
close("Example 44.6.9: 48 x 120 mesh, m = 1: outer-equator error 0.110",
      np.abs(ch44.HeatMethod(V48, F48).distance(0)[js48] - 2.8 * 2 * np.pi * js48 / 120).max(), 0.110, 6e-4)
# the smooth geodesics from x(0, 0): the only one reaching x(0, pi/4) within length 3 is the equator


def rhs(Y, R=2.0, r=0.8):
    u, v, du, dv = Y
    rho = R + r * np.cos(u)
    return np.array([du, dv, -(rho * np.sin(u) / r) * dv ** 2, 2 * r * np.sin(u) / rho * du * dv])


psis = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 4001)
Y = np.array([0 * psis, 0 * psis, np.sin(psis) / 0.8, np.cos(psis) / 2.8])
hstep, tt, vt = 0.002, 0.0, np.pi / 4
hits = []
while tt < 3.0:
    k1 = rhs(Y)
    k2 = rhs(Y + hstep / 2 * k1)
    k3 = rhs(Y + hstep / 2 * k2)
    k4 = rhs(Y + hstep * k3)
    Yn = Y + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    cross = ((Y[1] - vt) * (Yn[1] - vt) <= 0) & (Y[1] != Yn[1])
    if cross.any():
        lam = (vt - Y[1][cross]) / (Yn[1][cross] - Y[1][cross])
        ucr = Y[0][cross] + lam * (Yn[0][cross] - Y[0][cross])
        uw = (ucr + np.pi) % (2 * np.pi) - np.pi
        for L_, uu_, ps_ in zip(tt + lam * hstep, uw, psis[cross]):
            if abs(uu_) < 0.02:
                hits.append((L_, ps_))
    Y, tt = Yn, tt + hstep
check("Example 44.6.9: geodesics from x(0, 0) reach x(0, pi/4) within length 3 only along the equator (length 2.1991)",
      len(hits) > 0 and all(abs(L_ - 2.8 * vt) < 1e-3 and abs(ps_) < 0.02 for L_, ps_ in hits), str(hits[:3]))
close("Exercise 44.6.3: threshold pi sqrt(r/(R + r)) = 1.679", np.pi * np.sqrt(0.8 / 2.8), 1.679, 5e-4)

# ---------------------------------------------------------------- Example 44.6.10 (time step)
Vc, Fc = ch44.unit_cube(16)
s = int(np.argmin(np.linalg.norm(Vc - [0.5, 0.5, 0], axis=1)))
ex = ch44.cube_distance_vec(Vc[s], Vc)[0]
mc = dm.mass_voronoi(Vc, Fc)
sweep = [errs(ch44.HeatMethod(Vc, Fc, m_t=mm).distance(s), ex, mc)[1] for mm in (1 / 16, 1 / 4, 1, 4, 16)]
close("Example 44.6.10: cube(16) mean errors 0.127, 0.047, 0.012, 0.020, 0.027", sweep, [0.127, 0.047, 0.012, 0.020, 0.027], 6e-4)
V4, F4 = dm.icosphere(4)
s4 = int(np.argmax(V4[:, 2]))
H16 = ch44.HeatMethod(V4, F4, m_t=16)
close("Example 44.6.10: icosphere(4) at m = 16: mean error 0.0078",
      errs(H16.distance(s4), np.arccos(np.clip(V4 @ V4[s4], -1, 1)), H16.m)[1], 0.0078, 6e-5)

# ---------------------------------------------------------------- Example 44.6.12 (boundary)
xs_ = np.linspace(0, 1, 33)
Vg, Fg = dm.grid_mesh(xs_, xs_)
sg = int(np.argmin(np.linalg.norm(Vg - [0.25, 0.375, 0], axis=1)))
exg = np.linalg.norm(Vg - Vg[sg], axis=1)
mg = dm.mass_voronoi(Vg, Fg)
vals = {mm: [errs(ch44.HeatMethod(Vg, Fg, m_t=mm, bc=bc).distance(sg), exg, mg)[1] for bc in ("neumann", "dirichlet", "average")]
        for mm in (1, 16)}
close("Example 44.6.12: m = 1: Neumann 0.0060, Dirichlet 0.0170, average 0.0070", vals[1], [0.0060, 0.0170, 0.0070], 6e-5)
close("Example 44.6.12: m = 16: Neumann 0.0334, Dirichlet 0.0372, average 0.0029", vals[16], [0.0334, 0.0372, 0.0029], 6e-5)
close("Example 44.6.12: Dijkstra mean error 0.025", errs(ch44.dijkstra(Vg, Fg, sg)[0], exg, mg)[1], 0.025, 6e-4)
close("Figure 44.6.3: average at m = 8 is 0.0021",
      errs(ch44.HeatMethod(Vg, Fg, m_t=8, bc="average").distance(sg), exg, mg)[1], 0.0021, 6e-5)
loop = dm.boundary_loop(Fg)
eb = {bc: np.abs(ch44.HeatMethod(Vg, Fg, m_t=16, bc=bc).distance(sg) - exg)[loop] for bc in ("neumann", "dirichlet", "average")}
Pl = Vg[loop][:, :2]
near = lambda c: np.linalg.norm(Pl - np.array(c, float), axis=1) < 0.1  # noqa: E731
close("Figure 44.6.3(a): Neumann / Dirichlet maxima near (1, 1): 0.106, 0.145; Dirichlet near (1, 0): 0.105",
      [eb["neumann"][near((1, 1))].max(), eb["dirichlet"][near((1, 1))].max(), eb["dirichlet"][near((1, 0))].max()],
      [0.106, 0.145, 0.105], 6e-4)
close("Figure 44.6.3(a): average: max 0.024 (at (0, 0)), 91% of boundary vertices below 0.01",
      [eb["average"].max(), np.mean(eb["average"] < 0.01)], [0.024, 0.91], 6e-3)
check("Figure 44.6.3(a): the average's maximum sits at the corner (0, 0)", np.linalg.norm(Pl[np.argmax(eb["average"])]) < 1e-12)
# Proposition 44.6.5(c) also on a mesh with boundary (the proof never uses closedness)
Xg = rng.normal(size=(len(Fg), 3))
Xg[:, 2] = 0
close("Prop 44.6.5(c): sum_i b_i = 0 on the square grid (mesh with boundary)", ch44.integrated_divergence(Vg, Fg, Xg).sum(), 0, 1e-12)

# ---------------------------------------------------------------- Exercises 44.6.4, 44.6.5
for name, (V, F) in (("icosphere(3)", dm.icosphere(3)), ("grid 16 x 16", dm.grid_mesh(np.linspace(0, 1, 17), np.linspace(0, 1, 17)))):
    w_ok = all(w >= -1e-14 for w in dm.cotan_weights(V, F).values())
    u = ch44.HeatMethod(V, F).heat(0)
    check(f"Exercise 44.6.4: {name}: cotan weights >= 0 and u > 0 everywhere", w_ok and u.min() > 0)
V, F = dm.icosphere(2)
w = rng.normal(size=len(V))
L = dm.cotan_laplacian(V, F)
m = dm.mass_voronoi(V, F)
bb = ch44.integrated_divergence(V, F, ch44.face_gradients(V, F, w))
phi = np.linalg.solve(-L + np.outer(m, m), -bb)
close("Exercise 44.6.5(a): step III reproduces w - min w from X = grad w", phi - phi.min(), w - w.min(), 1e-10)

# ---------------------------------------------------------------- Exercise 44.6.6 (Coding)
check("Exercise 44.6.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-6-6") <= 30)
ns6, out6 = ch44.run_printed(PAGE, "code-ex-44-6-6")
check("Exercise 44.6.6: printed output matches the page", out6.strip() == ch44.printed_block(PAGE, "out-ex-44-6-6").strip(), out6)
summary()
