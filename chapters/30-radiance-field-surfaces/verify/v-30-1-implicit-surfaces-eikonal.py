"""Verification for Section 30.1 (Implicit Surfaces and the Eikonal Equation).

Run from the project root:
``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/verify/v-30-1-implicit-surfaces-eikonal.py``
(deterministic; a few seconds)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

x, y, z = sp.symbols("x y z", real=True)
X3 = sp.Matrix([x, y, z])
r = sp.symbols("r", positive=True)


def grad(f, V=X3):
    return sp.Matrix([sp.diff(f, v) for v in V])


def critical_points_on_zero_set(f, V):
    """points with f = 0 and grad f = 0 (sympy solve)."""
    eqs = [f] + list(grad(f, V))
    return sp.solve(eqs, list(V), dict=True)


# ---------------------------------------------------------------- Proposition 30.1.1, Example 30.1.2, Non-example 30.1.3
# (a) sphere: 0 is a regular value of |x|^2 - r^2; the two defining functions give the same N
f_sq = x ** 2 + y ** 2 + z ** 2 - r ** 2
check("Example 30.1.2: grad(|x|^2 - r^2) = 0 only at the origin, which is not on S",
      critical_points_on_zero_set(f_sq, X3) == [])
N1 = grad(f_sq) / sp.sqrt(sum(c ** 2 for c in grad(f_sq)))
f_sdf = sp.sqrt(x ** 2 + y ** 2 + z ** 2) - r
N2 = grad(f_sdf) / sp.sqrt(sum(c ** 2 for c in grad(f_sdf)))
rho = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
sym_equal("Example 30.1.2: N = grad f/|grad f| = x/|x| for both f = |x|^2 - r^2 and f = |x| - r", N1, X3 / rho)
sym_equal("Example 30.1.2: second defining function gives the same N", N2, X3 / rho)

# (b) cone x^2 + y^2 - z^2: the origin is a critical point on the zero set
f_cone = x ** 2 + y ** 2 - z ** 2
check("Non-example 30.1.3(a): cone x^2+y^2-z^2 has a critical point on f = 0 (the vertex)",
      critical_points_on_zero_set(f_cone, X3) == [{x: 0, y: 0, z: 0}])
# 2D cross-section of Figure 30.1.1(b)
X2 = sp.Matrix([x, y])
check("Figure 30.1.1(b): x^2 - y^2 has its only critical point at the origin, on the zero set",
      critical_points_on_zero_set(x ** 2 - y ** 2, X2) == [{x: 0, y: 0}])

# (c) two tangent spheres (circles in Figure 30.1.1(c))
f_kiss = ((x - 1) ** 2 + y ** 2 - 1) * ((x + 1) ** 2 + y ** 2 - 1)
sols = critical_points_on_zero_set(f_kiss, X2)
check("Non-example 30.1.3(b): two tangent circles: the only critical point on f = 0 is the contact point (0, 0)",
      sols == [{x: 0, y: 0}])

# (d) f^2: zero set is a circle but grad(f^2) = 2 f grad f vanishes on it
g = (x ** 2 + y ** 2 - 1) ** 2
th = sp.symbols("theta", real=True)
on_circle = {x: sp.cos(th), y: sp.sin(th)}
check("Non-example 30.1.3(c): grad((x^2+y^2-1)^2) = 0 at every point of the unit circle",
      all(sp.simplify(c.subs(on_circle)) == 0 for c in grad(g, X2)))
# numbers on Figure 30.1.1(d): |grad g| at r = 0.8 and 1.2 along a ray
gn = sp.lambdify((x, y), sp.sqrt(sum(c ** 2 for c in grad(g, X2))), modules="numpy")
close("Figure 30.1.1(d): |grad g| = 4 r |r^2 - 1| = 1.152 at r = 0.8", gn(0.8, 0.0), 4 * 0.8 * 0.36, tol=1e-12)
close("Figure 30.1.1(d): |grad g| = 2.112 at r = 1.2", gn(1.2, 0.0), 4 * 1.2 * 0.44, tol=1e-12)

# ---------------------------------------------------------------- Example 30.1.6: sphere SDF
gs = grad(f_sdf)
sym_equal("Example 30.1.6: |grad(|x| - r)|^2 = 1 away from the origin", sum(c ** 2 for c in gs), 1)
# along the normal line: f(y + t N(y)) = t for t > -r
t = sp.symbols("t", real=True)
p0 = sp.Matrix([0, 0, r])
expr = f_sdf.subs({x: 0, y: 0, z: r + t})
check("Proposition 30.1.5(b), Example 30.1.6: f(p + tN) = t for -r < t (p on the sphere, N = outward)",
      sp.simplify(expr.subs(t, sp.Rational(-1, 3) * r) - (-r / 3)) == 0 and sp.simplify(expr.subs(t, 2 * r) - 2 * r) == 0)

# ---------------------------------------------------------------- Example 30.1.6, Figure 30.1.2: ellipse SDF and medial axis
A, B = 2.0, 1.0


def ellipse_closest(P, n_init=256, iters=40):
    """closest point parameters on (A cos s, B sin s) for points P (n, 2): global by dense init + Newton."""
    s0 = np.linspace(0, 2 * np.pi, n_init, endpoint=False)
    E = np.stack([A * np.cos(s0), B * np.sin(s0)], 1)
    d2 = ((P[:, None, :] - E[None]) ** 2).sum(-1)
    s = s0[np.argmin(d2, 1)]
    for _ in range(iters):
        c, si = np.cos(s), np.sin(s)
        dx, dy = A * c - P[:, 0], B * si - P[:, 1]
        g1 = -A * si * dx + B * c * dy                     # (1/2) d/ds |e(s) - p|^2
        g2 = (A * si) ** 2 + (B * c) ** 2 - A * c * dx - B * si * dy
        s = s - g1 / np.where(np.abs(g2) > 1e-12, g2, 1e-12)
    return s


def ellipse_sdf(P):
    s = ellipse_closest(P)
    Q = np.stack([A * np.cos(s), B * np.sin(s)], 1)
    dist = np.linalg.norm(P - Q, axis=1)
    inside = P[:, 0] ** 2 / A ** 2 + P[:, 1] ** 2 / B ** 2 < 1
    return np.where(inside, -dist, dist)


c_med = (A ** 2 - B ** 2) / A
close("Example 30.1.6: medial axis of the ellipse a=2, b=1 is |x| <= (a^2-b^2)/a = 1.5 on y = 0", c_med, 1.5, tol=0)
# a point on the medial axis has two closest points (0.8, 0): symmetric pair
q = np.array([[0.8, 0.0]])
s_up = ellipse_closest(q + np.array([[0.0, 1e-9]]))[0]
s_dn = ellipse_closest(q - np.array([[0.0, 1e-9]]))[0]
Qu = np.array([A * np.cos(s_up), B * np.sin(s_up)])
Qd = np.array([A * np.cos(s_dn), B * np.sin(s_dn)])
close("Figure 30.1.2: q = (0.8, 0) has two closest points at the same distance", np.linalg.norm(Qu - q[0]),
      np.linalg.norm(Qd - q[0]), tol=1e-9)
check("Figure 30.1.2: ... and they are distinct (mirror images)", abs(Qu[1] + Qd[1]) < 1e-9 and abs(Qu[1]) > 0.3)
print(f"   closest points of q = (0.8, 0): ({Qu[0]:.3f}, {Qu[1]:.3f}), ({Qd[0]:.3f}, {Qd[1]:.3f}), "
      f"distance {np.linalg.norm(Qu - q[0]):.4f}")
# beyond the medial axis endpoint the closest point is unique: (1.7, 0) -> vertex (2, 0)
q2 = np.array([[1.7, 0.0]])
close("Example 30.1.6: (1.7, 0) is beyond the endpoint: unique closest point (2, 0), f = -0.3", ellipse_sdf(q2)[0], -0.3,
      tol=1e-9)
# |grad f| = 1 off the medial axis (finite differences), < 1 across it
h = 1e-5
for P in [(0.3, 0.5), (-1.0, -0.4), (2.5, 1.0), (0.0, -1.6), (1.6, 0.2)]:
    P = np.array(P, float)
    gx = (ellipse_sdf((P + [h, 0])[None])[0] - ellipse_sdf((P - [h, 0])[None])[0]) / (2 * h)
    gy = (ellipse_sdf((P + [0, h])[None])[0] - ellipse_sdf((P - [0, h])[None])[0]) / (2 * h)
    close(f"Proposition 30.1.5(c): |grad f| = 1 at ({P[0]}, {P[1]}) (off the medial axis)", np.hypot(gx, gy), 1.0, tol=1e-5)
# slice x = 0: f(0, y) = |y| - 1 (kink at y = 0 on the medial axis)
ys = np.linspace(-2, 2, 81)
close("Figure 30.1.2: f(0, y) = |y| - 1 along the minor axis", ellipse_sdf(np.stack([0 * ys, ys], 1)), np.abs(ys) - 1,
      tol=1e-9)
# central difference across the medial axis at (0.5, 0) in y: one-sided slopes +-sin, average 0
P = np.array([0.5, 0.0])
hy = 1e-4
gy_c = (ellipse_sdf((P + [0, hy])[None])[0] - ellipse_sdf((P - [0, hy])[None])[0]) / (2 * hy)
close("Figure 30.1.2(b): the central difference in y across the medial axis is 0 (symmetric kink)", gy_c, 0.0, tol=1e-6)

# ---------------------------------------------------------------- Figure 30.1.3: eikonal solutions that are not the SDF
xs = np.linspace(-2, 2, 400001) + 1.234567e-6       # offset: no grid point is exactly a zero
sdf1 = np.abs(xs) - 1
zig = np.where(np.abs(xs) >= 0.6, np.abs(xs) - 1, 0.2 - np.abs(xs))
unsigned = np.abs(np.abs(xs) - 1)
for name, fz in [("SDF |x|-1", sdf1), ("zigzag", zig), ("unsigned ||x|-1|", unsigned)]:
    d = np.diff(fz) / np.diff(xs)
    frac = np.mean(np.abs(np.abs(d) - 1) < 1e-9)
    check(f"Figure 30.1.3(a): {name}: |f'| = 1 except at finitely many kinks (fraction {frac:.6f})", frac > 0.99998)
    close(f"Figure 30.1.3(a): {name} vanishes at x = +-1", [fz[np.argmin(np.abs(xs - 1))], fz[np.argmin(np.abs(xs + 1))]],
          [0, 0], tol=1e-5)
zeros_zig = xs[np.where(np.diff(np.sign(zig)) != 0)[0]]
check("Figure 30.1.3(a): the zigzag has extra zeros at x = +-0.2 (four sign changes in total)",
      len(zeros_zig) == 4 and np.allclose(np.sort(np.abs(zeros_zig)), [0.2, 0.2, 1.0, 1.0], atol=1e-4))
# 2D radial: f = r - 1 (r >= 0.7), 0.4 - r (r < 0.7)
rr = np.linspace(0, 2, 200001) + 1.234567e-6
frad = np.where(rr >= 0.7, rr - 1, 0.4 - rr)
check("Figure 30.1.3(b): radial counterexample is continuous at r = 0.7", abs((0.7 - 1) - (0.4 - 0.7)) < 1e-15)
zr = rr[np.where(np.diff(np.sign(frad)) != 0)[0]]
check("Figure 30.1.3(b): zero set = circles r = 0.4 (spurious) and r = 1", np.allclose(zr, [0.4, 1.0], atol=1e-4))

# a smooth f with f < 0 inside a closed curve must have an interior critical point: eikonal error >= 1 there
f_smooth = lambda X, Y: 0.5 * (X ** 2 + Y ** 2 - 1)        # grad = x, zero set = unit circle
check("What geometry explains (smooth f): f = (|x|^2-1)/2 has |grad f| = 1 on S and grad f = 0 at the center",
      f_smooth(0, 0) == -0.5)
# eikonal loss of this f over the unit disk, uniform: E(|x| - 1)^2 = int_0^1 (r-1)^2 2r dr = 1/6
rs = sp.symbols("rs", positive=True)
check("What geometry explains (smooth f): E_disk(|grad f| - 1)^2 = 1/6 for f = (|x|^2 - 1)/2",
      sp.integrate((rs - 1) ** 2 * 2 * rs, (rs, 0, 1)) == sp.Rational(1, 6))

# ---------------------------------------------------------------- viscosity: 1D check of the two candidates
# u = 1 - |x| on (-1, 1): smooth test functions touching from above at 0 have |phi'(0)| <= 1 (subsolution OK),
# none touches from below (vacuous). u = |x| - 1: phi = const touches from below at 0 with |phi'| - 1 = -1 < 0 (fails).
check("Derivation (viscosity, 1D): for u = |x| - 1, the test function phi = -1 touches from below at 0 and |phi'| - 1 = -1 < 0",
      np.all(np.abs(np.linspace(-1, 1, 101)) - 1 >= -1))

# ---------------------------------------------------------------- IGR plane reproduction (linear model)
rng = np.random.default_rng(301)
n, dd, lam = 300, 3, 0.1
Y = rng.normal(size=(n, dd))
Y[:, 0] = 0.0
Xp = Y + 0.01 * rng.normal(size=(n, dd))
Cm = Xp.T @ Xp
ev, U = np.linalg.eigh(Cm)
w = rng.normal(size=dd)
check("IGR linear model: step size below 2/(largest Hessian eigenvalue)", 1e-3 < 2 / (2 * ev[-1] + 8 * lam))
for _ in range(40000):
    w -= 1e-3 * (2 * Cm @ w + 4 * lam * (w @ w - 1) * w)
close("IGR [Gropp20] Theorem 1-2 (linear model): GD converges to +-sqrt(1 - lambda_1/(2 lambda)) u_1",
      abs(w @ U[:, 0]), np.sqrt(1 - ev[0] / (2 * lam)), tol=1e-6)
close("IGR linear model: the minimizer is (approximately) the unit normal of the plane", abs(w[0]) / np.linalg.norm(w), 1.0,
      tol=1e-5)

# ---------------------------------------------------------------- (30.1.5)-(30.1.8): NeuS and VolSDF along a ray
s, f, beta, ct = sp.symbols("s f beta c", positive=True)
Phi = 1 / (1 + sp.exp(-s * f))
phi_s = sp.diff(Phi, f)
sym_equal("NeuS: phi_s = s Phi_s (1 - Phi_s) (logistic density)", phi_s, s * Phi * (1 - Phi))
# plane: f(p(t)) = -c (t - t*), c = |cos theta| ; rho = -(d/dt Phi_s(f))/Phi_s(f) = c phi_s(f)/Phi_s(f)
rho_neus = ct * phi_s / Phi
sym_equal("(30.1.8): NeuS opaque density for a plane = s |cos theta| Phi_s(-f)", rho_neus, s * ct / (1 + sp.exp(s * f)))

tt = np.linspace(0.0, 2.0, 400001)
dt = tt[1] - tt[0]
tstar = 1.0


def render_weights(sig):
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * dt)])
    T = np.exp(-tau)
    return T * sig, T


for cos_th in (1.0, 0.5):
    fr = cos_th * (tstar - tt)
    for sv in (10.0, float(np.exp(3.0)), 64.0):
        Ph = 1 / (1 + np.exp(-sv * fr))
        w_neus, T = render_weights(sv * cos_th * (1 - Ph))
        close(f"NeuS (30.1.8): T(t) = Phi_s(f(p(t)))/Phi_s(f(p(0))) (s = {sv:.1f}, cos = {cos_th})", T, Ph / Ph[0], tol=1e-8)
        close(f"NeuS Theorem 1: weight peaks at t* (s = {sv:.1f}, cos = {cos_th})", tt[np.argmax(w_neus)], tstar,
              tol=2 * dt)
        w_naive, _ = render_weights(sv * Ph * (1 - Ph))
        # naive sigma = phi_s(f) (no |cos| factor): w'/w = sigma'/sigma - sigma = 0 gives, with P = Phi_s(f) and
        # c = |cos|, c (2P - 1) = P (1 - P), i.e. P^2 + (2c - 1) P - c = 0; the peak is at f* = logit(P)/s > 0,
        # t = t* - f*/c (in front of the surface). Head-on (c = 1): P^2 + P - 1 = 0, f* = ln(golden ratio)/s.
        Pn = (-(2 * cos_th - 1) + np.sqrt((2 * cos_th - 1) ** 2 + 4 * cos_th)) / 2
        t_naive = tstar - np.log(Pn / (1 - Pn)) / (sv * cos_th)
        close(f"naive sigma = phi_s(f): weight peaks at t* - logit(P)/(s cos), P^2 + (2cos - 1)P - cos = 0 "
              f"(s = {sv:.1f}, cos = {cos_th})", tt[np.argmax(w_naive)], t_naive, tol=3 * dt)
        if cos_th == 1.0:
            close(f"naive head-on: t* - ln((1 + sqrt5)/2)/s (s = {sv:.1f})", t_naive,
                  tstar - np.log((1 + np.sqrt(5)) / 2) / sv, tol=1e-12)
close("NeuS init: s = exp(10 * 0.3) = 20.09", np.exp(3.0), 20.0855, tol=1e-4)
# naive sigma = phi_s(f): the head-on ray never becomes opaque: total weight 1 - 1/e
fr1 = tstar - tt
Ph1 = 1 / (1 + np.exp(-10.0 * fr1))
w_naive1, _ = render_weights(10.0 * Ph1 * (1 - Ph1))
close("Figure 30.1.4: naive sigma = phi_s(f): total weight on a head-on ray = 1 - 1/e = 0.632", w_naive1.sum() * dt,
      1 - np.exp(-1), tol=1e-4)

def volsdf_peak_offset(c, bv):
    """t_peak - t* for VolSDF on a plane, ray with |cos| = c (derived in the text, Exercise-level algebra).

    inside (c >= 1/2): with y = exp(-x)/2, x = (t - t*) c / beta, the peak solves c y = (1 - y)^2;
    outside (c < 1/2): the peak is at x = ln(2c) < 0.
    """
    if c >= 0.5:
        yv = ((2 + c) - np.sqrt((2 + c) ** 2 - 4)) / 2
        return -np.log(2 * yv) * bv / c
    return np.log(2 * c) * bv / c


for cos_th in (1.0, 0.8, 0.5, 0.3):
    for bv in (0.1, 0.05):
        dv = cos_th * (tstar - tt)                  # SDF along the ray, positive outside
        psi = np.where(-dv <= 0, 0.5 * np.exp(-dv / bv), 1 - 0.5 * np.exp(dv / bv))
        w_vol, _ = render_weights(psi / bv)
        close(f"VolSDF: weight peak offset (beta = {bv}, |cos| = {cos_th}) = {volsdf_peak_offset(cos_th, bv) / bv:+.4f} beta",
              tt[np.argmax(w_vol)] - tstar, volsdf_peak_offset(cos_th, bv), tol=2 * dt)
close("VolSDF: head-on offset = -ln(3 - sqrt 5) beta = +0.2693 beta", volsdf_peak_offset(1.0, 1.0), -np.log(3 - np.sqrt(5)),
      tol=1e-12)
close("VolSDF: offset is exactly 0 at |cos| = 1/2 (60 deg)", volsdf_peak_offset(0.5, 1.0), 0.0, tol=1e-12)
check("VolSDF: offset is negative (peak in front of the surface) for |cos| < 1/2", volsdf_peak_offset(0.3, 1.0) < 0)
close("VolSDF bias constant -ln(3 - sqrt 5) = 0.2693", -np.log(3 - np.sqrt(5)), 0.2693, tol=1e-4)

# official VolSDF code (density.py): alpha * (0.5 + 0.5 sign(sdf) expm1(-|sdf|/beta)), alpha = 1/beta  ==  Psi_beta(-sdf)/beta
sdf_s = np.linspace(-1, 1, 2001)
code = (1 / 0.1) * (0.5 + 0.5 * np.sign(sdf_s) * np.expm1(-np.abs(sdf_s) / 0.1))
paper = np.where(-sdf_s <= 0, 0.5 * np.exp(-sdf_s / 0.1), 1 - 0.5 * np.exp(sdf_s / 0.1)) / 0.1
close("VolSDF code density_func == (1/beta) Psi_beta(-d) of (30.1.6)", code, paper, tol=1e-12)
# official NeuS code: alpha = (Phi(prev) - Phi(next))/Phi(prev), clipped to [0,1] == 1 - exp(-int rho) for a plane
sv, cos_th = 20.0, 0.7
f_prev, f_next = 0.03, 0.03 - cos_th * 0.01
a_code = (1 / (1 + np.exp(-sv * f_prev)) - 1 / (1 + np.exp(-sv * f_next))) / (1 / (1 + np.exp(-sv * f_prev)))
ff = np.linspace(f_prev, f_next, 20001)
integral = np.trapezoid(sv * cos_th / (1 + np.exp(sv * ff)), dx=0.01 / 20000)
close("NeuS code alpha_i = (Phi(f_i) - Phi(f_{i+1}))/Phi(f_i) = 1 - exp(-int rho dt) on a plane", a_code,
      1 - np.exp(-integral), tol=1e-8)
# VolSDF: sigma -> alpha 1_Omega as beta -> 0
close("VolSDF: beta * sigma -> 1 inside, 0 outside as beta -> 0", [1 - 0.5 * np.exp(-0.2 / 1e-3), 0.5 * np.exp(-0.2 / 1e-3)],
      [1, 0], tol=1e-12)

# ---------------------------------------------------------------- TSDF along the line of sight (Exercise 30.1.4)
th_ = np.radians(60.0)
n_pl = np.array([0.0, 0.0, 1.0])
ray = np.array([np.sin(th_), 0.0, -np.cos(th_)])            # angle 60 deg with -n
Xpt = np.array([0.3, 0.2, 0.5])                               # above the plane z = 0
t_hit = -Xpt[2] / ray[2]
d_ray = t_hit
close("Exercise 30.1.4: distance along the ray = Euclidean distance / cos(theta) = 0.5/0.5 = 1.0", d_ray, 0.5 / np.cos(th_),
      tol=1e-12)
# gradient of the along-ray distance field D(x) = z / cos(theta): |grad D| = 1/cos(theta) = 2
close("Exercise 30.1.4: |grad D| = 1/cos(theta) = 2 for theta = 60 deg", 1 / np.cos(th_), 2.0, tol=1e-12)

# ---------------------------------------------------------------- Exercise 30.1.5: chord vs geodesic on the unit sphere
ang = np.radians([60.0, 180.0])
close("Exercise 30.1.5: chord 2 sin(theta/2) for 60 deg = 1", 2 * np.sin(ang[0] / 2), 1.0, tol=1e-12)
close("Exercise 30.1.5: antipodal: chord 2, geodesic pi", [2 * np.sin(ang[1] / 2), ang[1]], [2.0, np.pi], tol=1e-12)
close("Exercise 30.1.5: geodesic 60 deg = pi/3 = 1.047", ang[0], 1.0472, tol=1e-4)

# ---------------------------------------------------------------- Exercise 30.1.1: x^2 + y^2 - z^2 + c
cc = sp.symbols("c", real=True)
crit = sp.solve(list(grad(x ** 2 + y ** 2 - z ** 2 + cc)), [x, y, z], dict=True)
check("Exercise 30.1.1: the only critical point is the origin, with value c", crit == [{x: 0, y: 0, z: 0}])

summary()
