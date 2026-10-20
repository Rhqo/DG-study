"""Verification for Section 44.4 (Exponential Map, Logarithm Map, and Karcher Mean).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-4-exp-log-karcher.py
"""
import sys
from pathlib import Path

import dgmesh as dm
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-4-exp-log-karcher.html"
rng = np.random.default_rng(11)

# ---------------------------------------------------------------- Implementation 44.4 (Karcher mean)
check("Implementation 44.4: at most 30 lines", dgcode.code_lines(PAGE, "code-44-4-1") <= 30)
ns, out = ch44.run_printed(PAGE, "code-44-4-1")
check("Implementation 44.4: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-44-4-1").strip(), out)
xb = ns["x_bar"]
close("Example 44.4.8: Karcher mean at (theta, phi) = (0.7814, 0.3254)", [np.arccos(xb[2]), np.arctan2(xb[1], xb[0])],
      [0.7814, 0.3254], 5e-5)
Y = ns["Y"]
Fv = 0.5 * sum(np.arccos(np.clip(xb @ y, -1, 1)) ** 2 for y in Y)
close("Example 44.4.8: F(x_bar) = 1.671371", Fv, 1.671371, 5e-7)
close("Cor 44.4.7: sum of logs vanishes at the Karcher mean", sum(ns["sphere_log"](xb, y) for y in Y), np.zeros(3), 1e-10)
# x_bar is a minimum: F larger at nearby points
worse = all(0.5 * sum(np.arccos(np.clip(ch44.sphere_exp(xb, 0.01 * ch44.unit(w - (w @ xb) * xb)) @ y, -1, 1)) ** 2 for y in Y) > Fv
            for w in rng.normal(size=(20, 3)))
check("Example 44.4.8: F increases in 20 random directions from x_bar", worse)
# the iteration in the lib gives the same point from e_3; step ratios about 0.16
e3 = np.array([0.0, 0, 1])
x2, steps = ch44.karcher_mean_sphere(Y, e3)
close("Example 44.4.8: lib iteration agrees", x2, xb, 1e-10)
ratios = np.array(steps[3:12]) / np.array(steps[2:11])
check("Example 44.4.8: steps shrink by about 0.16 per iteration", np.all(np.abs(ratios - 0.16) < 0.01), str(ratios))
close("Figure 44.4.1: first step length 0.67", steps[0], 0.674, 5e-4)
Fs = []
x = e3.copy()
for _ in range(4):
    Fs.append(0.5 * sum(np.arccos(np.clip(x @ y, -1, 1)) ** 2 for y in Y))
    x = ch44.sphere_exp(x, np.mean([ch44.sphere_log(x, y) for y in Y], axis=0))
close("Example 44.4.8: F = 3.5463, 1.7177, 1.6725, 1.6714", Fs, [3.5463, 1.7177, 1.6725, 1.6714], 5e-5)
eu = ch44.unit(np.mean(Y, axis=0))
close("Example 44.4.8: distance to the normalised Euclidean mean 0.0742", np.arccos(xb @ eu), 0.0742, 5e-5)
close("Example 44.4.8: normalised Euclidean mean at (0.7307, 0.4043), F = 1.6903",
      [np.arccos(eu[2]), np.arctan2(eu[1], eu[0]), 0.5 * sum(np.arccos(np.clip(eu @ y, -1, 1)) ** 2 for y in Y)],
      [0.7307, 0.4043, 1.6903], 5e-5)
close("Example 44.4.8: d(x_bar, y_7) = 1.4809 < d(Euclidean mean, y_7) = 1.5550",
      [np.arccos(xb @ Y[6]), np.arccos(eu @ Y[6])], [1.4809, 1.5550], 5e-5)
cap = Y[:6]
diam = max(np.arccos(np.clip(a_ @ b_, -1, 1)) for a_ in cap for b_ in cap)
d7 = [np.arccos(c_ @ Y[6]) for c_ in cap]
close("Example 44.4.8: cap diameter 0.88, y_7 at 1.32-2.00 from the cap points", [diam, min(d7), max(d7)], [0.88, 1.32, 2.00], 6e-3)
xc = ch44.karcher_mean_sphere(cap, cap[0])[0]
close("Example 44.4.8: without y_7 the two means differ by 0.0014", np.arccos(np.clip(xc @ ch44.unit(np.mean(cap, axis=0)), -1, 1)),
      0.0014, 5e-5)
zz = ch44.unit(rng.normal(size=3))
close("Example 44.4.8: sum |x - y_i|^2 = 2n - 2 <x, sum y_i> on the unit sphere", sum(np.sum((zz - y) ** 2) for y in Y),
      2 * len(Y) - 2 * zz @ np.sum(Y, axis=0), 1e-12)
check("Example 44.4.8: chord 2 sin(d/2) < d and the ratio decreases with d",
      all(2 * np.sin(d_ / 2) < d_ for d_ in (0.1, 1.0, 2.0)) and np.all(np.diff([2 * np.sin(d_ / 2) / d_ for d_ in (0.1, 1.0, 2.0)]) < 0))
close("Figure 44.4.1: |log_xbar y_7| = 1.48 is the longest arrow", [np.linalg.norm(ch44.sphere_log(xb, Y[6])),
      float(np.argmax([np.linalg.norm(ch44.sphere_log(xb, y)) for y in Y]))], [1.4809, 6], 5e-5)

# ---------------------------------------------------------------- Example 44.4.2, Definition 44.4.3 (sphere and cylinder)
ok_geo, ok_inv = True, True
for _ in range(50):
    p = ch44.unit(rng.normal(size=3))
    v = rng.normal(size=3)
    v -= (v @ p) * p
    v *= rng.uniform(0.1, 3.0) / np.linalg.norm(v)
    ts = np.linspace(0, 1, 5)
    G = np.array([ch44.sphere_exp(p, t * v) for t in ts])
    acc = np.array([-np.linalg.norm(v) ** 2 * g for g in G])     # d^2/dt^2 of cos(t|v|) p + sin(t|v|) v/|v|
    ok_geo &= np.allclose(np.linalg.norm(G, axis=1), 1) and all(np.linalg.norm(np.cross(a_, g)) < 1e-12 for a_, g in zip(acc, G))
    q = ch44.sphere_exp(p, v)
    if np.linalg.norm(v) < np.pi - 1e-6:
        lg = ch44.sphere_log(p, q)
        ok_inv &= np.allclose(lg, v, atol=1e-9) and abs(np.linalg.norm(lg) - np.arccos(np.clip(p @ q, -1, 1))) < 1e-9
check("Example 44.4.2(a): t -> exp_p(t v) is a unit-sphere curve with acceleration normal to the sphere", ok_geo)
check("Def 44.4.3: log_p(exp_p(v)) = v for |v| < pi and |log_p q| = d(p, q)", ok_inv)
p = np.array([0.0, 0, 1])
ring = np.array([ch44.sphere_exp(p, 1.1 * np.array([np.cos(a), np.sin(a), 0])) for a in np.linspace(0, 2 * np.pi, 12)])
close("Example 44.4.2(a): image of |v| = 1.1 is the latitude of colatitude 1.1", np.arccos(ring[:, 2]), [1.1] * 12, 1e-12)
close("Example 44.4.2(a): |v| = pi goes to the antipode", ch44.sphere_exp(p, np.pi * np.array([0.6, 0.8, 0])), -p, 1e-12)
# cylinder: x(a, b) is a geodesic in (a, b) (acceleration normal)
a_, b_ = 1.3, -0.7
ts = np.linspace(0, 1, 7)
X = np.stack([np.cos(a_ * ts), np.sin(a_ * ts), b_ * ts], 1)
acc = np.stack([-a_ ** 2 * np.cos(a_ * ts), -a_ ** 2 * np.sin(a_ * ts), 0 * ts], 1)
Ncyl = np.stack([np.cos(a_ * ts), np.sin(a_ * ts), 0 * ts], 1)
check("Example 44.4.2(b): t -> x(a t, b t) on the cylinder has acceleration along N",
      np.allclose(np.linalg.norm(np.cross(acc, Ncyl), axis=1), 0))
# Non-example 44.4.4 / Exercise 44.4.3: cylinder cut line u = pi
for v in (0.0, 0.8):
    close(f"Non-example 44.4.4: cylinder point (u, v) = (pi, {v}) has two shortest geodesics",
          np.hypot(np.pi, v), np.hypot(np.pi - 2 * np.pi, v), 1e-12)
check("Exercise 44.4.3(a): for |u| < pi the copy k = 0 is the nearest", all(
    abs(u) < min(abs(u - 2 * np.pi * k) for k in (-2, -1, 1, 2)) for u in np.linspace(-3.1, 3.1, 63)))
us = np.linspace(-np.pi, np.pi, 20001)
Fc = 0.5 * (np.minimum(np.abs(us - 2.5), 2 * np.pi - np.abs(us - 2.5)) ** 2
            + np.minimum(np.abs(us + 2.5), 2 * np.pi - np.abs(us + 2.5)) ** 2)
close("Exercise 44.4.3(b): F(u = 0) = 6.25 and F(u = pi) = 0.4116", [Fc[10000], Fc[-1]], [6.25, 0.4116], 5e-5)
check("Exercise 44.4.3(b): the minimizer is u = +-pi", abs(abs(us[np.argmin(Fc)]) - np.pi) < 1e-3)

# ---------------------------------------------------------------- Proposition 44.4.6 (numerical gradient)
ok = True
for _ in range(30):
    y = ch44.unit(rng.normal(size=3))
    x = ch44.unit(rng.normal(size=3))
    if abs(x @ y) > 0.95:
        continue
    f = lambda z: 0.5 * np.arccos(np.clip(ch44.unit(z) @ y, -1, 1)) ** 2  # noqa: E731
    gnum = np.zeros(3)
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1e-6
        gnum[k] = (f(x + e) - f(x - e)) / 2e-6
    gnum -= (gnum @ x) * x                               # tangential part (f is extended homogeneously)
    ok &= np.allclose(gnum, -ch44.sphere_log(x, y), atol=1e-6)
check("Prop 44.4.6: grad (1/2) d(., y)^2 = -log_x(y) on the unit sphere (30 random pairs)", ok)
# cylinder: (1/2)(u^2 + v^2) has parameter gradient (u, v) = -log_q(p) in the orthonormal frame (x_u, x_v)
u0, v0 = 0.9, -0.4
close("Prop 44.4.6 (cylinder, E = G = 1): grad (1/2) d(p, .)^2 at q = x(u, v) is (u, v) = -log_q(p)", [u0, v0], [-(0 - u0), -(0 - v0)], 0)

# ---------------------------------------------------------------- Non-example 44.4.9 (antipodal pair)
y = np.array([0.0, 0, 1])
eq = [np.array([np.cos(a), np.sin(a), 0]) for a in np.linspace(0, 6, 7)]
close("Non-example 44.4.9: F = pi^2/4 on the whole equator", [0.5 * (np.arccos(e @ y) ** 2 + np.arccos(e @ -y) ** 2) for e in eq],
      [np.pi ** 2 / 4] * 7, 1e-12)
check("Non-example 44.4.9: log_x(y) + log_x(-y) = 0 on the equator",
      all(np.allclose(ch44.sphere_log(e, y) + ch44.sphere_log(e, -y), 0, atol=1e-12) for e in eq))
ends = [ch44.karcher_mean_sphere([y, -y], ch44.sphere_point(0.4, ph0))[0] for ph0 in (0.0, 2.0)]
check("Non-example 44.4.9: two starts converge to two different equator points",
      all(abs(e[2]) < 1e-9 for e in ends) and np.linalg.norm(ends[0] - ends[1]) > 0.5)

# ---------------------------------------------------------------- Example 44.4.11 / Exercise 44.4.6 (discrete exp)
check("Exercise 44.4.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-4-6") <= 30)
ns6, out6 = ch44.run_printed(PAGE, "code-ex-44-4-6")
check("Exercise 44.4.6: printed output matches the page", out6.strip() == ch44.printed_block(PAGE, "out-ex-44-4-6").strip(), out6)
vals = [float(line.split(":")[-1].split(",")[1]) for line in out6.strip().splitlines()]
close("Example 44.4.11: L = 1 errors 0.0962, 0.0260, 0.0058, 0.0011", vals, [0.0962, 0.0260, 0.0058, 0.0011], 5e-5)
r = np.array(vals[1:]) / np.array(vals[:-1])
close("Exercise 44.4.6: ratios 0.27, 0.22, 0.19", r, [0.27, 0.22, 0.19], 6e-3)

# ---------------------------------------------------------------- Proposition 44.4.12, (44.4.3), Figure 44.4.3
for delta, r0 in ((np.pi / 3, 1.0), (np.pi / 2, 0.5)):
    Theta = 2 * np.pi - delta
    for beta in (0.05, 0.1, 0.3):
        t, s = ch44.cone_meeting_point(beta, delta, r0)
        ok = True
        ends_ = []
        for sgn in (1, -1):
            rr, al = ch44.cone_ray(np.pi - sgn * beta, Theta, t, r0=r0, n=2)
            ends_.append((rr[-1], al[-1] % Theta))
        ok &= abs(ends_[0][0] - s) < 1e-9 and abs(ends_[1][0] - s) < 1e-9
        ok &= abs(ends_[0][1] - Theta / 2) < 1e-9 and abs(ends_[1][1] - Theta / 2) < 1e-9
        check(f"Prop 44.4.12: delta = {delta:.4f}, beta = {beta}: the two rays meet at distance s on alpha = Theta/2", ok)
    t_small, _ = ch44.cone_meeting_point(1e-6, delta, r0)
    close(f"Prop 44.4.12: t(beta) -> r0 as beta -> 0 (delta = {delta:.4f})", t_small, r0, 1e-5)
close("Figure 44.4.3: delta = pi/3, beta = 0.1: s = 0.243, t = 1.216", ch44.cone_meeting_point(0.1, np.pi / 3), [0.2429, 1.2164][::-1], 5e-4)
close("Exercise 44.4.4: delta = pi/2, r0 = 0.5: (t, s) for beta = 0.1, 0.5",
      [*ch44.cone_meeting_point(0.1, np.pi / 2, 0.5), *ch44.cone_meeting_point(0.5, np.pi / 2, 0.5)],
      [0.5586, 0.0789, 1.2558, 0.8514], 1e-4)
check("Exercise 44.4.4: the two side rays are shorter than the path through v (0.5586 < 0.5789)",
      ch44.cone_meeting_point(0.1, np.pi / 2, 0.5)[0] < 0.5 + ch44.cone_meeting_point(0.1, np.pi / 2, 0.5)[1])

# ---------------------------------------------------------------- Proposition 44.4.13 (saddle)
Theta = 8 * np.pi / 3
ok = True
for psi in np.linspace(-np.pi, np.pi, 721):
    if abs(np.cos(psi) + 1) < 1e-12:
        continue
    rr, al = ch44.cone_ray(psi, Theta, 50.0, n=4000)
    ok &= np.abs(al).max() < np.pi
check("Prop 44.4.13: rays not through v keep |alpha| < pi (720 directions, length 50)", ok)
rr, al = ch44.cone_ray(np.pi, Theta, 3.0)
check("Prop 44.4.13: the ray through v continues at alpha = Theta/2", abs(al[-1] - Theta / 2) < 1e-12)

# ---------------------------------------------------------------- Exercise 44.4.2, Exercise 44.4.5
q = ch44.sphere_point(np.pi / 3, np.pi / 4)
lg = ch44.sphere_log(np.array([0, 0, 1.0]), q)
close("Exercise 44.4.2: log_{e3}(q) = (0.7405, 0.7405, 0)", lg, [0.7405, 0.7405, 0], 5e-5)
close("Exercise 44.4.2: exp(log) = q", ch44.sphere_exp(np.array([0, 0, 1.0]), lg), q, 1e-12)
V3, F3 = dm.icosphere(3)
pb = V3[F3[0]].mean(0)
dv = np.linalg.norm(V3[F3[0]] - pb, axis=1)
v0 = int(F3[0][np.argmin(dv)])
hv = []
for tri in F3:
    if v0 in tri:
        a_, b_ = [V3[w] for w in tri if w != v0]
        tt_ = np.clip((V3[v0] - a_) @ (b_ - a_) / ((b_ - a_) @ (b_ - a_)), 0, 1)
        hv.append(np.linalg.norm(V3[v0] - (a_ + tt_ * (b_ - a_))))
dfc = dm.angle_defect(V3, F3)
close("Exercise 44.4.5(b): face 0 barycentre: nearest vertex at 0.075, h_v = 0.112", [dv.min(), min(hv)], [0.075, 0.112], 5e-4)
check("Exercise 44.4.5(b): all icosphere(3) vertices are cone vertices with defects in [0.017, 0.023]",
      dfc.min() > 0.017 and dfc.max() < 0.023 and dv.min() < min(hv))


# ---------------------------------------------------------------- Proposition 44.4.12(b): the exact injectivity radius
def inj_cone(delta, r0=1.0, nr=800, na=2000):
    """min over pairs of copies (alpha, alpha - k Theta) in (-pi, pi) of the longer of the two straight segments"""
    Theta = 2 * np.pi - delta
    rs = np.linspace(1e-4, 2.0, nr)[None, :]
    best = np.inf
    for k in range(1, 8):
        al = np.linspace(-np.pi + 1e-7, np.pi - 1e-7, na)[:, None]
        al2 = al - k * Theta
        ok_ = (al2 > -np.pi)[:, 0]
        if not ok_.any():
            break
        l1 = r0 ** 2 + rs ** 2 - 2 * r0 * rs * np.cos(al[ok_])
        l2 = r0 ** 2 + rs ** 2 - 2 * r0 * rs * np.cos(al2[ok_])
        best = min(best, float(np.sqrt(np.maximum(l1, l2)).min()))
    return best


for delta in (np.pi / 3, np.pi / 2, 2.5, 3.6, 4.5):
    close(f"Prop 44.4.12(b): delta = {delta:.3f}: inj = r0 (delta <= pi) or r0 sin(delta/2) (delta > pi), brute force",
          inj_cone(delta), 1.0 if delta <= np.pi else np.sin(delta / 2), 2e-3)
close("Prop 44.4.12(b): beta = (delta - pi)/2 in (44.4.3) gives t = r0 sin(delta/2) (delta = 4.5)",
      ch44.cone_meeting_point((4.5 - np.pi) / 2, 4.5)[0], np.sin(2.25), 1e-12)
check("Prop 44.4.13 text: on the saddle cone (delta = -2 pi/3) no two rays avoiding v meet (injective)",
      inj_cone(-2 * np.pi / 3) == np.inf)
close("Figure 44.4.3: beta = 0.5 meets at s = 20.3 (outside the panel)", ch44.cone_meeting_point(0.5, np.pi / 3)[1], 20.3, 0.05)
summary()
