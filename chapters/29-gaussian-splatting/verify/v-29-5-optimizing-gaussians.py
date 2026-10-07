"""Verification for Section 29.5 (Optimizing Gaussians).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-5-optimizing-gaussians.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(295)

# ---------------------------------------------------------------- parameter count
counts = {"mean": 3, "quaternion": 4, "log-scale": 3, "opacity logit": 1, "SH": 16 * 3}
check("parameters per Gaussian: 3 + 4 + 3 + 1 + 48 = 59", sum(counts.values()) == 59)
check("geometric manifold dimension: 3 + 3 + 3 + 1 + 48 = 58 (S^3 instead of R^4 minus 0)", sum(counts.values()) - 1 == 58)

# ---------------------------------------------------------------- (29.5.1)-(29.5.2): coordinates and metrics
s, l, c, eta = sp.symbols("s ell c eta", positive=True)
Lf = sp.Function("L")
# gradient descent in ell = log s, written in s: s_new = s exp(-eta s L'(s)) = s - eta s^2 L'(s) + O(eta^2)
step_log = s * sp.exp(-eta * s * sp.Symbol("Lp"))
sym_equal("(29.5.2): GD in log s moves s by -eta s^2 L'(s) to first order",
          sp.diff(step_log, eta).subs(eta, 0), -s ** 2 * sp.Symbol("Lp"))
# metric ds^2/s^2: the gradient w.r.t. g = 1/s^2 is s^2 L'(s)
g = 1 / s ** 2
sym_equal("(29.5.2): grad_g L = g^{-1} dL = s^2 L'(s) for g = ds^2/s^2", (1 / g) * sp.Symbol("Lp"), s ** 2 * sp.Symbol("Lp"))
# scale invariance: pulling back ds^2/s^2 by s -> c s gives the same metric
sym_equal("(29.5.2): ds^2/s^2 is invariant under s -> c s", (c ** 2) / (c * s) ** 2, 1 / s ** 2)
# general: GD in coordinates phi with theta = h(phi) is GD in theta with metric (Dh Dh^T)^{-1}
A = rng.normal(size=(2, 2)) + 2 * np.eye(2)          # Dh at a point (linear h)
gradth = rng.normal(size=2)
step_phi = -1e-3 * A @ (A.T @ gradth)                    # theta-step induced by a phi-step -eta grad_phi
metric = np.linalg.inv(A @ A.T)
close("(29.5.1): GD in phi = GD in theta with metric (Dh Dh^T)^{-1}", step_phi, -1e-3 * np.linalg.solve(metric, gradth),
      tol=1e-15)

# ---------------------------------------------------------------- Fisher metric of N(mu, s^2)
mu, x = sp.symbols("mu x", real=True)
logp = -sp.log(s) - sp.log(sp.sqrt(2 * sp.pi)) - (x - mu) ** 2 / (2 * s ** 2)
pdf = sp.exp(logp)
F = sp.zeros(2, 2)
pars = [mu, s]
for i in range(2):
    for j in range(2):
        F[i, j] = sp.simplify(sp.integrate(sp.diff(logp, pars[i]) * sp.diff(logp, pars[j]) * pdf, (x, -sp.oo, sp.oo)))
sym_equal("(29.5.3): Fisher metric of N(mu, s^2) is diag(1/s^2, 2/s^2)", F, sp.diag(1 / s ** 2, 2 / s ** 2))
# Gaussian curvature of the Fisher metric (orthogonal metric E dmu^2 + G ds^2): constant -1/2 (twice the U^2 metric)
EF, GF = F[0, 0], F[1, 1]
WF = sp.sqrt(EF * GF)
KF = sp.simplify(-1 / (2 * WF) * (sp.diff(GF.diff(mu) / WF, mu) + sp.diff(EF.diff(s) / WF, s)))
sym_equal("(29.5.3): the Fisher metric has constant curvature -1/2", KF, sp.Rational(-1, 2))
# natural gradient is coordinate-free: in (mu, ell = log s) the Fisher metric is diag(1/s^2, 2) and the induced
# s-step agrees with the (mu, s) natural-gradient s-step
gs = rng.normal()
s0 = 0.7
step_s = -(s0 ** 2 / 2) * gs                              # natural gradient in (mu, s)
gl = s0 * gs                                              # dL/d ell = s dL/ds
step_l = -(1 / 2) * gl                                    # Fisher in ell: 2 -> inverse 1/2
close("natural gradient: same first-order step in (mu, s) and (mu, log s)", step_s, s0 * step_l, tol=1e-15)

# ---------------------------------------------------------------- Figure 29.5.2: toy trajectories
SP = np.sqrt(np.pi)


def Ltoy(m, sd):
    a = sd * sd + 1.0
    return SP * sd + SP - 2 * np.sqrt(2 * np.pi) * sd / np.sqrt(a) * np.exp(-(m - 1.0) ** 2 / (2 * a))


# closed form of the L2 loss between footprints (peak 1) checked by quadrature
xs = np.linspace(-20, 20, 400001)
num = np.sum((np.exp(-(xs + 0.3) ** 2 / (2 * 0.49)) - np.exp(-(xs - 1.0) ** 2 / 2)) ** 2) * (xs[1] - xs[0])
close("Figure 29.5.2: closed-form toy loss agrees with quadrature", Ltoy(-0.3, 0.7), num, tol=1e-8)


def gtoy(m, sd, h=1e-6):
    return np.array([(Ltoy(m + h, sd) - Ltoy(m - h, sd)) / (2 * h), (Ltoy(m, sd + h) - Ltoy(m, sd - h)) / (2 * h)])


def run(kind, m, sd, eta, n):
    P = [(m, sd)]
    for _ in range(n):
        gm, gsd = gtoy(m, sd)
        if kind == "s":
            m, sd = m - eta * gm, sd - eta * gsd
        elif kind == "log":
            m, sd = m - eta * gm, sd * np.exp(-eta * sd * gsd)
        else:
            m, sd = m - eta * sd * sd * gm, sd - eta * sd * sd / 2 * gsd
        P.append((m, sd))
        if sd <= 0:
            break
    return np.array(P)


def iters_to(P, tol=0.01):
    d = np.hypot(P[:, 0] - 1, P[:, 1] - 1)
    return int(np.argmax(d < tol)) if np.any(d < tol) else -1


expected = {((-0.6, 0.35), "s"): 63, ((-0.6, 0.35), "log"): 66, ((-0.6, 0.35), "nat"): 160,
            ((-0.4, 2.6), "s"): 69, ((-0.4, 2.6), "log"): 62, ((-0.4, 2.6), "nat"): 60,
            ((2.6, 0.5), "s"): 61, ((2.6, 0.5), "log"): 61, ((2.6, 0.5), "nat"): 99}
for (start, kind), k in expected.items():
    P = run(kind, *start, eta=0.05, n=3000)
    check(f"Figure 29.5.2: start {start}, {kind}: within 0.01 of the minimum after {k} iterations", iters_to(P) == k)
# large step: GD in s leaves the domain s > 0; log-scale stays inside and converges
P = run("s", -1.5, 0.3, eta=0.3, n=5)
check("text: eta = 0.3 from (-1.5, 0.3): GD in s jumps to s < 0 in one step", P[1, 1] < 0)
close("text: that first step lands at s = -0.12", P[1, 1], -0.12, tol=0.005)
P = run("log", -1.5, 0.3, eta=0.3, n=400)
check("text: the same start with log-scale converges (37 iterations)", iters_to(P) == 37)
P = run("nat", -1.5, 0.3, eta=0.3, n=400)
check("text: natural gradient from (-1.5, 0.3) shrinks s toward 0 and stalls (s < 0.02, mu < -1.4)",
      P[-1, 1] < 0.02 and P[-1, 0] < -1.4)

# ---------------------------------------------------------------- (29.5.5): gradient through q / |q|
qv = sp.Matrix(sp.symbols("q0:4", real=True))
nq = sp.sqrt(sum(qi ** 2 for qi in qv))
qh = qv / nq
Jq = qh.jacobian(qv)
sym_equal("(29.5.4): d(q/|q|)/dq = (I - q^ q^T)/|q|", Jq, (sp.eye(4) - qh * qh.T) / nq)
sym_equal("(29.5.4): the Jacobian kills q (fiber = ray)", Jq * qv, sp.zeros(4, 1))


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Rq(q):
    q = q / np.linalg.norm(q)
    return np.eye(3) + 2 * q[0] * hat(q[1:]) + 2 * hat(q[1:]) @ hat(q[1:])


sc = np.array([1.0, 0.6, 0.3])
qs = rng.normal(size=4); qs /= np.linalg.norm(qs)
St = Rq(qs) @ np.diag(sc ** 2) @ Rq(qs).T


def Lq(q):
    Rm = Rq(q)
    return 0.5 * np.sum((Rm @ np.diag(sc ** 2) @ Rm.T - St) ** 2)


def gq(q, h=1e-7):
    return np.array([(Lq(q + h * e) - Lq(q - h * e)) / (2 * h) for e in np.eye(4)])


q = rng.normal(size=4)
qhat = q / np.linalg.norm(q)
gh = np.array([(Lq(qhat + 1e-7 * e) - Lq(qhat - 1e-7 * e)) / 2e-7 for e in np.eye(4)])
close("(29.5.4): grad_q L = (I - q^ q^T) grad_q^ L / |q| (numeric)", gq(q),
      (np.eye(4) - np.outer(qhat, qhat)) @ gh / np.linalg.norm(q), tol=1e-6)
close("(29.5.4): <grad_q L, q> = 0", gq(q) @ q, 0.0, tol=1e-7)
d = -0.3 * gq(q)
close("(29.5.5): |q + delta|^2 = |q|^2 + |delta|^2 for a GD step", np.linalg.norm(q + d) ** 2,
      np.linalg.norm(q) ** 2 + np.linalg.norm(d) ** 2, tol=1e-7)
# |grad_q L| scales like 1/|q|: doubling q halves the gradient
close("(29.5.4): grad at 2q is half the grad at q", gq(2 * q), 0.5 * gq(q), tol=1e-7)


def run_gd(eta, n=400):
    q = np.array([1.0, 0, 0, 0]); out = [1.0]
    for _ in range(n):
        q = q - eta * gq(q); out.append(np.linalg.norm(q))
    return np.array(out), q


def run_adam(lr, n=400, b1=0.9, b2=0.999, eps=1e-15):
    q = np.array([1.0, 0, 0, 0]); m = np.zeros(4); v = np.zeros(4); out = [1.0]
    for k in range(1, n + 1):
        g_ = gq(q); m = b1 * m + (1 - b1) * g_; v = b2 * v + (1 - b2) * g_ * g_
        q = q - lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps); out.append(np.linalg.norm(q))
    return np.array(out), q


for eta, fin in ((0.05, 1.0137), (0.2, 1.0567), (0.5, 1.3845)):
    nr, qf = run_gd(eta)
    check(f"Figure 29.5.3: GD eta = {eta}: |q| never decreases", np.all(np.diff(nr) >= -1e-12))
    close(f"Figure 29.5.3: GD eta = {eta}: final |q| = {fin}", nr[-1], fin, tol=1e-4)
    check(f"Figure 29.5.3: GD eta = {eta}: converges (loss < 1e-9)", Lq(qf) < 1e-9)
na, qa = run_adam(0.01)
check("Figure 29.5.3: Adam lr = 0.01: |q| decreases on some steps", np.sum(np.diff(na) < -1e-12) > 0)
close("Figure 29.5.3: Adam lr = 0.01: min |q| = 0.748, final 0.765", [na.min(), na[-1]], [0.748, 0.765], tol=1e-3)
na2, _ = run_adam(0.05)
close("Figure 29.5.3: Adam lr = 0.05: final |q| = 1.29, min 0.86", [na2[-1], na2.min()], [1.291, 0.856], tol=1e-3)
# effective angular step of GD: |delta_perp| / |q| = eta |P grad_q^| / |q|^2
qq = 1.7 * qhat
step = -0.1 * gq(qq)
close("Proposition 29.5.2: angular step = eta |P grad| / |q|^2", np.linalg.norm(step) / np.linalg.norm(qq),
      0.1 * np.linalg.norm((np.eye(4) - np.outer(qhat, qhat)) @ gh) / 1.7 ** 2, tol=1e-6)
# stochastic gradients: |q|^2 grows by the sum of squared steps
qn = np.array([1.0, 0, 0, 0]); acc = 1.0
for _ in range(200):
    dlt = rng.normal(size=4); dlt -= (dlt @ qn) / (qn @ qn) * qn; dlt *= 0.05
    acc += dlt @ dlt; qn = qn + dlt
close("text: with tangent noise, |q|^2 = 1 + sum |delta_k|^2", qn @ qn, acc, tol=1e-9)

# ---------------------------------------------------------------- Adam: invariance to diagonal rescaling (eps -> 0)
def adam_steps(grads, lr, eps=1e-15):
    m = np.zeros_like(grads[0]); v = np.zeros_like(grads[0]); out = []
    for k, g_ in enumerate(grads, 1):
        m = 0.9 * m + 0.1 * g_; v = 0.999 * v + 0.001 * g_ * g_
        out.append(-lr * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + eps))
    return np.array(out)


D = np.array([10.0, 0.1, 3.0])
grads = [rng.normal(size=3) for _ in range(20)]
st1 = adam_steps(grads, 0.01)                          # in theta
st2 = adam_steps([g_ / D for g_ in grads], 0.01)       # in phi = D theta (gradient wrt phi is g / D)
close("Adam: steps in phi = D theta equal steps in theta (scale-free per coordinate)", st2, st1, tol=1e-12)
check("Adam: so a theta-step from phi is D^{-1} times larger/smaller (metric changes)", not np.allclose(st2 / D, st1))
check("Adam: per-coordinate step bounded by about lr in the first steps", np.all(np.abs(st1[:5]) <= 0.01 * 1.0001))

# ---------------------------------------------------------------- densification: NDC vs pixel
Wpx, Hpx = 1920, 1080
v = sp.symbols("v", real=True)
Wsym = sp.symbols("W", positive=True)
pix = ((v + 1) * Wsym - 1) / 2
sym_equal("densification: d pixel / d ndc = W/2", sp.diff(pix, v), Wsym / 2)
gpx = np.array([1e-7, 1e-7])
gndc = np.array([Wpx / 2, Hpx / 2]) * gpx
close("densification: |grad_ndc| = |diag(W/2, H/2) grad_px|", np.linalg.norm(gndc), np.hypot(960e-7, 540e-7), tol=1e-15)
tau = 0.0002
close("Figure 29.5.4: threshold ellipse semi-axes 2 tau/W, 2 tau/H at 1920 x 1080 (2.08e-7, 3.70e-7)",
      [2 * tau / 1920, 2 * tau / 1080], [2.0833e-7, 3.7037e-7], tol=1e-10)
# Figure 29.5.4(b): two pixel gradients of the same length 3e-7, horizontal and vertical, at 1920 x 1080
close("Figure 29.5.4(b): NDC norms of (3, 0) and (0, 3) x 1e-7 are 2.88e-4 > tau and 1.62e-4 < tau",
      [960 * 3e-7, 540 * 3e-7], [2.88e-4, 1.62e-4], tol=1e-12)
check("Figure 29.5.4(b): horizontal densified, vertical not", 960 * 3e-7 > tau > 540 * 3e-7)


# Figure 29.5.4(a): toy splat rendered at several resolutions (same scene): pixel-unit gradient ~ 1/W, NDC ~ constant.
# Independent implementation: finite differences of the L1 loss (the figure script uses the analytic gradient).
def l1_loss(W, H, c_ndc, S_ndc=np.diag([0.03 ** 2, 0.05 ** 2]), tgt=(0.112, -0.046), o=0.8):
    to_px = lambda vv, S_: ((vv + 1) * S_ - 1) / 2
    A_ = np.diag([W / 2, H / 2])
    Si_ = np.linalg.inv(A_ @ S_ndc @ A_)
    Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))

    def im(c):
        dx_, dy_ = Xg - to_px(c[0], W), Yg - to_px(c[1], H)
        return o * np.exp(-0.5 * (Si_[0, 0] * dx_ ** 2 + 2 * Si_[0, 1] * dx_ * dy_ + Si_[1, 1] * dy_ ** 2))
    return np.mean(np.abs(im(c_ndc) - im(tgt)))


def grad_ndc(W, H, c=(0.1, -0.05), h=1e-6):
    return np.array([(l1_loss(W, H, (c[0] + h, c[1])) - l1_loss(W, H, (c[0] - h, c[1]))) / (2 * h),
                     (l1_loss(W, H, (c[0], c[1] + h)) - l1_loss(W, H, (c[0], c[1] - h))) / (2 * h)])


gn = {W: grad_ndc(W, W * 9 // 16) for W in (240, 480, 960)}
gp = {W: gn[W] / np.array([W / 2, (W * 9 // 16) / 2]) for W in gn}       # chain rule back to pixel units
close("Figure 29.5.4(a): NDC gradient norm is the same at W = 240, 480, 960 (rel tol 1%)",
      [np.linalg.norm(gn[W]) / np.linalg.norm(gn[960]) for W in gn], [1.0, 1.0, 1.0], tol=0.01)
close("Figure 29.5.4(a): pixel-unit gradient norm scales like 1/W (rel tol 1%)",
      [np.linalg.norm(gp[W]) * W / (np.linalg.norm(gp[960]) * 960) for W in gp], [1.0, 1.0, 1.0], tol=0.01)
# cancellation: norm of a sum vs sum of norms
contrib = np.array([[1.0, 0.2], [-0.9, -0.1], [0.2, 0.0]])
check("densification: |sum of per-pixel covectors| <= sum of their norms",
      np.linalg.norm(contrib.sum(0)) <= np.linalg.norm(contrib, axis=1).sum())
close("densification: example |sum| = 0.32, sum of norms = 2.13", [round(np.linalg.norm(contrib.sum(0)), 2),
                                                                  round(np.linalg.norm(contrib, axis=1).sum(), 2)],
      [0.32, 2.13], tol=1e-9)

# ---------------------------------------------------------------- Exercises
# 29.5.1: one GD step in s and in log s from s = 2 with L'(s) = 0.5, eta = 0.1
close("Exercise 29.5.1: GD in s: 2 - 0.1*0.5 = 1.95", 2 - 0.1 * 0.5, 1.95, tol=1e-12)
close("Exercise 29.5.1: GD in log s: 2 exp(-0.1*2*0.5) = 1.8097", 2 * np.exp(-0.1 * 2 * 0.5), 1.8097, tol=1e-4)
close("Exercise 29.5.1: from s = 0.02 with L' = 0.5: s-step -0.03, log-step 0.01998", [0.02 - 0.1 * 0.5, 0.02 * np.exp(-0.1 * 0.02 * 0.5)],
      [-0.03, 0.01998], tol=1e-6)
# 29.5.2: |q| after k orthogonal steps of size 0.1 from |q| = 1
close("Exercise 29.5.2: |q| after 100 tangent steps of length 0.1 is sqrt(2)", np.sqrt(1 + 100 * 0.01), np.sqrt(2), tol=1e-12)
# 29.5.3: half resolution: (a) the same pixel gradient has half the NDC norm, (b) the actual pixel gradient doubles,
#         (c) so the NDC norm is unchanged (toy above)
close("Exercise 29.5.3(a): half resolution halves the NDC norm of the same pixel gradient",
      np.linalg.norm(np.array([480, 270]) * gpx) / np.linalg.norm(np.array([960, 540]) * gpx), 0.5, tol=1e-12)
close("Exercise 29.5.3(b): the scaling argument gives 1/4 * 4 * 2 = 2", 0.25 * 4 * 2, 2.0, tol=1e-15)
close("Exercise 29.5.3(b, c): in the toy the pixel gradient doubles and the NDC norm stays (W 960 -> 480)",
      [np.linalg.norm(gp[480]) / np.linalg.norm(gp[960]), np.linalg.norm(gn[480]) / np.linalg.norm(gn[960])], [2.0, 1.0],
      tol=0.02)
summary()
