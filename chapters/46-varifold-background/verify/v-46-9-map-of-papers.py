"""Verification for Section 46.9 (A Map of the Four Papers).

Checks the conventions used in the symbol table and the numbers in the exercises.
Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-9-map-of-papers.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(469)

# ---------------------------------------------------------------- symbol table: plane metrics
for _ in range(5):
    n1, n2 = rng.normal(size=3), rng.normal(size=3)
    P, Q = v.plane_proj(n1), v.plane_proj(n2)
    close("table: hyperplanes in R^3: Frobenius = sqrt2 x operator norm", v.fro_dist(P, Q), math.sqrt(2) * v.op_dist(P, Q), 1e-12)
    a, b = rng.uniform(0, math.pi, size=2)
    close("table: lines in R^2: Frobenius = sqrt2 x operator norm", v.fro_dist(v.line_proj(a), v.line_proj(b)),
          math.sqrt(2) * v.op_dist(v.line_proj(a), v.line_proj(b)), 1e-12)
# ---------------------------------------------------------------- symbol table: mean curvature conventions on a sphere
r = 1.7
x = np.array([0.3, -0.5, 0.81])
x = r * x / np.linalg.norm(x)
N = x / r
H_mean = -1 / r                      # study set, outward N: H = (k1 + k2)/2 with k = -1/r
Hvec = 2 * H_mean * N                # GMT / the four papers: sum of principal curvatures times N
close("table: vec H = 2 H N = -2x/r^2 on the sphere (inward, length 2/r)", Hvec, -2 * x / r ** 2, 1e-15)
close("table: |vec H| = 2/r", np.linalg.norm(Hvec), 2 / r, 1e-15)

# ---------------------------------------------------------------- Exercise 46.9.2: Frobenius version of the sphere bound
h16, eT16 = 0.1386, 0.1383
close("Exercise 46.9.2: eps_T in Frobenius norm = sqrt2 x 0.1383 = 0.1956", math.sqrt(2) * eT16, 0.1956, 5e-5)
close("Exercise 46.9.2: ManifoldGS-style bound 4 pi (h + sqrt2 eps_T) = 4.20 (n_theta = 16)", 4 * math.pi * (h16 + math.sqrt(2) * eT16),
      4.200, 5e-3)
close("Exercise 46.9.2: operator-norm bound 4 pi (h + eps_T) = 3.48", 4 * math.pi * (h16 + eT16), 3.480, 5e-3)


# ---------------------------------------------------------------- Exercise 46.9.3: the helicoid has H = 0 but MAC > 0
def helicoid(u, w, c=1.0):
    return np.array([w * math.cos(u), w * math.sin(u), c * u])


def principal_curvatures(f, u, w, h=1e-4):
    fu = (f(u + h, w) - f(u - h, w)) / (2 * h)
    fw = (f(u, w + h) - f(u, w - h)) / (2 * h)
    fuu = (f(u + h, w) - 2 * f(u, w) + f(u - h, w)) / h ** 2
    fww = (f(u, w + h) - 2 * f(u, w) + f(u, w - h)) / h ** 2
    fuw = (f(u + h, w + h) - f(u + h, w - h) - f(u - h, w + h) + f(u - h, w - h)) / (4 * h * h)
    n = np.cross(fu, fw)
    n /= np.linalg.norm(n)
    I = np.array([[fu @ fu, fu @ fw], [fu @ fw, fw @ fw]])
    II = np.array([[fuu @ n, fuw @ n], [fuw @ n, fww @ n]])
    return np.sort(np.linalg.eigvals(np.linalg.solve(I, II)).real)


for (u0, w0, c) in ((0.3, 0.0, 1.0), (1.1, 0.7, 1.0), (2.0, -0.4, 0.5)):
    k = principal_curvatures(lambda a, b: helicoid(a, b, c), u0, w0)
    close(f"Exercise 46.9.3: helicoid c = {c}, v = {w0}: principal curvatures +-c/(c^2 + v^2)", k,
          [-c / (c * c + w0 * w0), c / (c * c + w0 * w0)], 1e-5)
    close(f"Exercise 46.9.3: mean curvature 0, MAC = c/(c^2 + v^2) (v = {w0})", [0.5 * k.sum(), 0.5 * np.abs(k).sum()],
          [0.0, c / (c * c + w0 * w0)], 1e-5)

# ---------------------------------------------------------------- Exercise 46.9.4: prune by transport
for trial in range(6):
    K = 12
    mu = rng.normal(size=(K, 3))
    Ps = np.array([v.plane_proj(rng.normal(size=3)) for _ in range(K)])
    q = rng.uniform(0.2, 1.0, size=K)
    removed = rng.choice(K, size=3, replace=False)
    keep = np.setdiff1d(np.arange(K), removed)
    q_new = q.copy()
    assign = {}
    for i in removed:
        j = keep[np.argmin(np.linalg.norm(mu[keep] - mu[i], axis=1))]
        assign[i] = j
        q_new[j] += q[i]
    q_new[removed] = 0.0
    check("Exercise 46.9.4: total mass is kept", abs(q_new.sum() - q.sum()) < 1e-12)
    dmom = (q_new[:, None] * mu).sum(0) - (q[:, None] * mu).sum(0)
    close("Exercise 46.9.4: first-moment change = sum q_i (mu_a(i) - mu_i)", dmom,
          sum(q[i] * (mu[assign[i]] - mu[i]) for i in removed), 1e-12)
    bound = sum(q[i] * (np.linalg.norm(mu[i] - mu[assign[i]]) + v.op_dist(Ps[i], Ps[assign[i]])) for i in removed)
    tests = v.random_lipschitz_tests(10, seed=100 + trial, n=3)
    gaps = [abs(v.atoms_integral(f, mu, Ps, q_new) - v.atoms_integral(f, mu, Ps, q)) for f in tests]
    check("Exercise 46.9.4: |V_after(phi) - V_before(phi)| <= sum q_i (|mu_i - mu_a(i)| + ||P_i - P_a(i)||)", max(gaps) <= bound + 1e-12,
          f"{max(gaps):.4f} <= {bound:.4f}")

summary()
