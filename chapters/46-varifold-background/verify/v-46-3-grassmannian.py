"""Verification for Section 46.3 (The Grassmannian: Unoriented Planes as Projections).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-3-grassmannian.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(46)


def orth(M):
    Q, _ = np.linalg.qr(M)
    return Q


# ---------------------------------------------------------------- Proposition 46.3.2: planes <-> projection matrices
for n, d in ((2, 1), (3, 1), (3, 2), (5, 3)):
    U = orth(rng.normal(size=(n, d)))
    P = U @ U.T
    check(f"Proposition 46.3.2: U U^T is symmetric, idempotent, trace d (n={n}, d={d})", v.is_projection(P, d))
    R = orth(rng.normal(size=(d, d)))                      # another orthonormal basis of the same plane
    check(f"Proposition 46.3.2: P does not depend on the basis (n={n}, d={d})", np.allclose((U @ R) @ (U @ R).T, P))
    # conversely: a symmetric idempotent matrix of trace d has eigenvalues 1 (d times) and 0
    w = np.sort(np.linalg.eigvalsh(P))
    check(f"Proposition 46.3.2: eigenvalues are 0 ({n - d} times) and 1 ({d} times)",
          np.allclose(w, np.r_[np.zeros(n - d), np.ones(d)]))
N = rng.normal(size=3)
N /= np.linalg.norm(N)
check("Proposition 46.3.2: I - N N^T = I - (-N)(-N)^T (no orientation)", np.allclose(v.plane_proj(N), v.plane_proj(-N)))
check("Proposition 46.3.2: I - N N^T is the projection onto N-perp", v.is_projection(v.plane_proj(N), 2)
      and np.allclose(v.plane_proj(N) @ N, 0))

# ---------------------------------------------------------------- Example 46.3.3: G_{1,2} as a circle
for deg in (0, 30, 45, 90, 137):
    t = math.radians(deg)
    P = v.line_proj(t)
    half = 0.5 * (np.eye(2) + np.array([[math.cos(2 * t), math.sin(2 * t)], [math.sin(2 * t), -math.cos(2 * t)]]))
    check(f"Example 46.3.3: P({deg} deg) = (I + R(2 theta)) / 2", np.allclose(P, half))
    check(f"Example 46.3.3: P(theta) = P(theta + pi) at {deg} deg", np.allclose(P, v.line_proj(t + math.pi)))
check("Example 46.3.3: P(0) -> (1, 0), P(90) -> (-1, 0), P(45) -> (0, 1)",
      np.allclose(v.proj_coords(v.line_proj(0.0)), [1, 0]) and np.allclose(v.proj_coords(v.line_proj(math.pi / 2)), [-1, 0])
      and np.allclose(v.proj_coords(v.line_proj(math.pi / 4)), [0, 1]))
th = np.linspace(0, math.pi, 4001)[:-1]
check("Example 46.3.3 / Exercise 46.3.3: average of P(theta) over all lines is I/2",
      np.allclose(v.line_proj(th).mean(axis=0), np.eye(2) / 2, atol=1e-12))
check("Exercise 46.3.3: (P_h + P_v)/2 = (P_45 + P_135)/2 = I/2",
      np.allclose(0.5 * (v.P_H + v.P_V), np.eye(2) / 2)
      and np.allclose(0.5 * (v.line_proj(math.pi / 4) + v.line_proj(3 * math.pi / 4)), np.eye(2) / 2))
check("Exercise 46.3.3: I/2 is not a projection ((I/2)^2 = I/4)", not v.is_projection(np.eye(2) / 2, 1))

# ---------------------------------------------------------------- Proposition 46.3.4: distances
for deg in (10, 30, 60, 90):
    t = math.radians(deg)
    P, Q = v.line_proj(0.0), v.line_proj(t)
    close(f"Proposition 46.3.4: lines at {deg} deg: ||P-Q||_op = sin", v.op_dist(P, Q), math.sin(t), 1e-12)
    close(f"Proposition 46.3.4: lines at {deg} deg: ||P-Q||_F = sqrt2 sin", v.fro_dist(P, Q), math.sqrt(2) * math.sin(t), 1e-12)
    # half chord in the disk picture
    chord = np.linalg.norm(v.proj_coords(P) - v.proj_coords(Q))
    close(f"Proposition 46.3.4: ||P-Q||_op is half the chord in Figure 46.3.1 ({deg} deg)", v.op_dist(P, Q), chord / 2, 1e-12)
close("Exercise 46.3.2: 30 degrees: operator 0.5", v.op_dist(v.line_proj(0.0), v.line_proj(math.pi / 6)), 0.5, 1e-12)
close("Exercise 46.3.2: 30 degrees: Frobenius 0.7071", v.fro_dist(v.line_proj(0.0), v.line_proj(math.pi / 6)), 0.7071, 5e-5)
# hyperplanes in R^3 with normals at angle theta
for deg in (15, 30, 75):
    t = math.radians(deg)
    n1 = np.array([0.0, 0.0, 1.0])
    n2 = np.array([math.sin(t), 0.0, math.cos(t)])
    P, Q = v.plane_proj(n1), v.plane_proj(n2)
    close(f"Proposition 46.3.4: planes in R^3, normals at {deg} deg: operator = sin", v.op_dist(P, Q), math.sin(t), 1e-12)
    close(f"Proposition 46.3.4: planes in R^3, normals at {deg} deg: Frobenius = sqrt2 sin", v.fro_dist(P, Q),
          math.sqrt(2) * math.sin(t), 1e-12)
# general d: operator distance = sine of the largest principal angle
for n, d in ((3, 2), (4, 2), (6, 3)):
    for _ in range(5):
        U, W = orth(rng.normal(size=(n, d))), orth(rng.normal(size=(n, d)))
        s = np.linalg.svd(U.T @ W, compute_uv=False)
        theta_max = math.acos(min(1.0, s.min()))
        close(f"Proposition 46.3.4: n={n}, d={d}: ||P-Q||_op = sin(largest principal angle)",
              v.op_dist(U @ U.T, W @ W.T), math.sin(theta_max), 1e-10)
        check(f"Proposition 46.3.4: n={n}, d={d}: ||P-Q||_op <= 1", v.op_dist(U @ U.T, W @ W.T) <= 1 + 1e-12)
close("Remark: ||P||_F = sqrt(d) (G_{d,n} is bounded)", np.linalg.norm(v.plane_proj(N), "fro"), math.sqrt(2), 1e-12)

# ---------------------------------------------------------------- Proposition 46.3.6: stretching rate
A = np.array([[0.8, 0.5], [-0.1, -0.4]])


def expm(M, t):
    w, Q = np.linalg.eig(M)
    return (Q @ np.diag(np.exp(t * w)) @ np.linalg.inv(Q)).real


for deg, want in zip((0, 30, 60, 90, 120, 150), (0.8, 0.6732, 0.0732, -0.4, -0.2732, 0.3268)):
    P = v.line_proj(math.radians(deg))
    close(f"Figure 46.3.2: div_P X at {deg} deg = {want}", v.tangential_div(P, A), want, 5e-5)
th = np.linspace(0, math.pi, 7201)
r = np.array([v.tangential_div(P, A) for P in v.line_proj(th)])
close("Figure 46.3.2: max rate 0.8325", r.max(), 0.2 + math.sqrt(0.4), 1e-6)
close("Figure 46.3.2: min rate -0.4325", r.min(), 0.2 - math.sqrt(0.4), 1e-6)
close("Figure 46.3.2: average rate tr(A)/2 = 0.2", r[:-1].mean(), 0.2, 1e-12)
for n, d in ((2, 1), (3, 2), (3, 1), (4, 2)):
    M = rng.normal(size=(n, n))
    U = orth(rng.normal(size=(n, d)))
    P = U @ U.T
    h = 1e-6
    vol = lambda t: math.sqrt(np.linalg.det((expm(M, t) @ U).T @ (expm(M, t) @ U)))  # noqa: E731
    fd = (vol(h) - vol(-h)) / (2 * h)
    close(f"Proposition 46.3.6: d/dt vol_d(exp(tM) cube) = tr(P M) (n={n}, d={d})", fd, v.tangential_div(P, M), 1e-6)
    close(f"Definition 46.3.5: tr(P DX) = sum_i <DX u_i, u_i> (n={n}, d={d})", v.tangential_div(P, M),
          sum(U[:, i] @ M @ U[:, i] for i in range(d)), 1e-12)
    # BLM17 form: sum_j <P grad X_j, e_j>, grad X_j = row j of DX
    blm = sum((P @ M[j])[j] for j in range(n))
    close(f"Definition 46.3.5: equals sum_j <P grad X_j, e_j> (n={n}, d={d})", blm, v.tangential_div(P, M), 1e-12)
# average over all lines of R^n of div_P X is (d/n) div X; check n = 2, d = 1 above and n = 3, d = 2 by Monte Carlo
M = rng.normal(size=(3, 3))
Ns = rng.normal(size=(200000, 3))
Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
vals = np.trace(M) - np.einsum("ij,jk,ik->i", Ns, M, Ns)          # tr((I - nn^T) M)
close("Remark: average of div_P X over planes of R^3 is (2/3) div X", vals.mean(), 2 / 3 * np.trace(M), 0.01)

# ---------------------------------------------------------------- Exercise 46.3.4: directions that are not stretched
for t in (math.atan(2.0), 3 * math.pi / 4):
    u = np.array([math.cos(t), math.sin(t)])
    close(f"Exercise 46.3.4: <Au, u> = 0 at {math.degrees(t):.2f} deg", u @ A @ u, 0.0, 1e-12)
close("Exercise 46.3.4: arctan 2 = 63.43 deg", math.degrees(math.atan(2.0)), 63.43, 5e-3)

# ---------------------------------------------------------------- Example 46.3.7: the sphere
r0 = 1.7
for _ in range(5):
    x = rng.normal(size=3)
    x = r0 * x / np.linalg.norm(x)
    P = np.eye(3) - np.outer(x, x) / r0 ** 2
    close("Example 46.3.7: div_P x = 2 on the sphere", v.tangential_div(P, np.eye(3)), 2.0, 1e-12)
    close("Example 46.3.7: div_P (constant field) = 0", v.tangential_div(P, np.zeros((3, 3))), 0.0, 1e-15)
    E3 = np.zeros((3, 3))
    E3[2, 2] = 1.0
    close("Example 46.3.7: div_P (0, 0, z) = 1 - z^2/r^2", v.tangential_div(P, E3), 1 - x[2] ** 2 / r0 ** 2, 1e-12)
# integral of 1 - z^2/r^2 over the sphere: 4 pi r^2 - 4 pi r^2 / 3 = 8 pi r^2 / 3
val = v.integrate_1d(lambda u: 2 * np.pi * r0 ** 2 * np.sin(u) * (1 - np.cos(u) ** 2), 0, np.pi)
close("Example 46.3.7: int (1 - z^2/r^2) dA = 8 pi r^2 / 3", val, 8 * math.pi * r0 ** 2 / 3, 1e-10)


def prolate_area(a, c):
    e = math.sqrt(1 - a * a / (c * c))
    return 2 * math.pi * a * a * (1 + c / (a * e) * math.asin(e))


h = 1e-5
fd = (prolate_area(r0, r0 * (1 + h)) - prolate_area(r0, r0 * (1 + 2 * h))) / (-h)
close("Example 46.3.7: d/dt area of the spheroid (r, r, r(1+t)) at t = 0 is 8 pi r^2 / 3", fd, 8 * math.pi * r0 ** 2 / 3, 1e-3)

# ---------------------------------------------------------------- Exercise 46.3.1
P = v.dir_proj(np.array([3.0, 4.0]))
check("Exercise 46.3.1: line through (3, 4): P = [[0.36, 0.48], [0.48, 0.64]]", np.allclose(P, [[0.36, 0.48], [0.48, 0.64]]))
P = v.plane_proj(np.ones(3))
check("Exercise 46.3.1: plane with normal (1,1,1): P = I - J/3", np.allclose(P, np.eye(3) - np.ones((3, 3)) / 3)
      and v.is_projection(P, 2))

# ---------------------------------------------------------------- Exercise 46.3.5: 1 - |cos| versus |sin|
for deg in (1, 5, 20):
    t = math.radians(deg)
    check(f"Exercise 46.3.5: at {deg} deg, 1 - cos ~ theta^2/2 is much smaller than sin ~ theta",
          abs((1 - math.cos(t)) - t * t / 2) < t ** 4 / 20 and abs(math.sin(t) - t) < t ** 3 / 5)
close("Exercise 46.3.5: 5 degrees: 1 - cos = 0.0038", 1 - math.cos(math.radians(5)), 0.0038, 5e-5)
close("Exercise 46.3.5: 5 degrees: sin = 0.0872", math.sin(math.radians(5)), 0.0872, 5e-5)

summary()
