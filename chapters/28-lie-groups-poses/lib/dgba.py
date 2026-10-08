"""Chapter 28 helper for Section 28.3: a small bundle-adjustment problem and a Sim(3)/SE(3) pose graph.

Bundle adjustment
  * cameras T_i = T_cw (world -> camera), points X_j in world coordinates, normalized pinhole phi(x) = (x/z, y/z);
  * local coordinates: left (camera-frame) perturbation Exp(xi_i) T_i for cameras (xi = (v, w), as (28.2.3)) and
    X_j + dX_j for points;  parameter vector = (xi_1, ..., xi_N, dX_1, ..., dX_M).
  * gauge generators: the infinitesimal similarity (w, u, sigma) of the world, x -> x + w x x + u + sigma x.

Pose graph (scale drift)
  * vertices: keyframe-to-map transforms X_k (SE(3) or Sim(3)); edge residual Log(Z_ab^{-1} X_a^{-1} X_b)
    (the form of [Strasdat10, (23)] and [MurArtal15, (8)] with the frames written as keyframe -> map);
  * Gauss-Newton with right perturbations X_k Exp(d_k), vertex 0 fixed (gauge), numerical edge Jacobians.
"""

import numpy as np

import dglie as L


# ----------------------------------------------------------------------------- bundle adjustment

def look_at_Tcw(center, target, up=(0.0, -1.0, 0.0)):
    """World->camera pose of a camera at `center` looking at `target` (OpenCV axes: x right, y down, z forward)."""
    z = np.asarray(target, float) - center
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rwc = np.column_stack([x, y, z])
    return L.inv_T(L.make_T(Rwc, center))


def make_ba(N=5, M=40, seed=28):
    rng = np.random.default_rng(seed)
    Xs = np.c_[rng.uniform(-1.2, 1.2, M), rng.uniform(-1.0, 1.0, M), rng.uniform(4.0, 6.0, M)]
    cams = []
    for i in range(N):
        a = np.radians(-30 + 60 * i / max(N - 1, 1))
        center = np.array([5.0 * np.sin(a), 0.4 * np.cos(2.0 * i), 5.0 - 5.0 * np.cos(a)])
        cams.append(look_at_Tcw(center, [0.0, 0.0, 5.0]))
    return cams, Xs


def phi(xc):
    return xc[:2] / xc[2]


def Dphi(xc):
    x, y, z = xc
    return np.array([[1 / z, 0, -x / z ** 2], [0, 1 / z, -y / z ** 2]])


def residuals(cams, Xs, obs):
    out = []
    for i, T in enumerate(cams):
        for j, X in enumerate(Xs):
            out.append(phi(T[:3, :3] @ X + T[:3, 3]) - obs[i, j])
    return np.concatenate(out)


def project_all(cams, Xs):
    return np.array([[phi(T[:3, :3] @ X + T[:3, 3]) for X in Xs] for T in cams])


def jacobian(cams, Xs, depth=False):
    """Stacked Jacobian (28.2.3). With depth=True, each observation also has a depth residual z_c - d."""
    N, M = len(cams), len(Xs)
    rows_per = 3 if depth else 2
    J = np.zeros((rows_per * N * M, 6 * N + 3 * M))
    r = 0
    for i, T in enumerate(cams):
        R = T[:3, :3]
        for j, X in enumerate(Xs):
            xc = R @ X + T[:3, 3]
            A = np.hstack([np.eye(3), -L.hat3(xc)])
            J[r:r + 2, 6 * i:6 * i + 6] = Dphi(xc) @ A
            J[r:r + 2, 6 * N + 3 * j:6 * N + 3 * j + 3] = Dphi(xc) @ R
            if depth:
                J[r + 2, 6 * i:6 * i + 6] = A[2]
                J[r + 2, 6 * N + 3 * j:6 * N + 3 * j + 3] = R[2]
            r += rows_per
    return J


def gauge_generators(cams, Xs):
    """7 columns: the infinitesimal similarities (w: 3, u: 3, sigma: 1) in the local coordinates.

    camera i (left perturbation): w_i = -R_i w,  v_i = sigma t_i - R_i u + (R_i w) x t_i;
    point j: dX_j = w x X_j + u + sigma X_j.
    """
    N, M = len(cams), len(Xs)
    G = np.zeros((6 * N + 3 * M, 7))
    basis = [(e, np.zeros(3), 0.0) for e in np.eye(3)] + [(np.zeros(3), e, 0.0) for e in np.eye(3)] \
        + [(np.zeros(3), np.zeros(3), 1.0)]
    for k, (w, u, sg) in enumerate(basis):
        for i, T in enumerate(cams):
            R, t = T[:3, :3], T[:3, 3]
            G[6 * i:6 * i + 3, k] = sg * t - R @ u + np.cross(R @ w, t)
            G[6 * i + 3:6 * i + 6, k] = -R @ w
        for j, X in enumerate(Xs):
            G[6 * N + 3 * j:6 * N + 3 * j + 3, k] = np.cross(w, X) + u + sg * X
    return G


def apply_similarity(cams, Xs, s, Q, u):
    """X -> s Q X + u, T_cw -> (R Q^T, s t - R Q^T u): camera coordinates scale by s, images unchanged."""
    Xn = (s * (Q @ Xs.T)).T + u
    cn = [L.make_T(T[:3, :3] @ Q.T, s * T[:3, 3] - T[:3, :3] @ Q.T @ u) for T in cams]
    return cn, Xn


def principal_angles(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(sv, -1, 1))


# ----------------------------------------------------------------------------- scale drift and pose graphs

def drift_experiment(K=60, radius=10.0, seed=11, trend=-0.006, walk=0.01, rot_noise_deg=0.3, trans_noise=0.02):
    """Circle of K + 1 keyframes (k = K coincides with k = 0). Odometry translations measured at drifting scale s_k."""
    rng = np.random.default_rng(seed)
    th = 2 * np.pi * np.arange(K + 1) / K
    P = np.stack([radius * np.sin(th), radius * (1 - np.cos(th)), np.zeros(K + 1)], 1)
    Rs = [L.expSO3([0, 0, t]) for t in th]
    logs = np.concatenate([[0.0], np.cumsum(trend + walk * rng.normal(size=K))])
    s = np.exp(logs)
    Z = []
    for k in range(K):
        dR = Rs[k].T @ Rs[k + 1]
        dt = Rs[k].T @ (P[k + 1] - P[k])
        dRn = L.expSO3(np.radians(rot_noise_deg) * rng.normal(size=3)) @ dR
        dtn = s[k] * dt + trans_noise * s[k] * rng.normal(size=3)
        Z.append(L.make_T(dRn, dtn))
    DR = [np.eye(4)]
    for k in range(K):
        DR.append(DR[-1] @ Z[k])
    return dict(P=P, Rs=Rs, s=s, Z=Z, DR=DR, K=K)


def to_sim(T, scale=1.0):
    S = np.eye(4)
    S[:3, :3] = scale * T[:3, :3]
    S[:3, 3] = T[:3, 3]
    return S


def pose_graph(exp_data, group, iters=15, tol=1e-10):
    """Gauss-Newton pose graph over SE(3) or Sim(3) with the loop edge K -> 0. Returns vertices and costs."""
    Z, DR, K, s = exp_data["Z"], exp_data["DR"], exp_data["K"], exp_data["s"]
    if group == "sim":
        Zs = [to_sim(z) for z in Z]
        Zl = to_sim(np.eye(4), 1.0 / s[-1])      # 3D-3D alignment of the two local maps: scale ratio s_0/s_K
        dim, Exp, Log, inv = 7, L.expSim3, L.logSim3, np.linalg.inv
        X = [to_sim(T) for T in DR]
    else:
        Zs = list(Z)
        Zl = np.eye(4)                           # rigid loop constraint (same place, same orientation)
        dim, Exp, Log, inv = 6, L.expSE3, L.logSE3, L.inv_T
        X = [T.copy() for T in DR]
    edges = [(k, k + 1, Zs[k]) for k in range(K)] + [(0, K, Zl)]
    costs = []
    h = 1e-6
    z0 = np.zeros(dim)
    for _ in range(iters):
        n = dim * K
        Hm = np.zeros((n, n))
        g = np.zeros(n)
        cost = 0.0
        for a, b, Zab in edges:
            def res(da, db, a=a, b=b, Zab=Zab):
                return Log(inv(Zab) @ inv(X[a] @ Exp(da)) @ (X[b] @ Exp(db)))
            r0 = res(z0, z0)
            cost += r0 @ r0
            Ja = np.column_stack([(res(h * e, z0) - res(-h * e, z0)) / (2 * h) for e in np.eye(dim)])
            Jb = np.column_stack([(res(z0, h * e) - res(z0, -h * e)) / (2 * h) for e in np.eye(dim)])
            blocks = [(a, Ja), (b, Jb)]
            for i, Ji in blocks:
                if i == 0:
                    continue
                si = slice(dim * (i - 1), dim * i)
                g[si] += Ji.T @ r0
                for j, Jj in blocks:
                    if j == 0:
                        continue
                    Hm[si, slice(dim * (j - 1), dim * j)] += Ji.T @ Jj
        costs.append(cost)
        d = -np.linalg.solve(Hm, g)
        for i in range(1, K + 1):
            X[i] = X[i] @ Exp(d[dim * (i - 1):dim * i])
        if np.linalg.norm(d) < tol:
            break
    return X, costs


def umeyama(A, B):
    """Similarity (s, R, t) minimizing sum |s R a_i + t - b_i|^2."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    sc = np.trace(np.diag(S) @ D) / np.mean(np.sum(A0 ** 2, 1))
    return sc, R, mb - sc * R @ ma


def rms(A, B):
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, 1))))
