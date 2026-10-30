"""Verification for Section 48.7 (Assessment).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-7-assessment.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "48-7-assessment.html"


def junction(N, M=4000):
    """N half-lines of length 1 in the directions of a regular N-gon, midpoint sampling with spacing 1/M,
    masses 1/M, exact planes (n = 2, d = 1)."""
    h = 1.0 / M
    s = h * (np.arange(M) + 0.5)
    X, P = [], []
    for l in range(N):
        a = 2 * np.pi * l / N
        u = np.array([np.cos(a), np.sin(a)])
        X.append(s[:, None] * u)
        P.append(np.broadcast_to(np.outer(u, u), (M, 2, 2)))
    return np.concatenate(X), np.concatenate(P), np.full(N * M, h)


eps = 0.1
res = {}
for N in (3, 5, 7):
    X, P, m = junction(N)
    A0 = v.wsff(np.zeros(2), X, P, m, eps, 2, 1)
    c0 = v.c_eps(np.zeros(2), X, P, m, eps)
    i = int(np.argmin(np.abs(X[:4000, 0] - 0.03)))                  # an atom on branch 0 at s = 0.03
    Ap = v.wsff_perp(X[i], P[i], X, P, m, eps, 2, 1)
    Aw = v.wsff(X[i], X, P, m, eps, 2, 1)
    H0 = v.H_from_beta(v.beta_tensor(np.zeros(2), X, P, m, eps, 2, 1))
    res[N] = dict(A0=np.abs(A0).max(), c0=np.linalg.eigvalsh(c0), Ap=np.abs(Ap).max(), Aw=np.abs(Aw).max(),
                  H0=np.linalg.norm(H0))
    print(N, {k: (np.round(val, 4).tolist() if np.ndim(val) else float("%.3e" % val)) for k, val in res[N].items()})

check("Our Evaluation 5 / Exercise 48.7.2 (a): regular pentagon and heptagon junctions: epsilon-WSFF = 0 at the junction",
      res[5]["A0"] < 1e-12 and res[7]["A0"] < 1e-12)
check("Exercise 48.7.2 (a): pentagon: epsilon-WSFF and epsilon-WSFF-perp about 1e-7 on a branch (s = 0.03)",
      res[5]["Ap"] < 1e-6 and res[5]["Aw"] < 1e-6)
close("Exercise 48.7.2 (c): Y junction: max |A^eps(0)| = 4.14 (eps = 0.1)", res[3]["A0"], 4.14, 0.005)
check("Exercise 48.7.2 (c): Y junction: epsilon-WSFF-perp nonzero on a branch", res[3]["Ap"] > 1.0)
check("Exercise 48.7.2 (b): c^{V,eps}(0) has eigenvalues 1/2, 1/2 for N = 3, 5, 7 (not a projection)",
      all(np.allclose(res[N]["c0"], [0.5, 0.5], atol=1e-12) for N in (3, 5, 7)))
check("Exercise 48.7.2 (c): mean curvature 0 at the junction for N = 3 and 5 (stationary)",
      res[3]["H0"] < 1e-12 and res[5]["H0"] < 1e-12)
al = 2 * np.pi * np.arange(5) / 5
check("Exercise 48.7.2 (b): sum cos^2 = sum sin^2 = 5/2, sum cos sin = 0 for the pentagon",
      np.isclose(np.sum(np.cos(al) ** 2), 2.5) and np.isclose(np.sum(np.sin(al) ** 2), 2.5)
      and abs(np.sum(np.cos(al) * np.sin(al))) < 1e-12)
check("Proposition 48.4.4 for the pentagon: sum e^{i a} = sum e^{3 i a} = 0; Y junction: sum e^{3 i a} = 3",
      abs(np.exp(1j * al).sum()) < 1e-12 and abs(np.exp(3j * al).sum()) < 1e-12
      and np.isclose(np.exp(3j * 2 * np.pi * np.arange(3) / 3).sum(), 3))

# ---------------------------------------------------------------- numbers quoted from Section 48.6
close("Table 48.7.1, row 6: noise 0.05 is 37% of eps = 0.134", 0.05 / 0.134, 0.373, 5e-4)
html = PAGE.read_text(encoding="utf-8")
body = html.split('id="tab-48-7-2"')[1].split("</table>")[0]
rows = re.findall(r"<tr><td>(\d+)</td>", body)
check("Table 48.7.2 has 15 numbered rows (Summary: 15 typos and inconsistencies)",
      rows == [str(k) for k in range(1, 16)])
summary()
