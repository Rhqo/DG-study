"""Figure 37.3.1: the Frenet frame and a Bishop frame along the helix (a, b) = (1, 0.3).

(a) Frenet frames (t, n, b) at seven points of one and a half turns; (b) the Bishop frame (t, u, w) with u(0) = n(0):
u = cos(theta) n + sin(theta) b, w = -sin(theta) n + cos(theta) b with theta(s) = -tau s (Proposition 37.3.10).
Also writes an interactive plotly version (-interactive.html).

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-3-1-helix-frames.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
a, b = 1.0, 0.3
kap, tau = ddg37.helix_kappa_tau(a, b)
c = np.sqrt(a * a + b * b)


def frenet(s):
    """Example 5.3.3: gamma, t, n, b of the arc-length parametrized helix."""
    g = np.array([a * np.cos(s / c), a * np.sin(s / c), b * s / c])
    T = np.array([-a / c * np.sin(s / c), a / c * np.cos(s / c), b / c])
    N = -np.array([np.cos(s / c), np.sin(s / c), 0.0])
    B = np.array([b / c * np.sin(s / c), -b / c * np.cos(s / c), a / c])
    return g, T, N, B


S = np.linspace(0, 3 * np.pi * c, 7)
frames = []
for s in S:
    g, T, N, B = frenet(s)
    assert np.allclose(np.cross(T, N), B) and abs(np.dot(T, N)) < 1e-12
    th = -tau * s
    U = np.cos(th) * N + np.sin(th) * B
    W = -np.sin(th) * N + np.cos(th) * B
    assert np.allclose(np.cross(T, U), W)
    frames.append((g, T, N, B, U, W))
# Bishop check: u' has no w component, by finite differences
h = 1e-6
for s in (0.7, 2.3):
    def uw(s_):
        g, T, N, B = frenet(s_)
        th = -tau * s_
        return np.cos(th) * N + np.sin(th) * B, -np.sin(th) * N + np.cos(th) * B
    up = (uw(s + h)[0] - uw(s - h)[0]) / (2 * h)
    assert abs(np.dot(up, uw(s)[1])) < 1e-8

ss = np.linspace(0, 3 * np.pi * c, 400)
curve = np.array([frenet(s)[0] for s in ss])
SC = 0.5

fig = plt.figure(figsize=(7.0, 3.7))
for j, (title, kind) in enumerate([("(a)", "frenet"), ("(b)", "bishop")]):
    ax = dgfig.axes3d(fig, 121 + j, elev=30, azim=-60)
    dgfig.curve3d(ax, curve, lw=1.6)
    for k, (g, T, N, B, U, W) in enumerate(frames):
        lab = k == 0
        dgfig.arrow3d(ax, g, SC * T, role="tangent", label=r"$\mathbf{t}$" if lab else None, head=8, lw=1.3,
                      fontsize=11)
        if kind == "frenet":
            dgfig.arrow3d(ax, g, SC * N, role="normal", label=r"$\mathbf{n}$" if lab else None, head=8, lw=1.3,
                          fontsize=11)
            dgfig.arrow3d(ax, g, SC * B, role="third", label=r"$\mathbf{b}$" if lab else None, head=8, lw=1.3,
                          fontsize=11)
        else:
            dgfig.arrow3d(ax, g, SC * U, role="accent", label=r"$\mathbf{u}$" if lab else None, head=8, lw=1.3,
                          fontsize=11)
            dgfig.arrow3d(ax, g, SC * W, role="covector", label=r"$\mathbf{w}$" if lab else None, head=8, lw=1.3,
                          fontsize=11)
    dgfig.equal_aspect(ax, np.vstack([curve + 0.45, curve - 0.45]), zoom=1.3)
    ax.set_title(title, fontsize=12, loc="left")
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93, wspace=0.0)
dgfig.save(fig, __file__)

# interactive version: the helix with both frames at 13 points
import plotly.graph_objects as go  # noqa: E402

S2 = np.linspace(0, 3 * np.pi * c, 13)
tr = [go.Scatter3d(x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], mode="lines", line=dict(color="#000000", width=4),
                   hoverinfo="skip", name="helix")]
P = np.array([frenet(s)[0] for s in S2])
for idx, (col, name) in enumerate([(C["tangent"], "t"), (C["normal"], "n (Frenet)"), (C["third"], "b (Frenet)")]):
    D = np.array([frenet(s)[1 + idx] for s in S2]) * SC
    tr += dm.plotly_arrows(P, D, color=col, cone=0.12, name=name)
U2 = np.array([np.cos(-tau * s) * frenet(s)[2] + np.sin(-tau * s) * frenet(s)[3] for s in S2]) * SC
W2 = np.array([-np.sin(-tau * s) * frenet(s)[2] + np.cos(-tau * s) * frenet(s)[3] for s in S2]) * SC
tr += dm.plotly_arrows(P, U2, color=C["accent"], cone=0.12, name="u (Bishop)")
tr += dm.plotly_arrows(P, W2, color=C["covector"], cone=0.12, name="w (Bishop)")
dm.plotly_save(tr, __file__, "Helix (a, b) = (1, 0.3): Frenet frame t (blue), n (vermilion), b (green); "
               "Bishop frame u (orange), w (purple)", "fig-37-3-1")
