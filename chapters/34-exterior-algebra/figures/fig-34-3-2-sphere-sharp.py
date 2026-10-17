"""Figure 34.3.2: the same covector components at two points of the unit sphere give different vectors under
sharp (Section 34.3, Example 34.3.5).

Sphere S^2(r), r = 1, x(theta, phi) = (sin th cos ph, sin th sin ph, cos th) (GUIDELINES 7). At
p1 = x(pi/2, 0.35) (equator) and p2 = x(pi/6, 0.35), the covector beta^2 (dual to x_phi, components (0, 1))
is sharpened with the first fundamental form G = diag(r^2, r^2 sin^2 th): (beta^2)^sharp = x_phi / (r^2 sin^2 th).
Blue: x_phi (length r sin th = 1 and 0.5); orange: (beta^2)^sharp (length 1/(r sin th) = 1 and 2). Both drawn at
half length (scale 0.5). Purple curves: meridians phi = 0.35 + 0.18 j (j = -3..3), the level curves of phi;
they crowd together near the pole, so phi changes fast there and its sharp is long.
Self-checks: G at both points, g((beta^2)^sharp, w) = beta^2(w) for w = x_theta, x_phi, and the lengths.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
r = 1.0


def X(th, ph):
    return r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def X_th(th, ph):
    return r * np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])


def X_ph(th, ph):
    return r * np.array([-np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), 0.0])


ph0 = 0.35
pts = {"p1": np.pi / 2, "p2": np.pi / 6}
sharp = {}
for name, th in pts.items():
    xt, xp = X_th(th, ph0), X_ph(th, ph0)
    G = np.array([[xt @ xt, xt @ xp], [xp @ xt, xp @ xp]])
    assert np.allclose(G, np.diag([r ** 2, r ** 2 * np.sin(th) ** 2]))
    comp = np.linalg.solve(G, np.array([0.0, 1.0]))          # components of (beta^2)^sharp in (x_th, x_ph)
    s = comp[0] * xt + comp[1] * xp
    assert np.allclose(s, xp / (r ** 2 * np.sin(th) ** 2))
    assert abs(s @ xt - 0.0) < 1e-12 and abs(s @ xp - 1.0) < 1e-12   # g(sharp, x_th) = 0, g(sharp, x_ph) = 1
    sharp[name] = s
assert abs(np.linalg.norm(sharp["p1"]) - 1.0) < 1e-12 and abs(np.linalg.norm(sharp["p2"]) - 2.0) < 1e-12
assert abs(np.linalg.norm(X_ph(np.pi / 6, ph0)) - 0.5) < 1e-12

fig = plt.figure(figsize=(5.0, 4.6))
ax = dgfig.axes3d(fig, elev=22, azim=30)
th = np.linspace(0, np.pi, 61)
ph = np.linspace(0, 2 * np.pi, 121)
TH, PH = np.meshgrid(th, ph, indexing="ij")
S = r * np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False)
view = dgfig.view_vector(ax)

# meridians (level curves of phi) and the equator for orientation
tt = np.linspace(0.02, np.pi - 0.02, 200)
for j in range(-3, 4):
    curve = np.array([X(t_, ph0 + 0.18 * j) for t_ in tt])
    dgfig.curve3d(ax, curve, role="covector", lw=1.0 if j else 1.4, visible=curve @ view > 0, hidden=None, zorder=4)
eq = np.array([X(np.pi / 2, p_) for p_ in ph])
dgfig.curve3d(ax, eq, role="aux", lw=0.6, visible=eq @ view > 0, hidden="dashed", zorder=3)

scale = 0.5
for name, th0 in pts.items():
    p = X(th0, ph0)
    dgfig.point3d(ax, p, zorder=12)
    dgfig.arrow3d(ax, p, X_ph(th0, ph0), role="tangent", scale=scale, zorder=13)
    dgfig.arrow3d(ax, p, sharp[name], role="accent", scale=scale, zorder=12)
lab = {"p1": (0.0, 0.0, -0.22), "p2": (0.0, -0.05, 0.12)}
ax.text(*(X(pts["p1"], ph0) + np.array(lab["p1"])), r"$p_1$", fontsize=12, ha="center")
ax.text(*(X(pts["p2"], ph0) + np.array(lab["p2"])), r"$p_2$", fontsize=12, ha="center")
tip2 = X(pts["p2"], ph0) + scale * sharp["p2"]
ax.text(*(tip2 + np.array([0.0, 0.05, 0.1])), r"$(\beta^2)^\sharp$", color=C["accent"], fontsize=12, ha="center")
tipx = X(pts["p1"], ph0) + scale * X_ph(pts["p1"], ph0)
ax.text(*(tipx + np.array([0.0, 0.12, -0.12])), r"$\mathbf{x}_\varphi = (\beta^2)^\sharp$", color=C["tangent"],
        fontsize=11, ha="left")
tipx2 = X(pts["p2"], ph0) + scale * X_ph(pts["p2"], ph0)
ax.text(*(tipx2 + np.array([0.0, -0.05, -0.16])), r"$\mathbf{x}_\varphi$", color=C["tangent"], fontsize=11, ha="center")

dgfig.equal_aspect(ax, S.reshape(-1, 3), np.array([tip2]), zoom=1.05)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
dgfig.save(fig, __file__)
