"""Figure 29.4.1: SPD(2) is an open convex cone (Section 29.4).

Coordinates (x, y, z) on Sym(2): Sigma = [[z + x, y], [y, z - x]]. Then det Sigma = z^2 - x^2 - y^2 and the
eigenvalues are z +- sqrt(x^2 + y^2), so SPD(2) = {z > sqrt(x^2 + y^2)}: the inside of a round cone.
(a) 3D: the boundary cone z = sqrt(x^2 + y^2) for 0 <= z <= 1.6 (gray), its axis (isotropic matrices z I, dashed),
    and the slice z = 1 (trace 2, blue circle).
(b) The slice z = 1 seen from above: at points (x, y) = r (cos psi, sin psi), r < 1, the 1-sigma ellipse of Sigma drawn
    as a small glyph (semi-axes 0.13 sqrt(1 +- r), major axis at angle psi/2). The circle r = 1 is the boundary
    (det = 0: degenerate, infinitely flat Gaussians).
Also writes an interactive version with the four geodesics of Figure 29.4.3 drawn inside the cone.
Self-checks: glyph eigen-decomposition agrees with z +- r and angle psi/2.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-4-1-spd-cone.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def to_xyz(S):
    return np.array([(S[0, 0] - S[1, 1]) / 2, S[0, 1], (S[0, 0] + S[1, 1]) / 2])


def from_xyz(p):
    x, y, z = p
    return np.array([[z + x, y], [y, z - x]])


fig = plt.figure(figsize=(7.8, 3.9))
ax = dgfig.axes3d(fig, pos=121, elev=18, azim=-60)
zz, pp = np.meshgrid(np.linspace(0, 1.6, 25), np.linspace(0, 2 * np.pi, 61), indexing="ij")
X, Y, Z = zz * np.cos(pp), zz * np.sin(pp), zz
dgfig.surface(ax, X, Y, Z, alpha=0.3, grid_every=4, zorder=1)
tt = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(tt), np.sin(tt), np.ones_like(tt), color=C["tangent"], lw=1.6, zorder=5)
ax.plot([0, 0], [0, 0], [0, 1.75], color=C["main"], lw=1.0, ls=(0, (4, 3)), zorder=6)
ax.text(0.05, 0, 1.8, r"axis: $zI$ (isotropic)", fontsize=10, zorder=7)
ax.text(1.25, -0.9, 0.55, r"$\det\Sigma = 0$", fontsize=10, color="#555555", zorder=7)
ax.text(-1.45, 0.5, 1.05, r"slice $z = 1$", fontsize=10, color=C["tangent"], zorder=7)
ax.scatter([0], [0], [0], color=C["main"], s=10)
ax.text(0.08, 0.05, -0.12, r"$0$", fontsize=10)
dgfig.equal_aspect(ax, np.stack([X, Y, Z], -1).reshape(-1, 3), zoom=1.3)
ax.set_title(r"(a) $\mathrm{SPD}(2) = \{z > \sqrt{x^2 + y^2}\}$", fontsize=11)

axb = fig.add_subplot(122)
axb.add_patch(plt.Circle((0, 0), 1.0, fc=C["region"], alpha=0.12, ec=C["tangent"], lw=1.6))
glyph = 0.13
t = np.linspace(0, 2 * np.pi, 80)
for r in (0.0, 0.35, 0.65, 0.9):
    psis = [0.0] if r == 0 else np.linspace(0, 2 * np.pi, {0.35: 6, 0.65: 10, 0.9: 14}[r], endpoint=False)
    for psi in psis:
        p = np.array([r * np.cos(psi), r * np.sin(psi), 1.0])
        S = from_xyz(p)
        w, V = np.linalg.eigh(S)
        assert np.allclose(w, [1 - r, 1 + r])
        if r > 0:
            ang = np.arctan2(V[1, 1], V[0, 1]) % np.pi
            assert abs(np.sin(ang - psi / 2)) < 1e-9
        E = p[:2] + glyph * (np.stack([np.cos(t), np.sin(t)], 1) * np.sqrt(w)) @ V.T
        axb.fill(*E.T, color=C["main"], alpha=0.75, lw=0)
axb.annotate(r"center: $\Sigma = I$" + "\n(circle)", xy=(0, 0), xytext=(-0.55, -1.42), fontsize=10, ha="center",
             arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
axb.annotate("near the boundary:\nalmost flat", xy=(0.9 * np.cos(np.radians(-26)), 0.9 * np.sin(np.radians(-26))),
             xytext=(0.8, -1.42), fontsize=10, ha="center", arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
axb.text(0.72, 0.86, r"$r = 1$: $\det\Sigma = 0$", fontsize=10, color=C["tangent"])
dgfig.schematic_axes(axb, (-1.3, 1.3), (-1.75, 1.25))
axb.set_title(r"(b) slice $z = 1$: glyph = $1\sigma$ ellipse", fontsize=11)
fig.subplots_adjust(wspace=0.05)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: cone + four geodesics of Figure 29.4.3
import plotly.graph_objects as go


def fun(S, f):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(f(w)) @ V.T


def geo(S0, S1, s, kind):
    if kind == "Euclidean":
        return (1 - s) * S0 + s * S1
    if kind == "log-Euclidean":
        return fun((1 - s) * fun(S0, np.log) + s * fun(S1, np.log), np.exp)
    h = fun(S0, np.sqrt); hi = np.linalg.inv(h)
    if kind == "affine-invariant":
        return h @ fun(hi @ S1 @ hi, lambda w: w ** s) @ h
    T = hi @ fun(h @ S1 @ h, np.sqrt) @ hi
    M = (1 - s) * np.eye(2) + s * T
    return M @ S0 @ M


def R2(d):
    a = np.radians(d)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


F0 = np.diag([1.0, 0.15 ** 2])
F1 = R2(70) @ F0 @ R2(70).T
cols = {"Euclidean": "#000000", "log-Euclidean": "#0072B2", "affine-invariant": "#E69F00", "Bures-Wasserstein": "#009E73"}
zz2, pp2 = np.meshgrid(np.linspace(0, 0.75, 20), np.linspace(0, 2 * np.pi, 73), indexing="ij")
traces = [go.Surface(x=zz2 * np.cos(pp2), y=zz2 * np.sin(pp2), z=zz2, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]],
                     showscale=False, opacity=0.35, hoverinfo="skip", name="det = 0")]
for kind, col in cols.items():
    P = np.array([to_xyz(geo(F0, F1, s, kind)) for s in np.linspace(0, 1, 101)])
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=col, width=6), name=kind))
E01 = np.array([to_xyz(F0), to_xyz(F1)])
traces.append(go.Scatter3d(x=E01[:, 0], y=E01[:, 1], z=E01[:, 2], mode="markers", marker=dict(size=5, color="#D55E00"),
                           name="Sigma_0, Sigma_1"))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x = (a-c)/2", yaxis_title="y = b",
                              zaxis_title="z = (a+c)/2"),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="SPD(2) as the cone z > sqrt(x^2 + y^2), with four geodesics between two flat "
                                   "ellipses 70 degrees apart", x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-29-4-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
