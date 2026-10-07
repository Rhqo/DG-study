"""Figure 29.6.1: the 16 real spherical harmonics Y_lm, l = 0..3, as colors on the unit sphere (Section 29.6).

Row l = 0, 1, 2, 3 (2l + 1 functions each), column m = -l..l, with the polynomial forms of Table (29.6.3).
Color: Y_lm(d) / max |Y_lm| on the diverging colormap RdBu_r (red positive, blue negative, white 0).
All spheres use the same view (elevation 20 deg, azimuth -60 deg); the small triad at the upper left shows x, y, z.
Under each sphere: a polynomial proportional to Y_lm on the sphere (self-checked).
Also writes an interactive plotly version (fig-29-6-1-sh-basis-interactive.html).
Self-checks: orthonormality of the 16 functions by exact quadrature.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-6-1-sh-basis.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

C = dgfig.COLORS
pi = np.pi
SH = [
    (0, 0, lambda x, y, z: 0 * x + 1 / (2 * np.sqrt(pi))),
    (1, -1, lambda x, y, z: np.sqrt(3 / (4 * pi)) * y), (1, 0, lambda x, y, z: np.sqrt(3 / (4 * pi)) * z),
    (1, 1, lambda x, y, z: np.sqrt(3 / (4 * pi)) * x),
    (2, -2, lambda x, y, z: np.sqrt(15 / pi) / 2 * x * y), (2, -1, lambda x, y, z: np.sqrt(15 / pi) / 2 * y * z),
    (2, 0, lambda x, y, z: np.sqrt(5 / pi) / 4 * (2 * z * z - x * x - y * y)),
    (2, 1, lambda x, y, z: np.sqrt(15 / pi) / 2 * x * z), (2, 2, lambda x, y, z: np.sqrt(15 / pi) / 4 * (x * x - y * y)),
    (3, -3, lambda x, y, z: np.sqrt(35 / (2 * pi)) / 4 * y * (3 * x * x - y * y)),
    (3, -2, lambda x, y, z: np.sqrt(105 / pi) / 2 * x * y * z),
    (3, -1, lambda x, y, z: np.sqrt(21 / (2 * pi)) / 4 * y * (4 * z * z - x * x - y * y)),
    (3, 0, lambda x, y, z: np.sqrt(7 / pi) / 4 * z * (2 * z * z - 3 * x * x - 3 * y * y)),
    (3, 1, lambda x, y, z: np.sqrt(21 / (2 * pi)) / 4 * x * (4 * z * z - x * x - y * y)),
    (3, 2, lambda x, y, z: np.sqrt(105 / pi) / 4 * z * (x * x - y * y)),
    (3, 3, lambda x, y, z: np.sqrt(35 / (2 * pi)) / 4 * x * (x * x - 3 * y * y)),
]
# self-check: orthonormality
tq, wq = np.polynomial.legendre.leggauss(12)
phq = np.linspace(0, 2 * pi, 24, endpoint=False)
T, PH = np.meshgrid(tq, phq, indexing="ij")
W = (wq[:, None] * np.ones_like(PH) * (2 * pi / 24)).ravel()
X, Y, Z = (np.sqrt(1 - T ** 2) * np.cos(PH)).ravel(), (np.sqrt(1 - T ** 2) * np.sin(PH)).ravel(), T.ravel()
Ym = np.stack([f(X, Y, Z) for _, _, f in SH], 1)
assert np.allclose(Ym.T @ (Ym * W[:, None]), np.eye(16), atol=1e-12)

th, ph = np.meshgrid(np.linspace(0, pi, 41), np.linspace(0, 2 * pi, 81), indexing="ij")
sx, sy, sz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
cmap = plt.get_cmap("RdBu_r")
norm = Normalize(-1, 1)

# short polynomial forms shown under each sphere (proportional to Y_lm; on the sphere x^2 + y^2 + z^2 = 1 is used to
# shorten 2z^2 - x^2 - y^2 = 3z^2 - 1, 4z^2 - x^2 - y^2 = 5z^2 - 1, 2z^2 - 3x^2 - 3y^2 = 5z^2 - 3)
POLY = {(0, 0): ("1", lambda x, y, z: 1 + 0 * x), (1, -1): ("y", lambda x, y, z: y), (1, 0): ("z", lambda x, y, z: z),
        (1, 1): ("x", lambda x, y, z: x), (2, -2): ("xy", lambda x, y, z: x * y), (2, -1): ("yz", lambda x, y, z: y * z),
        (2, 0): ("3z^2-1", lambda x, y, z: 3 * z * z - 1), (2, 1): ("xz", lambda x, y, z: x * z),
        (2, 2): ("x^2-y^2", lambda x, y, z: x * x - y * y),
        (3, -3): ("y(3x^2-y^2)", lambda x, y, z: y * (3 * x * x - y * y)), (3, -2): ("xyz", lambda x, y, z: x * y * z),
        (3, -1): ("y(5z^2-1)", lambda x, y, z: y * (5 * z * z - 1)), (3, 0): ("z(5z^2-3)", lambda x, y, z: z * (5 * z * z - 3)),
        (3, 1): ("x(5z^2-1)", lambda x, y, z: x * (5 * z * z - 1)), (3, 2): ("z(x^2-y^2)", lambda x, y, z: z * (x * x - y * y)),
        (3, 3): ("x(x^2-3y^2)", lambda x, y, z: x * (x * x - 3 * y * y))}
P3 = np.random.default_rng(0).normal(size=(3, 50))
P3 /= np.linalg.norm(P3, axis=0)
for l, m, f in SH:   # self-check: each label polynomial is proportional to Y_lm on the unit sphere
    ratio = f(*P3) / POLY[(l, m)][1](*P3)
    assert np.allclose(ratio, ratio[0])

FW = 8.0
LM = 0.62                             # left margin (in) for the l labels
cw_in = (FW - LM - 0.1) / 7           # physical size of one (square) sphere cell
LAB = 0.26                            # space under each sphere for its polynomial
TM = 0.36                             # top margin for the m labels
FH = 4 * (cw_in + LAB) + TM + 0.05
fig = plt.figure(figsize=(FW, FH))
cell_w, cell_h, row_h = cw_in / FW, cw_in / FH, (cw_in + LAB) / FH
top = 1 - TM / FH
for l, m, f in SH:
    left = LM / FW + (3 + m) * cell_w
    bottom = top - l * row_h - cell_h
    ax = fig.add_axes([left, bottom, cell_w, cell_h], projection="3d", computed_zorder=False)
    vals = f(sx, sy, sz)
    vals = vals / np.max(np.abs(vals))
    ax.plot_surface(sx, sy, sz, facecolors=cmap(norm(vals)), rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False, rasterized=True)
    ax.view_init(elev=20, azim=-60)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()
    fig.text(left + 0.5 * cell_w, bottom + 0.02 / FH, r"$%s$" % POLY[(l, m)][0], fontsize=12, ha="center", va="top")
for l in range(4):
    fig.text(0.01, top - l * row_h - 0.5 * cell_h, r"$\ell = %d$" % l, fontsize=15, va="center")
for m in range(-3, 4):
    fig.text(LM / FW + (3 + m + 0.5) * cell_w, 0.995, r"$m = %d$" % m, fontsize=14, ha="center", va="top")
# axis triad
axt = fig.add_axes([LM / FW + 0.2 * cell_w, top - 1.15 * cell_h, 1.3 * cell_w, 1.1 * cell_h], projection="3d")
axt.view_init(elev=20, azim=-60)
for v, lab in ((np.array([1, 0, 0]), "x"), (np.array([0, 1, 0]), "y"), (np.array([0, 0, 1]), "z")):
    axt.plot([0, v[0]], [0, v[1]], [0, v[2]], color=C["main"], lw=1.3)
    axt.text(*(1.3 * v), f"${lab}$", fontsize=13)
axt.set_xlim(-0.2, 1); axt.set_ylim(-0.2, 1); axt.set_zlim(-0.2, 1)
axt.set_box_aspect((1, 1, 1))
axt.set_axis_off()
cax = fig.add_axes([LM / FW + 5.55 * cell_w, top - row_h - 0.9 * cell_h, 0.018, row_h + 0.8 * cell_h])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-1, 0, 1])
cb.ax.set_yticklabels(["$-1$ (blue)", "$0$", "$+1$ (red)"])
cb.set_label(r"$Y_{\ell m}/\max|Y_{\ell m}|$", fontsize=13)
cb.ax.tick_params(labelsize=12)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go
from plotly.subplots import make_subplots

th2, ph2 = np.meshgrid(np.linspace(0, pi, 41), np.linspace(0, 2 * pi, 81), indexing="ij")
X2, Y2, Z2 = np.sin(th2) * np.cos(ph2), np.sin(th2) * np.sin(ph2), np.cos(th2)
specs = [[{"type": "scene"} if abs(c - 3) <= l else None for c in range(7)] for l in range(4)]
titles = []
for l in range(4):
    for c in range(7):
        if abs(c - 3) <= l:
            titles.append(f"l={l}, m={c - 3}")
pfig = make_subplots(rows=4, cols=7, specs=specs, subplot_titles=titles, horizontal_spacing=0.0, vertical_spacing=0.03)
for l, m, f in SH:
    v = f(X2, Y2, Z2)
    v = v / np.max(np.abs(v))
    pfig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, surfacecolor=v, colorscale="RdBu", reversescale=True, cmin=-1, cmax=1,
                              showscale=(l == 0), colorbar=dict(title="Y/max|Y|", len=0.5),
                              hovertemplate=f"l={l}, m={m}<br>value %{{surfacecolor:.2f}}<extra></extra>"),
                   row=l + 1, col=m + 4)
for i in range(1, 17):
    key = "scene" if i == 1 else f"scene{i}"
    pfig.layout[key].update(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                            camera=dict(eye=dict(x=1.1, y=-1.9, z=0.75)))
pfig.update_layout(height=820, width=1100, margin=dict(l=0, r=0, t=60, b=0),
                   title=dict(text="Real spherical harmonics Y_lm, l = 0..3 (red positive, blue negative); "
                                   "drag each sphere to rotate it", x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-29-6-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
