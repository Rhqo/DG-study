"""Figure 44.1.1: geodesics on the unit sphere and on the cylinder (Section 44.1).

(a) unit sphere, view from elev 22, azim 0: the points p = x(pi/3, -1.1), q = x(pi/3, 1.1) on the circle of latitude
    theta = pi/3; the short great-circle arc pq (orange, solid), the rest of that great circle (orange, dashed: the
    long arc), and the latitude arc between p and q (black). Lengths are printed in the caption.
(b) the cylinder of radius 1 cut along u = 0 and unrolled to the (u, v) plane (isometry, Example 7.2.9): the helix
    (a, b) = (1, 0.3) becomes the line v = 0.3 u; the points p = x(0, 0) and q = x(0, 0.6 pi) (one turn of the helix
    above p) have the lifts (2 pi k, 0.6 pi); the three lines from (0, 0) to (0, 0.6 pi), (2 pi, 0.6 pi),
    (-2 pi, 0.6 pi) are three geodesics from p to q.
Self-checks: the great arc is shorter than the latitude arc; the lines have the lengths printed.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-1-1-two-features.py
"""

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS


def X(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th) + 0 * ph], -1)


th0, ph0 = np.pi / 3, 1.1
p, q = X(th0, -ph0), X(th0, ph0)
ang = float(np.arccos(p @ q))
lat_len = np.sin(th0) * 2 * ph0
assert ang < lat_len
print(f"great arc {ang:.4f}, long arc {2 * np.pi - ang:.4f}, latitude arc {lat_len:.4f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.0, 1.25]})
cam = dm.Ortho(elev=22, azim=0)


def draw3(ax, P, front_style, back_style, **kw):
    P = np.asarray(P)
    vis = P @ cam.view > -1e-9
    Q = cam(P)
    for mask, style in ((vis, front_style), (~vis, back_style)):
        seg = np.where(mask, 1.0, np.nan)
        ax.plot(Q[:, 0] * seg, Q[:, 1] * seg, **{**kw, **style})


# sphere outline and a few grid circles
tt = np.linspace(0, 2 * np.pi, 400)
axa.plot(np.cos(tt), np.sin(tt), color=col["main"], lw=1.0)
for th in (np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6):
    draw3(axa, X(th, tt), dict(ls="-"), dict(ls=":"), color=col["aux"], lw=0.4)
for ph in np.linspace(0, np.pi, 6, endpoint=False):
    draw3(axa, X(tt, ph + 0 * tt), dict(ls="-"), dict(ls=":"), color=col["aux"], lw=0.4)
# great circle through p, q
n = np.cross(p, q)
n /= np.linalg.norm(n)
e2 = np.cross(n, p)
s = np.linspace(0, ang, 200)
short = np.outer(np.cos(s), p) + np.outer(np.sin(s), e2)
s2 = np.linspace(ang, 2 * np.pi, 400)
long_ = np.outer(np.cos(s2), p) + np.outer(np.sin(s2), e2)
draw3(axa, long_, dict(ls="--"), dict(ls=":"), color=col["accent"], lw=1.4)
draw3(axa, short, dict(ls="-"), dict(ls=":"), color=col["accent"], lw=2.4)
lat = X(th0, np.linspace(-ph0, ph0, 200))
draw3(axa, lat, dict(ls="-"), dict(ls=":"), color=col["main"], lw=1.8)
for P_, lab, off in ((p, r"$p$", (-0.13, -0.02)), (q, r"$q$", (0.05, -0.02))):
    c = cam(P_[None])[0]
    axa.plot(*c, "o", color=col["main"], ms=4, zorder=5)
    axa.text(c[0] + off[0], c[1] + off[1], lab, fontsize=12)
mid_s = cam(short[100][None])[0]
axa.annotate("great arc", xy=mid_s, xytext=(mid_s[0] - 0.8, mid_s[1] + 0.33), fontsize=10, color=col["accent"],
             arrowprops=dict(arrowstyle="-", color=col["accent"], lw=0.8))
mid_l = cam(lat[100][None])[0]
axa.annotate("latitude", xy=mid_l, xytext=(mid_l[0] + 0.2, mid_l[1] - 0.5), fontsize=10,
             arrowprops=dict(arrowstyle="-", color=col["main"], lw=0.8))
axa.text(-0.98, -1.12, "long arc (dashed)", fontsize=9.5, color=col["accent"])
axa.set_aspect("equal")
axa.set_xlim(-1.15, 1.15)
axa.set_ylim(-1.2, 1.15)
axa.set_axis_off()
axa.set_title("(a) unit sphere", fontsize=11)

# (b) unrolled cylinder
H = 0.6 * np.pi
axb.axvspan(0, 2 * np.pi, color=col["region"], alpha=0.18, lw=0)
for k in (-1, 0, 1, 2):
    axb.axvline(2 * np.pi * k, color=col["aux"], lw=0.6, ls="--")
uu = np.linspace(-2 * np.pi, 4 * np.pi, 10)
axb.plot(uu, 0.3 * uu, color=col["aux"], lw=0.8)
ends = [(0.0, H), (2 * np.pi, H), (-2 * np.pi, H)]
styles = [dict(color=col["accent"], lw=2.4), dict(color=col["tangent"], lw=1.8), dict(color=col["third"], lw=1.8)]
lengths = []
for (u1, v1), st in zip(ends, styles):
    axb.plot([0, u1], [0, v1], **st)
    lengths.append(float(np.hypot(u1, v1)))
assert abs(lengths[0] - H) < 1e-12 and abs(lengths[1] - 2 * np.pi * np.sqrt(1 + 0.09)) < 1e-12
print("geodesics p -> q on the cylinder:", [round(L_, 4) for L_ in lengths])
for (u1, v1) in [(0, 0)] + ends:
    axb.plot(u1, v1, "o", color=col["main"], ms=4, zorder=5)
axb.text(0.15, -0.28, r"$p$", fontsize=12)
axb.text(0.15, H + 0.08, r"$q$", fontsize=12)
axb.text(2 * np.pi + 0.1, H + 0.08, r"$q$", fontsize=12)
axb.text(-2 * np.pi + 0.1, H + 0.08, r"$q$", fontsize=12)
axb.text(3.6, 0.3 * 3.6 - 0.32, r"$v = 0.3u$", fontsize=10, color=col["aux"])
axb.set_xlim(-2 * np.pi - 0.4, 2 * np.pi + 1.4)
axb.set_ylim(-0.5, H + 0.5)
axb.set_xlabel(r"$u$ (cut along $u = 0$, period $2\pi$)")
axb.set_ylabel(r"$v$")
axb.set_xticks([-2 * np.pi, 0, 2 * np.pi])
axb.set_xticklabels([r"$-2\pi$", r"$0$", r"$2\pi$"])
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b) unrolled cylinder, $r = 1$ (axes not to scale)", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.17, wspace=0.12)
dgfig.save(fig, __file__)
