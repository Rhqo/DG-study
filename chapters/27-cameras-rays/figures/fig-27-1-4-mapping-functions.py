"""Figure 27.1.4: every central camera is a chart r(theta) of the sphere of directions (Section 27.1).

Image radius r/f as a function of the angle theta between the ray and the optical axis, for the five projections
listed by Kannala and Brandt [Kannala06]: perspective tan(theta) (the gnomonic chart, i.e. the pinhole camera),
stereographic 2 tan(theta/2), equidistant theta, equisolid 2 sin(theta/2), orthographic sin(theta).
Self-checks: the five curves agree to first order at theta = 0 (r ~ theta); tan blows up at 90 deg; sin has its
maximum at 90 deg; the values quoted in the text.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-1-4-mapping-functions.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
models = [
    ("perspective (pinhole) $\\tan\\theta$", np.tan, C["main"]),
    ("stereographic $2\\tan(\\theta/2)$", lambda t: 2 * np.tan(t / 2), C["tangent"]),
    ("equidistant $\\theta$", lambda t: t, C["accent"]),
    ("equisolid $2\\sin(\\theta/2)$", lambda t: 2 * np.sin(t / 2), C["third"]),
    ("orthographic $\\sin\\theta$", np.sin, C["covector"]),
]
small = 1e-4
for _, f, _ in models:
    assert abs(f(small) / small - 1) < 1e-6          # all charts agree to first order at the axis
assert abs(np.tan(np.radians(80)) - 5.671) < 1e-3 and abs(np.radians(95) - 1.658) < 1e-3

fig, ax = plt.subplots(figsize=(6.6, 4.3))
th = np.linspace(0, np.radians(179), 1000)
for name, f, col in models:
    y = f(th)
    if f is np.tan:
        tt = np.linspace(0, np.radians(88.5), 600)
        ax.plot(np.degrees(tt), np.tan(tt), color=col, lw=2.0, label=name)
    elif f is np.sin:
        tt = np.linspace(0, np.radians(90), 300)
        ax.plot(np.degrees(tt), np.sin(tt), color=col, lw=1.8, label=name)
        tt2 = np.linspace(np.radians(90), np.radians(179), 300)
        ax.plot(np.degrees(tt2), np.sin(tt2), color=col, lw=1.0, ls=":")
    else:
        ax.plot(np.degrees(th), y, color=col, lw=1.8, label=name)
ax.axvline(90, color=C["aux"], lw=1.0, ls="--")
ax.axvspan(90, 180, color=C["surface"], alpha=0.35, lw=0)
ax.text(93, 0.06, "behind the camera plane ($t_z<0$)", fontsize=10, color=C["aux"])
ax.annotate("$r\\to\\infty$ as $\\theta\\to 90^\\circ$", xy=(84.5, 3.6), xytext=(30, 3.6), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
ax.plot([95], [np.radians(95)], "o", color=C["accent"], ms=5)
ax.text(97, 1.12, r"$95^\circ$: $r=1.66\,f$", fontsize=10, color=C["accent"])
ax.set_xlim(0, 180)
ax.set_ylim(0, 4.2)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.set_xlabel(r"angle between ray and optical axis $\theta$ (deg)")
ax.set_ylabel(r"image radius $r/f$")
labels = [("perspective\n(pinhole) $\\tan\\theta$", (8, 2.3), C["main"]),
          ("stereographic\n$2\\tan(\\theta/2)$", (155, 3.6), C["tangent"]),
          ("equidistant $\\theta$", (150, 2.95), C["accent"]),
          ("equisolid $2\\sin(\\theta/2)$", (152, 1.6), C["third"]),
          ("orthographic $\\sin\\theta$", (30, 0.18), C["covector"])]
for txt, xy, col in labels:
    ax.text(*xy, txt, fontsize=10, color=col, ha="center" if xy[0] > 100 else "left")
ax.set_title("Lens projections as charts $r(\\theta)$ of the sphere of directions", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
