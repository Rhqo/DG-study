"""Shared helpers for the animations of Chapter 32 (GUIDELINES.md 12.8). Not a render target (the name does not start
with ``anim-``, so render_anims.sh skips it).

Copied and trimmed from chapters/31-meshes/anim/_helpers_c.py (Scene2D with the frame check and the timeline log,
seg, arrow, dot, polyline, Plot, text, math, swap), because an animation must not import another chapter's helpers.
Added here: dashed() for a dashed closed polyline and ring() for a circle drawn as a polyline.

Import from an animation script with::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _helpers as hp
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
    import ddg32
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402

import dganim as da  # noqa: E402
from manim import (BLACK, Arrow, DashedVMobject, Dot, FadeIn, FadeOut, Line, SVGMobject, Text, VGroup,  # noqa: E402
                   VMobject, config)

C = da.C
HEX = da.COLORS
# the four discrete curvatures keep the colours of the static figures (lib/ddg32.KIND_STYLE)
KIND = {"A": da.C["tangent"], "B": da.C["third"], "C": da.C["covector"], "D": da.C["accent"]}


# ---------------------------------------------------------------- small drawing helpers
def seg(p, q, color=BLACK, width=3.0, dashed=False, **kw):
    ln = Line(p, q, color=color, stroke_width=width, **kw)
    return DashedVMobject(ln, num_dashes=max(2, int(np.linalg.norm(np.asarray(q) - np.asarray(p)) / 0.12))) if dashed \
        else ln


def arrow(p, q, color=BLACK, width=5.0, tip=0.22):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q - p) + 1e-12
    return Arrow(p, q, buff=0, color=color, stroke_width=width, max_tip_length_to_length_ratio=min(0.5, tip / L),
                 max_stroke_width_to_length_ratio=100)


def dot(p, color=BLACK, r=0.07):
    return Dot(p, radius=r, color=color)


def to3(P):
    """(m, 2) -> (m, 3) with z = 0."""
    P = np.asarray(P, float)
    return np.c_[P, np.zeros(len(P))] if P.ndim == 2 else np.r_[P, 0.0]


def polyline(points, color=BLACK, width=3.0, closed=False):
    P = np.asarray(points, float)
    if P.shape[-1] == 2:
        P = to3(P)
    if closed:
        P = np.vstack([P, P[:1]])
    m = VMobject(stroke_color=color, stroke_width=width)
    m.set_points_as_corners(P)
    return m


def dashed(points, color=BLACK, width=3.0, closed=True, dash=0.14, ratio=0.55):
    """dashed polyline: dash length about ``dash`` scene units."""
    m = polyline(points, color=color, width=width, closed=closed)
    P = np.asarray(points, float)
    L = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1).sum()
    return DashedVMobject(m, num_dashes=max(3, int(L / dash)), dashed_ratio=ratio)


def ring(center, radius, color=BLACK, width=2.0, dashed_=False, m=240):
    a = np.linspace(0, 2 * np.pi, m, endpoint=False)
    P = np.asarray(center, float)[:2] + radius * np.stack([np.cos(a), np.sin(a)], 1)
    return dashed(P, color=color, width=width) if dashed_ else polyline(P, color=color, width=width, closed=True)


class Plot:
    """Minimal 2D axes (no LaTeX): data (x, y) -> scene points inside the box [x0, x1] x [y0, y1] (scene units)."""

    def __init__(self, xlim, ylim, box, logx=False, logy=False):
        self.xlim, self.ylim, self.box = xlim, ylim, box
        self.logx, self.logy = logx, logy

    def _t(self, v, lim, log):
        v = np.asarray(v, float)
        if log:
            return (np.log(v) - np.log(lim[0])) / (np.log(lim[1]) - np.log(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def __call__(self, x, y):
        (bx0, by0), (bx1, by1) = self.box
        tx, ty = self._t(x, self.xlim, self.logx), self._t(y, self.ylim, self.logy)
        X = bx0 + tx * (bx1 - bx0)
        Y = by0 + ty * (by1 - by0)
        return np.stack([X, Y, np.zeros_like(np.asarray(X, float))], -1)

    def axes(self, color=BLACK, width=2.0, xtick=(), ytick=(), size=30, xlabels=None, ylabels=None):
        (bx0, by0), (bx1, by1) = self.box
        g = VGroup(Line([bx0, by0, 0], [bx1, by0, 0], color=color, stroke_width=width),
                   Line([bx0, by0, 0], [bx0, by1, 0], color=color, stroke_width=width))
        for k, x in enumerate(xtick):
            p = self(x, self.ylim[0])
            g.add(Line(p, p + [0, -0.1, 0], color=color, stroke_width=width))
            lab = xlabels[k] if xlabels else f"{x:g}"
            if lab is not None:
                m = lab if hasattr(lab, "move_to") else da.label(lab, size=size)
                g.add(m.next_to(p, np.array([0, -1.0, 0]), buff=0.18))
        for k, y in enumerate(ytick):
            p = self(self.xlim[0], y)
            g.add(Line(p, p + [-0.1, 0, 0], color=color, stroke_width=width))
            lab = ylabels[k] if ylabels else f"{y:g}"
            if lab is not None:
                m = lab if hasattr(lab, "move_to") else da.label(lab, size=size)
                g.add(m.next_to(p, np.array([-1.0, 0, 0]), buff=0.15))
        return g

    def curve(self, x, y, color=BLACK, width=3.0):
        return polyline(self(np.asarray(x), np.asarray(y)), color=color, width=width)


def text(s, size=32, color=BLACK, **kw):
    return da.label(s, size=size, color=color, **kw)


def math(expr, size=34, color=BLACK):
    return da.mtex(expr, size=size, color=color)


def swap(old, new):
    """Cross-fade two labels (morphing Text glyphs garbles them mid-way)."""
    return [FadeOut(old), FadeIn(new)]


class Scene2D(da.Scene2D):
    """dganim.Scene2D that asserts, after every play/wait, that all labels (Text, mtex) lie inside the frame
    (margin 0.2, GUIDELINES 12.8). With DGANIM_TIMELINE=path it appends "end time  what" lines to that file
    (used to write the time ranges of the captions)."""

    MARGIN = 0.2

    def _check_frame(self):
        W, H = config["frame_width"] / 2 - self.MARGIN, config["frame_height"] / 2 - self.MARGIN
        for m in self.mobjects:
            for sub in m.get_family():
                if not isinstance(sub, (Text, SVGMobject)):
                    continue
                parts = sub.family_members_with_points()
                if parts and max(q.get_fill_opacity() for q in parts) > 0.05:
                    lo, hi = sub.get_corner(np.array([-1, -1, 0])), sub.get_corner(np.array([1, 1, 0]))
                    assert lo[0] > -W and hi[0] < W and lo[1] > -H and hi[1] < H, \
                        f"label outside the frame: {getattr(sub, 'text', type(sub).__name__)!r} {lo[:2]} {hi[:2]}"

    def _log(self, what):
        path = os.environ.get("DGANIM_TIMELINE")
        if path:
            with open(path, "a") as fh:
                fh.write(f"{self.renderer.time:7.2f}  {what}\n")

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._check_frame()
        if args:
            self._log("play " + ", ".join(type(a).__name__ for a in args[:4]))

    def wait(self, *args, **kwargs):
        super().wait(*args, **kwargs)
        self._check_frame()
        self._log(f"wait {args[0] if args else kwargs.get('duration', 1.0)}")

    def mark(self, what):
        """Write a named mark into the timeline (no effect without DGANIM_TIMELINE)."""
        self._log("MARK " + what)
