"""Helpers for the animations of Sections 31.4-31.6 (GUIDELINES.md 12.8).

This file is not a render target (its name does not start with ``anim-``). It imports dgmesh (tools/dgmesh.py;
render_anims.sh puts tools on PYTHONPATH), so an animation script only needs::

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _helpers_d import *          # dm (dgmesh), da (dganim), np, manim names, and the helpers below

Contents
- cmap_rgb(values, vmin, vmax, name): matplotlib colormap -> (n, 3) RGB array (vectorised)
- Cam: orthographic camera (elev, azim in degrees, like mplot3d / dgmesh.Ortho) that maps world points to
  screen coordinates (x right, y up) plus a depth (toward the viewer)
- MeshView: a fixed pool of triangles drawn with the painter's algorithm (back-face culling optional,
  Lambert shading in camera space). ``set_state(V, rgb)`` moves and recolours the triangles in place, so it is
  cheap to call from an updater every frame. Built from dganim.mesh_group's Polygons (the same face style).
- Plot: a plain 2D plot frame (linear or log axes) with tick labels made by dganim.label / dganim.mtex
  (manim's Axes would need LaTeX for numbers).
- polyline, readout: small conveniences.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy (this machine freezes
    os.environ.setdefault(_k, "4")                                          # under long all-core loads)

import numpy as np  # noqa: E402

import dgmesh as dm  # noqa: E402,F401
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F401,F403
from manim import VGroup, VMobject, ORIGIN, LEFT, RIGHT, UP, DOWN, always_redraw  # noqa: E402

C = da.C
HEX = da.COLORS


# ---------------------------------------------------------------- text
def _mt_spaces(expr):
    """matplotlib writes a plain-text space as an SVG glyph without path data, which svgelements cannot parse
    (dganim.mtex fails on 'index $k$'). Replace spaces outside $...$ by the math space '$\\ $'."""
    import re
    parts = re.split(r"(\$[^$]*\$)", expr)
    return "".join(p if p.startswith("$") else p.replace(" ", r"$\ $") for p in parts)


def mt(expr, size=34, color=BLACK):
    """dganim.mtex with the space work-around above (text and math may be mixed: 'index $k$')."""
    return da.mtex(_mt_spaces(expr), size, color=color)


# ---------------------------------------------------------------- colours
def cmap_rgb(values, vmin, vmax, name="RdBu_r"):
    """Scalar values -> (n, 3) RGB in [0, 1] with a matplotlib colormap (values clipped to [vmin, vmax])."""
    import matplotlib
    matplotlib.use("Agg")
    cm = matplotlib.colormaps[name]
    v = np.asarray(values, float)
    t = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.asarray(cm(t))[..., :3]


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def rgb_hex(rgb):
    r = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(r)


# ---------------------------------------------------------------- camera
class Cam:
    """Orthographic camera. elev/azim as in mplot3d's view_init (and dgmesh.Ortho).

    world point P -> screen (center + scale * (P - pivot) . right, ... . up) and depth (P - pivot) . view.
    """

    def __init__(self, elev=20, azim=-60, scale=1.0, center=ORIGIN, pivot=(0.0, 0.0, 0.0)):
        e, a = np.radians(elev), np.radians(azim)
        self.view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        up = np.array([0.0, 0.0, 1.0])
        self.right = np.cross(up, self.view)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.view, self.right)
        self.scale = float(scale)
        self.center = np.asarray(center, float)
        self.pivot = np.asarray(pivot, float)

    def camera_coords(self, P):
        Q = np.asarray(P, float) - self.pivot
        return np.stack([Q @ self.right, Q @ self.up, Q @ self.view], -1)

    def __call__(self, P):
        """Screen points (…, 3) with z = 0."""
        Q = self.camera_coords(P)
        out = np.zeros_like(Q)
        out[..., 0] = self.center[0] + self.scale * Q[..., 0]
        out[..., 1] = self.center[1] + self.scale * Q[..., 1]
        return out

    def depth(self, P):
        return self.camera_coords(P)[..., 2]

    def facing(self, normals):
        """Component of world normals toward the viewer (> 0: front facing)."""
        return np.asarray(normals, float) @ self.view


# ---------------------------------------------------------------- mesh drawing
class MeshView(VGroup):
    """A pool of triangles (one per face) drawn back to front.

    set_state(V, rgb, cam): V (n0, 3) world positions, rgb (n2, 3) face colours in [0, 1].
    cull=True hides faces whose normal points away from the viewer (closed, consistently oriented meshes);
    with cull=False both sides are shaded by |cos| (open patches). Unused slots are made invisible.
    """

    def __init__(self, F, stroke_width=0.35, stroke_rgb=(0.2, 0.2, 0.2), opacity=1.0, cull=True,
                 light=(-0.35, 0.45, 0.82), ambient=0.42, stroke_opacity=1.0, pool=None):
        F = np.asarray(F, int)
        # the polygons are dganim.mesh_group's (same stroke/fill style); their points are replaced in set_state.
        # pool: number of triangles to allocate (default len(F)); set_state(..., F=other) may then draw another mesh
        dummy = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]])
        polys = da.mesh_group(dummy, np.zeros((pool or len(F), 3), int), edge_width=stroke_width, opacity=opacity)
        super().__init__(*polys)
        self.F = F
        self.cull = cull
        self.opacity = opacity
        self.stroke_rgba = np.array([[*stroke_rgb, stroke_opacity]])
        lt = np.asarray(light, float)
        self.light = lt / np.linalg.norm(lt)
        self.ambient = ambient
        self.visible = np.zeros(0, int)

    def set_state(self, V, rgb, cam, shade=True, stroke=True, F=None):
        V = np.asarray(V, float)
        F = self.F if F is None else np.asarray(F, int)
        assert len(F) <= len(self.submobjects), "MeshView pool too small"
        Q = cam.camera_coords(V)
        tri = Q[F]                                     # (n2, 3, 3) camera coordinates
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        nn = n / np.maximum(ln, 1e-300)[:, None]
        if self.cull:
            vis = np.where(nn[:, 2] > 1e-9)[0]
            k = self.ambient + (1 - self.ambient) * np.clip(nn @ self.light, 0, 1)
        else:
            vis = np.arange(len(F))
            k = self.ambient + (1 - self.ambient) * np.abs(nn @ self.light)
        k[ln < 1e-14] = self.ambient + 0.5 * (1 - self.ambient)
        order = vis[np.argsort(tri[vis, :, 2].mean(1), kind="stable")]
        rgb = np.asarray(rgb, float)
        if rgb.ndim == 1:
            rgb = np.tile(rgb, (len(F), 1))
        col = np.clip(rgb * (k[:, None] if shade else 1.0), 0, 1)
        S = np.zeros_like(tri)
        S[..., 0] = cam.center[0] + cam.scale * tri[..., 0]
        S[..., 1] = cam.center[1] + cam.scale * tri[..., 1]
        polys = self.submobjects
        srgba = self.stroke_rgba if stroke is True else np.array([[0, 0, 0, 0.0]])
        for slot, f in enumerate(order):
            p = polys[slot]
            p.set_points_as_corners([S[f, 0], S[f, 1], S[f, 2], S[f, 0]])
            p.fill_rgbas = np.array([[*col[f], self.opacity]])
            p.stroke_rgbas = np.array([[*col[f], 1.0]]) if stroke == "seal" else srgba
        for slot in range(len(order), len(polys)):
            p = polys[slot]
            c = S[0, 0]
            p.set_points_as_corners([c, c, c, c])
            p.fill_rgbas = np.array([[1.0, 1.0, 1.0, 0.0]])
            p.stroke_rgbas = np.array([[0, 0, 0, 0.0]])
        self.visible = order
        return self


class FlatMeshView(VGroup):
    """2D triangle mesh (e.g. a UV map) with fixed face order; set_state(UV, rgb) moves and recolours it.
    UV (n0, 2) are placed at center + scale * UV. stroke: True (edges), False (none) or "seal" (each face stroked in
    its own colour, which hides the hairline gaps between unstroked faces; give stroke_width > 0)."""

    def __init__(self, F, stroke_width=0.35, stroke_rgb=(0.2, 0.2, 0.2), opacity=1.0):
        F = np.asarray(F, int)
        dummy = np.array([[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0]])
        polys = da.mesh_group(dummy, np.zeros((len(F), 3), int), edge_width=stroke_width, opacity=opacity)
        super().__init__(*polys)
        self.F = F
        self.opacity = opacity
        self.stroke_rgba = np.array([[*stroke_rgb, 1.0]])

    def set_state(self, UV, rgb, center=ORIGIN, scale=1.0, order=None, stroke=True):
        UV = np.asarray(UV, float)
        P = np.zeros((len(UV), 3))
        P[:, 0] = center[0] + scale * UV[:, 0]
        P[:, 1] = center[1] + scale * UV[:, 1]
        rgb = np.asarray(rgb, float)
        if rgb.ndim == 1:
            rgb = np.tile(rgb, (len(self.F), 1))
        order = np.arange(len(self.F)) if order is None else order
        srgba = self.stroke_rgba if stroke is True else np.array([[0, 0, 0, 0.0]])
        for p, f in zip(self.submobjects, order):
            a, b, c = self.F[f]
            p.set_points_as_corners([P[a], P[b], P[c], P[a]])
            fc = np.clip(rgb[f], 0, 1)
            p.fill_rgbas = np.array([[*fc, self.opacity]])
            p.stroke_rgbas = np.array([[*fc, 1.0]]) if stroke == "seal" else srgba
        return self


# ---------------------------------------------------------------- 2D plots
class Plot:
    """A rectangle [x0, x0 + w] x [y0, y0 + h] in scene units showing data xlim x ylim (optionally log)."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _t(self, v, lim, log):
        v = np.asarray(v, float)
        if log:
            return (np.log10(v) - np.log10(lim[0])) / (np.log10(lim[1]) - np.log10(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def X(self, x):
        return self.x0 + self.w * self._t(x, self.xlim, self.xlog)

    def Y(self, y):
        return self.y0 + self.h * self._t(y, self.ylim, self.ylog)

    def p(self, x, y):
        return np.array([float(self.X(x)), float(self.Y(y)), 0.0])

    def pts(self, xs, ys):
        return np.stack([self.X(xs), self.Y(ys), np.zeros(len(xs))], -1)

    def axes(self, xticks=(), yticks=(), xlabel=None, ylabel=None, size=30, box=False, color=BLACK,
             tick_len=0.1, ylabel_buff=0.25):
        """xticks/yticks: lists of (value, label) where label is a str (dganim.label) or a Mobject."""
        g = VGroup()
        a0 = np.array([self.x0, self.y0, 0])
        g.add(Line(a0, a0 + RIGHT * self.w, stroke_width=2.5, color=color))
        g.add(Line(a0, a0 + UP * self.h, stroke_width=2.5, color=color))
        if box:
            g.add(Line(a0 + UP * self.h, a0 + UP * self.h + RIGHT * self.w, stroke_width=1.2, color=color))
            g.add(Line(a0 + RIGHT * self.w, a0 + UP * self.h + RIGHT * self.w, stroke_width=1.2, color=color))
        for v, lab in xticks:
            x = float(self.X(v))
            g.add(Line([x, self.y0, 0], [x, self.y0 - tick_len, 0], stroke_width=2, color=color))
            m = da.label(lab, size) if isinstance(lab, str) else lab
            m.next_to([x, self.y0 - tick_len, 0], DOWN, buff=0.08)
            g.add(m)
        for v, lab in yticks:
            y = float(self.Y(v))
            g.add(Line([self.x0, y, 0], [self.x0 - tick_len, y, 0], stroke_width=2, color=color))
            m = da.label(lab, size) if isinstance(lab, str) else lab
            m.next_to([self.x0 - tick_len, y, 0], LEFT, buff=0.08)
            g.add(m)
        if xlabel is not None:
            m = da.label(xlabel, size) if isinstance(xlabel, str) else xlabel
            m.next_to([self.x0 + self.w / 2, self.y0 - tick_len, 0], DOWN, buff=0.55)
            g.add(m)
        if ylabel is not None:
            m = da.label(ylabel, size) if isinstance(ylabel, str) else ylabel
            m.rotate(np.pi / 2).next_to([self.x0 - tick_len, self.y0 + self.h / 2, 0], LEFT, buff=ylabel_buff)
            g.add(m)
        return g

    def curve(self, xs, ys, color=BLACK, width=4, clip=True, **kw):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if clip:
            lo, hi = sorted(self.ylim)
            ys = np.clip(ys, lo, hi)
        return polyline(self.pts(xs, ys), color=color, width=width, **kw)


def polyline(P, color=BLACK, width=4, **kw):
    m = VMobject(stroke_color=color, stroke_width=width, **kw)
    m.set_points_as_corners(np.asarray(P, float))
    return m


def readout(fn, pos, size=32, color=BLACK, edge=LEFT):
    """A Text label recomputed every frame: fn() -> str, placed with its `edge` at pos."""
    return always_redraw(lambda: da.label(fn(), size, color=color).move_to(pos, aligned_edge=edge))


def title(text, size=36):
    return da.label(text, size).to_edge(UP, buff=0.3)


def swap_text(old, new_text, size=32, color=BLACK, edge=LEFT):
    """A new label at the same anchor (edge) as `old` (for Transform / become)."""
    m = da.label(new_text, size, color=color)
    anchor = old.get_critical_point(edge)
    m.move_to(anchor, aligned_edge=edge)
    return m


def colorbar(x0, y0, w, h, name, vmin, vmax, ticks=(), size=30, n=64, label=None, vertical=False):
    """A colour bar from matplotlib colormap `name` made of n rectangles; ticks: (value, str or Mobject)."""
    g = VGroup()
    vals = np.linspace(vmin, vmax, n)
    rgb = cmap_rgb(vals, vmin, vmax, name)
    for k in range(n):
        if vertical:
            r = Rectangle(width=w, height=h / n, stroke_width=0, fill_color=rgb_hex(rgb[k]), fill_opacity=1)
            r.move_to([x0 + w / 2, y0 + (k + 0.5) * h / n, 0])
        else:
            r = Rectangle(width=w / n, height=h, stroke_width=0, fill_color=rgb_hex(rgb[k]), fill_opacity=1)
            r.move_to([x0 + (k + 0.5) * w / n, y0 + h / 2, 0])
        g.add(r)
    g.add(Rectangle(width=w, height=h, stroke_width=1.5, stroke_color=BLACK).move_to([x0 + w / 2, y0 + h / 2, 0]))
    for v, lab in ticks:
        m = da.label(lab, size) if isinstance(lab, str) else lab
        t = (v - vmin) / (vmax - vmin)
        if vertical:
            m.next_to([x0 + w, y0 + t * h, 0], RIGHT, buff=0.1)
        else:
            m.next_to([x0 + t * w, y0, 0], DOWN, buff=0.1)
        g.add(m)
    if label is not None:
        m = da.label(label, size) if isinstance(label, str) else label
        if vertical:
            m.next_to([x0 + w / 2, y0 + h, 0], UP, buff=0.15)
        else:
            m.next_to([x0 + w / 2, y0 + h, 0], UP, buff=0.12)
        g.add(m)
    return g
