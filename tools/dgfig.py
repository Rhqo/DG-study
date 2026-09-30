"""DG-study 그림 공용 모듈 (GUIDELINES.md §12, 부록 D·E).

그림 스크립트의 기본 틀::

    import dgfig
    dgfig.setup()                       # 반드시 matplotlib/3D 사용 전에 호출
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 4.5))
    ax = dgfig.axes3d(fig, elev=20, azim=35)
    ...
    dgfig.save(fig, __file__)           # 스크립트 옆에 같은 이름의 .svg

색은 역할(role)로 고른다: ``COLORS[role]``. 역할 이름은
main, surface, tangent, normal, third, accent, region, aux.
라벨은 mathtext만 쓴다(그림 안에 한글 금지, §12.3).
"""

import os
from pathlib import Path

import numpy as np

# §12.3 색 규칙 (Okabe–Ito)
COLORS = {
    "main": "#000000",      # 주 대상: 곡선, 곡면 윤곽, 점
    "surface": "#D0D0D0",   # 곡면의 면 (alpha 0.35–0.6)
    "tangent": "#0072B2",   # 접벡터
    "normal": "#D55E00",    # 법벡터
    "third": "#009E73",     # 세 번째 벡터 (b, 두 번째 벡터장 Y)
    "accent": "#E69F00",    # 강조 곡선·보조 벡터 (측지선, 평행이동 결과)
    "region": "#56B4E9",    # 영역 채우기 (U, 차트의 치역), 접평면 면
    "aux": "#777777",       # 보조선, 숨은선
    "covector": "#CC79A7",  # 여벡터·1-형식 (등위선), 01장 파일럿에서 추가
}

# §12.3 선 굵기 (pt)
LW = {"main": 1.8, "vector": 1.5, "grid": 0.3, "aux": 0.8}

ALPHA = {"surface": 0.45, "region": 0.25, "plane": 0.2}

_SETUP_DONE = False


def _color(role):
    return COLORS.get(role, role)


# ---------------------------------------------------------------------------
# 설정과 저장
# ---------------------------------------------------------------------------

def setup():
    """백엔드, mplot3d 경로 문제(부록 E), rcParams를 설정한다. 여러 번 불러도 안전하다."""
    global _SETUP_DONE
    import matplotlib
    matplotlib.use("Agg")

    # 부록 E: 시스템의 오래된 mpl_toolkits(3.5용)가 사용자 matplotlib 3.9의 것을 가린다.
    import mpl_toolkits
    own = os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)), "mpl_toolkits")
    if os.path.isdir(own):
        mpl_toolkits.__path__ = [own]
    import mpl_toolkits.mplot3d  # noqa: F401  (projection="3d" 등록)

    matplotlib.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": ["DejaVu Serif", "Noto Serif CJK JP"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": LW["main"],
        "axes.linewidth": LW["aux"],
        "patch.linewidth": LW["aux"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 200,
        "svg.fonttype": "path",
        "svg.hashsalt": "dg-study",   # SVG 출력 결정적 (재생성 시 같은 파일)
        "path.simplify": True,
    })
    _SETUP_DONE = True


def save(fig, script_file, fmt="svg", png=False, outdir=None, stem=None):
    """그림을 스크립트 옆(또는 ``outdir``)에 스크립트와 같은 이름으로 저장한다.

    반환값: 저장한 경로 리스트. SVG가 1.5MB를 넘으면 경고를 출력한다 (§12.3).
    """
    import matplotlib.pyplot as plt

    script = Path(script_file).resolve()
    out = Path(outdir).resolve() if outdir else script.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or script.stem
    paths = []
    main = out / f"{stem}.{fmt}"
    meta = {"Date": None} if fmt in ("svg", "pdf") else None
    fig.savefig(main, format=fmt, metadata=meta)
    paths.append(main)
    if png and fmt != "png":
        p = out / f"{stem}.png"
        fig.savefig(p, format="png", dpi=200)
        paths.append(p)
    if fmt == "svg":
        _make_svg_ids_deterministic(main)
    for p in paths:
        size = p.stat().st_size
        print(f"saved {p} ({size / 1024:.0f} KB)")
        if p.suffix == ".svg" and size > 1.5 * 1024 * 1024:
            print(f"WARNING: {p.name}가 1.5MB를 넘는다. 격자 해상도를 낮추거나 PNG로 저장할 것 (§12.3).")
    plt.close(fig)
    return paths


def _make_svg_ids_deterministic(path):
    """matplotlib는 경로 clip의 id를 파이썬 객체 id로 만들어 실행마다 달라진다.

    ``p0123456789`` 꼴(문자 1개 + 16진수 10자리)의 id를 처음 나타난 순서대로 다시 붙여
    같은 스크립트가 항상 같은 SVG를 내게 한다 (§12.1 재생성).
    """
    import re

    text = Path(path).read_text(encoding="utf-8")
    pat = re.compile(r'(?<=["#])([a-z])([0-9a-f]{10})(?=["\)])')
    mapping = {}
    for m in pat.finditer(text):
        tok = m.group(0)
        if tok not in mapping:
            mapping[tok] = f"{m.group(1)}{len(mapping):010d}"
    if mapping:
        text = pat.sub(lambda m: mapping[m.group(0)], text)
        Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 3D 헬퍼
# ---------------------------------------------------------------------------

def axes3d(fig, pos=111, elev=20, azim=35, axis_off=True):
    """3D 축을 만든다. computed_zorder=False이므로 그리는 순서와 zorder로 앞뒤를 정한다."""
    ax = fig.add_subplot(pos, projection="3d", computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)
    if axis_off:
        ax.set_axis_off()
    return ax


def equal_aspect(ax, *arrays, pad=0.02, zoom=1.0):
    """주어진 좌표 배열들(모양 (..., 3) 또는 X, Y, Z 따로)이 모두 들어가도록 x:y:z 실제 비율을 맞춘다.

    ``zoom`` > 1이면 3D 상자를 확대해 그림 둘레의 빈 공간을 줄인다.
    """
    if len(arrays) == 3 and all(np.ndim(a) >= 1 for a in arrays) and np.shape(arrays[0]) == np.shape(arrays[1]):
        pts = np.stack([np.ravel(a) for a in arrays], axis=1)
    else:
        pts = np.concatenate([np.reshape(np.asarray(a, float), (-1, 3)) for a in arrays], axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    lo, hi = lo - pad * span.max(), hi + pad * span.max()
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=zoom)


def view_vector(ax):
    """보는 사람 쪽을 향하는 단위벡터 (ax.elev, ax.azim에서 계산)."""
    e, a = np.deg2rad(ax.elev), np.deg2rad(ax.azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def visible_mask(ax, pts, normals):
    """볼록한 곡면 위 점들의 가시성: <normal, view> > 0인 점을 보이는 것으로 본다."""
    return np.asarray(normals) @ view_vector(ax) > 0


def _arrow3d_class():
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, p0, p1, **kw):
            super().__init__((0, 0), (0, 0), **kw)
            self._verts3d = np.array([p0, p1], dtype=float).T

        def do_3d_projection(self, renderer=None):
            xs, ys, zs = proj3d.proj_transform(*self._verts3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    return Arrow3D


def arrow3d(ax, base, vec, role="tangent", label=None, scale=1.0, label_offset=(0, 0, 0),
            lw=None, head=11, zorder=10, fontsize=12, **text_kw):
    """3D 화살표 (FancyArrowPatch, 교과서식 화살촉). 벡터 길이는 ``scale``배.

    배율을 1이 아닌 값으로 쓰면 캡션에 적는다 (§12.4).
    """
    Arrow3D = _arrow3d_class()
    base = np.asarray(base, float)
    tip = base + scale * np.asarray(vec, float)
    arr = Arrow3D(base, tip, arrowstyle="-|>", mutation_scale=head,
                  color=_color(role), lw=lw or LW["vector"], shrinkA=0, shrinkB=0,
                  zorder=zorder)
    ax.add_artist(arr)
    if label:
        lp = tip + 0.08 * (tip - base) / max(np.linalg.norm(tip - base), 1e-12) + np.asarray(label_offset, float)
        ax.text(*lp, label, color=_color(role), fontsize=fontsize, zorder=zorder + 1,
                ha=text_kw.pop("ha", "center"), va=text_kw.pop("va", "center"), **text_kw)
    return arr


def _dilate(mask):
    d = mask.copy()
    d[1:] |= mask[:-1]
    d[:-1] |= mask[1:]
    return d


def curve3d(ax, pts, role="main", lw=None, ls="-", visible=None, hidden="dashed", zorder=5, **kw):
    """3D 곡선. ``visible``(불리언 배열)을 주면 보이는 부분은 실선으로 그린다.

    가려진 부분은 ``hidden="dashed"``이면 옅은 회색 점선(숨은선), ``hidden=None``이면 그리지 않는다.
    """
    pts = np.asarray(pts, float)
    if visible is None:
        return ax.plot(*pts.T, color=_color(role), lw=lw or LW["main"], ls=ls, zorder=zorder, **kw)
    visible = np.asarray(visible, bool)
    front = np.where(_dilate(visible)[:, None], pts, np.nan)
    back = np.where(_dilate(~visible)[:, None], pts, np.nan)
    lines = []
    if hidden == "dashed":
        lines += ax.plot(*back.T, color=COLORS["aux"], lw=0.6, ls=(0, (3, 2.5)), alpha=0.55,
                         zorder=zorder - 1, **kw)
    lines += ax.plot(*front.T, color=_color(role), lw=lw or LW["main"], ls=ls, zorder=zorder, **kw)
    return lines


def surface(ax, X, Y, Z, color=None, alpha=None, grid=True, grid_color="#8C8C8C",
            rstride=1, cstride=1, grid_every=4, zorder=1, shade=True, rasterized=True, **kw):
    """반투명 곡면 + 얇은 좌표격자 (교과서 선화 느낌).

    면은 기본적으로 래스터화한다(``rasterized=True``): SVG가 작아지고, 반투명 다각형 사이의
    이음새 선이 생기지 않는다. 선(격자, 곡선, 벡터)은 벡터로 남는다.
    """
    ax.plot_surface(X, Y, Z, color=color or COLORS["surface"],
                    alpha=ALPHA["surface"] if alpha is None else alpha,
                    rstride=rstride, cstride=cstride, linewidth=0, antialiased=True,
                    shade=shade, zorder=zorder, rasterized=rasterized, **kw)
    if grid:
        ax.plot_wireframe(X, Y, Z, rstride=grid_every, cstride=grid_every,
                          color=grid_color, linewidth=LW["grid"], zorder=zorder + 0.5)


def tangent_plane(ax, p, e1, e2, size=1.0, zorder=3, edge=True):
    """점 p를 중심으로 e1, e2 방향(정규화)의 평행사변형 접평면을 그린다."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    p = np.asarray(p, float)
    a = size * np.asarray(e1, float) / np.linalg.norm(e1)
    b = size * np.asarray(e2, float) / np.linalg.norm(e2)
    corners = [p - a - b, p + a - b, p + a + b, p - a + b]
    poly = Poly3DCollection([corners], facecolor=COLORS["region"], alpha=ALPHA["plane"],
                            edgecolor=COLORS["tangent"] if edge else "none",
                            linewidth=LW["aux"], zorder=zorder)
    ax.add_collection3d(poly)
    return np.array(corners)


def point3d(ax, p, label=None, role="main", size=18, label_offset=(0, 0, 0), zorder=12, fontsize=12):
    """3D 점과 라벨."""
    p = np.asarray(p, float)
    ax.scatter(*p, s=size, color=_color(role), depthshade=False, zorder=zorder)
    if label:
        ax.text(*(p + np.asarray(label_offset, float)), label, color=_color(role),
                fontsize=fontsize, zorder=zorder + 1)


# ---------------------------------------------------------------------------
# 2D 개념도 헬퍼
# ---------------------------------------------------------------------------

def blob(ax, center=(0.0, 0.0), radius=1.0, seed=0, n_modes=4, amp=0.13, n=400,
         fill=None, fill_alpha=None, edge="main", lw=None, zorder=2):
    """개념도용 '얼룩' 모양 다양체: 푸리에 섭동으로 만든 매끈한 닫힌 곡선.

    반환값: 경계 점 배열 (n, 2). ``fill``에 역할 이름이나 색을 주면 안을 칠한다.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    rr = np.ones_like(t)
    for k in range(2, 2 + n_modes):
        rr += amp / (k - 1) * rng.uniform(0.5, 1.0) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
    pts = np.stack([center[0] + radius * rr * np.cos(t), center[1] + radius * rr * np.sin(t)], axis=1)
    if fill:
        ax.fill(*pts.T, color=_color(fill),
                alpha=ALPHA["region"] if fill_alpha is None else fill_alpha, lw=0, zorder=zorder - 1)
    if edge:
        # 닫힌 다각형으로 그려야 시작점과 끝점의 이음새에 홈이 생기지 않는다.
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(pts[:-1], closed=True, fill=False, edgecolor=_color(edge),
                             lw=lw or LW["main"], joinstyle="round", zorder=zorder))
    return pts


def schematic_axes(ax, xlim=None, ylim=None):
    """개념도용 2D 축: 눈금·테두리 없음, 같은 비율."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def map_arrow(fig, ax_from, ax_to, label, xy_from=(1.0, 0.5), xy_to=(0.0, 0.5),
              coords="axes fraction", rad=-0.35, role="main", lw=None, fontsize=13,
              label_offset=(0.0, 0.0)):
    """두 패널 사이의 사상 화살표 (곡선 화살표 + 라벨). 라벨 위치는 figure 좌표의 중점 + 오프셋.

    ``coords``: "axes fraction"(기본) 또는 "data".
    """
    from matplotlib.patches import ConnectionPatch

    con = ConnectionPatch(xyA=xy_from, coordsA=coords, axesA=ax_from,
                          xyB=xy_to, coordsB=coords, axesB=ax_to,
                          arrowstyle="-|>", mutation_scale=14, shrinkA=4, shrinkB=4,
                          connectionstyle=f"arc3,rad={rad}", color=_color(role),
                          lw=lw or LW["vector"], zorder=20)
    fig.add_artist(con)

    def to_fig(ax, xy):
        tr = ax.transAxes if coords == "axes fraction" else ax.transData
        return fig.transFigure.inverted().transform(tr.transform(xy))

    # arc3의 꼭짓점(베지에 곡선의 중앙)은 중점 + ½·rad·(dy, −dx) (display 좌표). 그 바깥쪽에 라벨.
    def disp(ax, xy):
        tr = ax.transAxes if coords == "axes fraction" else ax.transData
        return tr.transform(xy)

    a, b = disp(ax_from, xy_from), disp(ax_to, xy_to)
    d = b - a
    perp = np.array([d[1], -d[0]])
    apex = (a + b) / 2 + 0.5 * rad * perp
    out = np.sign(rad) * perp / max(np.linalg.norm(perp), 1e-12) if rad else np.array([0.0, 1.0])
    lab = apex + out * 0.9 * fontsize * fig.dpi / 72
    lab = fig.transFigure.inverted().transform(lab) + np.asarray(label_offset)
    fig.text(*lab, label, ha="center", va="center", fontsize=fontsize, color=_color(role))
    return con
