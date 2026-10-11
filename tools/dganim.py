"""manim 애니메이션 공용 도구 (GUIDELINES.md §12.8).

애니메이션 스크립트(chapters/NN-slug/anim/anim-N-M-K-slug.py)는 이 모듈을 불러 쓴다.
실행은 tools/render_anims.sh가 한다(.manim/ 환경, `python3 -S`). 직접 돌릴 때:

    PYTHONPATH=.manim/site:/usr/lib/python3/dist-packages:tools python3 -S chapters/…/anim/anim-….py [--check|--preview]

스크립트의 형식:

    import dganim as da
    from manim import *
    ... 데이터 계산과 assert (모듈 수준. --check는 여기까지만 실행한다) ...
    class Anim(da.Scene2D):           # 또는 da.Scene3D
        def construct(self): ...
    if __name__ == "__main__":
        da.main(Anim, __file__, poster_t=6.0)

출력: 스크립트와 같은 폴더의 anim-N-M-K-slug.mp4와 anim-N-M-K-slug-poster.png.

주의
- 정적 `self.wait()`는 frozen frame이라 시간에 의존하지 않는 updater가 돌지 않는다. updater를 단 mobject는
  add 직후 `m.update()`를 한 번 부른다.
- 영상을 멈추면 브라우저의 control bar가 화면 아래쪽 약 12%(720p에서 약 90px)를 덮는다. 좁은 화면에서는
  음영이 20%까지 넓어진다. poster에서 읽혀야 하는 글자는 그 위(되도록 아래 15% 위)에 둔다(§12.8).
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CACHE = os.path.join(_ROOT, ".manim", "cache")

import manim as _m  # noqa: E402
from manim import (BLACK, WHITE, DOWN, LEFT, ORIGIN, RIGHT, UP, Polygon, Scene, SVGMobject,  # noqa: E402
                   Text, ThreeDScene, VGroup, config, tempconfig)
from manim import ManimColor  # noqa: E402
from manim.camera.three_d_camera import ThreeDCamera  # noqa: E402

# ---------------------------------------------------------------- 색과 글꼴 (dgfig.COLORS와 같다)
COLORS = {
    "main": "#000000",      # 주 대상
    "surface": "#D0D0D0",   # 곡면의 면
    "tangent": "#0072B2",   # 접벡터 (파랑)
    "normal": "#D55E00",    # 법벡터 (주홍)
    "third": "#009E73",     # 세 번째 벡터 (초록)
    "accent": "#E69F00",    # 강조 (주황)
    "region": "#56B4E9",    # 영역 (하늘색)
    "aux": "#777777",       # 보조선 (회색)
    "covector": "#CC79A7",  # 여벡터 (자주)
}
C = {k: ManimColor(v) for k, v in COLORS.items()}
FONT = "DejaVu Serif"
# 1280×720 화면에서 최소 글자 크기. 400px 폭에서도 읽히려면 이보다 작게 쓰지 않는다(§12.8).
MIN_FONT = 30

# 흰 배경. 기본 선·글자 색은 검정
config.background_color = WHITE
for cls in (_m.VMobject, _m.Line, _m.Arrow, _m.Dot, _m.Polygon, _m.Circle, _m.Arc, _m.DashedLine):
    try:
        cls.set_default(color=BLACK)
    except Exception:  # pragma: no cover - 버전 차이
        pass


def label(text, size=36, color=BLACK, **kw):
    """영어 라벨(그림과 같은 글꼴). 한글은 쓰지 않는다(§12.3).

    폭 어림: size 30에서 글자당 약 0.22 scene unit(화면 폭 14.2 unit)이다. 6 unit 열에 약 27자가 들어간다.
    """
    assert size >= MIN_FONT, f"font size {size} < MIN_FONT {MIN_FONT} (§12.8)"
    assert not any("가" <= ch <= "힣" for ch in text), "no Korean inside animations (§12.3)"
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def mtex(expr, size=36, color=BLACK):
    """matplotlib mathtext로 만든 수식 (LaTeX가 없으므로 MathTex 대신 쓴다).

    expr는 matplotlib mathtext 문법이다. 예: r"$\\delta_i = 2\\pi - \\sum_j \\theta_{ij}$".
    `$…$` 밖의 일반 글자(공백 포함)도 섞을 수 있다: r"$\\oint\\omega = 0$  (no hole)".
    mathtext에는 `\\lt`, `\\gt`가 없다. `<`, `>`를 그대로 쓴다.
    size는 Text의 font_size와 같은 눈높이가 되도록 맞춘다. 색은 matplotlib에서 칠한다
    (SVG의 배경 사각형까지 칠하지 않도록 set_fill을 쓰지 않는다).

    식마다 SVG 하나가 .manim/cache에 남는다. 매 frame 바뀌는 숫자에는 mtex 대신 label()을 쓰고,
    음수는 Unicode minus(−)로 쓴다. label()은 Text라서 `_`나 `^`를 첨자로 바꾸지 않는다
    (T_xS²는 글자 그대로 나온다). 첨자가 필요하면 mtex를 쓴다.
    """
    assert size >= MIN_FONT, f"font size {size} < MIN_FONT {MIN_FONT} (§12.8)"
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from matplotlib import pyplot as plt
    hexcol = ManimColor(color).to_hex()
    os.makedirs(_CACHE, exist_ok=True)
    key = hashlib.sha1((expr + hexcol).encode()).hexdigest()[:16]
    path = os.path.join(_CACHE, f"mtex-{key}.svg")
    if not os.path.exists(path):
        matplotlib.rcParams.update({"mathtext.fontset": "cm", "svg.fonttype": "path",
                                    "font.family": "DejaVu Serif"})
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_visible(False)
        fig.text(0, 0, expr, fontsize=20, color=hexcol)
        fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    _fix_empty_paths(path)
    mob = SVGMobject(path, height=None, stroke_width=0)
    # 눈높이 맞추기(2026-10-11 측정): 20pt mathtext의 대문자 H는 SVG에서 13.66 단위,
    # Text("H", font_size=48)은 0.486 단위다. 같은 size에서 대문자 높이가 같도록 scale한다
    mob.scale(0.4859375 / 13.6625 * size / 48)
    return mob


def _fix_empty_paths(path):
    """matplotlib은 공백 글리프를 d 속성이 없는 <path>로 쓴다. svgelements가 TypeError를 내므로 d=""를 넣는다."""
    import re
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    t = re.sub(r"<((?:\w+:)?path)(?![^>]*\sd=)(\s)", r'<\1 d=""\2', s)
    if t != s:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(t)


# ---------------------------------------------------------------- 3D mesh
def shade(color, normal, light=(0.35, -0.45, 0.82), ambient=0.38):
    """Lambert 음영: face normal과 광원 방향의 cos으로 밝기를 정한다."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-15)
    l = np.asarray(light, float)
    l = l / np.linalg.norm(l)
    k = ambient + (1 - ambient) * abs(float(n @ l))
    rgb = np.array(ManimColor(color).to_rgb())
    return ManimColor(tuple(np.clip(rgb * k + (1 - k) * 0.0, 0, 1)))


def mesh_group(V, F, color=None, colors=None, face_values=None, vertex_values=None,
               vmin=None, vmax=None, cmap="coolwarm", edge_color=None, edge_width=0.6, opacity=0.95,
               scale=1.0, light=(0.35, -0.45, 0.82)):
    """triangle mesh → 음영을 넣은 Polygon들의 VGroup (Scene3D에서 쓴다).

    face 색: colors(face마다 ManimColor) > face_values > vertex_values(face의 세 vertex 평균) > color.
    값은 value_colors(…, vmin, vmax, cmap)로 칠한다.
    깊이 정렬: ThreeDCamera는 shade_in_3d가 켜진 mobject만 정렬하므로 face마다 켠다. 음영은 이미
    shade()로 넣었으므로 Scene3D의 카메라(_Camera3D)가 이 face에는 manim 음영을 다시 넣지 않는다.
    face 중심 기준의 painter 정렬이라 서로 관통하는 면이나 크기가 아주 다른 face에서는 틀릴 수 있다.
    정확한 숨은면 처리가 필요하면 2D 장면에서 직접 투영하고 정렬한다(chapters/*/anim/_helpers_*.py).
    """
    V = np.asarray(V, float) * scale
    F = np.asarray(F)
    color = color or C["region"]
    if colors is None and vertex_values is not None:
        face_values = np.asarray(vertex_values, float)[F].mean(axis=1)
    if colors is None and face_values is not None:
        colors = value_colors(face_values, vmin, vmax, cmap)
    if colors is not None:
        assert len(colors) == len(F), "colors must have one entry per face"
    edge_color = edge_color if edge_color is not None else ManimColor("#2b2b2b")
    g = VGroup()
    for k, f in enumerate(F):
        p = V[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        base = colors[k] if colors is not None else color
        poly = Polygon(*p, stroke_width=edge_width, stroke_color=edge_color,
                       fill_color=shade(base, n, light), fill_opacity=opacity)
        poly.shade_in_3d = True
        poly.dg_preshaded = True
        g.add(poly)
    return g


def value_colors(values, vmin=None, vmax=None, cmap="coolwarm"):
    """스칼라 값 → ManimColor 목록 (matplotlib colormap)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm as _cm
    v = np.asarray(values, float)
    lo = v.min() if vmin is None else vmin
    hi = v.max() if vmax is None else vmax
    t = np.clip((v - lo) / (hi - lo + 1e-15), 0, 1)
    cmap = matplotlib.colormaps[cmap]
    return [ManimColor(tuple(cmap(x)[:3])) for x in t]


# ---------------------------------------------------------------- Scene 기본형
class Scene2D(Scene):
    """흰 배경 2D 장면."""


class _Camera3D(ThreeDCamera):
    """mesh_group의 face(dg_preshaded)에는 manim 음영을 다시 넣지 않는다. 나머지는 ThreeDCamera와 같다."""

    def modified_rgbas(self, vmobject, rgbas):
        if getattr(vmobject, "dg_preshaded", False):
            return rgbas
        return super().modified_rgbas(vmobject, rgbas)


class Scene3D(ThreeDScene):
    """흰 배경 3D 장면. 화면에 고정된 글자는 self.add_fixed_in_frame_mobjects로 넣는다."""

    def __init__(self, **kwargs):
        kwargs.setdefault("camera_class", _Camera3D)
        super().__init__(**kwargs)


# ---------------------------------------------------------------- 렌더
QUALITY = {"final": dict(pixel_width=1280, pixel_height=720, frame_rate=30),
           "preview": dict(pixel_width=854, pixel_height=480, frame_rate=15)}


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def main(scene_cls, script_file, poster_t, quality=None):
    """스크립트의 __main__에서 부른다.

    --check: 모듈 수준의 계산과 assert만 확인하고 렌더하지 않는다(이미 import에서 끝났다).
    --preview: 480p15로 빠르게 렌더해 scratch에 둔다(출력 파일은 바꾸지 않는다).
    """
    args = sys.argv[1:]
    if "--check" in args:
        print(f"CHECK OK {os.path.basename(script_file)}")
        return
    preview = "--preview" in args
    q = QUALITY["preview" if preview else (quality or "final")]
    stem = os.path.splitext(os.path.abspath(script_file))[0]
    base = os.path.basename(stem)
    assert base.startswith("anim-"), "animation scripts are named anim-N-M-K-slug.py (§12.8)"
    tmp = tempfile.mkdtemp(prefix="dganim-")
    try:
        with tempconfig({"media_dir": tmp, "disable_caching": True, "output_file": base,
                         "background_color": WHITE, "verbosity": "WARNING", "progress_bar": "none",
                         **q}):
            scene = scene_cls()
            scene.render()
            out = str(scene.renderer.file_writer.movie_file_path)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
        assert 0 < poster_t < dur, f"poster_t {poster_t} outside 0..{dur:.2f} s"
        dest = (os.path.join(tempfile.gettempdir(), base + "-preview.mp4") if preview else stem + ".mp4")
        # 웹 재생용: moov atom을 앞으로(faststart), 오디오 없음
        _ffmpeg("-i", out, "-c:v", "copy", "-an", "-movflags", "+faststart", dest)
        if not preview:
            _ffmpeg("-ss", f"{poster_t}", "-i", dest, "-frames:v", "1", stem + "-poster.png")
        size = os.path.getsize(dest) / 1e6
        print(f"RENDERED {base}: {dur:.1f} s, {size:.2f} MB -> {dest}")
        assert size <= 4.0, f"{base}.mp4 is {size:.1f} MB > 4 MB (§12.8)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
