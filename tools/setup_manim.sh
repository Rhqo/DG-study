#!/usr/bin/env bash
# manim(애니메이션, GUIDELINES.md §12.8)을 프로젝트 안의 .manim/ 에 설치한다. .manim/ 은 git에서 제외한다.
#
#   bash tools/setup_manim.sh        설치 (이미 있으면 import만 확인)
#   bash tools/setup_manim.sh --force 지우고 다시 설치
#
# 이 컴퓨터의 사정 때문에 표준 방법(venv, pip install manim)이 안 된다 (2026-10-11 확인).
#   - python3-venv(ensurepip)가 없다.
#   - /usr/local/lib/python3.10/dist-packages/pyzed-5.2.dist-info 를 읽을 수 없어,
#     site-packages를 훑는 pip와 pkg_resources가 PermissionError로 멈춘다.
#     그래서 pip를 `python3 -S`(site 없음)로 돌리고 sys.path에 /usr/lib/python3/dist-packages만 넣는다.
#   - pycairo는 Linux wheel이 없다. 시스템의 python3-cairo(1.20)를 쓴다.
#   - manimpango와 srt는 sdist뿐이다. PyPI에서 받아 setuptools 69 + Cython<3.1로 직접 wheel을 만든다
#     (최신 setuptools에서는 manimpango 빌드가 Compiler.__init__ TypeError로 실패한다).
#   - LaTeX가 없으므로 manim의 MathTex/Tex는 쓰지 않는다. 수식은 tools/dganim.py의 mtex()(matplotlib mathtext)로 만든다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/.manim"
PY=python3
MANIM_VERSION=0.19.1

if [ "${1:-}" = "--force" ]; then rm -rf "$M"; fi
# 실행은 언제나 `python3 -S`로 한다(tools/render_anims.sh도 같다). -S가 없으면 /usr/local의 읽을 수 없는
# cython.py를 scipy가 import하다가 PermissionError로 멈춘다(부록 E의 "scipy 사용 불가"와 같은 원인).
check() { PYTHONPATH="$M/site:/usr/lib/python3/dist-packages" $PY -S -c \
  "import cairo, manim, manimpango; print('manim', manim.__version__, '| manimpango', manimpango.__version__, '| pycairo', cairo.version)"; }
if [ -d "$M/site/manim" ] && check 2>/dev/null; then exit 0; fi

mkdir -p "$M/site" "$M/buildenv" "$M/src" "$M/wheels"
cat > "$M/pip.py" <<'EOF'
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
for p in sys.argv[1].split(':'):
    if p:
        sys.path.insert(0, p)
from pip._internal.cli.main import main
sys.exit(main(sys.argv[2:]))
EOF
pipx() { $PY -S "$M/pip.py" "$@"; }

echo "== build tools"
pipx "" install -q --only-binary=:all: --target "$M/buildenv" "setuptools==69.5.1" "Cython<3.1" wheel

echo "== sdist-only packages: manimpango, srt"
for spec in "manimpango 0.7.0" "srt 3.5.3"; do
  set -- $spec
  url=$(curl -fsSL "https://pypi.org/pypi/$1/$2/json" | $PY -c \
    "import sys,json; d=json.load(sys.stdin); print([u['url'] for u in d['urls'] if u['packagetype']=='sdist'][0])")
  (cd "$M/src" && curl -fsSLO "$url" && tar xzf "$(basename "$url")")
done
for d in "$M"/src/*/; do
  (cd "$d" && rm -rf build && $PY -S -c "
import sys; sys.path[:0] = ['$M/buildenv', '/usr/lib/python3/dist-packages']
import setuptools.build_meta as b
print('built', b.build_wheel('$M/wheels'))")
done

echo "== manim $MANIM_VERSION and dependencies"
pipx "" install -q --no-deps --only-binary=:all: --target "$M/site" "manim==$MANIM_VERSION"
pipx "" install -q --upgrade --only-binary=:all: --find-links "$M/wheels" --target "$M/site" \
  "av<14.0.0,>=9.0.0" "beautifulsoup4>=4.12" "click>=8.0" "cloup>=2.0.0" "decorator>=4.3.2" \
  "isosurfaces>=0.1.0" "manimpango==0.7.0" "mapbox-earcut>=1.0.0" "moderngl-window>=2.0.0" \
  "moderngl<6.0.0,>=5.0.0" "networkx>=2.6" "numpy>=2.1" "pillow>=9.1" "pydub>=0.20.0" \
  "pygments>=2.0.0" "rich>=12.0.0" "scipy>=1.13.0" "screeninfo>=0.7" "skia-pathops>=0.7.0" \
  "srt==3.5.3" "svgelements>=1.8.0" "tqdm>=4.0.0" "typing-extensions>=4.12.0" "watchdog>=2.0.0" \
  "matplotlib>=3.8"
check
