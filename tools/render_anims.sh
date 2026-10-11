#!/usr/bin/env bash
# 애니메이션을 렌더한다 (GUIDELINES.md §12.8).
#
#   bash tools/render_anims.sh                    모든 chapters/*/anim/anim-*.py 렌더
#   bash tools/render_anims.sh --chapter 31       한 장만
#   bash tools/render_anims.sh --check [...]      렌더하지 않고 데이터 계산과 assert만 확인
#   bash tools/render_anims.sh FILE...            지정한 스크립트만
#
# 렌더는 느려서 check_all.sh에 넣지 않았다. 애니메이션 스크립트를 고쳤으면 commit 전에 이것을 돌린다.
# manim 환경은 tools/setup_manim.sh가 .manim/ 에 만든다. 실행은 `python3 -S`로 한다(setup_manim.sh 주석).
# 이 컴퓨터는 모든 코어를 오래 쓰면 멈추므로 렌더를 CPU 0–7에 묶는다.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d .manim/site/manim ]; then
  echo "manim is not installed: run  bash tools/setup_manim.sh"; exit 2
fi
export PYTHONPATH="$ROOT/.manim/site:/usr/lib/python3/dist-packages:$ROOT/tools"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONWARNINGS=ignore

MODE=""; FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="--check" ;;
    --chapter) shift; CH=$(printf '%02d' "$((10#$1))"); FILES+=(chapters/"$CH"-*/anim/anim-*.py) ;;
    *) FILES+=("$1") ;;
  esac
  shift
done
[ ${#FILES[@]} -eq 0 ] && FILES=(chapters/*/anim/anim-*.py)

pass=0; fail=0; failed=()
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  stem="${f%.py}"
  if out=$(taskset -c 0-7 nice -n 5 python3 -S "$f" $MODE 2>&1); then
    if [ -z "$MODE" ] && { [ ! -s "$stem.mp4" ] || [ ! -s "$stem-poster.png" ]; }; then
      echo "FAIL  $f — missing $stem.mp4 or -poster.png"; fail=$((fail+1)); failed+=("$f"); continue
    fi
    echo "PASS  $f  $(echo "$out" | grep -E '^(RENDERED|CHECK OK)' | tail -1)"
    pass=$((pass+1))
  else
    echo "FAIL  $f"; echo "$out" | tail -8 | sed 's/^/      /'
    fail=$((fail+1)); failed+=("$f")
  fi
done
echo "render_anims: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
