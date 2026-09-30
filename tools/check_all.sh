#!/usr/bin/env bash
# DG-study 전체 검사 (GUIDELINES.md §15.4, 부록 D).
#
#   bash tools/check_all.sh              전체 검사 (통합 담당)
#   bash tools/check_all.sh --chapter 07 한 장만 검사 (장 작성·검토 agent)
#
# --chapter 모드: 그 장의 페이지만 check_site/check_math로 검사하고, 그 장의 그림·verify만 실행한다.
#   build_index는 저장소 전체를 --check-only로 검사한다(중복 정의는 전역 문제이므로). 공유 파일
#   concepts.html은 다시 쓰지 않는다. 검사기 테스트와 dgnum 테스트는 건너뛴다.
#
# 순서: build_index → check_site → check_math → dgsym 자체 테스트 → 검사기 테스트 → dgnum 테스트
#       → 모든 그림 스크립트(재생성 확인) → 모든 verify 스크립트.
# 실패해도 끝까지 돌리고 마지막에 요약한다. 하나라도 실패하면 종료코드 1.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

CHAPTER=""
if [ "${1:-}" = "--chapter" ]; then
  CHAPTER=$(printf '%02d' "$((10#${2:?'--chapter NN'}))")
  CHDIRS=(chapters/"$CHAPTER"-*/)
  if [ ! -d "${CHDIRS[0]}" ]; then echo "chapter $CHAPTER not found"; exit 2; fi
  CHDIR="${CHDIRS[0]%/}"
  echo "chapter mode: $CHDIR"
fi

names=()
results=()
failed=0

record() {   # record 이름 결과(PASS|FAIL|SKIP) [설명]
  names+=("$1"); results+=("$2${3:+ — $3}")
  [ "$2" = "FAIL" ] && failed=1
  return 0
}

step() {     # step 이름 명령...
  local name=$1; shift
  echo
  echo "==> $name"
  if "$@"; then record "$name" PASS; else record "$name" FAIL "exit $?"; fi
}

shopt -s nullglob
if [ -n "$CHAPTER" ]; then
  PAGES=("$CHDIR"/*.html)
  step "build_index (--check-only)" python3 tools/build_index.py --check-only
  step "check_site ($CHDIR)"        python3 tools/check_site.py "${PAGES[@]}"
else
  PAGES=()
  step "build_index (concepts.html)" python3 tools/build_index.py
  step "check_site"                  python3 tools/check_site.py
fi
if [ -d tools/node_modules/mathjax-full ]; then
  step "check_math"                node tools/check_math.mjs "${PAGES[@]}"
else
  echo; echo "==> check_math"; echo "mathjax-full is not installed: run  cd tools && npm install"
  record "check_math" FAIL "run: cd tools && npm install"
fi

if [ -f tools/dgsym.py ]; then
  step "dgsym self-test"           python3 tools/dgsym.py
else
  record "dgsym self-test" SKIP "tools/dgsym.py not found"
fi
if [ -n "$CHAPTER" ]; then
  record "checker tests" SKIP "chapter mode"
else
  step "checker tests"             bash tools/tests/test_checkers.sh
fi
if [ -n "$CHAPTER" ]; then
  record "dgnum tests" SKIP "chapter mode"
elif [ -f tools/tests/test_dgnum.py ]; then
  step "dgnum tests"               python3 tools/tests/test_dgnum.py
else
  record "dgnum tests" SKIP "tools/tests/test_dgnum.py not found"
fi

# 그림: 스크립트를 실행하고, 같은 이름의 출력(.svg/.png)이 새로 만들어졌는지 확인한다
FIGS=(chapters/*/figures/*.py); VERS=(chapters/*/verify/*.py)
if [ -n "$CHAPTER" ]; then FIGS=("$CHDIR"/figures/*.py); VERS=("$CHDIR"/verify/*.py); fi
nfig=0
for script in "${FIGS[@]}"; do
  nfig=$((nfig + 1))
  base="${script%.py}"
  echo
  echo "==> figure $script"
  stamp=$(mktemp)
  if python3 "$script"; then
    newest=""
    for out in "$base.svg" "$base.png"; do
      # find -newer는 나노초 단위로 비교한다
      if [ -f "$out" ] && [ -n "$(find "$out" -newer "$stamp" 2>/dev/null)" ]; then newest="$out"; fi
    done
    if [ -n "$newest" ]; then
      record "figure $script" PASS
    else
      echo "no fresh $base.svg or $base.png after running the script"
      record "figure $script" FAIL "output not regenerated"
    fi
  else
    record "figure $script" FAIL "exit $?"
  fi
  rm -f "$stamp"
done
[ "$nfig" -eq 0 ] && record "figures" SKIP "no chapters/*/figures/*.py yet"

nver=0
for script in "${VERS[@]}"; do
  nver=$((nver + 1))
  step "verify $script" python3 "$script"
done
[ "$nver" -eq 0 ] && record "verify scripts" SKIP "no chapters/*/verify/*.py yet"

echo
echo "================ check_all 요약 ================"
for i in "${!names[@]}"; do
  printf '%-8s %s\n' "${results[$i]%% *}" "${names[$i]}${results[$i]#${results[$i]%% *}}"
done
if [ "$failed" -eq 0 ]; then
  echo "check_all: OK"
else
  echo "check_all: FAILED"
fi
exit $failed
