#!/usr/bin/env bash
# 검사 도구(check_site.py, build_index.py, check_math.mjs) 자체 테스트.
#
#   bash tools/tests/test_checkers.sh
#
# fixtures/site     : 계약을 모두 지키는 작은 사이트 (페이지 작성의 참고 예시이기도 하다)
# fixtures/site_bad : 일부러 심은 오류가 모두 보고되는지 확인한다
set -u
cd "$(dirname "$0")/.."            # tools/
FIX=tests/fixtures
GOOD=$FIX/site
BAD=$FIX/site_bad
BROKEN=$BAD/chapters/07-first-fundamental-form/7-1-broken.html
fail=0
out=""
code=0

pass() { echo "PASS  $1"; }
bad()  { echo "FAIL  $1"; fail=1; }
run()  { out=$("$@" 2>&1); code=$?; }
expect_ok() {
  local name=$1; shift; run "$@"
  if [ "$code" -eq 0 ]; then pass "$name"; else bad "$name (exit $code)"; echo "$out" | sed 's/^/      /'; fi
}
expect_fail() {
  local name=$1; shift; run "$@"
  if [ "$code" -ne 0 ]; then pass "$name"; else bad "$name (expected a non-zero exit)"; fi
}
expect_msg() {   # 직전 run의 출력에 문자열이 있어야 한다
  local name=$1 pattern=$2
  if grep -qF -- "$pattern" <<<"$out"; then pass "$name"; else bad "$name — missing: $pattern"; fi
}

echo "== 정상 사이트 (fixtures/site)"
expect_ok   "build_index: valid site"            python3 build_index.py --root "$GOOD"
expect_ok   "check_site: valid site"             python3 check_site.py --root "$GOOD"
expect_msg  "check_site: no warnings"            "0 errors, 0 warnings"
expect_msg  "check_site: todo counted"           "todo 1, REVIEW 1"
expect_ok   "check_math: valid site"             node check_math.mjs --root "$GOOD"
C=$GOOD/concepts.html
run cat "$C"
expect_msg  "concepts: term listed"              "<strong>리만 계량</strong> (Riemannian metric)"
expect_msg  "concepts: generalization chain"     "일반화하는 개념: 제1기본형식"
expect_msg  "concepts: reverse chain"            "더 일반적인 개념: 리만 계량"
expect_msg  "concepts: backlinks"                "쓰이는 곳:"
expect_msg  "concepts: compat list"              "명제 21.1.2</a> (제1기본형식과의 호환성)"

echo "== 빈 사이트"
TMP=$(mktemp -d)
ln -s "$(cd ../assets && pwd)" "$TMP/assets"
cp "$GOOD/index.html" "$TMP/index.html"   # concepts.html의 breadcrumb 대상 (검사 대상은 concepts.html만)
expect_ok   "build_index: empty site"            python3 build_index.py --root "$TMP"
expect_ok   "check_site: generated concepts.html" python3 check_site.py --root "$TMP" "$TMP/concepts.html"
run cat "$TMP/concepts.html"
expect_msg  "concepts: empty message"            "아직 정의된 용어가 없다."
rm -rf "$TMP"

echo "== 깨진 절 (check_site)"
expect_fail "check_site: broken page fails"      python3 check_site.py --root "$BAD" "$BROKEN"
expect_msg  "missing viewport"                   'missing <meta name="viewport"'
expect_msg  "dg-id mismatch"                     "dg-id '7.9' must be '7.1'"
expect_msg  "bad dg-status"                      "dg-status 'done' must be one of"
expect_msg  "bad dg-updated"                     "dg-updated '2026-13-01' is not a YYYY-MM-DD date"
expect_msg  "load order"                         "load order must be style.css → mathjax-config.js → MathJax CDN"
expect_msg  "extra script"                       "<script src='extra.js'> is not allowed"
expect_msg  "<style> element"                    "<style> elements are not allowed"
expect_msg  "style attribute"                    "style= attributes are not allowed"
expect_msg  "unknown class"                      "class 'fancy' is not in the class contract"
expect_msg  "h1 number"                          "<h1> must start with '7.1 '"
expect_msg  "dollar math"                        "'\$' in text"
expect_msg  "raw < in math"                      "raw '<' inside math"
expect_msg  "duplicate id"                       "duplicate id 'sec-dup'"
expect_msg  "item numbering"                     "item 2 in document order must have id 'i-7-1-2'"
expect_msg  "thm-head word/number"               "thm-head must read '정리 7.1.2'"
expect_msg  "equation tag"                       "equation eq-7-1-1 must contain \\tag{7.1.1}"
expect_msg  "tag outside equation"               "\\tag{…} outside div.equation"
expect_msg  "figure never referenced"            "figure 'fig-7-1-1' is never referenced"
expect_msg  "missing image"                      "broken link 'figures/fig-7-1-1-missing.svg'"
expect_msg  "missing figure source"              "fig-7-1-1-missing.py is missing"
expect_msg  "broken file link"                   "broken link '7-9-nope.html': file does not exist"
expect_msg  "broken anchor"                      "no element with id 'i-7-1-99'"
expect_msg  "block inside p"                     "<ul> inside <p>"
expect_msg  "unclosed tag"                       "<span> is never closed"
expect_msg  "macro definition"                   "\\newcommand in a page is not allowed"
expect_msg  "summary p.next"                     'section.summary must end with <p class="next">'
expect_msg  "exercise solution"                  'exercise needs <details class="solution">'
expect_msg  "unclosed inline math"               "unclosed \\( (no matching \\))"

expect_msg  "unlinked ref: 정리"                 "unlinked cross-reference '정리 7.1.1'"
expect_msg  "unlinked ref: 식"                   "unlinked cross-reference '식 (7.1.2)'"
expect_msg  "unlinked ref: 따름정리 (not 정리)"  "unlinked cross-reference '따름정리 7.1.3'"
if grep -qF -- "cross-reference '예 7.1.4'" <<<"$out"; then bad "unlinked ref: <code> must be exempt"; else pass "unlinked ref: <code> exempt"; fi
if [ "$(grep -cF -- "unlinked cross-reference" <<<"$out")" -eq 3 ]; then pass "unlinked ref: exactly 3 warnings"; else bad "unlinked ref: expected exactly 3 warnings"; fi

echo "== 깨진 절 (check_math)"
expect_fail "check_math: broken page fails"      node check_math.mjs --root "$BAD" "$BROKEN"
expect_msg  "undefined macro"                    "Undefined control sequence \\foo"
expect_msg  "brace error"                        "Missing close brace"
expect_msg  "macro definition (math)"            "\\newcommand is not allowed in pages"
expect_msg  "unclosed delimiter (math)"          "unclosed \\("

echo "== 연결·일관성 (build_index)"
expect_fail "build_index: bad site fails"        python3 build_index.py --root "$BAD" --check-only
expect_msg  "duplicate definition"               "term '제1기본형식' is defined more than once without data-generalizes"
expect_msg  "missing generalizes target"         "data-generalizes target '7-9-missing.html#i-7-9-1' does not exist"
expect_msg  "missing compat"                     "chapter 13 is 'draft' but has no div.thm.compat item"

echo
if [ "$fail" -eq 0 ]; then echo "test_checkers: all passed"; else echo "test_checkers: FAILED"; fi
exit $fail
