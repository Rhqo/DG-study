#!/usr/bin/env bash
# 검사 도구(check_site.py, build_index.py, check_math.mjs, term_guard.py) 자체 테스트.
#
#   bash tools/tests/test_checkers.sh
#
# fixtures/site       : 계약을 모두 지키는 작은 사이트. 영어 라벨·용어, 한국어 본문(§8)의 참고 예시이기도 하다
# fixtures/site_bad   : 일부러 심은 오류와 남은 한국어 라벨이 모두 보고되는지 확인한다
# fixtures/term_guard : 변환 전(before) / 올바른 변환(after) / 잘못된 변환(after_bad)
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
expect_msg  "concepts: English title"            "<title>Index — DG Study</title>"
expect_msg  "concepts: breadcrumb"               '<a href="index.html">Contents</a> ›'
expect_msg  "concepts: letter group"             '<h3 id="sec-letter-r">R</h3>'
expect_msg  "concepts: term listed"              "<strong>Riemannian metric</strong> (리만 계량)"
expect_msg  "concepts: definition link"          'html#i-21-1-1">Definition 21.1.1</a>'
expect_msg  "concepts: generalization chain"     "Generalizes: first fundamental form"
expect_msg  "concepts: reverse chain"            "Generalized by: Riemannian metric"
expect_msg  "concepts: backlinks"                "Used in:"
expect_msg  "concepts: section label"            ">Section 7.1 · Worked example</a>"
expect_msg  "concepts: compat title"             "<h2>Compatibility propositions</h2>"
expect_msg  "concepts: compat list"              "Proposition 21.1.2</a> (compatibility with the first fundamental form)"
# 알파벳순(대소문자 무시): coefficient … < first … < Riemannian …
order=$(grep -o '<strong>[^<]*</strong>' <<<"$out" | tr '\n' '|')
if [ "$order" = "<strong>coefficient of the first fundamental form</strong>|<strong>first fundamental form</strong>|<strong>Riemannian metric</strong>|" ]; then
  pass "concepts: case-insensitive order"; else bad "concepts: case-insensitive order (got $order)"; fi

echo "== 빈 사이트"
TMP=$(mktemp -d)
ln -s "$(cd ../assets && pwd)" "$TMP/assets"
cp "$GOOD/index.html" "$TMP/index.html"   # concepts.html의 breadcrumb 대상 (검사 대상은 concepts.html만)
expect_ok   "build_index: empty site"            python3 build_index.py --root "$TMP"
expect_ok   "check_site: generated concepts.html" python3 check_site.py --root "$TMP" "$TMP/concepts.html"
run cat "$TMP/concepts.html"
expect_msg  "concepts: empty message"            "No terms have been defined yet."
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
expect_msg  "thm-head word/number"               "thm-head must read 'Theorem 7.1.2'"
expect_msg  "fig-head word"                      "fig-head must read 'Figure 7.1.1.' (got '그림 7.1.1.')"
expect_msg  "exercise head word"                 "exercise head must read 'Exercise 7.1.1' (got '연습 7.1.1')"
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

expect_msg  "unlinked ref: Theorem"              "unlinked cross-reference 'Theorem 7.1.1'"
expect_msg  "unlinked ref: Corollary"            "unlinked cross-reference 'Corollary 7.1.3'"
expect_msg  "unlinked ref: Non-example"          "unlinked cross-reference 'Non-example 7.1.4'"
expect_msg  "unlinked ref: bare (N.M.K)"         "unlinked cross-reference '(7.1.2)'"
expect_msg  "unlinked ref: Equation (N.M.K)"     "unlinked cross-reference 'Equation (7.1.2)'"
for x in "Example 7.1.4" "(7.1.5)" "(7.1.6)" "(2026.10.04)" "(7.1.1)"; do
  if grep -qF -- "cross-reference '$x'" <<<"$out"; then bad "unlinked ref: '$x' must be exempt (code/math/date/link)"
  else pass "unlinked ref: '$x' exempt"; fi
done
if [ "$(grep -cF -- "unlinked cross-reference" <<<"$out")" -eq 5 ]; then pass "unlinked ref: exactly 5 warnings"
else bad "unlinked ref: expected exactly 5 warnings"; fi

expect_msg  "Korean label: breadcrumb"           "Korean label '전체 목차' in breadcrumb: use 'Contents'"
expect_msg  "Korean label: p.meta"               "Korean label '대응 교재' in p.meta: use 'References'"
expect_msg  "Korean label: h2"                   "Korean label '선수 지식' in <h2>: use 'Prerequisites'"
expect_msg  "Korean label: h2 exercises"         "Korean label '연습문제' in <h2>: use 'Exercises'"
expect_msg  "Korean label: proof-head"           "Korean label '증명' in .proof-head: use 'Proof'"
expect_msg  "Korean label: box-title"            "Korean label '직관' in .box-title: use 'Intuition'"
expect_msg  "Korean label: fig-head"             "Korean label '그림' in .fig-head: use 'Figure'"
expect_msg  "Korean label: exercise head"        "Korean label '연습' in .thm-head: use 'Exercise'"
expect_msg  "Korean label: summary"              "Korean label '힌트' in <summary>: use 'Hint'"
expect_msg  "Korean label: pager"                "Korean label '장 개요' in pager: use 'Chapter overview'"
expect_msg  "Korean title: h1"                   "Korean text in <h1> '제목 번호가 틀린 절': titles are English"
expect_msg  "Korean ref: item in link"           "Korean reference '정리 7.1.1': write 'Theorem 7.1.1'"
expect_msg  "Korean ref: 식 + linked number"     "Korean reference '식 (7.1.1)': write '(7.1.1)'"
expect_msg  "Korean ref: chapter"                "Korean reference '7장': write 'Chapter 7'"
expect_msg  "Korean ref: section"                "Korean reference '6.3절': write 'Section 6.3'"

echo "== 남은 한국어 상태 라벨 (check_site, 장 개요)"
expect_fail "check_site: chapter index fails"    python3 check_site.py --root "$BAD" "$BAD/chapters/07-first-fundamental-form/index.html"
expect_msg  "status label"                       "status label for 'draft' must be 'Draft' (got '초안')"
expect_msg  "Korean label: status"               "Korean label '초안' in span.status: use 'Draft'"
expect_msg  "Korean label: h2 sections"          "Korean label '절 목록' in <h2>: use 'Sections'"

echo "== 깨진 절 (check_math)"
expect_fail "check_math: broken page fails"      node check_math.mjs --root "$BAD" "$BROKEN"
expect_msg  "undefined macro"                    "Undefined control sequence \\foo"
expect_msg  "brace error"                        "Missing close brace"
expect_msg  "macro definition (math)"            "\\newcommand is not allowed in pages"
expect_msg  "unclosed delimiter (math)"          "unclosed \\("

echo "== 연결·일관성 (build_index)"
expect_fail "build_index: bad site fails"        python3 build_index.py --root "$BAD" --check-only
expect_msg  "duplicate (case-insensitive)"       "ERROR chapters/07-first-fundamental-form/7-2-dup.html:22: term 'first fundamental form' is defined more than once without data-generalizes"
expect_msg  "duplicate (old-style dfn)"          "ERROR chapters/13-tangent-spaces/13-1-derivations.html:26: term 'first fundamental form' is defined more than once"
expect_msg  "missing generalizes target"         "data-generalizes target '7-9-missing.html#i-7-9-1' does not exist"
expect_msg  "missing compat"                     "chapter 13 is 'draft' but has no div.thm.compat item"
expect_msg  "homonym warning"                    "WARN chapters/13-tangent-spaces/13-1-derivations.html:29: term 'trace' is defined more than once with different Korean translations (대각합, 자취"
if grep -qF -- "ERROR chapters/13-tangent-spaces/13-1-derivations.html:29" <<<"$out"; then bad "homonym must not be an error"; else pass "homonym is not an error"; fi
TMP=$(mktemp -d)
run python3 build_index.py --root "$BAD" --output "$TMP/concepts.html"
run cat "$TMP/concepts.html"
expect_msg  "old-style dfn: English side"        "<strong>tangent space</strong> (접공간)"
expect_msg  "homonyms listed apart"              "<strong>trace</strong> (자취)"
rm -rf "$TMP"

echo "== 변환 안전장치 (term_guard)"
TG=$FIX/term_guard
expect_ok   "term_guard: identical file"         python3 term_guard.py --against "$TG/after.html" "$TG/after.html"
expect_ok   "term_guard: correct conversion"     python3 term_guard.py --against "$TG/before.html" "$TG/after.html"
expect_msg  "note: Hangul-only \\text change"     "math changed only in Hangul \\text"
expect_msg  "term left"                          "Korean term '선형범함수' → linear functional"
expect_msg  "single-syllable term + particle"    "Korean term '핵' → kernel"
expect_msg  "label left"                         "Korean label '풀이' in <summary>: use 'Solution'"
expect_msg  "term count"                         "2 Korean terms left"
for x in "'쌍대공간'" "'사영'" "'상'" "'핵심'" "'이상'" "'대상'"; do
  if grep -qF -- "$x" <<<"$out"; then bad "term_guard: $x must not be reported"; else pass "term_guard: $x not reported"; fi
done
expect_fail "term_guard: bad conversion fails"   python3 term_guard.py --against "$TG/before.html" "$TG/after_bad.html"
expect_msg  "id removed"                         "id 'i-1-3-1' was removed"
expect_msg  "id added"                           "id 'i-1-3-01' was added"
expect_msg  "math removed"                       "math segment removed or changed (line 18 in "
expect_msg  "math added"                         'math segment added or changed: \( V^{*} \)'
expect_msg  "non-Hangul change near \\text"      'math segment added or changed: \[ \beta^i(b_j) = \delta^i_j \qquad'
expect_msg  "figure count"                       "number of figure.dg-figure changed: 1 → 0"
expect_msg  "img count"                          "number of img changed: 1 → 0"
expect_msg  "href removed"                       'href="1-2-linear-maps.html#i-1-2-1" was removed or changed'
expect_msg  "href added"                         'href="1-2-linear-maps.html#i-1-2-2" was added or changed'
expect_msg  "src removed"                        'src="figures/fig-1-3-1-covector.svg" was removed or changed'
# git 모드(--rev): 임시 저장소에 변환 전 판을 커밋하고 비교한다
TMP=$(mktemp -d)
cp "$TG/before.html" "$TMP/page.html"
if git -C "$TMP" init -q && git -C "$TMP" add page.html \
   && git -C "$TMP" -c user.name=t -c user.email=t@example.com commit -q -m before; then
  cp "$TG/after.html" "$TMP/page.html"
  expect_ok   "term_guard --rev HEAD: good"      python3 term_guard.py "$TMP/page.html"
  cp "$TG/after_bad.html" "$TMP/page.html"
  expect_fail "term_guard --rev HEAD: bad"       python3 term_guard.py --rev HEAD "$TMP/page.html"
  expect_msg  "git mode: id removed"             "id 'i-1-3-1' was removed (line 16 in HEAD)"
  echo "<p>new</p>" > "$TMP/new.html"
  expect_fail "term_guard: file not in REV"      python3 term_guard.py "$TMP/new.html"
  expect_msg  "not in REV message"               "cannot read HEAD:new.html"
else
  bad "term_guard: could not create a temporary git repository"
fi
rm -rf "$TMP"

echo
if [ "$fail" -eq 0 ]; then echo "test_checkers: all passed"; else echo "test_checkers: FAILED"; fi
exit $fail
