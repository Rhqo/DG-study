#!/usr/bin/env python3
"""영어 용어·라벨 전환(GUIDELINES.md §8, PLAN.md §9)의 안전장치.

변환한 페이지를 변환 전 판(git의 REV, 기본 HEAD)과 비교해서, 글자만 바뀌고 구조·수식·링크는
그대로인지 확인한다. 표준 라이브러리와 tools/check_site.py만 쓴다.

사용법:
    python3 tools/term_guard.py [--rev REV] [--quiet] FILE...
    python3 tools/term_guard.py --against OLD.html [--quiet] NEW.html

    --rev REV        각 FILE을 `git show REV:FILE`과 비교한다 (기본 HEAD)
    --against OLD    git 대신 이 파일과 비교한다 (FILE은 하나만). 테스트용
    --quiet          WARN/NOTE 줄은 빼고 파일별 요약과 ERROR만 출력한다

ERROR (하나라도 있으면 종료코드 1)
  - 요소 id 집합이 바뀌었다
  - 수식 구간 \\( … \\), \\[ … \\]의 다중집합이 바뀌었다. 공백 차이는 무시한다.
    \\text{…}·\\mbox{…} 안의 한글만 바뀐 경우(번역)는 같은 것으로 보고 NOTE로 알린다
  - div.thm, figure.dg-figure, div.exercise, div.equation, img 개수가 바뀌었다
  - href, src, data-generalizes, data-file 대상이 바뀌었다 (다중집합 비교. 링크 글자는 바뀌어도 된다)
WARN
  - 남은 한국어 라벨·제목 (check_site.py의 korean_label_issues와 같은 검사)
  - tools/terms.tsv의 한국어 용어가 본문(텍스트, <title>, img alt)에 남았다.
    </dfn>·</strong> 바로 뒤 괄호 "(한국어)", tools/terms_keep.txt의 낱말, code/pre/script/style은 보지 않는다.
    한 글자 용어(상, 핵, 공, 틀)는 앞이 한글이 아니고 뒤에 조사(상은/상이/상을/상의/상과/상에 …)가
    붙은 독립된 낱말일 때만 잡는다. 겹치는 후보는 가장 긴 낱말을 고른다(정사영 ≠ 사영, 미분동형사상 = 한 용어).
"""

import argparse
import html
import os
import re
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_site import (NO_MATH_TAGS, DomBuilder, Node, Text, blank_regions,  # noqa: E402
                        find_math_spans, korean_label_issues, line_of)

TOOLS = os.path.dirname(os.path.abspath(__file__))
TERMS_TSV = os.path.join(TOOLS, "terms.tsv")
TERMS_KEEP = os.path.join(TOOLS, "terms_keep.txt")

HANGUL = re.compile(r"[가-힣]")
COUNTED = (("div.thm", lambda n: n.tag == "div" and n.has_class("thm")),
           ("figure.dg-figure", lambda n: n.tag == "figure" and n.has_class("dg-figure")),
           ("div.exercise", lambda n: n.tag == "div" and n.has_class("exercise")),
           ("div.equation", lambda n: n.tag == "div" and n.has_class("equation")),
           ("img", lambda n: n.tag == "img"))
TARGET_ATTRS = ("href", "src", "data-generalizes", "data-file")
TEXT_CMD_RE = re.compile(r"\\(?:text|mbox|textrm|textit|textbf|textsf|texttt|textnormal|hbox)\s*\{")
# 한 글자 용어 뒤에 올 수 있는 조사. 조사 뒤는 낱말 경계여야 한다("상이한"은 잡지 않는다).
PARTICLES = ("에서의", "으로의", "이라고", "에서", "으로", "이다", "이고", "이며", "이면", "이라", "이란",
             "과의", "와의", "에는", "에도", "까지", "부터", "보다", "처럼", "이나", "만큼",
             "은", "는", "이", "가", "을", "를", "의", "과", "와", "에", "로", "도", "만", "인")
PARTICLE_RE = re.compile("(?:" + "|".join(PARTICLES) + r")(?![가-힣])")


# ---------------------------------------------------------------- 파싱

class Doc:
    def __init__(self, raw, label):
        self.raw = raw
        self.label = label
        b = DomBuilder()
        b.feed(raw)
        b.finish()
        self.dom = b.root

    def ids(self):
        return {n.id for n in self.dom.iter() if n.id}

    def counts(self):
        nodes = list(self.dom.iter())
        return {name: sum(1 for n in nodes if pred(n)) for name, pred in COUNTED}

    def targets(self):
        c = Counter()
        lines = {}
        for n in self.dom.iter():
            for a in TARGET_ATTRS:
                v = n.get(a)
                if v is not None:
                    c[(a, v)] += 1
                    lines.setdefault((a, v), n.line)
        return c, lines

    def math(self):
        """[(줄, 정규화된 수식)] — 구분자 포함, 엔티티 해제, 공백 정규화."""
        text = blank_regions(self.raw)
        out = []
        for start, end, display, tex, err in find_math_spans(text):
            line = line_of(text, start)
            if tex is None:
                tail = text[start:start + 60].split("\n", 1)[0]
                out.append((line, "UNCLOSED " + _norm_ws(tail)))
                continue
            o, c = ("\\[", "\\]") if display else ("\\(", "\\)")
            out.append((line, o + " " + _norm_ws(html.unescape(tex)) + " " + c))
        return out


def _norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def _short(s, n=90):
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------- 수식 비교

def text_parts(tex):
    """(뼈대, [\\text{…} 내용들]). 뼈대에서는 \\text 류의 내용을 \\x00으로 바꾼다."""
    parts, skel, i = [], [], 0
    while True:
        m = TEXT_CMD_RE.search(tex, i)
        if not m:
            skel.append(tex[i:])
            break
        skel.append(tex[i:m.end()])
        depth, j = 1, m.end()
        while j < len(tex) and depth:
            ch = tex[j]
            if ch == "\\":
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth:                       # 괄호가 닫히지 않음: 비교를 그대로 한다
            skel.append(tex[m.end():])
            break
        parts.append(tex[m.end():j - 1].strip())
        skel.append("\x00}")
        i = j
    return "".join(skel), parts


def compare_math(old, new, base="old"):
    """(errors, notes). old/new는 Doc.math()의 결과. 없어진 것은 base(옛 판)의 줄 번호를 적는다."""
    oc = Counter(k for _, k in old)
    nc = Counter(k for _, k in new)
    common = oc & nc

    def rest(items, counter):
        left = dict(common)
        out = []
        for line, k in items:
            if left.get(k, 0) > 0:
                left[k] -= 1
            else:
                out.append((line, k))
        return out

    o_rest, n_rest = rest(old, oc), rest(new, nc)
    notes, errors = [], []
    used = set()
    for ol, ok in o_rest:
        osk, oparts = text_parts(ok)
        match = None
        for idx, (nl, nk) in enumerate(n_rest):
            if idx in used:
                continue
            nsk, nparts = text_parts(nk)
            if nsk != osk or len(nparts) != len(oparts):
                continue
            diffs = [(a, b) for a, b in zip(oparts, nparts) if a != b]
            if diffs and all(HANGUL.search(a) or HANGUL.search(b) for a, b in diffs):
                match = (idx, nl, diffs)
                break
        if match:
            idx, nl, diffs = match
            used.add(idx)
            what = "; ".join(f"\\text{{{a}}} → \\text{{{b}}}" for a, b in diffs)
            notes.append((nl, f"math changed only in Hangul \\text: {_short(what)}"))
        else:
            errors.append((0, f"math segment removed or changed (line {ol} in {base}): {_short(ok)}"))
    for idx, (nl, nk) in enumerate(n_rest):
        if idx not in used:
            errors.append((nl, f"math segment added or changed: {_short(nk)}"))
    return errors, notes


# ---------------------------------------------------------------- 용어

def _strip_qualifier(s):
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def load_terms(tsv=TERMS_TSV, keep=TERMS_KEEP):
    """(정규식, {공백 뺀 낱말: ('term', 영어) | ('keep', None)})"""
    keep_words = set()
    if os.path.exists(keep):
        with open(keep, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keep_words.add(_strip_qualifier(line))
    terms = {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            ko, en = _strip_qualifier(cols[0]), cols[1].strip()
            if ko and HANGUL.search(ko):
                terms.setdefault(ko, [])
                if en not in terms[ko]:
                    terms[ko].append(en)
    table = {}
    for w in keep_words:
        table[re.sub(r"\s+", "", w)] = ("keep", None)
    for ko, ens in terms.items():
        key = re.sub(r"\s+", "", ko)
        if key not in table:            # terms_keep.txt가 이긴다
            table[key] = ("term", " / ".join(ens))
    phrases = sorted(set(keep_words) | set(terms), key=lambda p: -len(re.sub(r"\s+", "", p)))
    alts = [r"\s*".join(re.escape(w) for w in p.split()) for p in phrases]
    return re.compile("|".join(alts)), table


def term_hits(text, regex, table):
    """text 안의 남은 한국어 용어: [(위치, 한국어, 영어)]"""
    hits = []
    for m in regex.finditer(text):
        key = re.sub(r"\s+", "", m.group(0))
        kind, en = table.get(key, ("keep", None))
        if kind != "term":
            continue
        if len(key) == 1:
            before = text[m.start() - 1] if m.start() > 0 else " "
            if HANGUL.match(before) or not PARTICLE_RE.match(text, m.end()):
                continue
        hits.append((m.start(), key if " " not in m.group(0) else _norm_ws(m.group(0)), en))
    return hits


def _skip_text(t):
    for a in _ancestors(t):
        if a.tag in NO_MATH_TAGS:
            return True
        if a.tag == "head":
            return t.parent is None or t.parent.tag != "title"
    return False


def _ancestors(t):
    p = t.parent
    while p is not None:
        yield p
        p = p.parent


def _dfn_paren_len(t):
    """t가 </dfn> 바로 뒤의 텍스트이고 '(…)'로 시작하면 그 괄호 끝까지의 길이.
    Chapter 0(Tour)은 dfn 대신 <strong>을 쓰므로(§2.3) </strong> 뒤도 같게 본다."""
    parent = t.parent
    if parent is None:
        return 0
    kids = parent.children
    idx = next((i for i, c in enumerate(kids) if c is t), None)
    if not idx or not isinstance(kids[idx - 1], Node) or kids[idx - 1].tag not in ("dfn", "strong"):
        return 0
    m = re.match(r"\s*\([^()]*\)", t.data)
    return m.end() if m else 0


def term_issues(doc, regex, table):
    """[(줄, 메시지)] — 줄마다 묶는다."""
    by_line = {}
    for t in doc.dom.iter_text():
        if _skip_text(t):
            continue
        data = t.data
        cut = _dfn_paren_len(t)
        if cut:
            data = " " * cut + data[cut:]
        for pos, ko, en in term_hits(data, regex, table):
            by_line.setdefault(t.line + data.count("\n", 0, pos), []).append((ko, en))
    for n in doc.dom.iter():
        if n.tag == "img" and n.get("alt"):
            for pos, ko, en in term_hits(html.unescape(n.get("alt")), regex, table):
                by_line.setdefault(n.line, []).append((ko + " (alt)", en))
    out = []
    for line in sorted(by_line):
        seen, items = set(), []
        for ko, en in by_line[line]:
            if ko not in seen:
                seen.add(ko)
                items.append(f"'{ko}' → {en}")
        out.append((line, "Korean term " + ", ".join(items)))
    total = Counter(ko for v in by_line.values() for ko, _ in v)
    return out, total


# ---------------------------------------------------------------- 비교

def compare(old, new, regex, table):
    errors, warns, notes = [], [], []
    oi, ni = old.ids(), new.ids()
    for i in sorted(oi - ni):
        node = next(n for n in old.dom.iter() if n.id == i)
        errors.append((0, f"id '{i}' was removed (line {node.line} in {old.label})"))
    for i in sorted(ni - oi):
        node = next(n for n in new.dom.iter() if n.id == i)
        errors.append((node.line, f"id '{i}' was added"))
    e, n = compare_math(old.math(), new.math(), old.label)
    errors += e
    notes += n
    oc, nc = old.counts(), new.counts()
    for name, _ in COUNTED:
        if oc[name] != nc[name]:
            errors.append((0, f"number of {name} changed: {oc[name]} → {nc[name]}"))
    (ot, olines), (nt, nlines) = old.targets(), new.targets()
    for (a, v), k in sorted((ot - nt).items()):
        errors.append((0, f"{a}=\"{v}\" was removed or changed (line {olines[(a, v)]} in {old.label})"
                          + (f" ({k}×)" if k > 1 else "")))
    for (a, v), k in sorted((nt - ot).items()):
        errors.append((nlines.get((a, v), 0), f"{a}=\"{v}\" was added or changed" + (f" ({k}×)" if k > 1 else "")))
    for line, msg in korean_label_issues(new.dom):
        warns.append((line, msg))
    tw, total = term_issues(new, regex, table)
    warns += tw
    stats = {"ids": len(ni), "math": len(new.math()), "counts": nc, "targets": sum(nt.values()),
             "terms": total}
    return errors, warns, notes, stats


def git_show(path, rev):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    top = subprocess.run(["git", "-C", d, "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if top.returncode != 0:
        return None, f"not in a git repository: {path}"
    top = top.stdout.strip()
    rel = os.path.relpath(path, top).replace(os.sep, "/")
    r = subprocess.run(["git", "-C", top, "show", f"{rev}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return None, f"cannot read {rev}:{rel} ({r.stderr.decode('utf-8', 'replace').strip()})"
    return r.stdout.decode("utf-8"), None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare converted pages with their previous version "
                                             "(ids, math, structure counts, link targets) and report "
                                             "leftover Korean labels and terms.")
    ap.add_argument("--rev", default="HEAD", help="git revision to compare with (default HEAD)")
    ap.add_argument("--against", metavar="OLD", help="compare with this file instead of git (one FILE only)")
    ap.add_argument("--quiet", action="store_true", help="print only errors and the per-file summary")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    if args.against and len(args.files) != 1:
        ap.error("--against takes exactly one FILE")
    regex, table = load_terms()

    total_err = total_warn = 0
    for f in args.files:
        name = os.path.relpath(os.path.abspath(f))
        if not os.path.exists(f):
            print(f"ERROR {name}: file does not exist")
            total_err += 1
            continue
        with open(f, encoding="utf-8") as fh:
            new_raw = fh.read()
        if args.against:
            with open(args.against, encoding="utf-8") as fh:
                old_raw, why, base = fh.read(), None, args.against
        else:
            (old_raw, why), base = git_show(f, args.rev), args.rev
        if old_raw is None:
            print(f"ERROR {name}: {why}")
            total_err += 1
            continue
        errors, warns, notes, st = compare(Doc(old_raw, base), Doc(new_raw, name), regex, table)
        print(f"== {name} (vs {base})")
        for line, msg in errors:
            print(f"ERROR {name}:{line}: {msg}")
        if not args.quiet:
            for line, msg in sorted(warns):
                print(f"WARN {name}:{line}: {msg}")
            for line, msg in notes:
                print(f"NOTE {name}:{line}: {msg}")
        c = st["counts"]
        top_terms = ", ".join(f"{k}×{v}" for k, v in st["terms"].most_common(8))
        print(f"   ids {st['ids']} · math {st['math']} · thm {c['div.thm']} · fig {c['figure.dg-figure']} · "
              f"ex {c['div.exercise']} · eq {c['div.equation']} · img {c['img']} · targets {st['targets']}")
        nterm = sum(st["terms"].values())
        nlabel = len(warns) - len([w for w in warns if w[1].startswith("Korean term")])
        print(f"   {'OK' if not errors else 'FAILED'}: {len(errors)} errors, {nlabel} label warnings, "
              f"{nterm} Korean terms left" + (f" ({top_terms})" if top_terms else "")
              + f", {len(notes)} notes")
        total_err += len(errors)
        total_warn += len(warns)
    print(f"term_guard: {len(args.files)} files, {total_err} errors, {total_warn} warnings")
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
