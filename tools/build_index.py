#!/usr/bin/env python3
"""DG-study 찾아보기(concepts.html) 생성기와 연결·일관성 검사 (GUIDELINES.md §3.3, §3.4, §3.6).

사용법:
    python3 tools/build_index.py [--root DIR] [--check-only] [--output PATH]

    --root DIR     사이트 루트 (기본: tools/의 상위 폴더)
    --check-only   concepts.html을 쓰지 않고 검사만 한다
    --output PATH  출력 경로 (기본: ROOT/concepts.html)

수집: 절 페이지의 <dfn>(용어, 영어, 정의 위치, data-generalizes), div.thm 항목, 내부 링크 그래프.
오류:
  - 같은 용어의 <dfn>이 data-generalizes 없이 두 번 이상 나옴
  - data-generalizes가 가리키는 파일·id가 없음
  - §3.4 표의 장(11 12 13 14 16 18 21 22 23 24 25 26) 중 장 개요 dg-status가
    draft|reviewed|final인데 div.thm.compat 항목이 없음
오류가 있어도 concepts.html은 쓴다(링크가 깨지지 않게). 종료코드는 1.
"""

import argparse
import datetime
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_site import Node, Text, parse_page, repo_root, site_pages  # noqa: E402

COMPAT_CHAPTERS = (11, 12, 13, 14, 16, 18, 21, 22, 23, 24, 25, 26)
CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
# 찾아보기 묶음: 된소리는 예사소리 묶음에 넣는다
GROUP_OF = {"ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ"}
GROUP_ORDER = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ") + ["A–Z", "기호", "기타"]


def norm_term(s):
    return re.sub(r"\s+", " ", s).strip()


def group_of(term):
    t = term.lstrip()
    if t.startswith("\\(") or t.startswith("\\["):
        return "기호"
    if not t:
        return "기타"
    ch = t[0]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        cho = CHOSEONG[(code - 0xAC00) // 588]
        return GROUP_OF.get(cho, cho)
    if ch.isascii() and ch.isalpha():
        return "A–Z"
    return "기타"


def location_map(page):
    """각 요소 → 정의 위치 id (가장 가까운 조상 id, 없으면 문서 순서상 바로 앞의 id)."""
    loc = {}
    last = None
    for n in page.dom.iter():
        anc = n.id or next((a.id for a in n.ancestors() if a.id), None)
        loc[id(n)] = anc or last
        if n.id:
            last = n.id
    return loc


def english_after(dfn):
    """</dfn> 바로 뒤 텍스트가 '(…)'로 시작하면 괄호 안을 영어 용어로 본다."""
    parent = dfn.parent
    kids = parent.children
    idx = next(i for i, c in enumerate(kids) if c is dfn)
    if idx + 1 < len(kids) and isinstance(kids[idx + 1], Text):
        m = re.match(r"\s*\(([^()]*)\)", kids[idx + 1].data)
        if m:
            return norm_term(m.group(1))
    return None


class Index:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.errors = []     # (rel, line, msg)
        self.warnings = []
        self.terms = []      # dict per <dfn>
        self.items = {}      # (rel, id) -> dict
        self.links = []      # dict per internal link
        self.chapters = {}   # chapter number -> {"status", "rel", "dir"}
        self.pages = {}

    def collect(self):
        for path in site_pages(self.root):
            page = parse_page(self.root, path)
            if page.kind in ("excluded", "unknown"):
                continue
            if page.rel == "concepts.html":
                continue
            self.pages[page.rel] = page
            if page.kind == "chapter" and page.info.get("dir_ok"):
                self.chapters[page.info["chapter"]] = {
                    "status": page.meta.get("dg-status"), "rel": page.rel, "dir": page.info["dir"]}
            loc = location_map(page)
            for n in page.dom.iter():
                if n.tag == "div" and n.has_class("thm") and n.id:
                    head = n.find(cls="thm-head")
                    name = n.find(cls="thm-name")
                    self.items[(page.rel, n.id)] = {
                        "rel": page.rel, "id": n.id, "line": n.line,
                        "head": head.text() if head else n.id,
                        "name": name.text() if name else "",
                        "compat": n.has_class("compat"),
                        "chapter": page.info.get("n"), "page": page}
                if n.tag == "dfn" and page.kind == "section":
                    self.terms.append({
                        "term": norm_term(n.text()), "english": english_after(n),
                        "rel": page.rel, "line": n.line, "loc": loc[id(n)],
                        "generalizes": n.get("data-generalizes"), "page": page})
                if n.tag == "a" and n.get("href"):
                    target = self.resolve(page, n.get("href"))
                    if target is not None:
                        self.links.append({
                            "src_rel": page.rel, "src_loc": loc[id(n)], "line": n.line,
                            "target": target, "page": page})

    def resolve(self, page, url):
        """내부 링크 → (대상 rel, fragment) 또는 None."""
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("//"):
            return None
        path, _, frag = url.partition("#")
        path = path.split("?", 1)[0]
        target = page.path if path == "" else os.path.normpath(os.path.join(page.dir, path))
        if not target.startswith(self.root + os.sep):
            return None
        return os.path.relpath(target, self.root).replace(os.sep, "/"), frag

    def check(self):
        # 1) 중복 정의
        by_term = {}
        for t in self.terms:
            by_term.setdefault(t["term"], []).append(t)
        for term, defs in by_term.items():
            plain = [d for d in defs if not d["generalizes"]]
            if len(plain) > 1:
                where = ", ".join(f"{d['rel']}:{d['line']}" for d in plain)
                for d in plain[1:]:
                    self.errors.append((d["rel"], d["line"],
                                        f"term '{term}' is defined more than once without data-generalizes "
                                        f"({where}); link to the first definition instead (§3.3)"))
        # 2) data-generalizes 대상
        for t in self.terms:
            g = t["generalizes"]
            if g is None:
                continue
            if not g.strip():
                self.errors.append((t["rel"], t["line"], "empty data-generalizes"))
                t["gen_target"] = None
                continue
            res = self.resolve(t["page"], g)
            ok = False
            if res is not None:
                trel, frag = res
                tpath = os.path.join(self.root, trel)
                if os.path.exists(tpath) and frag:
                    tpage = parse_page(self.root, tpath)
                    ok = frag in tpage.ids
                elif os.path.exists(tpath) and not frag:
                    self.errors.append((t["rel"], t["line"],
                                        f"data-generalizes '{g}' must point to a definition id (…#i-N-M-K)"))
                    t["gen_target"] = None
                    continue
            if not ok:
                self.errors.append((t["rel"], t["line"],
                                    f"data-generalizes target '{g}' does not exist (§3.3)"))
                t["gen_target"] = None
            else:
                t["gen_target"] = res
        # 3) 호환성 명제
        for ch in COMPAT_CHAPTERS:
            info = self.chapters.get(ch)
            if not info or info["status"] not in ("draft", "reviewed", "final"):
                continue
            has = any(it["compat"] for it in self.items.values() if it["chapter"] == ch)
            if not has:
                self.errors.append((info["rel"], 1,
                                    f"chapter {ch} is '{info['status']}' but has no div.thm.compat item "
                                    f"(필수 호환성 명제, §3.4)"))

    # ------------------------------------------------------------ 출력

    def item_label(self, rel, loc):
        page = self.pages.get(rel)
        it = self.items.get((rel, loc))
        dg = page.meta.get("dg-id", rel) if page else rel
        if it:
            return it["head"]
        base = f"{dg}절" if page is not None and page.kind == "section" else dg
        node = page.ids.get(loc) if (page is not None and loc) else None
        if node is not None:
            heading = node if node.tag in ("h1", "h2", "h3", "h4") else (node.find("h2") or node.find("h3"))
            if heading is not None and heading.text():
                return f"{base} · {heading.text()}"
        return base

    def href(self, rel, frag=None):
        return rel + (f"#{frag}" if frag else "")

    def render(self, today):
        e = html.escape
        # 정의 위치별 역참조
        back = {}
        for ln in self.links:
            trel, frag = ln["target"]
            if not frag:
                continue
            if (ln["src_rel"], ln["src_loc"]) == (trel, frag):
                continue
            back.setdefault((trel, frag), []).append(ln)
        # 용어 → 일반화 관계
        terms_at = {}
        for t in self.terms:
            terms_at.setdefault((t["rel"], t["loc"]), []).append(t)
        generalized_by = {}
        for t in self.terms:
            tgt = t.get("gen_target")
            if tgt:
                generalized_by.setdefault(tgt, []).append(t)

        groups = {}
        for t in sorted(self.terms, key=lambda t: (t["term"], t["rel"], t["line"])):
            groups.setdefault(group_of(t["term"]), []).append(t)

        out = []
        w = out.append
        w("<!doctype html>")
        w('<html lang="ko">')
        w("<head>")
        w('  <meta charset="utf-8">')
        w('  <meta name="viewport" content="width=device-width, initial-scale=1">')
        w("  <title>찾아보기 — 미분기하 스터디</title>")
        w('  <meta name="dg-id" content="concepts">')
        w(f'  <meta name="dg-updated" content="{today}">')
        w('  <link rel="stylesheet" href="assets/style.css">')
        w('  <script src="assets/mathjax-config.js"></script>')
        w('  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>')
        w("</head>")
        w("<body>")
        w('  <nav class="breadcrumb">')
        w('    <a href="index.html">전체 목차</a> ›')
        w("    <span>찾아보기</span>")
        w("  </nav>")
        w("")
        w("  <main>")
        w("    <header>")
        w("      <h1>찾아보기</h1>")
        w('      <p class="meta">용어가 정의된 곳, 일반화 관계, 쓰이는 곳을 모은다. '
          "이 페이지는 <code>tools/build_index.py</code>가 자동으로 만들므로 직접 고치지 않는다.</p>")
        w("    </header>")
        w("")
        w('    <section id="sec-terms">')
        w("      <h2>용어</h2>")
        if not self.terms:
            w("      <p>아직 정의된 용어가 없다.</p>")
        k = 0
        for gi, g in enumerate(GROUP_ORDER, 1):
            if g not in groups:
                continue
            w(f'      <h3 id="sec-group-{gi}">{e(g)}</h3>')
            w("      <ul>")
            for t in groups[g]:
                k += 1
                parts = [f"<strong>{e(t['term'])}</strong>"]
                if t["english"]:
                    parts.append(f" ({e(t['english'])})")
                parts.append(" — 정의: " + self._link(t["rel"], t["loc"]))
                tgt = t.get("gen_target")
                if tgt:
                    gen_terms = terms_at.get(tgt, [])
                    label = ", ".join(e(x["term"]) for x in gen_terms) or e(self.item_label(*tgt))
                    parts.append(f" · 일반화하는 개념: {label} ({self._link(*tgt)})")
                ups = generalized_by.get((t["rel"], t["loc"]), [])
                if ups:
                    parts.append(" · 더 일반적인 개념: " + ", ".join(
                        f"{e(u['term'])} ({self._link(u['rel'], u['loc'])})" for u in ups))
                uses = back.get((t["rel"], t["loc"]), [])
                seen, links = set(), []
                for u in sorted(uses, key=lambda u: (u["src_rel"], u["line"])):
                    key = (u["src_rel"], u["src_loc"])
                    if key in seen:
                        continue
                    seen.add(key)
                    links.append(self._link(u["src_rel"], u["src_loc"]))
                if links:
                    parts.append(" · 쓰이는 곳: " + ", ".join(links))
                w(f'        <li id="term-{k}">' + "".join(parts) + "</li>")
            w("      </ul>")
        w("    </section>")
        w("")
        w('    <section id="sec-compat">')
        w("      <h2>호환성 명제</h2>")
        w("      <p>Part I의 개념과 Part II·III의 일반화가 일치함을 보이는 명제들이다(GUIDELINES.md §3.4).</p>")
        compat = sorted((it for it in self.items.values() if it["compat"]),
                        key=lambda it: (it["chapter"] or 0, it["rel"], it["line"]))
        if not compat:
            w("      <p>아직 작성된 호환성 명제가 없다.</p>")
        else:
            w("      <ul>")
            for it in compat:
                name = f" {e(it['name'])}" if it["name"] else ""
                w(f"        <li>{self._link(it['rel'], it['id'])}{name}</li>")
            w("      </ul>")
        w("    </section>")
        w("  </main>")
        w("</body>")
        w("</html>")
        return "\n".join(out) + "\n"

    def _link(self, rel, loc):
        return f'<a href="{html.escape(self.href(rel, loc), quote=True)}">{html.escape(self.item_label(rel, loc))}</a>'


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build concepts.html and check §3 consistency rules")
    ap.add_argument("--root", default=repo_root())
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    idx = Index(args.root)
    idx.collect()
    idx.check()
    if not args.check_only:
        out = args.output or os.path.join(idx.root, "concepts.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(idx.render(datetime.date.today().isoformat()))
    for rel, line, msg in sorted(idx.errors):
        print(f"ERROR {rel}:{line}: {msg}")
    for rel, line, msg in sorted(idx.warnings):
        print(f"WARN {rel}:{line}: {msg}")
    ncompat = sum(1 for it in idx.items.values() if it["compat"])
    print(f"build_index: {len(idx.pages)} pages, {len(idx.terms)} terms, {len(idx.items)} items, "
          f"{ncompat} compat, {len(idx.errors)} errors"
          + ("" if args.check_only else " — wrote concepts.html"))
    return 1 if idx.errors else 0


if __name__ == "__main__":
    sys.exit(main())
