#!/usr/bin/env python3
"""DG-study 사이트 HTML 검사기.

GUIDELINES.md §11(HTML 작성 규칙)의 페이지 계약을 검사한다. 표준 라이브러리만 쓴다.

사용법:
    python3 tools/check_site.py [--root DIR] [--no-warn] [파일 ...]

    --root DIR   사이트 루트 (기본: tools/의 상위 폴더)
    --no-warn    WARN 줄을 출력하지 않는다 (개수는 요약에 나온다)
    파일 ...     이 파일들만 검사한다 (링크 대상은 루트 기준으로 확인)

출력: `ERROR 경로:줄: 메시지` / `WARN 경로:줄: 메시지`, 마지막에 요약.
오류가 하나라도 있으면 종료코드 1.

build_index.py가 이 파일의 파서(parse_page, site_pages 등)를 가져다 쓴다.
"""

import argparse
import datetime
import os
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote

# ---------------------------------------------------------------- 상수

MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"
ROOT_PAGES = ("index", "notation", "glossary", "references", "concepts")
STATUSES = ("planned", "draft", "reviewed", "final")
STATUS_LABEL = {"planned": "계획", "draft": "초안", "reviewed": "검토 완료", "final": "확정"}
THM_WORD = {
    "definition": "정의", "theorem": "정리", "proposition": "명제", "lemma": "보조정리",
    "corollary": "따름정리", "example": "예", "nonexample": "비예", "remark": "비고",
}
REQUIRED_META = {
    "section": ("dg-id", "dg-status", "dg-prereq", "dg-refs", "dg-updated"),
    "chapter": ("dg-id", "dg-status", "dg-updated"),
    "root": ("dg-id", "dg-updated"),
}
KNOWN_META = {"dg-id", "dg-status", "dg-prereq", "dg-refs", "dg-updated"}

# §11.3 클래스 계약. style.css에서 읽은 클래스와 합쳐 허용 목록으로 쓴다.
CONTRACT_CLASSES = set("""
    thm definition theorem proposition lemma corollary example nonexample remark compat
    thm-head thm-name proof proof-idea proof-head
    box intuition warning convention forward backward history box-title
    equation dg-figure fig-head interactive cite todo verified
    exercise level hint solution goals prereq summary exercises next
    toc toc-title toc-note status planned draft reviewed final
    inherits questions handoff threads breadcrumb pager meta
""".split())

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
        "param", "source", "track", "wbr"}
BLOCK = {"address", "article", "aside", "blockquote", "details", "div", "dl", "fieldset",
         "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
         "header", "hr", "main", "nav", "ol", "p", "pre", "section", "table", "ul"}
NO_MATH_TAGS = {"script", "style", "code", "pre", "textarea", "noscript"}
SKIP_DIRS = {"tools", "refs", "assets", "node_modules", ".git"}

SECTION_RE = re.compile(r"^(\d+)-(\d+)-[a-z0-9]+(?:-[a-z0-9]+)*\.html$")
CHAPTER_DIR_RE = re.compile(r"^(\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------- DOM

class Text:
    __slots__ = ("data", "line", "parent")

    def __init__(self, data, line, parent):
        self.data, self.line, self.parent = data, line, parent


class Node:
    __slots__ = ("tag", "attrs", "line", "parent", "children", "classes", "order")

    def __init__(self, tag, attrs, line, parent, order):
        self.tag = tag
        self.attrs = {}
        for k, v in attrs:
            self.attrs.setdefault(k, "" if v is None else v)
        self.line = line
        self.parent = parent
        self.children = []
        self.classes = (self.attrs.get("class") or "").split()
        self.order = order

    def get(self, key, default=None):
        return self.attrs.get(key, default)

    @property
    def id(self):
        return self.attrs.get("id")

    def has_class(self, cls):
        return cls in self.classes

    def iter(self):
        """자기 자신을 포함한 전위 순회(문서 순서)."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed([c for c in node.children if isinstance(c, Node)]))

    def iter_text(self):
        for c in self.children:
            if isinstance(c, Text):
                yield c
            else:
                yield from c.iter_text()

    def text(self):
        return re.sub(r"\s+", " ", "".join(t.data for t in self.iter_text())).strip()

    def find_all(self, tag=None, cls=None):
        return [n for n in self.iter()
                if n is not self and (tag is None or n.tag == tag)
                and (cls is None or cls in n.classes)]

    def find(self, tag=None, cls=None):
        for n in self.iter():
            if n is not self and (tag is None or n.tag == tag) and (cls is None or cls in n.classes):
                return n
        return None

    def ancestors(self):
        p = self.parent
        while p is not None:
            yield p
            p = p.parent

    def inside(self, tags):
        return any(a.tag in tags for a in self.ancestors())


class DomBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.order = 0
        self.root = Node("#document", [], 1, None, 0)
        self.stack = [self.root]
        self.problems = []      # (line, message) — 구조 오류
        self.comments = []      # (line, text)
        self.doctype = None

    def _new(self, tag, attrs):
        self.order += 1
        line = self.getpos()[0]
        parent = self.stack[-1]
        if tag in BLOCK and any(n.tag == "p" for n in self.stack):
            self.problems.append((line, f"<{tag}> inside <p>: the browser closes the <p> early; "
                                        f"end the paragraph before this element"))
        node = Node(tag, attrs, line, parent, self.order)
        parent.children.append(node)
        return node

    def handle_starttag(self, tag, attrs):
        node = self._new(tag, attrs)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self._new(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        line = self.getpos()[0]
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                for n in self.stack[i + 1:]:
                    self.problems.append((n.line, f"<{n.tag}> is never closed (closed implicitly by </{tag}> on line {line})"))
                del self.stack[i:]
                return
        self.problems.append((line, f"stray </{tag}> with no matching open tag"))

    def handle_data(self, data):
        self.stack[-1].children.append(Text(data, self.getpos()[0], self.stack[-1]))

    def handle_comment(self, data):
        self.comments.append((self.getpos()[0], data))

    def handle_decl(self, decl):
        self.doctype = decl

    def finish(self):
        self.close()
        for n in self.stack[1:]:
            self.problems.append((n.line, f"<{n.tag}> is never closed"))
        self.stack = [self.root]


class Page:
    """파싱된 페이지 하나."""

    def __init__(self, root, path):
        self.root_dir = root
        self.path = os.path.abspath(path)
        self.rel = os.path.relpath(self.path, root).replace(os.sep, "/")
        with open(self.path, encoding="utf-8") as f:
            self.raw = f.read()
        builder = DomBuilder()
        builder.feed(self.raw)
        builder.finish()
        self.dom = builder.root
        self.problems = builder.problems
        self.comments = builder.comments
        self.doctype = builder.doctype
        self.ids = {}
        self.dup_ids = []
        for n in self.dom.iter():
            i = n.id
            if i is None:
                continue
            if i in self.ids:
                self.dup_ids.append((n, self.ids[i]))
            else:
                self.ids[i] = n
        self.kind, self.info = classify(root, self.path)
        self.meta = {}
        for m in self.dom.find_all("meta"):
            name = m.get("name")
            if name:
                self.meta.setdefault(name, m.get("content", ""))

    @property
    def dir(self):
        return os.path.dirname(self.path)


_page_cache = {}


def parse_page(root, path):
    key = os.path.abspath(path)
    if key not in _page_cache:
        _page_cache[key] = Page(root, key)
    return _page_cache[key]


# ---------------------------------------------------------------- 사이트 구조

def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify(root, path):
    """(종류, 정보) — 종류는 root | chapter | section | excluded | unknown."""
    rel = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
    parts = rel.split("/")
    name = parts[-1]
    if name.endswith("-interactive.html"):
        return "excluded", {}
    if len(parts) == 1:
        return "root", {"stem": name[:-5]}
    if parts[0] == "chapters" and len(parts) == 3:
        m_dir = CHAPTER_DIR_RE.match(parts[1])
        chap = int(m_dir.group(1)) if m_dir else None
        info = {"dir": parts[1], "chapter": chap, "dir_ok": bool(m_dir)}
        if name == "index.html":
            return "chapter", info
        m = SECTION_RE.match(name)
        if m:
            info.update(n=int(m.group(1)), m=int(m.group(2)))
            return "section", info
    return "unknown", {}


def site_pages(root):
    """검사 대상 HTML 전부 (GUIDELINES.md: tools/, refs/, assets/, node_modules, *-interactive.html 제외)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        else:
            dirnames[:] = [d for d in dirnames if d != "node_modules" and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".html") and not fn.endswith("-interactive.html"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# ---------------------------------------------------------------- 원문 기반 수식 스캔

_BLANK_RE = re.compile(
    r"<!--.*?-->"
    r"|<head\b[^>]*>.*?</head\s*>"
    r"|<(script|style|code|pre|textarea|noscript)\b[^>]*>.*?</\1\s*>",
    re.S | re.I)


def blank_regions(raw):
    """MathJax가 처리하지 않는 영역(주석, head, script/style/code/pre/textarea/noscript)을
    줄바꿈을 보존한 공백으로 바꾼다."""
    return _BLANK_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), raw)


def _unescaped(text, pos):
    """text[pos]의 백슬래시가 다른 백슬래시에 의해 이스케이프되지 않았는가."""
    return pos == 0 or text[pos - 1] != "\\"


def find_math_spans(text):
    """(start, end, display, tex, error) 목록. end/tex는 닫히지 않으면 None."""
    spans = []
    i, n = 0, len(text)
    opener_re = re.compile(r"\\[(\[]")
    while True:
        m = opener_re.search(text, i)
        if not m:
            break
        p = m.start()
        if not _unescaped(text, p):
            i = p + 2
            continue
        display = text[p + 1] == "["
        close_tok = "\\]" if display else "\\)"
        open_tok = "\\[" if display else "\\("
        j = p + 2
        end = None
        nested = None
        while True:
            c = text.find(close_tok, j)
            o = text.find(open_tok, p + 2)
            if c == -1:
                break
            if not _unescaped(text, c):
                j = c + 2
                continue
            if o != -1 and o < c and _unescaped(text, o):
                nested = o
            end = c
            break
        if end is None:
            spans.append((p, None, display, None, f"unclosed {open_tok} (no matching {close_tok})"))
            i = p + 2
            continue
        err = None
        if nested is not None:
            err = f"{open_tok} opened again before the previous one was closed"
        spans.append((p, end + 2, display, text[p + 2:end], err))
        i = end + 2
    return spans


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


# ---------------------------------------------------------------- 보고

class Reporter:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, rel, line, msg):
        self.errors.append((rel, line or 0, msg))

    def warn(self, rel, line, msg):
        self.warnings.append((rel, line or 0, msg))


def css_classes(root):
    path = os.path.join(root, "assets", "style.css")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        css = f.read()
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    css = re.sub(r"url\([^)]*\)", " ", css)
    css = re.sub(r"\"[^\"]*\"|'[^']*'", " ", css)
    return set(re.findall(r"\.(-?[_a-zA-Z][_a-zA-Z0-9-]*)", css))


# ---------------------------------------------------------------- 검사

class SiteChecker:
    def __init__(self, root, rep):
        self.root = os.path.abspath(root)
        self.rep = rep
        css = css_classes(self.root)
        self.allowed_classes = None if css is None else css | CONTRACT_CLASSES
        self.todo_total = 0
        self.review_total = 0
        self.pages_checked = 0

    # -- 공통

    def check(self, path):
        page = parse_page(self.root, path)
        rep, rel = self.rep, page.rel
        self.pages_checked += 1
        if page.kind == "excluded":
            return
        if page.kind == "unknown":
            rep.error(rel, 1, "unexpected page location or file name (expected root page, "
                              "chapters/NN-slug/index.html or chapters/NN-slug/N-M-slug.html)")
            return
        for line, msg in page.problems:
            rep.error(rel, line, msg)
        for node, first in page.dup_ids:
            rep.error(rel, node.line, f"duplicate id '{node.id}' (first used on line {first.line})")
        self.check_head(page)
        self.check_forbidden(page)
        self.check_links(page)
        self.check_math_text(page)
        self.count_markers(page)
        if page.kind == "root":
            self.check_root(page)
        elif page.kind == "chapter":
            self.check_chapter(page)
        elif page.kind == "section":
            self.check_section(page)

    def rel_prefix(self, page):
        return "" if page.kind == "root" else "../../"

    def check_head(self, page):
        rep, rel = self.rep, page.rel
        if not page.doctype or page.doctype.strip().lower() != "doctype html":
            rep.error(rel, 1, "missing <!doctype html>")
        html = page.dom.find("html")
        if html is None:
            rep.error(rel, 1, "missing <html>")
            return
        if html.get("lang") != "ko":
            rep.error(rel, html.line, '<html> must have lang="ko"')
        head = html.find("head")
        if head is None:
            rep.error(rel, html.line, "missing <head>")
            return
        metas = head.find_all("meta")
        if not any((m.get("charset") or "").lower() == "utf-8" for m in metas):
            rep.error(rel, head.line, 'missing <meta charset="utf-8">')
        if not any(m.get("name") == "viewport" and "width=device-width" in (m.get("content") or "")
                   for m in metas):
            rep.error(rel, head.line, 'missing <meta name="viewport" content="width=device-width, initial-scale=1">')
        title = head.find("title")
        if title is None or not title.text():
            rep.error(rel, head.line, "missing or empty <title>")

        # dg-* 메타
        kind = page.kind
        for name in REQUIRED_META[kind]:
            if name not in page.meta:
                rep.error(rel, head.line, f'missing <meta name="{name}"> (§11.6)')
        for m in metas:
            name = m.get("name") or ""
            if name.startswith("dg-") and name not in KNOWN_META:
                rep.warn(rel, m.line, f"unknown metadata '{name}'")
            if name.startswith("dg-") and name not in REQUIRED_META[kind]:
                rep.warn(rel, m.line, f"'{name}' is not used on {kind} pages (§11.6)")
        upd = page.meta.get("dg-updated")
        if upd is not None:
            ok = bool(DATE_RE.match(upd))
            if ok:
                try:
                    datetime.date.fromisoformat(upd)
                except ValueError:
                    ok = False
            if not ok:
                rep.error(rel, head.line, f"dg-updated '{upd}' is not a YYYY-MM-DD date")
        st = page.meta.get("dg-status")
        if st is not None and st not in STATUSES:
            rep.error(rel, head.line, f"dg-status '{st}' must be one of {'|'.join(STATUSES)}")

        # 리소스 로드 순서
        pre = self.rel_prefix(page)
        want_css = pre + "assets/style.css"
        want_cfg = pre + "assets/mathjax-config.js"
        loads = []
        for n in page.dom.iter():
            if n.tag == "link" and "stylesheet" in (n.get("rel") or "").split():
                loads.append(n)
            elif n.tag == "script":
                loads.append(n)
        styles = [n for n in loads if n.tag == "link"]
        scripts = [n for n in loads if n.tag == "script"]
        for n in styles:
            if n.get("href") != want_css:
                rep.error(rel, n.line, f"unexpected stylesheet '{n.get('href')}' (only {want_css} is allowed)")
        if not any(n.get("href") == want_css for n in styles):
            rep.error(rel, head.line, f'missing <link rel="stylesheet" href="{want_css}">')
        for n in scripts:
            src = n.get("src")
            if src is None:
                rep.error(rel, n.line, "inline <script> is not allowed (only MathJax config and CDN)")
            elif src not in (want_cfg, MATHJAX_CDN):
                rep.error(rel, n.line, f"<script src='{src}'> is not allowed (only {want_cfg} and the MathJax CDN)")
        cfg = [n for n in scripts if n.get("src") == want_cfg]
        cdn = [n for n in scripts if n.get("src") == MATHJAX_CDN]
        if not cfg:
            rep.error(rel, head.line, f'missing <script src="{want_cfg}"></script>')
        if not cdn:
            rep.error(rel, head.line, f'missing <script defer src="{MATHJAX_CDN}"></script>')
        for n in cdn:
            if "defer" not in n.attrs:
                rep.error(rel, n.line, "the MathJax CDN <script> must have the defer attribute")
        for group in (styles, cfg, cdn):
            for n in group:
                if not n.inside({"head"}):
                    rep.error(rel, n.line, f"<{n.tag}> for {n.get('href') or n.get('src')} must be inside <head>")
        good = [x for x in (styles[:1], cfg[:1], cdn[:1]) if x]
        if len(good) == 3:
            a, b, c = styles[0].order, cfg[0].order, cdn[0].order
            if not (a < b < c):
                rep.error(rel, cfg[0].line, "load order must be style.css → mathjax-config.js → MathJax CDN")

    def check_forbidden(self, page):
        rep, rel = self.rep, page.rel
        for n in page.dom.iter():
            if n.tag == "style":
                rep.error(rel, n.line, "<style> elements are not allowed; add classes to assets/style.css")
            if "style" in n.attrs and n.tag != "#document":
                rep.error(rel, n.line, "style= attributes are not allowed (§11.1)")
            if self.allowed_classes is not None:
                for c in n.classes:
                    if c not in self.allowed_classes:
                        rep.error(rel, n.line, f"class '{c}' is not in the class contract / assets/style.css (§11.3)")

    def resolve(self, page, url):
        """상대 URL → (절대경로, fragment). 외부 URL이면 None."""
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith("//"):
            return None
        path, _, frag = url.partition("#")
        path = path.split("?", 1)[0]
        target = page.path if path == "" else os.path.normpath(os.path.join(page.dir, unquote(path)))
        return target, frag

    def check_links(self, page):
        rep, rel = self.rep, page.rel
        for n in page.dom.iter():
            for attr in ("href", "src"):
                url = n.get(attr)
                if url is None or n.tag not in ("a", "img", "iframe", "link", "script", "source"):
                    continue
                if url.strip() == "":
                    rep.error(rel, n.line, f"empty {attr} on <{n.tag}>")
                    continue
                if url.lower().startswith("javascript:"):
                    rep.error(rel, n.line, "javascript: URLs are not allowed")
                    continue
                res = self.resolve(page, url)
                if res is None:
                    continue
                target, frag = res
                if not (target == self.root or target.startswith(self.root + os.sep)):
                    rep.error(rel, n.line, f"link '{url}' points outside the site root")
                    continue
                if url.endswith("#"):
                    rep.error(rel, n.line, f"link '{url}' has an empty fragment")
                    continue
                if os.path.isdir(target):
                    rep.error(rel, n.line, f"link '{url}' points to a directory; link the file (e.g. index.html)")
                    continue
                if not os.path.exists(target):
                    rep.error(rel, n.line, f"broken link '{url}': file does not exist")
                    continue
                if frag:
                    if target.endswith(".html"):
                        tpage = parse_page(self.root, target)
                        if unquote(frag) not in tpage.ids:
                            rep.error(rel, n.line, f"broken link '{url}': no element with id '{frag}' in {tpage.rel}")

    def check_math_text(self, page):
        rep, rel = self.rep, page.rel
        # 1) '$' 금지 (code/pre/script/style/textarea 밖의 텍스트)
        for t in page.dom.iter_text():
            if "$" in t.data and not _text_inside(t, NO_MATH_TAGS):
                off = t.data.index("$")
                rep.error(rel, t.line + t.data.count("\n", 0, off),
                          "'$' in text: use \\( … \\) / \\[ … \\] for math (§11.4)")
        # 2) 원문 기준 수식 구간
        text = blank_regions(page.raw)
        for start, end, display, tex, err in find_math_spans(text):
            line = line_of(text, start)
            if tex is None:
                rep.error(rel, line, err)
                continue
            if err:
                rep.error(rel, line, err)
            if "<" in tex:
                rep.error(rel, line + tex.count("\n", 0, tex.index("<")),
                          "raw '<' inside math: write \\lt (or &lt;) — the browser parses '<' as a tag (§11.4)")
        # 3) \tag는 div.equation 안에서만
        for t in page.dom.iter_text():
            if "\\tag{" in t.data and not any(a.tag == "div" and a.has_class("equation") for a in _text_ancestors(t)) \
                    and not _text_inside(t, NO_MATH_TAGS):
                rep.error(rel, t.line, "\\tag{…} outside div.equation (§11.3)")
        # 4) 페이지 안의 매크로 정의 금지
        for m in re.finditer(r"\\(newcommand|renewcommand|def|let)\b", text):
            rep.error(rel, line_of(text, m.start()), f"\\{m.group(1)} in a page is not allowed; request a macro for 부록 C (§11.4)")

    def count_markers(self, page):
        todo = sum(1 for n in page.dom.iter() if n.tag == "span" and n.has_class("todo"))
        review = sum(1 for _, c in page.comments if c.strip().startswith("REVIEW"))
        self.todo_total += todo
        self.review_total += review
        page.todo_count, page.review_count = todo, review

    # -- 공용 페이지

    def check_root(self, page):
        rep, rel = self.rep, page.rel
        stem = page.info["stem"]
        if stem not in ROOT_PAGES:
            rep.warn(rel, 1, f"unexpected root page '{stem}.html' (expected one of {', '.join(ROOT_PAGES)})")
        dgid = page.meta.get("dg-id")
        if dgid is not None and dgid != stem:
            rep.error(rel, 1, f"dg-id '{dgid}' must be '{stem}' on this page")
        if stem != "index":
            nav = _find_nav(page, "breadcrumb")
            if nav is None:
                rep.error(rel, 1, "missing <nav class=\"breadcrumb\">")
            elif not any(a.get("href") == "index.html" for a in nav.find_all("a")):
                rep.error(rel, nav.line, "breadcrumb must link to index.html (전체 목차)")
        self.check_main(page)

    def check_main(self, page):
        main = page.dom.find("main")
        if main is None:
            self.rep.error(page.rel, 1, "missing <main>")
        return main

    # -- 장 개요

    def check_chapter(self, page):
        rep, rel, info = self.rep, page.rel, page.info
        if not info["dir_ok"]:
            rep.error(rel, 1, f"chapter folder '{info['dir']}' must look like NN-slug (two digits, lowercase kebab-case)")
            return
        n = info["chapter"]
        dgid = page.meta.get("dg-id")
        if dgid is not None and dgid != str(n):
            rep.error(rel, 1, f"dg-id '{dgid}' must be '{n}' for folder {info['dir']}")
        nav = _find_nav(page, "breadcrumb")
        if nav is None:
            rep.error(rel, 1, "missing <nav class=\"breadcrumb\">")
        elif not any(a.get("href") == "../../index.html" for a in nav.find_all("a")):
            rep.error(rel, nav.line, "breadcrumb must link to ../../index.html (전체 목차)")
        main = self.check_main(page)
        if main is None:
            return
        h1 = _header_h1(main)
        if h1 is None:
            rep.error(rel, main.line, "missing <header><h1>…</h1></header> in <main>")
        elif not h1.text().startswith(f"{n}. "):
            rep.error(rel, h1.line, f"<h1> must start with '{n}. ' (got '{h1.text()}')")
        if main.find("p", "threads") is None:
            rep.error(rel, main.line, "missing <p class=\"threads\"> (줄기 위치, §3.2)")
        for cls, label in (("questions", "이 장의 질문"), ("inherits", "이어받는 것"), ("handoff", "넘겨주는 것")):
            if main.find("section", cls) is None:
                rep.error(rel, main.line, f"missing <section class=\"{cls}\"> ({label}, §3.2)")
        toc = main.find("ol", "toc")
        if toc is None:
            rep.error(rel, main.line, "missing <ol class=\"toc\"> (절 목록)")
        else:
            self.check_toc(page, toc, n)
        pager = _find_nav(page, "pager")
        if pager is None:
            rep.error(rel, 1, "missing <nav class=\"pager\"> (이전/다음 장)")
        elif not pager.find_all("a"):
            rep.warn(rel, pager.line, "pager has no links")

    def check_toc(self, page, toc, n):
        rep, rel = self.rep, page.rel
        lis = [c for c in toc.children if isinstance(c, Node) and c.tag == "li"]
        if not lis:
            rep.error(rel, toc.line, "ol.toc has no <li> entries")
        any_planned = False
        for j, li in enumerate(lis, 1):
            want_id = f"toc-{n}-{j}"
            if li.id != want_id:
                rep.error(rel, li.line, f"toc entry {j} must have id '{want_id}' (got '{li.id}')")
            dfile = li.get("data-file")
            fre = re.compile(rf"^{n}-{j}-[a-z0-9]+(?:-[a-z0-9]+)*\.html$")
            if not dfile or not fre.match(dfile):
                rep.error(rel, li.line, f"toc entry {j} needs data-file=\"{n}-{j}-slug.html\" (got '{dfile}')")
            title = li.find(cls="toc-title")
            status = [s for s in li.find_all("span", "status")]
            if title is None:
                rep.error(rel, li.line, "toc entry needs a .toc-title (span if unwritten, a if written)")
            elif not title.text().startswith(f"{n}.{j} "):
                rep.error(rel, title.line, f"toc title must start with '{n}.{j} ' (got '{title.text()}')")
            st = None
            if len(status) != 1:
                rep.error(rel, li.line, "toc entry needs exactly one <span class=\"status …\">")
            else:
                s = status[0]
                kinds = [c for c in s.classes if c in STATUSES]
                if len(kinds) != 1:
                    rep.error(rel, s.line, f"status span needs one of {'|'.join(STATUSES)} as a class")
                else:
                    st = kinds[0]
                    if s.text() != STATUS_LABEL[st]:
                        rep.error(rel, s.line, f"status label for '{st}' must be '{STATUS_LABEL[st]}' (got '{s.text()}')")
            if st == "planned":
                any_planned = True
            exists = bool(dfile) and os.path.exists(os.path.join(page.dir, dfile))
            if title is not None and title.tag == "a":
                if dfile and title.get("href") != dfile:
                    rep.error(rel, title.line, f"toc link href '{title.get('href')}' must equal data-file '{dfile}'")
                if st == "planned":
                    rep.warn(rel, li.line, f"{dfile} is linked but still marked 계획 (planned)")
                if exists and st:
                    sec = parse_page(self.root, os.path.join(page.dir, dfile))
                    sst = sec.meta.get("dg-status")
                    if sst and sst != st:
                        rep.warn(rel, li.line, f"toc status '{st}' differs from {dfile} dg-status '{sst}'")
            elif title is not None and title.tag == "span":
                if exists:
                    rep.warn(rel, li.line, f"{dfile} exists but its toc entry is not linked")
                if st and st != "planned":
                    rep.error(rel, li.line, "an unlinked toc entry must have status planned (계획)")
        cst = page.meta.get("dg-status")
        if cst in ("draft", "reviewed", "final") and any_planned:
            rep.warn(rel, 1, f"chapter dg-status is '{cst}' but some sections are still planned (§11.6)")

    # -- 절

    def check_section(self, page):
        rep, rel, info = self.rep, page.rel, page.info
        n, m = info["n"], info["m"]
        if not info["dir_ok"]:
            rep.error(rel, 1, f"chapter folder '{info['dir']}' must look like NN-slug")
        elif info["chapter"] != n:
            rep.error(rel, 1, f"section {n}.{m} is in folder '{info['dir']}' (chapter {info['chapter']})")
        dgid = page.meta.get("dg-id")
        if dgid is not None and dgid != f"{n}.{m}":
            rep.error(rel, 1, f"dg-id '{dgid}' must be '{n}.{m}' to match the file name")
        pre = page.meta.get("dg-prereq")
        if pre is not None and pre.strip():
            for tok in pre.split(","):
                if not re.match(r"^\d+(\.\d+)?$", tok.strip()):
                    rep.error(rel, 1, f"dg-prereq entry '{tok.strip()}' must look like N.M or N")
        if page.meta.get("dg-refs", "x").strip() == "":
            rep.warn(rel, 1, "dg-refs is empty (대응 교재를 적는다)")

        nav = _find_nav(page, "breadcrumb")
        if nav is None:
            rep.error(rel, 1, "missing <nav class=\"breadcrumb\">")
        else:
            hrefs = {a.get("href") for a in nav.find_all("a")}
            if "../../index.html" not in hrefs:
                rep.error(rel, nav.line, "breadcrumb must link to ../../index.html (전체 목차)")
            if "index.html" not in hrefs:
                rep.error(rel, nav.line, "breadcrumb must link to index.html (장 개요)")
        if _find_nav(page, "pager") is None:
            rep.error(rel, 1, "missing <nav class=\"pager\"> (이전/다음 절)")
        main = self.check_main(page)
        if main is None:
            return
        h1 = _header_h1(main)
        if h1 is None:
            rep.error(rel, main.line, "missing <header><h1>…</h1></header> in <main>")
        elif not h1.text().startswith(f"{n}.{m} "):
            rep.error(rel, h1.line, f"<h1> must start with '{n}.{m} ' (got '{h1.text()}')")
        for cls, label in (("goals", "이 절의 목표"), ("prereq", "선수 지식"),
                           ("summary", "요약"), ("exercises", "연습문제")):
            if main.find("section", cls) is None:
                rep.error(rel, main.line, f"missing <section class=\"{cls}\"> ({label})")
        summ = main.find("section", "summary")
        if summ is not None and summ.find("p", "next") is None:
            rep.error(rel, summ.line, "section.summary must end with <p class=\"next\">다음 절에서는 …</p> (§3.2)")
        self.check_numbering(page, n, m)

    def check_numbering(self, page, n, m):
        rep, rel = self.rep, page.rel
        k_item = k_eq = k_fig = k_ex = 0
        figures = []
        for node in page.dom.iter():
            if node.tag == "#document":
                continue
            # 번호 항목
            if node.has_class("thm"):
                k_item += 1
                want = f"i-{n}-{m}-{k_item}"
                if node.tag != "div":
                    rep.error(rel, node.line, ".thm must be a <div>")
                types = [c for c in node.classes if c in THM_WORD]
                if len(types) != 1:
                    rep.error(rel, node.line, f".thm needs exactly one type class ({', '.join(THM_WORD)})")
                if node.id != want:
                    rep.error(rel, node.line, f"item {k_item} in document order must have id '{want}' (got '{node.id}')")
                head = node.find(cls="thm-head")
                if head is None:
                    rep.error(rel, node.line, "missing <span class=\"thm-head\">")
                elif len(types) == 1:
                    want_head = f"{THM_WORD[types[0]]} {n}.{m}.{k_item}"
                    if head.text() != want_head:
                        rep.error(rel, head.line, f"thm-head must read '{want_head}' (got '{head.text()}')")
                if node.has_class("compat") and "proposition" not in node.classes:
                    rep.warn(rel, node.line, "compat is meant for propositions (호환성 명제, §3.4)")
            elif node.has_class("equation"):
                k_eq += 1
                want = f"eq-{n}-{m}-{k_eq}"
                if node.tag != "div":
                    rep.error(rel, node.line, ".equation must be a <div>")
                if node.id != want:
                    rep.error(rel, node.line, f"equation {k_eq} must have id '{want}' (got '{node.id}')")
                raw_text = "".join(t.data for t in node.iter_text())
                if f"\\tag{{{n}.{m}.{k_eq}}}" not in raw_text:
                    rep.error(rel, node.line, f"equation {want} must contain \\tag{{{n}.{m}.{k_eq}}}")
            elif node.tag == "figure":
                if not node.has_class("dg-figure"):
                    rep.error(rel, node.line, "<figure> must have class dg-figure")
                    continue
                k_fig += 1
                want = f"fig-{n}-{m}-{k_fig}"
                if node.id != want:
                    rep.error(rel, node.line, f"figure {k_fig} must have id '{want}' (got '{node.id}')")
                figures.append(node)
                self.check_figure(page, node, f"그림 {n}.{m}.{k_fig}.", want)
            elif node.has_class("exercise"):
                k_ex += 1
                want = f"ex-{n}-{m}-{k_ex}"
                if node.id != want:
                    rep.error(rel, node.line, f"exercise {k_ex} must have id '{want}' (got '{node.id}')")
                head = node.find(cls="thm-head")
                want_head = f"연습 {n}.{m}.{k_ex}"
                if head is None or head.text() != want_head:
                    rep.error(rel, node.line, f"exercise head must read '{want_head}' "
                                              f"(got '{head.text() if head else None}')")
                lvl = node.get("data-level")
                if lvl not in ("1", "2", "3"):
                    rep.error(rel, node.line, f"exercise data-level must be 1, 2 or 3 (got '{lvl}')")
                else:
                    stars = node.find("span", "level")
                    if stars is None:
                        rep.error(rel, node.line, "exercise needs <span class=\"level\">★…</span>")
                    elif stars.text().count("★") != int(lvl):
                        rep.warn(rel, stars.line, f"level shows {stars.text().count('★')} stars but data-level is {lvl}")
                if not any(d.tag == "details" and d.has_class("solution") for d in node.iter()):
                    rep.error(rel, node.line, "exercise needs <details class=\"solution\"> (완전한 풀이, §14)")
            # 번호 접두사를 다른 요소에 쓰지 않았는가
            i = node.id or ""
            if re.match(r"^(i|eq|fig|ex)-\d", i):
                ok = ((i.startswith("i-") and node.has_class("thm"))
                      or (i.startswith("eq-") and node.has_class("equation"))
                      or (i.startswith("fig-") and node.tag == "figure")
                      or (i.startswith("ex-") and node.has_class("exercise")))
                if not ok:
                    rep.error(rel, node.line, f"id '{i}' uses a numbering prefix on the wrong element (§11.5)")
        refs = {a.get("href") for a in page.dom.find_all("a") if a.get("href")}
        for fig in figures:
            if fig.id and f"#{fig.id}" not in refs:
                rep.error(rel, fig.line, f"figure '{fig.id}' is never referenced in the text (<a href=\"#{fig.id}\">, §12.6)")

    def check_figure(self, page, fig, want_head, want_id):
        rep, rel = self.rep, page.rel
        cap = fig.find("figcaption")
        head = cap.find(cls="fig-head") if cap else None
        if head is None:
            rep.error(rel, fig.line, "figure needs <figcaption><span class=\"fig-head\">그림 N.M.K.</span> …")
        elif head.text() != want_head:
            rep.error(rel, head.line, f"fig-head must read '{want_head}' (got '{head.text()}')")
        imgs = fig.find_all("img")
        frames = fig.find_all("iframe")
        if not imgs and not frames:
            rep.error(rel, fig.line, "figure needs an <img> (static figure is always required)")
        for img in imgs:
            if not (img.get("alt") or "").strip():
                rep.error(rel, img.line, "img needs a non-empty alt (한국어 요약, §12.6)")
            src = img.get("src") or ""
            res = self.resolve(page, src) if src else None
            if res is None:
                continue
            target = res[0]
            base, ext = os.path.splitext(target)
            if os.path.basename(os.path.dirname(target)) == "figures" and ext.lower() in (".svg", ".png"):
                if not os.path.exists(base + ".py"):
                    rep.error(rel, img.line, f"figure source {os.path.relpath(base, self.root)}.py is missing (§12.1)")
                stem = os.path.basename(base)
                prefix = want_id  # fig-N-M-K
                if not (stem == prefix or stem.startswith(prefix + "-")):
                    rep.warn(rel, img.line, f"figure file '{stem}' should start with '{prefix}-'")


# ---------------------------------------------------------------- 도우미

def _text_ancestors(t):
    p = t.parent
    while p is not None:
        yield p
        p = p.parent


def _text_inside(t, tags):
    return any(a.tag in tags for a in _text_ancestors(t))


def _find_nav(page, cls):
    for n in page.dom.iter():
        if n.tag == "nav" and n.has_class(cls):
            return n
    return None


def _header_h1(main):
    header = main.find("header")
    if header is None:
        return None
    return header.find("h1")


def main(argv=None):
    ap = argparse.ArgumentParser(description="DG-study HTML page checker (GUIDELINES.md §11)")
    ap.add_argument("--root", default=repo_root(), help="site root (default: parent of tools/)")
    ap.add_argument("--no-warn", action="store_true", help="do not print WARN lines")
    ap.add_argument("files", nargs="*", help="check only these files")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    files = [os.path.abspath(f) for f in args.files] if args.files else site_pages(root)

    rep = Reporter()
    checker = SiteChecker(root, rep)
    if checker.allowed_classes is None:
        rep.warn("assets/style.css", 0, "style.css not found; class contract not checked")
    for f in files:
        if not os.path.exists(f):
            rep.error(os.path.relpath(f, root), 0, "file does not exist")
            continue
        checker.check(f)

    for rel, line, msg in sorted(rep.errors):
        print(f"ERROR {rel}:{line}: {msg}")
    if not args.no_warn:
        for rel, line, msg in sorted(rep.warnings):
            print(f"WARN {rel}:{line}: {msg}")
    for f in files:
        p = _page_cache.get(os.path.abspath(f))
        if p is not None and (getattr(p, "todo_count", 0) or getattr(p, "review_count", 0)):
            print(f"INFO {p.rel}: todo={p.todo_count} REVIEW={p.review_count}")
    print(f"check_site: {checker.pages_checked} pages, {len(rep.errors)} errors, "
          f"{len(rep.warnings)} warnings, todo {checker.todo_total}, REVIEW {checker.review_total}")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
