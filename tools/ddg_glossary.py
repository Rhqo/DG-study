"""Regenerate the Track 2 section of glossary.html (#sec-glossary-ddg) from the <dfn>s in chapters 32–45.

    python3 tools/ddg_glossary.py          rewrite the section in glossary.html
    python3 tools/ddg_glossary.py --print  print the rows only

Each <dfn>English</dfn>(한국어) in a written Track 2 section page becomes one row:
English | 한국어 | note (from tools/terms.tsv, if any) | first place it appears (the nearest preceding id).
Rows are sorted case-insensitively by the English term. The integrator runs this after each chapter is
integrated (PLAN.md §13). Terms defined twice would already fail build_index, so the first one wins here.
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = range(32, 46)
DFN_RE = re.compile(r"<dfn(?:\s[^>]*)?>(.*?)</dfn>\s*\(([^)]*)\)", re.S)
ID_RE = re.compile(r'\sid="([^"]+)"')
HEAD_RE = re.compile(r'<span class="thm-head">([^<]+)</span>')
SECTION_FILE_RE = re.compile(r"^(\d+)-(\d+)-.*\.html$")


def strip_tags(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def terms_notes():
    notes = {}
    for line in (ROOT / "tools/terms.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip():
            notes.setdefault(parts[1].strip().lower(), parts[2].strip())
    return notes


THM_RE = re.compile(r'<div class="thm [^"]*" id="(i-[^"]+)"')


def enclosing_item(text, pos):
    """id of the numbered item (div.thm) that is still open at pos, or ("", 0)."""
    starts = [m for m in THM_RE.finditer(text, 0, pos)]
    if not starts:
        return "", 0
    m = starts[-1]
    seg = text[m.start():pos]
    depth = len(re.findall(r"<div\b", seg)) - len(re.findall(r"</div>", seg))
    return (m.group(1), m.end()) if depth > 0 else ("", 0)


def label_for(text, pos, anchor, n, m):
    """'Definition 33.1.9' for an item id, 'Section 33.1' otherwise."""
    if anchor.startswith("i-"):
        h = HEAD_RE.search(text, pos)
        if h:
            return h.group(1).strip()
    if anchor.startswith("ex-"):
        return "Exercise " + anchor[3:].replace("-", ".")
    return f"Section {n}.{m}"


def rows():
    notes = terms_notes()
    seen, out = set(), []
    for ch in CHAPTERS:
        for folder in sorted(ROOT.glob(f"chapters/{ch}-*")):
            for page in sorted(folder.glob("*.html")):
                mm = SECTION_FILE_RE.match(page.name)
                if not mm:
                    continue
                text = page.read_text(encoding="utf-8")
                st = re.search(r'<meta name="dg-status" content="([^"]+)"', text)
                if not st or st.group(1) == "planned":
                    continue
                for d in DFN_RE.finditer(text):
                    en, ko = strip_tags(d.group(1)), strip_tags(d.group(2))
                    if not re.search(r"[가-힣]", ko) or en.lower() in seen:
                        continue
                    seen.add(en.lower())
                    anchor, apos = enclosing_item(text, d.start())
                    if not anchor:
                        ids = list(ID_RE.finditer(text, 0, d.start()))
                        anchor, apos = (ids[-1].group(1), ids[-1].end()) if ids else ("", 0)
                    lab = label_for(text, apos, anchor, *mm.groups())
                    href = f"chapters/{folder.name}/{page.name}" + (f"#{anchor}" if anchor else "")
                    out.append((en, ko, notes.get(en.lower(), ""), href, lab))
    out.sort(key=lambda r: r[0].lower())
    return out


def render(rs):
    lines = [
        '    <section id="sec-glossary-ddg">',
        "      <h2>Track 2: Discrete Differential Geometry</h2>",
        "      <p>Track 2(Chapter 32–45)가 정의하는 용어다. 각 절의 <code>&lt;dfn&gt;</code>에서 "
        "<code>tools/ddg_glossary.py</code>가 만든다. Track 3이 정의하는 smooth 개념은 그 장의 표에 있다.</p>",
        "      <table>",
        "        <thead>",
        "          <tr><th>English</th><th>한국어</th><th>Note</th><th>First appears</th></tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for en, ko, note, href, lab in rs:
        lines.append(f'          <tr><td lang="en">{html.escape(en)}</td><td>{html.escape(ko)}</td>'
                     f'<td>{html.escape(note)}</td><td><a href="{href}">{html.escape(lab)}</a></td></tr>')
    lines += ["        </tbody>", "      </table>", "    </section>"]
    return "\n".join(lines) + "\n"


def main():
    rs = rows()
    if "--print" in sys.argv:
        print(render(rs))
        return
    p = ROOT / "glossary.html"
    s = p.read_text(encoding="utf-8")
    block = render(rs)
    m = re.search(r'    <section id="sec-glossary-ddg">.*?</section>\n', s, re.S)
    if m:
        s = s[:m.start()] + block + s[m.end():]
    else:
        anchor = s.index('    <section id="sec-glossary-3d-vision">')
        end = s.index("</section>\n", anchor) + len("</section>\n")
        s = s[:end] + "\n" + block + s[end:]
    p.write_text(s, encoding="utf-8")
    print(f"ddg_glossary: {len(rs)} rows")


if __name__ == "__main__":
    main()
