"""Run the code printed in a page from its verify script (GUIDELINES §2.5, Implementation box).

Track 2 pages print Python in ``<pre id="code-…"><code>…</code></pre>``. A verify script runs
exactly that text, so the printed code and the checked code cannot drift apart::

    import dgcode
    ns = dgcode.run_block(PAGE, "code-36-4-1")      # exec in a fresh namespace
    check("Implementation 36.4: d1 d0 = 0", ns["ok"])

Entities (&lt; &gt; &amp;) are decoded before execution. First written for Chapter 32 (lib/ddg32.py).
"""

import html
import re
from pathlib import Path


def code_block(page, block_id):
    """the text of <pre id="block_id"><code>...</code></pre> in the page (HTML entities decoded)."""
    text = Path(page).read_text(encoding="utf-8")
    m = re.search(r'<pre id="%s"><code>(.*?)</code></pre>' % re.escape(block_id), text, re.S)
    if m is None:
        raise KeyError(f"code block {block_id} not found in {page}")
    return html.unescape(m.group(1))


def run_block(page, block_id, namespace=None):
    """exec the code block in a fresh namespace (or the given one) and return the namespace."""
    ns = {} if namespace is None else namespace
    exec(compile(code_block(page, block_id), f"{Path(page).name}#{block_id}", "exec"), ns)
    return ns


def code_lines(page, block_id):
    """number of lines of the block (the Implementation box allows at most 30)."""
    return len(code_block(page, block_id).strip("\n").splitlines())
