"""Frequently Asked Questions (FAQ) for the app.

Renders the curated Q&A blocks from ``fyq.txt`` (project root), the single source
of truth for the questions shown in the app's FyQ modal. Each entry keeps the
question, a short technical answer, the code reference (file + function/class) and
the bibliographic reference curated from ``app/references.py``. The ``## ...``
section headers in ``fyq.txt`` become the grouping headings in the UI.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

_FYQ_PATH = Path(__file__).resolve().parent.parent / "fyq.txt"


# ----------------------------------------------------------------------------
# FAQ entry model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FaqEntry:
    """One FAQ item parsed from ``fyq.txt``.

    ``category`` is the ``## ...`` heading that groups questions in the UI;
    ``code`` and ``reference`` are the optional ``Código`` / ``Referencia`` lines.
    """

    category: str
    question: str
    answer_md: str
    code: str = ""
    reference: str = ""


# ----------------------------------------------------------------------------
# fyq.txt parser (single source of truth)
# ----------------------------------------------------------------------------
def _make_entry(data: dict[str, str], category: str) -> FaqEntry:
    """Build a :class:`FaqEntry` from an already-parsed field dict."""
    return FaqEntry(
        category=category,
        question=data.get("question", ""),
        answer_md=data.get("answer_md", ""),
        code=data.get("code", ""),
        reference=data.get("reference", ""),
    )


def _read_entries() -> list[FaqEntry]:
    """Parse ``fyq.txt`` into :class:`FaqEntry` objects in display order.

    Recognized lines: ``##`` (section heading), ``P:`` (question, starts a new
    entry), ``R:`` / ``Código:`` / ``Referencia:`` (fields of the current entry).
    Blank lines and ``#`` comment lines are skipped; any other line continues the
    last field, so multi-line answers survive intact.
    """
    entries: list[FaqEntry] = []
    category = "General"
    current: dict[str, str] | None = None
    field: str = ""

    for raw in _FYQ_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            category = line[3:].strip()
            continue
        if not line or line.startswith("#"):
            continue
        if current is None:
            if not line.startswith("P: "):
                continue
            current = {"question": line[3:].strip()}
            field = "question"
        elif line.startswith("P: "):
            entries.append(_make_entry(current, category))
            current = {"question": line[3:].strip()}
            field = "question"
        elif line.startswith("R: "):
            current["answer_md"] = line[3:].strip()
            field = "answer_md"
        elif line.startswith("Código: "):
            current["code"] = line[len("Código: ") :].strip()
            field = "code"
        elif line.startswith("Referencia: "):
            current["reference"] = line[len("Referencia: ") :].strip()
            field = "reference"
        elif field:
            current[field] = f"{current[field]} {line}"

    if current is not None:
        entries.append(_make_entry(current, category))
    return entries


# ----------------------------------------------------------------------------
# The FAQ registry (single source is ``fyq.txt``)
# ----------------------------------------------------------------------------
def build_faq() -> list[FaqEntry]:
    """Return the FAQ entries in display order (the ``fyq.txt`` file order)."""
    return _read_entries()


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
def _inline_md(text: str) -> str:
    """Minimal inline-Markdown -> HTML for the FAQ content (bold, italic, code, links).

    Only the subset used by the FAQ answers is handled: ``**bold**``, ``*italic*``,
    ```backticks``` and ``[label](url)`` links. Everything else is escaped so
    untrusted/wonky text never injects markup.
    """
    text = html.escape(text)

    def _link(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        return f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*\s][^*]*)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def render_faq_html() -> str:
    """Render the FAQ as compact, clearly separated question cards per section.

    Each ``## ...`` heading in ``fyq.txt`` becomes a section heading; every question
    is its own bordered card with the answer, the code reference and the source.
    """
    blocks = ['<div class="fq-list">']
    previous: str | None = None
    within = 0

    for entry in build_faq():
        if entry.category != previous:
            blocks.append(
                "<div class='fq-group-title'>"
                f"{html.escape(entry.category)}"
                "</div>"
            )
            previous = entry.category
            within = 0
        within += 1

        blocks.append("<div class='fq-card'>")
        blocks.append(
            "<div class='fq-q'>"
            f"<span class='fq-num'>{within}</span>"
            f"<span>{html.escape(entry.question)}</span>"
            "</div>"
        )
        blocks.append(f"<div class='fq-answer'>{_inline_md(entry.answer_md)}</div>")
        if entry.code:
            blocks.append(
                "<div class='fq-refs'>"
                "<span class='fq-refs-label'>Código:</span>"
                f"<span>{_inline_md(entry.code)}</span>"
                "</div>"
            )
        if entry.reference:
            blocks.append(
                "<div class='fq-refs'>"
                "<span class='fq-refs-label'>Fuente:</span>"
                f"<span>{_inline_md(entry.reference)}</span>"
                "</div>"
            )
        blocks.append("</div>")

    blocks.append("</div>")
    return "".join(blocks)
