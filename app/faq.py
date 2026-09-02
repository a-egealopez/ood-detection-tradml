"""Frequently Asked Questions (FAQ) for the app.

One plain-language entry per concept: question, an intuitive answer, and the
curated references that back it (reused from ``references.py`` as a single source
of papers / DOIs). Each answer points to the code that implements the idea, so a
curious visitor can go from "how does it work?" to the actual source file.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from references import Resource


# ----------------------------------------------------------------------------
# FAQ entry model
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FaqEntry:
    """One FAQ item: question, plain answer, and clicking sources.

    ``references`` is a ``(Resource, note)`` list where ``note`` is the short
    "[n]" citation label shown in the UI.
    """

    question: str
    answer_md: str
    references: tuple[Resource, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------
# The FAQ registry (start with one entry to validate the UX before expanding)
# ----------------------------------------------------------------------------
def _synthetic_generation_refs() -> tuple[Resource, ...]:
    """Curated sources that back the synthetic-house-generation answer."""
    from references import RESOURCES

    titles = {
        "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition",
        "Anomaly Detection for Discrete Sequences: A Survey",
    }
    return tuple(res for res in RESOURCES if res.title in titles)


def build_faq() -> tuple[FaqEntry, ...]:
    """Return the FAQ entries in display order."""
    return (
        FaqEntry(
            question="How is a synthetic house generated?",
            answer_md=(
                "The synthetic CASAS houses are drawn by a **first-order Markov model** "
                "of a resident moving room to room. The movement graph is asymmetric "
                "(a directed cycle Bedroom -> Kitchen -> LivingRoom, with near-zero "
                "backward edges) and depends on the hour band, so normal days follow a "
                "consistent routine. A **sticky latent day regime** (a day stays in the "
                "same 'quiet / typical / active' state with probability about 0.75) adds "
                "day-to-day autocorrelation. The asymmetry matters: a reversed day only "
                "looks abnormal because the backward transitions are rare here — on "
                "symmetric data this anomaly would be invisible. See "
                "`src/ingestion/markov_generator.py`."
            ),
            references=_synthetic_generation_refs(),
        ),
    )


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
def _inline_md(text: str) -> str:
    """Minimal inline-Markdown -> HTML for the FAQ content (bold, code, links).

    Only the subset used by the FAQ answers is handled: ``**bold**``, ```backticks```
    and ``[label](url)`` links. Everything else is escaped so untrusted/wonky text
    never injects markup.
    """
    import re

    text = html.escape(text)

    def _link(m):
        label, url = m.group(1), m.group(2)
        return f'<a href="{html.escape(url)}" target="_blank" rel="noopener">{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def render_faq_html() -> str:
    """Render the FAQ as compact, clearly separated question cards.

    Each question is its own bordered card; references are minimal short links
    (the full title is kept as a hover tooltip so the block stays tiny).
    """
    blocks = ['<div class="fq-list">']
    for idx, entry in enumerate(build_faq(), start=1):
        blocks.append("<div class='fq-card'>")
        blocks.append(
            "<div class='fq-q'>"
            f"<span class='fq-num'>{idx}</span>"
            f"<span>{html.escape(entry.question)}</span>"
            "</div>"
        )
        blocks.append(f"<div class='fq-answer'>{_inline_md(entry.answer_md)}</div>")
        if entry.references:
            refs = []
            for n, res in enumerate(entry.references, start=1):
                refs.append(
                    f"<a class='fq-ref' href='{html.escape(res.url)}' "
                    f"target='_blank' rel='noopener' title='{html.escape(res.title)}'>[{n}]</a>"
                )
            blocks.append(
                "<div class='fq-refs'>"
                "<span class='fq-refs-label'>Fuente:</span>"
                + " ".join(refs)
                + "</div>"
            )
        blocks.append("</div>")
    blocks.append("</div>")
    return "".join(blocks)
