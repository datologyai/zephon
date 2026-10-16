# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Build a page-local contents list out of run-in paragraph lead-ins.

The guide carries its structure in ``**Lead-in.**`` paragraphs rather than
subheadings, so Sphinx sees one section per page and furo, finding nothing to
list, hides the right-hand table of contents entirely. This collects those
paragraphs, gives each one an anchor, and hands furo a contents list built from
them. Pages with real subheadings are left with the contents list Sphinx builds.
"""

from __future__ import annotations

import re
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

# A lead-in is a short run-in sentence; anything longer is ordinary bold prose
# that happens to open a paragraph.
MAX_LEADIN_CHARS = 90
SENTENCE_END = (".", "?", "!", ":")


def _leadin_title(node: nodes.paragraph) -> str | None:
    """The run-in heading this paragraph opens with, if it opens with one."""
    # Only top-level prose: bold also opens list items, notes and table cells,
    # and those are not page structure.
    if not isinstance(node.parent, (nodes.section, nodes.document)):
        return None
    # MyST puts an empty text node ahead of the first inline element.
    children = [
        child
        for child in node.children
        if not (isinstance(child, nodes.Text) and not child.astext().strip())
    ]
    if not children or not isinstance(children[0], nodes.strong):
        return None
    text = children[0].astext().strip()
    if not text or len(text) > MAX_LEADIN_CHARS:
        return None
    # Either it introduces the prose that follows, or it stands alone.
    if len(children) > 1 and not text.endswith(SENTENCE_END):
        return None
    return text


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _unique(slug: str, document: nodes.document) -> str:
    if slug not in document.ids:
        return slug
    suffix = 2
    while f"{slug}-{suffix}" in document.ids:
        suffix += 1
    return f"{slug}-{suffix}"


def _collect(app: Sphinx, doctree: nodes.document) -> None:
    """Anchor every lead-in and remember it for this page's contents list."""
    entries: list[tuple[str, str]] = []
    for paragraph in doctree.findall(nodes.paragraph):
        heading = _leadin_title(paragraph)
        if heading is None:
            continue
        title = heading.rstrip(".")
        # An explicit ``(anchor)=`` label already sits on the paragraph; reuse it
        # so the anchor the guide cross-references stays the canonical one.
        if paragraph["ids"]:
            anchor = paragraph["ids"][0]
        else:
            anchor = _unique(_slug(title), doctree)
            paragraph["ids"].append(anchor)
            doctree.ids[anchor] = paragraph
        entries.append((title, anchor))

    if entries:
        app.env.leadin_toc[app.env.docname] = entries


def _page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    entries = app.env.leadin_toc.get(pagename)
    if not entries:
        return
    # A page with real subheadings already has a contents list worth showing.
    if app.env.toc_num_entries.get(pagename, 0) > 1:
        return

    items = "\n".join(
        f'<li><a class="reference internal" href="#{anchor}">{title}</a></li>'
        for title, anchor in entries
    )
    context["toc"] = (
        f'<ul>\n<li><a class="reference internal" href="#">{context.get("title", "")}</a>'
        f"<ul>\n{items}\n</ul>\n</li>\n</ul>\n"
    )
    context["furo_hide_toc"] = False


def _init(app: Sphinx, env: BuildEnvironment, docnames: list[str]) -> None:
    if not hasattr(env, "leadin_toc"):
        env.leadin_toc = {}


def _purge(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    getattr(env, "leadin_toc", {}).pop(docname, None)


def _merge(
    app: Sphinx,
    env: BuildEnvironment,
    docnames: list[str],
    other: BuildEnvironment,
) -> None:
    env.leadin_toc.update(getattr(other, "leadin_toc", {}))


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("env-before-read-docs", _init)
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    app.connect("doctree-read", _collect)
    # Furo decides `furo_hide_toc` in its own handler; this has to run after it.
    app.connect("html-page-context", _page_context, priority=900)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
