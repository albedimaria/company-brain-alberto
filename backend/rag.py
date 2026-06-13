"""Knowledge base retrieval over backend/data/kb/.

BM25 at WHOLE-DOCUMENT granularity (the spec warns the 35 docs are small and
near-identical; chunking hurts and splits shelf-life from allergens). Lexical
matching also nails code-exact tokens (SKU, DOC ids). Index built lazily once.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

_KB_DIR = Path(__file__).resolve().parent / "data" / "kb"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    # keep SKU/DOC-like tokens intact (PAS-SPA-500, DOC-015, RAW-SEM-003)
    return _TOKEN_RE.findall(text.lower())


@lru_cache
def _index() -> tuple[list[str], list[str], BM25Okapi]:
    doc_ids: list[str] = []
    contents: list[str] = []
    for path in sorted(_KB_DIR.glob("*.md")):
        doc_ids.append(path.stem)  # 'DOC-015'
        contents.append(path.read_text(encoding="utf-8"))
    corpus = [_tokenize(c) for c in contents]
    return doc_ids, contents, BM25Okapi(corpus)


def kb_search(query: str, k: int = 4) -> list[tuple[str, str]]:
    """Return top-k (doc_id, content) by BM25 score for the query."""
    doc_ids, contents, bm25 = _index()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[str, str]] = []
    for i in ranked[:k]:
        if scores[i] <= 0:
            continue
        out.append((doc_ids[i], contents[i]))
    # always return at least the top doc so the model has something to ground on
    if not out and ranked:
        i = ranked[0]
        out.append((doc_ids[i], contents[i]))
    return out


def get_kb_document(doc_id: str) -> str | None:
    doc_ids, contents, _ = _index()
    try:
        return contents[doc_ids.index(doc_id)]
    except ValueError:
        return None
