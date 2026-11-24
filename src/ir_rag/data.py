"""Corpus loading: the bundled synthetic sample, or a CNN-articles CSV (Kaggle "cnn-articles-after-basic-cleaning").

Both loaders return LangChain ``Document`` objects with at least ``doc_id`` and ``title`` in their metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

# CNN articles can be longer than Python's default csv field limit (131072 characters).
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CNN_TEXT_COLUMN = "Article text"
CNN_METADATA_COLUMNS = ("Headline", "Author", "Date published", "Category", "Section", "Url", "Description")


def load_sample() -> List[Document]:
    """The 40 short fictional news articles shipped with the package (see ``docs`` for how they were built)."""
    with resources.files("ir_rag.resources").joinpath("sample_corpus.jsonl").open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [
        Document(page_content=r["text"], metadata={"doc_id": r["id"], "title": r["title"], "section": r["section"], "date": r["date"]})
        for r in rows
    ]


def load_cnn_csv(path: str | Path, limit: Optional[int] = None) -> List[Document]:
    """Load a CNN-articles CSV. Rows with an empty article text are skipped (and counted in the returned metadata).

    ``doc_id`` is ``cnn-<Index>`` (the CSV's own index column when present, otherwise the row number), so ids stay
    stable between runs. Only a small, useful subset of the columns is kept as metadata.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    docs: List[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or CNN_TEXT_COLUMN not in reader.fieldnames:
            raise ValueError(f"column {CNN_TEXT_COLUMN!r} not found; columns are {reader.fieldnames}")
        for row_number, row in enumerate(reader):
            text = (row.get(CNN_TEXT_COLUMN) or "").strip()
            if not text:
                continue
            meta = {k.lower().replace(" ", "_"): row[k] for k in CNN_METADATA_COLUMNS if k in row}
            meta["doc_id"] = f"cnn-{row.get('Index') or row_number}"
            meta["title"] = row.get("Headline") or meta["doc_id"]
            docs.append(Document(page_content=text, metadata=meta))
            if limit is not None and len(docs) >= limit:
                break
    return docs


def corpus_fingerprint(docs: Iterable[Document]) -> str:
    """Short hash of the corpus content; stored with a saved index so a stale index is detected."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.metadata.get("doc_id", "").encode())
        h.update(b"\0")
        h.update(d.page_content.encode())
        h.update(b"\1")
    return h.hexdigest()[:16]
