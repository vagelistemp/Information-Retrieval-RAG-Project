"""Splitting documents into overlapping chunks for embedding and retrieval."""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_documents(docs: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, min_chars: int = 40) -> List[Document]:
    """Split documents into chunks of at most ``chunk_size`` characters with ``chunk_overlap`` characters of overlap.

    Differences from a default splitter configuration:

    * ``keep_separator="end"`` keeps a sentence's full stop with that sentence. With the default
      (``"start"``) the separator ``". "`` is glued to the *beginning* of the next chunk, so chunks start with a stray
      ". " (the original project's retrieved chunks all began with ". ").
    * Sentence separators for ``?`` and ``!`` are included, and the (unreachable) tab separator is dropped.
    * Chunks shorter than ``min_chars`` (for example a lone caption) are discarded.

    Every chunk gets ``chunk_id`` (``<doc_id>#<n>``) and ``chunk_index`` in its metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator="end",
        is_separator_regex=False,
    )
    chunks: List[Document] = []
    for doc in docs:
        n = 0
        for piece in splitter.split_text(doc.page_content):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            meta = dict(doc.metadata)
            meta["chunk_index"] = n
            meta["chunk_id"] = f"{meta.get('doc_id', 'doc')}#{n}"
            chunks.append(Document(page_content=piece, metadata=meta))
            n += 1
    return chunks
