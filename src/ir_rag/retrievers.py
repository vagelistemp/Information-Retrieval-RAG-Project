"""Chunk index (FAISS vectors + BM25) and a LangChain retriever with three modes: bm25, vector, hybrid.

* **vector**: dense retrieval. Chunks and the query are embedded with a Transformer encoder and compared by cosine
  similarity (inner product of unit vectors) in a FAISS flat index.
* **bm25**: classic lexical ranking over the same chunks.
* **hybrid**: both rankings are combined with *reciprocal rank fusion* (RRF): ``score(d) = sum_r w_r / (k + rank_r(d))``,
  which needs no score calibration between a similarity in [-1, 1] and an unbounded BM25 score.

Every returned chunk carries its ``vector_score`` (cosine similarity to the query), which the RAG chain uses to refuse
questions the corpus cannot answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .bm25 import BM25Index
from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from .data import corpus_fingerprint

MODES = ("bm25", "vector", "hybrid")


@dataclass
class Manifest:
    """Describes how an index was built; a saved index is only reused when all of it matches."""
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    corpus_fingerprint: str
    n_chunks: int


class ChunkIndex:
    def __init__(self, chunks: List[Document], store: FAISS, manifest: Manifest):
        self.chunks, self.store, self.manifest = chunks, store, manifest
        self.embeddings = store.embedding_function
        self.matrix: np.ndarray = np.asarray(store.index.reconstruct_n(0, store.index.ntotal), dtype=np.float32)
        self.bm25 = BM25Index([c.page_content for c in chunks])
        self._pos: Dict[str, int] = {c.metadata["chunk_id"]: i for i, c in enumerate(chunks)}

    # ------------------------------------------------------------------ build / save / load
    @classmethod
    def build(cls, docs: List[Document], embeddings: Embeddings, chunk_size: int = DEFAULT_CHUNK_SIZE,
              chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> "ChunkIndex":
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        if not chunks:
            raise ValueError("no chunks were produced; is the corpus empty?")
        store = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        manifest = Manifest(getattr(embeddings, "model_name", type(embeddings).__name__), chunk_size, chunk_overlap,
                            corpus_fingerprint(docs), len(chunks))
        return cls(chunks, store, manifest)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(directory))
        (directory / "manifest.json").write_text(json.dumps(self.manifest.__dict__, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embeddings: Embeddings) -> "ChunkIndex":
        """Load an index written by :meth:`save`.

        FAISS indexes in LangChain are stored with pickle for the document store, so ``allow_dangerous_deserialization``
        is required. Only load indexes that *you* created; never load one from an untrusted source.
        """
        directory = Path(directory)
        manifest = Manifest(**json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
        store = FAISS.load_local(str(directory), embeddings, allow_dangerous_deserialization=True,
                                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        chunks = [store.docstore.search(store.index_to_docstore_id[i]) for i in range(store.index.ntotal)]
        return cls(chunks, store, manifest)

    @classmethod
    def load_or_build(cls, docs: List[Document], embeddings: Embeddings, directory: Optional[str | Path] = None,
                      chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                      log=print) -> "ChunkIndex":
        """Reuse the saved index only if it was built from this corpus, model and chunking; otherwise rebuild it.

        The original project loaded whatever index was on disk, so changing the CSV, the embedding model or the chunk
        size silently kept serving stale vectors.
        """
        wanted = Manifest(getattr(embeddings, "model_name", type(embeddings).__name__), chunk_size, chunk_overlap,
                          corpus_fingerprint(docs), -1)
        if directory and (Path(directory) / "manifest.json").is_file():
            try:
                idx = cls.load(directory, embeddings)
                m = idx.manifest
                if (m.embedding_model, m.chunk_size, m.chunk_overlap, m.corpus_fingerprint) == (
                        wanted.embedding_model, wanted.chunk_size, wanted.chunk_overlap, wanted.corpus_fingerprint):
                    log(f"loaded index from {directory} ({m.n_chunks} chunks)")
                    return idx
                log(f"saved index in {directory} does not match the corpus/model/chunking; rebuilding")
            except Exception as exc:   # corrupt or incompatible index: fall back to a rebuild
                log(f"could not load index ({exc}); rebuilding")
        idx = cls.build(docs, embeddings, chunk_size, chunk_overlap)
        if directory:
            idx.save(directory)
            log(f"built and saved index in {directory} ({idx.manifest.n_chunks} chunks)")
        return idx

    # ------------------------------------------------------------------ rankings
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self.embeddings.embed_query(query), dtype=np.float32)

    def vector_scores(self, qvec: np.ndarray, indices: Sequence[int]) -> List[float]:
        return [float(self.matrix[i] @ qvec) for i in indices]

    def vector_ranking(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        pairs = self.store.similarity_search_with_score_by_vector(qvec.tolist(), k=k)
        return [(self._pos[d.metadata["chunk_id"]], float(score)) for d, score in pairs]

    def bm25_ranking(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self.bm25.search(query, k)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], weights: Sequence[float], k: int = 60) -> List[Tuple[int, float]]:
    """Fuse several best-first rankings (lists of ids): ``score(id) = sum_r weight_r / (k + rank_r(id))``, ranks from 1."""
    scores: Dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


class HybridRetriever(BaseRetriever):
    """LangChain retriever over a :class:`ChunkIndex`.

    ``k`` chunks are returned, at most ``max_per_doc`` of them from the same document (so one long article cannot fill
    the whole context window with near-duplicate chunks). ``pool`` candidates per ranker feed the fusion.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: ChunkIndex
    mode: str = "hybrid"
    k: int = 4
    pool: int = 20
    max_per_doc: int = 2
    rrf_k: int = 60
    weight_vector: float = 1.0
    weight_bm25: float = 1.0

    def _ranked(self, query: str, qvec: np.ndarray) -> List[int]:
        if self.mode == "vector":
            return [i for i, _ in self.index.vector_ranking(qvec, self.pool)]
        if self.mode == "bm25":
            return [i for i, _ in self.index.bm25_ranking(query, self.pool)]
        if self.mode == "hybrid":
            v = [i for i, _ in self.index.vector_ranking(qvec, self.pool)]
            b = [i for i, _ in self.index.bm25_ranking(query, self.pool)]
            return [i for i, _ in reciprocal_rank_fusion([v, b], [self.weight_vector, self.weight_bm25], self.rrf_k)]
        raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    def retrieve(self, query: str) -> List[Document]:
        """Return the top chunks, each annotated with ``rank`` and ``vector_score`` in its metadata."""
        qvec = self.index.embed_query(query)
        chosen: List[int] = []
        per_doc: Dict[str, int] = {}
        for i in self._ranked(query, qvec):
            doc_id = self.index.chunks[i].metadata["doc_id"]
            if per_doc.get(doc_id, 0) >= self.max_per_doc:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            chosen.append(i)
            if len(chosen) == self.k:
                break
        scores = self.index.vector_scores(qvec, chosen)
        out = []
        for rank, (i, s) in enumerate(zip(chosen, scores), start=1):
            c = self.index.chunks[i]
            out.append(Document(page_content=c.page_content, metadata={**c.metadata, "rank": rank, "vector_score": s}))
        return out

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.retrieve(query)
