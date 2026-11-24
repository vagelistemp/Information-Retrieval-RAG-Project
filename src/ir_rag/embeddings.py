"""Text embeddings.

``TransformerEmbeddings`` runs a Transformer *encoder* (default: ``sentence-transformers/all-MiniLM-L6-v2``, a 6-layer
MiniLM with 384-dimensional output) directly through Hugging Face ``transformers`` and does the pooling by hand:

    1. tokenize (WordPiece) and run the encoder -> one contextual vector per token;
    2. **mean pooling** over the real tokens (the attention mask excludes padding);
    3. **L2 normalisation**, so that the inner product of two embeddings equals their cosine similarity.

This is exactly what the sentence-transformers library does for this model (a unit test compares the two), but writing
the steps out is the point: it shows what an "embedding model" is.

``HashingEmbeddings`` is a dependency-free, deterministic bag-of-hashed-words embedding used for fast tests and offline
demos. It is *not* a semantic model and is never used for the reported results.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 32, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.device, self.batch_size, self.max_length = model_name, device, batch_size, max_length
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()

    @property
    def dimension(self) -> int:
        return int(self._model.config.hidden_size)

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        torch = self._torch
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state                 # (batch, tokens, hidden)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)        # (batch, tokens, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)   # mean over real tokens
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)         # unit length
            out.extend(pooled.cpu().tolist())
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


class HashingEmbeddings(Embeddings):
    """Feature-hashing bag-of-words embedding (signed hashing trick), L2-normalised. Deterministic, no downloads."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    @property
    def dimension(self) -> int:
        return self.dim

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)
