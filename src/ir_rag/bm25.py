"""A small, dependency-free BM25 (Okapi) ranking function over chunks.

    score(q, d) = sum over query terms t of  idf(t) * f(t,d) * (k1 + 1) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )
    idf(t)      = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

with k1 = 1.5 and b = 0.75 by default. Terms are lower-cased word tokens without stop words and with a light
suffix stemmer, so "strikes" and "striking" both match "strike".
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
STOP_WORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if in into is it its
just me more most my no nor not now of off on once only or other our out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when where which while who whom why will with
would you your""".split())


def stem(word: str) -> str:
    """Very light English suffix stripping (plural, -ing, -ed). Deliberately conservative."""
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("es") and word[-3] in "sxz":
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":   # "stopp" -> "stop"
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):                                   # "strike" and "striking" -> "strik"
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP_WORDS]


class BM25Index:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(t) for t in texts]
        self.n = len(self.doc_tokens)
        self.lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: List[Counter] = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf: Dict[str, float] = {t: math.log(1.0 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # inverted index: term -> list of (doc index, term frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for term, f in c.items():
                self.postings.setdefault(term, []).append((i, f))

    def scores(self, query: str) -> Dict[int, float]:
        """BM25 score of every document that contains at least one query term."""
        out: Dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, f in self.postings[term]:
                norm = f + self.k1 * (1.0 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1.0) / norm
        return out

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Top-k (document index, score), best first; ties are broken by the lower index for determinism."""
        ranked = sorted(self.scores(query).items(), key=lambda p: (-p[1], p[0]))
        return ranked[:k]
