"""Evaluation: retrieval quality (BM25 vs vector vs hybrid) and answer quality (RAG vs LLM-only).

Retrieval is scored at the *document* level against hand-labelled relevant articles; answers are scored by whether the
facts a correct answer must contain appear in the text, and (for unanswerable questions) by whether the system abstains.
Fact matching is a transparent substring test, not a judgement by another model, so it is crude but reproducible.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from importlib import resources
from typing import Dict, List, Sequence

from .retrievers import HybridRetriever, ChunkIndex, MODES

ABSTAIN_PATTERNS = re.compile(
    r"i don'?t know|do not know|not (?:contain|mention|provided|in the (?:provided )?(?:context|articles))|no information|"
    r"cannot (?:answer|find|provide)|can'?t (?:answer|find|provide)|not aware|unable to|i'?m not sure|i do not have",
    re.IGNORECASE)


def load_eval_queries() -> List[dict]:
    with resources.files("ir_rag.resources").joinpath("eval_queries.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def abstains(answer: str) -> bool:
    """Heuristic: does the answer say that the information is not available / not known?"""
    return bool(ABSTAIN_PATTERNS.search(answer))


# --------------------------------------------------------------------------- retrieval metrics
def _doc_ranking(retriever: HybridRetriever, query: str) -> List[str]:
    return [d.metadata["doc_id"] for d in retriever.retrieve(query)]


def retrieval_metrics(index: ChunkIndex, queries: Sequence[dict], ks: Sequence[int] = (1, 3, 5)) -> Dict[str, dict]:
    """Per mode and query type: Hit@k, Recall@k, MRR and nDCG@5, over the answerable queries.

    Documents (not chunks) are ranked: ``max_per_doc=1`` makes the retriever return distinct articles, so the ranking
    is directly comparable with the article-level relevance labels.
    """
    kmax = max(max(ks), 5)
    answerable = [q for q in queries if q["relevant"]]
    results: Dict[str, dict] = {}
    for mode in MODES:
        retriever = HybridRetriever(index=index, mode=mode, k=kmax, pool=max(20, kmax * 3), max_per_doc=1)
        rows = defaultdict(list)
        for q in answerable:
            rel = set(q["relevant"])
            ranking = _doc_ranking(retriever, q["query"])
            first = next((i for i, d in enumerate(ranking, start=1) if d in rel), None)
            row = {"mrr": 1.0 / first if first else 0.0}
            for k in ks:
                top = set(ranking[:k])
                row[f"hit@{k}"] = 1.0 if top & rel else 0.0
                row[f"recall@{k}"] = len(top & rel) / len(rel)
            dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:5], start=1) if d in rel)
            ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), 5) + 1))
            row["ndcg@5"] = dcg / ideal
            for group in ("all", q["type"]):
                rows[group].append(row)
        results[mode] = {g: {m: sum(r[m] for r in rs) / len(rs) for m in rs[0]} | {"queries": len(rs)} for g, rs in rows.items()}
    return results


def hybrid_weight_sweep(index: ChunkIndex, queries: Sequence[dict], bm25_weights: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0)) -> List[dict]:
    """Sensitivity of hybrid retrieval to the BM25 weight in the fusion (vector weight fixed at 1)."""
    answerable = [q for q in queries if q["relevant"]]
    out = []
    for w in bm25_weights:
        r = HybridRetriever(index=index, mode="hybrid", k=5, pool=30, max_per_doc=1, weight_bm25=w)
        hit1 = mrr = 0.0
        for q in answerable:
            rel = set(q["relevant"])
            ranking = _doc_ranking(r, q["query"])
            first = next((i for i, d in enumerate(ranking, start=1) if d in rel), None)
            hit1 += 1.0 if first == 1 else 0.0
            mrr += 1.0 / first if first else 0.0
        out.append({"weight_bm25": w, "hit@1": hit1 / len(answerable), "mrr": mrr / len(answerable)})
    return out


# --------------------------------------------------------------------------- abstention threshold
def top_vector_scores(index: ChunkIndex, queries: Sequence[dict]) -> List[dict]:
    """Best cosine similarity of any chunk to each query, with the query's type (answerable or not)."""
    out = []
    for q in queries:
        qvec = index.embed_query(q["query"])
        (_, score), = index.vector_ranking(qvec, 1)
        out.append({"id": q["id"], "type": q["type"], "answerable": bool(q["relevant"]), "top_vector_score": score})
    return out


def calibrate_threshold(scores: Sequence[dict]) -> dict:
    """Pick the abstention threshold that best separates answerable from unanswerable queries.

    Every candidate threshold t (a score value) is evaluated as "abstain if top score < t"; accuracy is the fraction of
    queries handled correctly. The midpoint of the best-scoring run of thresholds is returned.
    """
    values = sorted({s["top_vector_score"] for s in scores})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [1.0]
    best, best_acc = [], -1.0
    for t in candidates:
        correct = sum((s["top_vector_score"] < t) == (not s["answerable"]) for s in scores)
        acc = correct / len(scores)
        if acc > best_acc + 1e-12:
            best, best_acc = [t], acc
        elif abs(acc - best_acc) <= 1e-12:
            best.append(t)
    return {"threshold": best[len(best) // 2], "accuracy": best_acc,
            "min_answerable": min(s["top_vector_score"] for s in scores if s["answerable"]),
            "max_unanswerable": max((s["top_vector_score"] for s in scores if not s["answerable"]), default=None)}


# --------------------------------------------------------------------------- answer metrics
def fact_recall(answer: str, facts: Sequence[Sequence[str]]) -> float:
    """Fraction of required facts found in the answer. Each fact is a list of acceptable variants (case-insensitive)."""
    if not facts:
        return 1.0
    low = answer.lower()
    return sum(any(v.lower() in low for v in variants) for variants in facts) / len(facts)


def evaluate_answers(pipeline, queries: Sequence[dict], log=print) -> dict:
    """Run every query through RAG and through the LLM alone and aggregate simple, reproducible metrics."""
    per_query = []
    for q in queries:
        rag = pipeline.ask(q["query"])
        llm = pipeline.ask_llm_only(q["query"])
        row = {"id": q["id"], "type": q["type"], "query": q["query"], "rag_answer": rag.answer, "llm_answer": llm.answer,
               "rag_abstained": rag.abstained or abstains(rag.answer), "llm_abstained": abstains(llm.answer),
               "rag_seconds": rag.seconds, "llm_seconds": llm.seconds,
               "sources": [d.metadata["doc_id"] for d in rag.sources]}
        if q["relevant"]:
            row["rag_fact_recall"] = fact_recall(rag.answer, q["facts"])
            row["llm_fact_recall"] = fact_recall(llm.answer, q["facts"])
            row["source_hit"] = bool(set(row["sources"]) & set(q["relevant"]))
        per_query.append(row)
        log(f"  {q['id']} [{q['type']}] rag={'abstain' if row['rag_abstained'] else 'answer'} ({rag.seconds:.1f}s) llm-only={llm.seconds:.1f}s")
    ans = [r for r in per_query if "rag_fact_recall" in r]
    una = [r for r in per_query if "rag_fact_recall" not in r]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    summary = {
        "answerable_queries": len(ans), "unanswerable_queries": len(una),
        "rag_fact_recall": mean([r["rag_fact_recall"] for r in ans]),
        "llm_only_fact_recall": mean([r["llm_fact_recall"] for r in ans]),
        "rag_source_hit_rate": mean([1.0 if r["source_hit"] else 0.0 for r in ans]),
        "rag_false_abstention_rate": mean([1.0 if r["rag_abstained"] else 0.0 for r in ans]),
        "rag_abstains_on_unanswerable": mean([1.0 if r["rag_abstained"] else 0.0 for r in una]),
        "llm_only_abstains_on_unanswerable": mean([1.0 if r["llm_abstained"] else 0.0 for r in una]),
        "rag_mean_seconds": mean([r["rag_seconds"] for r in per_query]),
        "llm_only_mean_seconds": mean([r["llm_seconds"] for r in per_query]),
    }
    return {"summary": summary, "per_query": per_query}
