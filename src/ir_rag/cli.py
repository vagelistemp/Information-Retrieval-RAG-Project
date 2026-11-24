"""Command line interface:  python -m ir_rag <command>

    index                     build (or reuse) and save the chunk index
    ask "question"            answer one question with RAG (use --compare to also show the LLM-only answer)
    evaluate retrieval        BM25 vs vector vs hybrid retrieval metrics (no LLM needed)
    evaluate rag              RAG vs LLM-only answers (needs a running Ollama server)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chain import DEFAULT_LLM, RAGPipeline, check_ollama, make_llm
from .data import load_cnn_csv, load_sample
from .embeddings import DEFAULT_MODEL, HashingEmbeddings, TransformerEmbeddings
from .evaluate import (calibrate_threshold, evaluate_answers, hybrid_weight_sweep, load_eval_queries, retrieval_metrics,
                       top_vector_scores)
from .retrievers import MODES, ChunkIndex, HybridRetriever


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", default="sample", help="'sample' (bundled fictional articles) or the path of a CNN-articles CSV")
    p.add_argument("--limit", type=int, default=None, help="use only the first N articles of a CSV")
    p.add_argument("--embedding", choices=["transformer", "hashing"], default="transformer",
                   help="'transformer' = MiniLM encoder (downloads ~90 MB once); 'hashing' = offline toy embedding")
    p.add_argument("--embedding-model", default=DEFAULT_MODEL)
    p.add_argument("--index-dir", default=None, help="where to save/reuse the index (default: .index/<corpus>)")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)


def _build_index(args) -> ChunkIndex:
    docs = load_sample() if args.corpus == "sample" else load_cnn_csv(args.corpus, args.limit)
    emb = TransformerEmbeddings(args.embedding_model) if args.embedding == "transformer" else HashingEmbeddings()
    default_dir = Path(".index") / (("sample" if args.corpus == "sample" else Path(args.corpus).stem) + f"-{args.embedding}")
    directory = args.index_dir or default_dir
    print(f"{len(docs)} articles loaded")
    return ChunkIndex.load_or_build(docs, emb, directory, args.chunk_size, args.chunk_overlap)


def _print_sources(docs) -> None:
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}] {d.metadata.get('title', '')} ({d.metadata['doc_id']}, cosine {d.metadata['vector_score']:.2f})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ir_rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build or reuse the index")
    _add_common(p_index)

    p_ask = sub.add_parser("ask", help="ask one question")
    _add_common(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=MODES, default="vector", help="retrieval mode (vector was best on the bundled evaluation)")
    p_ask.add_argument("-k", type=int, default=4, help="number of chunks given to the model")
    p_ask.add_argument("--model", default=DEFAULT_LLM, help="Ollama model name")
    p_ask.add_argument("--min-score", type=float, default=0.25, help="abstain if the best chunk's cosine similarity is below this")
    p_ask.add_argument("--compare", action="store_true", help="also show the answer of the LLM alone")

    p_eval = sub.add_parser("evaluate", help="evaluate retrieval or answers")
    p_eval.add_argument("what", choices=["retrieval", "rag"])
    _add_common(p_eval)
    p_eval.add_argument("--model", default=DEFAULT_LLM)
    p_eval.add_argument("--mode", choices=MODES, default="vector", help="retrieval mode used for the RAG answers")
    p_eval.add_argument("--min-score", type=float, default=None, help="abstention threshold (default: calibrated on the queries)")
    p_eval.add_argument("--out", default=None, help="write the results as JSON to this file")

    args = ap.parse_args(argv)

    if args.command == "index":
        idx = _build_index(args)
        print(f"index ready: {idx.manifest.n_chunks} chunks, model {idx.manifest.embedding_model}")
        return 0

    if args.command == "ask":
        problem = check_ollama(args.model)
        if problem:
            print("error:", problem, file=sys.stderr)
            return 2
        idx = _build_index(args)
        pipe = RAGPipeline(HybridRetriever(index=idx, mode=args.mode, k=args.k), make_llm(args.model), args.min_score)
        res = pipe.ask(args.question)
        print(f"\nRAG answer ({res.seconds:.1f}s):\n{res.answer}\n\nSources:")
        _print_sources(res.sources)
        if args.compare:
            base = pipe.ask_llm_only(args.question)
            print(f"\nLLM-only answer ({base.seconds:.1f}s):\n{base.answer}")
        return 0

    # evaluate
    queries = load_eval_queries()
    idx = _build_index(args)
    if args.what == "retrieval":
        metrics = retrieval_metrics(idx, queries)
        scores = top_vector_scores(idx, queries)
        report = {"retrieval": metrics, "hybrid_weight_sweep": hybrid_weight_sweep(idx, queries),
                  "abstention": calibrate_threshold(scores), "top_vector_scores": scores}
        for mode, groups in metrics.items():
            a = groups["all"]
            print(f"{mode:7s} hit@1 {a['hit@1']:.2f}  hit@3 {a['hit@3']:.2f}  recall@5 {a['recall@5']:.2f}  MRR {a['mrr']:.2f}  nDCG@5 {a['ndcg@5']:.2f}")
        print("hybrid weight sweep:", [(s["weight_bm25"], round(s["hit@1"], 2), round(s["mrr"], 2)) for s in report["hybrid_weight_sweep"]])
        print("abstention:", report["abstention"])
    else:
        problem = check_ollama(args.model)
        if problem:
            print("error:", problem, file=sys.stderr)
            return 2
        threshold = args.min_score
        if threshold is None:
            threshold = calibrate_threshold(top_vector_scores(idx, queries))["threshold"]
        print(f"abstention threshold: {threshold:.3f}")
        pipe = RAGPipeline(HybridRetriever(index=idx, mode=args.mode, k=4), make_llm(args.model), threshold)
        report = evaluate_answers(pipe, queries)
        report["threshold"] = threshold
        report["model"] = args.model
        report["retrieval_mode"] = args.mode
        print(json.dumps(report["summary"], indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
