"""Choose the RAG prompt and context size on a *dev set* that is separate from the evaluation questions.

    python scripts/select_prompt.py [results/prompt_selection.json]

Each variant is run with the abstention guard switched off (min_score = 0), so that what is measured is the model's own
behaviour: does it answer when the context contains the answer (fact recall, false refusals), and does it refuse when the
retrieved text is unrelated? The evaluation questions in ir_rag/resources/eval_queries.json are not used here, so the
reported evaluation is not tuned on them. Needs a running Ollama server.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from ir_rag.chain import DEFAULT_LLM, RAG_PROMPTS, RAGPipeline, make_llm
from ir_rag.data import load_sample
from ir_rag.embeddings import TransformerEmbeddings
from ir_rag.evaluate import abstains, fact_recall
from ir_rag.retrievers import ChunkIndex, HybridRetriever

DEV = [  # (question, relevant doc, required facts)
    ("How many people were evacuated when the river Orsa flooded?", "doc-019", [["3,000", "3000"]]),
    ("Which player signed the richest contract in the league and for how much?", "doc-035", [["quill"], ["24 million"]]),
    ("By how much did living near busy roads raise the risk of heart attack?", "doc-024", [["14"]]),
    ("How many passengers did Verne airport handle last year?", "doc-010", [["31 million", "31"]]),
    ("What did the Selene-2 probe carry to search for water ice?", "doc-036", [["drill"]]),
    ("Where was the Roman-era market square found?", "doc-040", [["verne"], ["car park", "old town"]]),
    ("How many species were found near the hydrothermal vents?", "doc-038", [["14"]]),
    ("Who won the national cup and who scored the winner?", "doc-032", [["port calder"], ["varga"]]),
]
DEV_UNANSWERABLE = ["What is the population of Tokyo?", "Who invented the telephone?", "What is the tallest mountain on Earth?"]

SETTINGS = [("strict", 4), ("direct", 4), ("direct", 2), ("direct", 1)]   # (prompt variant, chunks given to the model)


def main(out: str | None) -> None:
    idx = ChunkIndex.load_or_build(load_sample(), TransformerEmbeddings(), ".index/sample-transformer", log=lambda *a: None)
    llm = make_llm(DEFAULT_LLM)
    rows = []
    for name, k in SETTINGS:
        pipe = RAGPipeline(HybridRetriever(index=idx, mode="vector", k=k), llm, min_score=0.0, prompt=RAG_PROMPTS[name])
        recalls, refused = [], 0
        for q, doc, facts in DEV:
            res = pipe.ask(q)
            recalls.append(fact_recall(res.answer, facts))
            refused += abstains(res.answer)
        refused_unans = sum(abstains(pipe.ask(q).answer) for q in DEV_UNANSWERABLE)
        row = {"prompt": name, "k": k, "fact_recall": sum(recalls) / len(recalls), "false_refusals": refused / len(DEV),
               "refuses_unanswerable": refused_unans / len(DEV_UNANSWERABLE)}
        rows.append(row)
        print(row)
    if out:
        Path(out).write_text(json.dumps({"dev_answerable": len(DEV), "dev_unanswerable": len(DEV_UNANSWERABLE), "settings": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
