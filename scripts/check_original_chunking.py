"""Reproduce the chunking defects of the original coursework script on a CNN-articles CSV.

    python scripts/check_original_chunking.py path/to/CNN_Articels_clean.csv [results/original_code_check.json]

The "original" configuration is the one printed in the coursework report (RecursiveCharacterTextSplitter with
chunk_size=1000, chunk_overlap=200 and separators ["\\n\\n", "\\n", ". ", " ", "", "\\t"], default keep_separator) and a plain
csv.DictReader. The script reports how many chunks start with a stray ". ", and compares with this project's chunker.
"""
import csv
import json
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ir_rag.chunking import chunk_documents
from ir_rag.data import load_cnn_csv


def main(csv_path: str, out_path: str | None) -> None:
    t0 = time.time()
    # --- original loading: csv.DictReader with the default field-size limit
    original_error = None
    docs = []
    csv.field_size_limit(131072)                      # Python's default
    try:
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                docs.append(Document(page_content=str(row.get("Article text"))))
    except csv.Error as exc:
        original_error = str(exc)
    # --- project loading
    ours = load_cnn_csv(csv_path)
    texts = [d.page_content for d in ours]
    raw = [Document(page_content=t) for t in texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len,
                                              is_separator_regex=False, separators=["\n\n", "\n", ". ", " ", "", "\t"])
    original_chunks = splitter.split_documents(raw)
    new_chunks = chunk_documents(ours, 1000, 200)
    lead = sum(1 for c in original_chunks if c.page_content.startswith(". "))
    lead_new = sum(1 for c in new_chunks if c.page_content.startswith("."))
    result = {
        "documents": len(ours),
        "empty_articles": sum(1 for d in docs if not d.page_content.strip() or d.page_content == "None"),
        "max_article_chars": max(len(t) for t in texts),
        "original_reader_error": original_error,
        "chunks": len(original_chunks),
        "chunks_starting_with_period": lead,
        "share_chunks_starting_with_period": lead / len(original_chunks),
        "new_chunker_chunks": len(new_chunks),
        "new_chunker_chunks_starting_with_period": lead_new,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=1))
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
