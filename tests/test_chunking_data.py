import csv
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ir_rag.chunking import chunk_documents
from ir_rag.data import corpus_fingerprint, load_cnn_csv, load_sample

TEXT = ". ".join(f"Sentence number {i} says something about the topic of the article" for i in range(60)) + "."


class ChunkingTests(unittest.TestCase):
    def test_no_chunk_starts_with_a_stray_full_stop(self):
        chunks = chunk_documents([Document(page_content=TEXT, metadata={"doc_id": "d"})], 300, 60)
        self.assertGreater(len(chunks), 3)
        self.assertFalse([c for c in chunks if c.page_content.startswith(".")])

    def test_original_splitter_configuration_does_start_chunks_with_a_full_stop(self):
        """Regression for the original project: separators=['. ', ...] with the default keep_separator='start'."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, separators=["\n\n", "\n", ". ", " ", "", "\t"])
        pieces = splitter.split_text(TEXT)
        self.assertTrue(any(p.startswith(". ") for p in pieces))

    def test_size_overlap_and_ids(self):
        doc = Document(page_content=TEXT, metadata={"doc_id": "x1", "title": "T"})
        chunks = chunk_documents([doc], 300, 60)
        self.assertTrue(all(len(c.page_content) <= 300 for c in chunks))
        self.assertEqual([c.metadata["chunk_id"] for c in chunks], [f"x1#{i}" for i in range(len(chunks))])
        self.assertTrue(all(c.metadata["title"] == "T" for c in chunks))
        # consecutive chunks share text (overlap), so no information is lost at a boundary
        shared = [set(a.page_content.split()) & set(b.page_content.split()) for a, b in zip(chunks, chunks[1:])]
        self.assertTrue(all(len(s) > 3 for s in shared))

    def test_tiny_chunks_are_dropped_and_short_documents_kept(self):
        short = Document(page_content="A short but complete article body that fits in one chunk easily.", metadata={"doc_id": "s"})
        tiny = Document(page_content="Photo", metadata={"doc_id": "t"})
        chunks = chunk_documents([short, tiny], 300, 60)
        self.assertEqual([c.metadata["doc_id"] for c in chunks], ["s"])

    def test_metadata_is_copied_not_shared(self):
        doc = Document(page_content=TEXT, metadata={"doc_id": "m"})
        chunks = chunk_documents([doc], 300, 60)
        chunks[0].metadata["x"] = 1
        self.assertNotIn("x", doc.metadata)
        self.assertNotIn("x", chunks[1].metadata)


class DataTests(unittest.TestCase):
    def test_sample_corpus(self):
        docs = load_sample()
        self.assertEqual(len(docs), 40)
        ids = [d.metadata["doc_id"] for d in docs]
        self.assertEqual(len(set(ids)), 40)
        self.assertTrue(all(d.page_content.strip() and d.metadata["title"] for d in docs))

    def _write_csv(self, rows, fields):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
        w = csv.DictWriter(tmp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink())
        return tmp.name

    def test_cnn_loader_skips_empty_text_and_handles_huge_fields(self):
        fields = ["Index", "Headline", "Article text", "Section"]
        huge = "word " * 100_000          # 500,000 characters, above the default csv field limit of 131,072
        path = self._write_csv([
            {"Index": "0", "Headline": "First", "Article text": "Some article text.", "Section": "world"},
            {"Index": "1", "Headline": "Empty", "Article text": "   ", "Section": "world"},
            {"Index": "2", "Headline": "Huge", "Article text": huge, "Section": "news"},
        ], fields)
        docs = load_cnn_csv(path)
        self.assertEqual([d.metadata["doc_id"] for d in docs], ["cnn-0", "cnn-2"])
        self.assertEqual(len(docs[1].page_content), len(huge.strip()))
        self.assertEqual(docs[0].metadata["title"], "First")
        self.assertEqual(docs[0].metadata["section"], "world")
        self.assertEqual(len(load_cnn_csv(path, limit=1)), 1)

    def test_cnn_loader_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_cnn_csv("does-not-exist.csv")
        path = self._write_csv([{"a": "1"}], ["a"])
        with self.assertRaises(ValueError):
            load_cnn_csv(path)

    def test_fingerprint_changes_with_content_and_ids(self):
        a = [Document(page_content="x", metadata={"doc_id": "1"})]
        b = [Document(page_content="y", metadata={"doc_id": "1"})]
        c = [Document(page_content="x", metadata={"doc_id": "2"})]
        self.assertEqual(corpus_fingerprint(a), corpus_fingerprint(list(a)))
        self.assertNotEqual(corpus_fingerprint(a), corpus_fingerprint(b))
        self.assertNotEqual(corpus_fingerprint(a), corpus_fingerprint(c))


if __name__ == "__main__":
    unittest.main()
