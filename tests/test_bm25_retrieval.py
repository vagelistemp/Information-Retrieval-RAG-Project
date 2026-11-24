import math
import tempfile
import unittest

from ir_rag.bm25 import BM25Index, stem, tokenize
from ir_rag.data import load_sample
from ir_rag.embeddings import HashingEmbeddings
from ir_rag.retrievers import ChunkIndex, HybridRetriever, reciprocal_rank_fusion


class BM25Tests(unittest.TestCase):
    def test_tokenizer_and_stemmer(self):
        self.assertEqual(tokenize("The strikes were STOPPED, and Striking workers' pay!"), ["strik", "stop", "strik", "worker", "pay"])
        self.assertEqual(stem("strikes"), stem("striking"))
        self.assertEqual(stem("flights"), "flight")
        self.assertEqual(stem("stopped"), "stop")
        self.assertEqual(stem("companies"), "company")
        self.assertEqual(stem("class"), "class")
        self.assertEqual(tokenize("the and of"), [])

    def test_score_matches_the_formula_by_hand(self):
        texts = ["apple banana apple", "banana cherry", "cherry cherry cherry date"]
        idx = BM25Index(texts, k1=1.5, b=0.75)
        n, avgdl = 3, (3 + 2 + 4) / 3
        idf_apple = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
        expected = idf_apple * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl))
        self.assertAlmostEqual(idx.scores("apple")[0], expected)
        self.assertEqual(set(idx.scores("apple")), {0})

    def test_ranking_prefers_rare_terms_and_shorter_documents(self):
        texts = ["rail strike at the port", "the port is busy " * 5 + "strike", "completely unrelated text about cooking"]
        idx = BM25Index(texts)
        ranked = [i for i, _ in idx.search("port strike")]
        self.assertEqual(ranked[0], 0)
        self.assertNotIn(2, ranked)

    def test_unknown_terms_empty_query_and_k(self):
        idx = BM25Index(["alpha beta", "beta gamma"])
        self.assertEqual(idx.search("zzz"), [])
        self.assertEqual(idx.search(""), [])
        self.assertEqual(len(idx.search("beta", k=1)), 1)

    def test_deterministic_tie_break(self):
        idx = BM25Index(["same words here", "same words here"])
        self.assertEqual([i for i, _ in idx.search("same words")], [0, 1])


class RRFTests(unittest.TestCase):
    def test_fusion_scores(self):
        fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], [1.0, 1.0], k=60))
        self.assertAlmostEqual(fused[1], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused[3], 1 / 63 + 1 / 61)
        self.assertAlmostEqual(fused[2], 1 / 62)

    def test_item_in_both_lists_beats_item_in_one(self):
        fused = reciprocal_rank_fusion([[10, 20], [30, 20]], [1, 1])
        self.assertEqual(fused[0][0], 20)

    def test_weights(self):
        fused = reciprocal_rank_fusion([[1], [2]], [1.0, 3.0])
        self.assertEqual(fused[0][0], 2)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_sample()
        cls.emb = HashingEmbeddings(512)
        cls.index = ChunkIndex.build(cls.docs, cls.emb, 600, 100)

    def retriever(self, **kw):
        return HybridRetriever(index=self.index, **kw)

    def test_every_mode_finds_an_article_by_its_own_headline(self):
        for mode in ("bm25", "vector", "hybrid"):
            r = self.retriever(mode=mode, k=3, max_per_doc=1)
            hits = 0
            for d in self.docs:
                top = [c.metadata["doc_id"] for c in r.retrieve(d.metadata["title"])]
                hits += d.metadata["doc_id"] in top
            self.assertGreaterEqual(hits, 36, mode)          # at least 36 of 40 headlines retrieve their article in the top 3

    def test_max_per_doc_and_k_are_respected(self):
        r = self.retriever(mode="hybrid", k=6, max_per_doc=1)
        ids = [c.metadata["doc_id"] for c in r.retrieve("airport flights runway passengers")]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 6)
        r2 = self.retriever(mode="bm25", k=2)
        self.assertLessEqual(len(r2.retrieve("airport")), 2)

    def test_results_carry_rank_and_vector_score(self):
        out = self.retriever(mode="hybrid", k=3).retrieve("central bank interest rate")
        self.assertEqual([c.metadata["rank"] for c in out], [1, 2, 3][: len(out)])
        self.assertTrue(all(-1.0001 <= c.metadata["vector_score"] <= 1.0001 for c in out))

    def test_vector_score_of_identical_text_is_one(self):
        chunk = self.index.chunks[5]
        qvec = self.index.embed_query(chunk.page_content)
        self.assertAlmostEqual(self.index.vector_scores(qvec, [5])[0], 1.0, places=4)

    def test_faiss_search_agrees_with_a_numpy_oracle(self):
        import numpy as np
        for query in ("solar farm output", "airport drone flights", "a vaccine for children"):
            qvec = self.index.embed_query(query)
            exact = np.argsort(-(self.index.matrix @ qvec), kind="stable")[:5]
            got = [i for i, _ in self.index.vector_ranking(qvec, 5)]
            scores = self.index.matrix @ qvec
            self.assertEqual(set(scores[got].round(5)), set(scores[exact].round(5)))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.retriever(mode="nope").retrieve("x")

    def test_langchain_invoke_interface(self):
        docs = self.retriever(mode="hybrid", k=2).invoke("tram line university")
        self.assertTrue(docs and docs[0].metadata["doc_id"])

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.index.save(tmp)
            loaded = ChunkIndex.load(tmp, self.emb)
            self.assertEqual(len(loaded.chunks), len(self.index.chunks))
            a = HybridRetriever(index=self.index, mode="vector", k=3).retrieve("solar farm")
            b = HybridRetriever(index=loaded, mode="vector", k=3).retrieve("solar farm")
            self.assertEqual([c.metadata["chunk_id"] for c in a], [c.metadata["chunk_id"] for c in b])

    def test_load_or_build_reuses_a_matching_index_and_rebuilds_a_stale_one(self):
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            ChunkIndex.load_or_build(self.docs, self.emb, tmp, 600, 100, log=logs.append)
            self.assertIn("built", logs[-1])
            ChunkIndex.load_or_build(self.docs, self.emb, tmp, 600, 100, log=logs.append)
            self.assertIn("loaded", logs[-1])
            ChunkIndex.load_or_build(self.docs, self.emb, tmp, 400, 100, log=logs.append)           # different chunk size
            self.assertIn("rebuilding", logs[-2])
            ChunkIndex.load_or_build(self.docs[:30], self.emb, tmp, 400, 100, log=logs.append)       # different corpus
            self.assertIn("rebuilding", logs[-2])
            ChunkIndex.load_or_build(self.docs[:30], HashingEmbeddings(128), tmp, 400, 100, log=logs.append)   # different model
            self.assertIn("rebuilding", logs[-2])

    def test_empty_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            ChunkIndex.build([], self.emb)


if __name__ == "__main__":
    unittest.main()
