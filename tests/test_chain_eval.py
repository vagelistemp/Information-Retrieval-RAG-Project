import math
import unittest

from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM

from ir_rag.chain import ABSTAIN_ANSWER, RAG_PROMPT, RAG_PROMPTS, RAGPipeline, check_ollama, format_context
from ir_rag.data import load_sample
from ir_rag.embeddings import HashingEmbeddings
from ir_rag.evaluate import (abstains, calibrate_threshold, fact_recall, load_eval_queries, retrieval_metrics,
                             top_vector_scores)
from ir_rag.retrievers import ChunkIndex, HybridRetriever


class EmbeddingTests(unittest.TestCase):
    def test_hashing_embeddings_are_deterministic_unit_vectors(self):
        e = HashingEmbeddings(128)
        a, b = e.embed_query("central bank rate"), e.embed_query("central bank rate")
        self.assertEqual(a, b)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in a)), 1.0)
        self.assertEqual(len(e.embed_documents(["x y", "z"])), 2)

    def test_transformer_embeddings_match_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            from ir_rag.embeddings import DEFAULT_MODEL, TransformerEmbeddings
            ours = TransformerEmbeddings(DEFAULT_MODEL)
            ref = SentenceTransformer(DEFAULT_MODEL, device="cpu")
        except Exception as exc:     # offline machine or packages missing
            self.skipTest(f"model not available: {exc}")
        texts = ["The central bank cut interest rates.", "A plane crashed near the runway.", "Short."]
        mine = ours.embed_documents(texts)
        theirs = ref.encode(texts, normalize_embeddings=True)
        for m, t in zip(mine, theirs):
            cos = sum(x * float(y) for x, y in zip(m, t))
            self.assertGreater(cos, 0.9999)
        self.assertEqual(ours.dimension, 384)
        # semantic sanity: a paraphrase is closer than an unrelated sentence
        q, para, other = (ours.embed_query(s) for s in ("interest rate decision", "the bank lowered its borrowing cost", "a marathon runner won the race"))
        dot = lambda u, v: sum(x * y for x, y in zip(u, v))
        self.assertGreater(dot(q, para), dot(q, other))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ChunkIndex.build(load_sample(), HashingEmbeddings(512), 600, 100)

    def pipeline(self, responses, min_score, **kw):
        llm = FakeListLLM(responses=responses)
        return RAGPipeline(HybridRetriever(index=self.index, mode="hybrid", k=3, **kw), llm, min_score), llm

    def test_answers_with_sources_when_evidence_is_strong(self):
        pipe, llm = self.pipeline(["The central bank set 3.75 percent [1].", "unused"], min_score=0.0)
        res = pipe.ask("What interest rate did the central bank set?")
        self.assertEqual(res.answer, "The central bank set 3.75 percent [1].")
        self.assertFalse(res.abstained)
        self.assertEqual(llm.i, 1)                                   # the model was called exactly once (FakeListLLM cycles its counter, hence two responses)
        self.assertTrue(any(d.metadata["doc_id"] == "doc-012" for d in res.sources))

    def test_abstains_without_calling_the_model_when_evidence_is_weak(self):
        pipe, llm = self.pipeline(["should never be returned", "nor this"], min_score=0.99)
        res = pipe.ask("What is the recipe for sourdough bread?")
        self.assertEqual(res.answer, ABSTAIN_ANSWER)
        self.assertTrue(res.abstained)
        self.assertEqual(llm.i, 0)                                   # no LLM call: nothing to hallucinate from

    def test_llm_only_baseline_uses_no_retrieval(self):
        pipe, llm = self.pipeline(["plain answer"], min_score=0.0)
        res = pipe.ask_llm_only("anything")
        self.assertEqual((res.answer, res.sources), ("plain answer", []))

    def test_prompt_contains_numbered_context_and_question(self):
        docs = [Document(page_content="alpha text", metadata={"doc_id": "d1", "title": "One"}),
                Document(page_content="beta text", metadata={"doc_id": "d2", "title": "Two"})]
        ctx = format_context(docs)
        self.assertIn("[1] One (d1)\nalpha text", ctx)
        self.assertIn("[2] Two (d2)\nbeta text", ctx)
        rendered = RAG_PROMPT.format(context=ctx, question="Why?")
        self.assertIn("Question: Why?", rendered)
        self.assertIn("Do not add anything that is not in the articles", rendered)
        self.assertIn("cite the article you used like [1]", rendered)

    def test_both_prompt_variants_render_and_default_is_direct(self):
        for name, prompt in RAG_PROMPTS.items():
            out = prompt.format(context="[1] A (d) text", question="Q?")
            self.assertIn("Q?", out, name)
        self.assertIs(RAG_PROMPT, RAG_PROMPTS["direct"])

    def test_check_ollama_reports_an_unreachable_server(self):
        msg = check_ollama("llama3.2:1b", base_url="http://127.0.0.1:9")
        self.assertIsNotNone(msg)
        self.assertIn("Ollama", msg)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ChunkIndex.build(load_sample(), HashingEmbeddings(512), 600, 100)
        cls.queries = load_eval_queries()

    def test_query_file_is_consistent_with_the_corpus(self):
        ids = {d.metadata["doc_id"] for d in load_sample()}
        self.assertEqual(len({q["id"] for q in self.queries}), len(self.queries))
        for q in self.queries:
            self.assertTrue(set(q["relevant"]) <= ids)
            self.assertEqual(bool(q["relevant"]), q["type"] != "unanswerable")
            self.assertEqual(bool(q["facts"]), bool(q["relevant"]))

    def test_every_fact_of_an_answerable_query_occurs_in_its_relevant_articles(self):
        text = {d.metadata["doc_id"]: d.page_content.lower() for d in load_sample()}
        for q in self.queries:
            if not q["relevant"]:
                continue
            body = " ".join(text[i] for i in q["relevant"])
            for variants in q["facts"]:
                self.assertTrue(any(v.lower() in body for v in variants), (q["id"], variants))

    def test_retrieval_metrics_structure_and_ranges(self):
        m = retrieval_metrics(self.index, self.queries)
        self.assertEqual(set(m), {"bm25", "vector", "hybrid"})
        for mode in m.values():
            self.assertEqual(set(mode), {"all", "lexical", "paraphrase", "multi"})
            for v in mode["all"].values():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 20.0)
            self.assertLessEqual(mode["all"]["hit@1"], mode["all"]["hit@3"])
            self.assertLessEqual(mode["all"]["hit@3"], mode["all"]["hit@5"])
            self.assertEqual(mode["all"]["queries"], 20)

    def test_perfect_retrieval_gives_perfect_scores(self):
        queries = [{"id": "t", "type": "lexical", "query": d.metadata["title"], "relevant": [d.metadata["doc_id"]], "facts": [["x"]]}
                   for d in load_sample()[:10]]
        bm = retrieval_metrics(self.index, queries)["bm25"]["all"]
        self.assertAlmostEqual(bm["hit@3"], 1.0)
        self.assertGreater(bm["mrr"], 0.9)

    def test_calibrate_threshold_separates_a_separable_set(self):
        scores = [{"answerable": True, "top_vector_score": s} for s in (0.6, 0.7, 0.8)] + \
                 [{"answerable": False, "top_vector_score": s} for s in (0.1, 0.2)]
        cal = calibrate_threshold(scores)
        self.assertEqual(cal["accuracy"], 1.0)
        self.assertTrue(0.2 < cal["threshold"] < 0.6)

    def test_top_vector_scores_cover_every_query(self):
        s = top_vector_scores(self.index, self.queries)
        self.assertEqual(len(s), len(self.queries))

    def test_fact_recall_and_abstention_detection(self):
        self.assertEqual(fact_recall("The rate is 3.75 percent.", [["3.75"], ["bank"]]), 0.5)
        self.assertEqual(fact_recall("anything", []), 1.0)
        self.assertEqual(fact_recall("ILSA BERG won", [["ilsa", "berg"]]), 1.0)
        for text in ("I don't know: the provided articles do not contain this information.",
                     "I'm not aware of any such event.", "The context does not mention this.", "I do not have that information."):
            self.assertTrue(abstains(text), text)
        self.assertFalse(abstains("The central bank set 3.75 percent."))


if __name__ == "__main__":
    unittest.main()
