"""The RAG chain, written with LangChain Expression Language (LCEL), and an LLM-only baseline.

    question ──► retriever ──► docs ──► [guard: enough evidence?] ──► prompt(context, question) ──► Ollama LLM ──► answer
                                              │ no
                                              └──► fixed "I don't know" answer (the LLM is not called)

The *guard* is the main behavioural fix compared with the original project: when the best retrieved chunk is only weakly
related to the question, a small model such as ``llama3.2:1b`` tends to invent a confident answer from irrelevant text
(the original report's answer about China's railways is an example). Refusing is cheaper and more honest.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

DEFAULT_LLM = "llama3.2:1b"
ABSTAIN_ANSWER = "I don't know: the provided articles do not contain this information."

# Prompt variants. "strict" is the first version of this project (modelled on the original coursework prompt, with
# citations added): it makes a 1B-parameter model refuse almost everything. They are compared on a separate dev
# set in scripts/select_prompt.py; "direct" is the one used.
RAG_PROMPTS = {
    "strict": ChatPromptTemplate.from_template(
        """You are a careful assistant that answers questions about news articles.
Use ONLY the numbered context below. Do not use any outside knowledge.
If the context does not contain the answer, reply exactly: "I don't know: the provided articles do not contain this information."
Answer in at most three sentences and cite the sources you used like [1] or [2].

Context:
{context}

Question: {question}

Answer:"""),
    "direct": ChatPromptTemplate.from_template(
        """Answer the question using the news articles below. Give the specific facts (names, numbers, places) from the articles and cite the article you used like [1].
Do not add anything that is not in the articles. Answer in at most three sentences.

Articles:
{context}

Question: {question}

Answer:"""),
}
RAG_PROMPT = RAG_PROMPTS["direct"]

LLM_ONLY_PROMPT = ChatPromptTemplate.from_template("Answer the following question in at most three sentences.\n\nQuestion: {question}\n\nAnswer:")


def format_context(docs: List[Document]) -> str:
    """Numbered context block; the numbers are the citation labels the model is asked to use."""
    parts = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "")
        doc_id = d.metadata.get("doc_id", "")
        parts.append(f"[{i}] {title} ({doc_id})\n{d.page_content}")
    return "\n\n".join(parts)


def make_llm(model: str = DEFAULT_LLM, base_url: Optional[str] = None, temperature: float = 0.0, num_predict: int = 300) -> BaseLLM:
    """An Ollama-served LLM through ``langchain-ollama``. Temperature 0 makes answers (nearly) reproducible."""
    from langchain_ollama import OllamaLLM

    kwargs: Dict[str, Any] = {"model": model, "temperature": temperature, "num_predict": num_predict}
    if base_url:
        kwargs["base_url"] = base_url
    return OllamaLLM(**kwargs)


def check_ollama(model: str = DEFAULT_LLM, base_url: Optional[str] = None) -> Optional[str]:
    """Return None if the Ollama server is reachable and has ``model``; otherwise a human-readable problem description."""
    try:
        import ollama

        client = ollama.Client(host=base_url) if base_url else ollama.Client()
        client.show(model)
        return None
    except ImportError:
        return "the 'ollama' Python package is not installed (pip install ollama)"
    except Exception as exc:
        text = str(exc).lower()
        if "not found" in text:
            return f"model {model!r} is not available locally; run: ollama pull {model}"
        return f"cannot reach the Ollama server ({exc}); start it with 'ollama serve' (or open the Ollama app)"


@dataclass
class RAGAnswer:
    question: str
    answer: str
    abstained: bool
    sources: List[Document] = field(default_factory=list)
    seconds: float = 0.0


class RAGPipeline:
    def __init__(self, retriever, llm: BaseLLM, min_score: float = 0.25, prompt: ChatPromptTemplate = None):
        """``min_score``: the best retrieved chunk must have at least this cosine similarity to the question."""
        self.retriever, self.llm, self.min_score = retriever, llm, min_score
        answer_chain = (prompt or RAG_PROMPT) | llm | StrOutputParser()

        def generate(inputs: Dict[str, Any]) -> str:
            docs: List[Document] = inputs["docs"]
            if not docs or max(d.metadata.get("vector_score", 0.0) for d in docs) < self.min_score:
                return ABSTAIN_ANSWER
            return answer_chain.invoke({"context": format_context(docs), "question": inputs["question"]}).strip()

        self.chain = RunnableParallel(question=RunnablePassthrough(), docs=retriever) | RunnablePassthrough.assign(answer=RunnableLambda(generate))
        self.llm_only_chain = LLM_ONLY_PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> RAGAnswer:
        t0 = time.perf_counter()
        out = self.chain.invoke(question)
        return RAGAnswer(question, out["answer"], out["answer"] == ABSTAIN_ANSWER or "i don't know" in out["answer"].lower()[:40],
                         out["docs"], time.perf_counter() - t0)

    def ask_llm_only(self, question: str) -> RAGAnswer:
        """Baseline: the same LLM with no retrieval."""
        t0 = time.perf_counter()
        answer = self.llm_only_chain.invoke({"question": question}).strip()
        return RAGAnswer(question, answer, False, [], time.perf_counter() - t0)
