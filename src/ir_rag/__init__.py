"""ir_rag: retrieval-augmented question answering over news articles (Transformer embeddings, BM25 + FAISS hybrid
retrieval, LangChain LCEL chain, Ollama LLM)."""
import os as _os
import sys as _sys

# Windows only: PyTorch and FAISS each ship their own OpenMP runtime. Whichever is loaded second aborts the process
# with "OMP: Error #15: Initializing libomp140 ... but found libiomp5md.dll already initialized" (this is a known
# problem of mixing the two wheels, and importing them in the other order does not help). Intel's documented
# workaround is KMP_DUPLICATE_LIB_OK=TRUE. It is "unsupported" because two runtimes may oversubscribe threads, so the
# test-suite checks the numerical results of both libraries (embeddings against sentence-transformers, FAISS search
# against a numpy oracle). Set the variable yourself (to FALSE) before importing ir_rag to opt out.
if _sys.platform == "win32":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
