"""Draw the figures used in the documentation from results/*.json (needs matplotlib).

    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, BLUE, LBLUE, GREEN, LGREEN, ORANGE, LORANGE, GREY, LGREY, RED = (
    "#1f2933", "#2b6cb0", "#dbe8f5", "#2f855a", "#dff0e6", "#c05621", "#fbe6d6", "#718096", "#edf0f3", "#d1495b")


def box(ax, x, y, w, h, text, fc, ec, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4, linestyle=ls))


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(0.1, 6.3, "Offline: build the index", fontsize=10, fontweight="bold", color=BLUE)
    box(ax, 0.1, 4.6, 2.2, 1.2, "Articles\n(sample or CNN CSV)", LGREY, GREY)
    box(ax, 3.0, 4.6, 2.2, 1.2, "Chunking\n1000 chars,\n200 overlap", LBLUE, BLUE)
    box(ax, 5.9, 5.0, 3.1, 0.9, "Transformer encoder\nMiniLM -> 384-d unit vectors", LORANGE, ORANGE)
    box(ax, 5.9, 3.9, 3.1, 0.9, "BM25 inverted index", LORANGE, ORANGE)
    box(ax, 9.7, 5.0, 2.4, 0.9, "FAISS inner-product\nindex (cosine)", LBLUE, BLUE)
    box(ax, 12.5, 4.4, 2.3, 1.2, "Saved index +\nmanifest (model,\ncorpus hash)", LGREY, GREY)
    arrow(ax, 2.3, 5.2, 3.0, 5.2)
    arrow(ax, 5.2, 5.4, 5.9, 5.45)
    arrow(ax, 5.2, 4.9, 5.9, 4.4)
    arrow(ax, 9.0, 5.45, 9.7, 5.45)
    arrow(ax, 12.1, 5.2, 12.5, 5.0)
    ax.text(0.1, 3.4, "Online: answer a question", fontsize=10, fontweight="bold", color=GREEN)
    box(ax, 0.1, 1.5, 1.9, 1.2, "Question", LGREY, GREY)
    box(ax, 2.7, 1.5, 3.2, 1.2, "Retriever (LangChain)\nvector | bm25 | hybrid (RRF)\nmax 2 chunks per article", LBLUE, BLUE)
    box(ax, 6.6, 1.5, 2.4, 1.2, "Guard: best cosine\n>= threshold?", LORANGE, ORANGE, bold=True)
    box(ax, 9.7, 1.5, 3.0, 1.2, "Prompt (numbered context)\n-> Ollama llama3.2:1b\n-> answer with [n] citations", LGREEN, GREEN)
    box(ax, 13.1, 1.5, 1.8, 1.2, "Answer +\nsources", LGREY, GREY)
    box(ax, 6.6, 0.1, 2.4, 0.8, "\"I don't know\"\n(LLM not called)", LGREY, RED)
    arrow(ax, 2.0, 2.1, 2.7, 2.1)
    arrow(ax, 5.9, 2.1, 6.6, 2.1)
    arrow(ax, 9.0, 2.1, 9.7, 2.1)
    arrow(ax, 12.7, 2.1, 13.1, 2.1)
    arrow(ax, 7.8, 1.5, 7.8, 0.9, color=RED)
    arrow(ax, 13.6, 4.4, 4.3, 2.7, color=BLUE, ls="--")
    ax.text(9.6, 3.55, "index lookup", fontsize=8, color=BLUE, style="italic", rotation=-17)
    fig.savefig(OUT / "pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def retrieval_chart() -> None:
    data = json.loads((ROOT / "results" / "retrieval_eval.json").read_text(encoding="utf-8"))["retrieval"]
    groups = ["lexical", "paraphrase", "multi", "all"]
    modes = [("bm25", ORANGE), ("vector", BLUE), ("hybrid", GREEN)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, title in ((axes[0], "hit@1", "Hit@1"), (axes[1], "mrr", "MRR")):
        w = 0.26
        for k, (mode, color) in enumerate(modes):
            ax.bar([i + (k - 1) * w for i in range(len(groups))], [data[mode][g][metric] for g in groups], w, color=color, label=mode)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\n(n={data['bm25'][g]['queries']})" for g in groups], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", color="#dde3ea")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Retrieval quality by question type (20 answerable questions, 40 articles)", y=1.02)
    fig.savefig(OUT / "retrieval_by_type.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    retrieval_chart()
    print("figures written to", OUT)
