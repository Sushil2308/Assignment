import json
from main import RAGEngine, build_graph
import csv

QUERIES = [
    "How does the system handle peak load?",
    "What similarity metric should I use for vector search?",
    "How does retrieval augmented generation work?",
]


def run_benchmark():
    engine = RAGEngine()
    graph = build_graph(engine)

    report = []

    for query in QUERIES:
        results_a = engine.strategy_a(query, top_k=3)
        results_b = engine.strategy_b(query, top_k=3)
        state = graph.invoke({"query": query, "results_a": [], "results_b": []})

        entry = {
            "query": query,
            "strategy_a": [
                {"rank": i + 1, "score": round(r["score"], 4), "text": r["text"][:120]}
                for i, r in enumerate(results_a)
            ],
            "strategy_b": [
                {"rank": i + 1, "score": round(r.get("rerank_score", r["score"]), 4), "text": r["text"][:120]}
                for i, r in enumerate(results_b)
            ],
            "graph_final_top3": [
                {"rank": i + 1, "score": round(r.get("rerank_score", r["score"]), 4), "text": r["text"][:120]}
                for i, r in enumerate(state["final"])
            ],
        }
        report.append(entry)

    return report


def write_markdown(report: list[dict]):
    lines = [
        "# Retrieval Benchmark: Strategy A vs Strategy B\n",
        "## Metric Choice: Cosine vs Euclidean\n",
        "Cosine similarity measures the angle between vectors, making it independent of magnitude. "
        "This matters because embedding magnitude encodes model confidence, not semantic direction. "
        "A short, highly relevant chunk and a long, moderately relevant one will produce vectors "
        "of different magnitudes; cosine correctly ranks them by angle, euclidean would not. "
        "For unit-normalised embeddings (as produced by sentence-transformers), "
        "cosine similarity equals dot-product, so FAISS IndexFlatIP is used here.\n",
        "## Vertex AI Migration\n",
        "1. Export embeddings as JSON-Lines to a GCS bucket: `{\"id\": \"doc_0\", \"embedding\": [...]}`\n"
        "2. Create a Vector Search index via REST specifying `DOT_PRODUCT_DISTANCE` and shard count.\n"
        "3. Deploy the index to an IndexEndpoint for online serving.\n"
        "4. Replace `VectorStore.search` calls with `endpoint.find_neighbors()`.\n"
        "5. Use `index.upsert_datapoints()` for streaming updates.\n",
        "---\n",
    ]

    for entry in report:
        lines.append(f"## Query: `{entry['query']}`\n")

        lines.append("### Strategy A — Raw Vector Search\n")
        lines.append("| Rank | Score | Text |\n|---|---|---|\n")
        for r in entry["strategy_a"]:
            lines.append(f"| {r['rank']} | {r['score']} | {r['text']}... |\n")

        lines.append("\n### Strategy B — Query Expansion + Dedup + Rerank\n")
        lines.append("| Rank | Score | Text |\n|---|---|---|\n")
        for r in entry["strategy_b"]:
            lines.append(f"| {r['rank']} | {r['score']} | {r['text']}... |\n")

        lines.append("\n### Graph Final Top-3 (Parallel A+B → Dedup → Rerank)\n")
        lines.append("| Rank | Score | Text |\n|---|---|---|\n")
        for r in entry["graph_final_top3"]:
            lines.append(f"| {r['rank']} | {r['score']} | {r['text']}... |\n")

        lines.append("\n---\n")

    with open("retrieval_benchmark.md", "w") as f:
        f.writelines(lines)
    print("Written: retrieval_benchmark.md")


if __name__ == "__main__":
    report = run_benchmark()
    print(json.dumps(report, indent=2))
    write_markdown(report)
