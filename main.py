import operator
from typing import Annotated, TypedDict
from unittest.mock import MagicMock

from langgraph.graph import StateGraph, END

from embedding import get_mock_vertex_embedding_model, embed_texts
from VectorStore import VectorStore


CORPUS = [
    "Distributed systems rely on horizontal scaling and load balancers to handle traffic spikes. "
    "Auto-scaling groups spin up new instances when CPU thresholds are breached, typically above 70 percent.",

    "Peak load management uses circuit breakers and rate limiting to shed excess requests gracefully. "
    "The system queues overflow traffic and returns 503 responses when the queue is saturated.",

    "Vector similarity search projects text into high-dimensional embedding space and retrieves nearest neighbours. "
    "Cosine similarity is the standard metric because it is invariant to vector magnitude.",

    "FAISS provides approximate nearest neighbour search using inverted file indices and product quantisation. "
    "For small corpora an exact flat index with inner product scoring is sufficient and highly accurate.",

    "Retrieval-Augmented Generation grounds LLM responses in external documents fetched at inference time. "
    "This reduces hallucinations because the model cites passages rather than relying on parametric memory.",

    "Query expansion rewrites the user query into a richer, keyword-dense form before embedding. "
    "Expanded queries cover more semantic territory and retrieve a broader set of relevant chunks.",

    "Sentence Transformers produce dense bi-encoder embeddings suitable for semantic search. "
    "The all-MiniLM-L6-v2 model balances speed and quality and outputs 384-dimensional unit vectors.",

    "Cosine similarity measures the angle between two vectors regardless of their magnitudes. "
    "Euclidean distance is magnitude-sensitive and can rank long documents above short relevant ones.",

    "Vertex AI Vector Search, formerly Matching Engine, stores billions of embeddings and serves sub-100 ms queries. "
    "Migrating from a local prototype requires exporting JSON-Lines embeddings to GCS and creating an index via REST.",

    "RAGAS evaluates RAG pipelines across four metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall. "
    "Each metric is computed reference-free using a judge LLM that inspects retrieved context and generated answers.",
]


def get_mock_generative_model():
    mock = MagicMock()
    mock.model_name = "gemini-pro"

    def generate_content(prompt: str):
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        query = lines[-1] if lines else prompt

        keywords = {
            "peak load": [
                "how does the system scale under high request volume",
                "what mechanisms prevent overload during traffic spikes",
                "how are excess requests queued or dropped",
            ],
            "similarity": [
                "what distance metric should be used for vector search",
                "how does cosine similarity differ from euclidean distance",
                "why is cosine preferred for normalised embeddings",
            ],
            "retrieval": [
                "how does RAG fetch external knowledge at inference time",
                "what embedding model is used for semantic retrieval",
                "how does query expansion improve recall",
            ],
        }

        for key, subs in keywords.items():
            if key in query.lower():
                expanded = " ".join(subs)
                break
        else:
            expanded = f"detailed technical explanation of: {query}"

        result = MagicMock()
        result.text = expanded
        return result

    mock.generate_content.side_effect = generate_content
    return mock


class RAGEngine:
    def __init__(self):
        self.embedding_model = get_mock_vertex_embedding_model()
        self.generative_model = get_mock_generative_model()
        self.store = VectorStore(dimension=384)
        self._ingest()

    def _ingest(self):
        embeddings = embed_texts(self.embedding_model, CORPUS)
        self.store.add(
            texts=CORPUS,
            embeddings=embeddings,
            metadata=[{"id": i} for i in range(len(CORPUS))],
        )

    def _expand_query(self, query: str) -> str:
        prompt = f"Rewrite this query for dense retrieval:\n{query}"
        result = self.generative_model.generate_content(prompt)
        return result.text

    def strategy_a(self, query: str, top_k: int = 3) -> list[dict]:
        [embedding] = embed_texts(self.embedding_model, [query])
        return self.store.search(embedding, top_k=top_k)

    def strategy_b(self, query: str, top_k: int = 3) -> list[dict]:
        expanded = self._expand_query(query)
        sub_queries = expanded.split("  ") if "  " in expanded else [expanded]
        seen = {}
        for sq in sub_queries:
            [embedding] = embed_texts(self.embedding_model, [sq])
            for result in self.store.search(embedding, top_k=top_k * 2):
                key = result["text"]
                if key not in seen or result["score"] > seen[key]["score"]:
                    seen[key] = result
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return self._rerank(query, merged)[:top_k]

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        [q_emb] = embed_texts(self.embedding_model, [query])
        import numpy as np
        q_vec = np.array(q_emb)
        for c in candidates:
            [c_emb] = embed_texts(self.embedding_model, [c["text"]])
            c_vec = np.array(c_emb)
            c["rerank_score"] = float(np.dot(q_vec, c_vec))
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


class GraphState(TypedDict):
    query: str
    results_a: list[dict]
    results_b: list[dict]
    merged: list
    final: list


def build_graph(engine: RAGEngine):
    def decompose(state: GraphState) -> dict:
        return {"query": state["query"], "results_a": [], "results_b": []}

    def search_a(state: GraphState) -> dict:
        return {"results_a": engine.strategy_a(state["query"], top_k=5)}

    def search_b(state: GraphState) -> dict:
        return {"results_b": engine.strategy_b(state["query"], top_k=5)}

    def dedup(state: GraphState) -> dict:
        seen = {}
        for doc in state["results_a"] + state["results_b"]:
            key = doc["metadata"]["id"]
            if key not in seen:
                seen[key] = doc
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return {"merged": merged}

    def rerank(state: GraphState) -> dict:
        top3 = engine._rerank(state["query"], state["merged"])[:3]
        return {"final": top3}

    g = StateGraph(GraphState)
    g.add_node("decompose", decompose)
    g.add_node("search_a", search_a)
    g.add_node("search_b", search_b)
    g.add_node("dedup", dedup)
    g.add_node("rerank", rerank)
    g.add_node("aggregate", lambda state: state)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "search_a")
    g.add_edge("decompose", "search_b")
    g.add_edge("search_a","aggregate")
    g.add_edge("search_b","aggregate")
    g.add_edge("aggregate", "dedup")
    g.add_edge("dedup", "rerank")
    g.add_edge("rerank", END)

    return g.compile()

engine = RAGEngine()

graph = build_graph(engine)

state = graph.invoke({
    "query": "peak load",
    "results_a": [],
    "results_b": [],
})

print("\nFINAL RESULTS:\n")

for doc in state["final"]:
    print(doc["text"])
    print("score:", doc.get("rerank_score", doc["score"]))
    print("-" * 50)