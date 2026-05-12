# Retrieval Benchmark: Strategy A vs Strategy B
## Metric Choice: Cosine vs Euclidean
Cosine similarity measures the angle between vectors, making it independent of magnitude. This matters because embedding magnitude encodes model confidence, not semantic direction. A short, highly relevant chunk and a long, moderately relevant one will produce vectors of different magnitudes; cosine correctly ranks them by angle, euclidean would not. For unit-normalised embeddings (as produced by sentence-transformers), cosine similarity equals dot-product, so FAISS IndexFlatIP is used here.
## Vertex AI Migration
1. Export embeddings as JSON-Lines to a GCS bucket: `{"id": "doc_0", "embedding": [...]}`
2. Create a Vector Search index via REST specifying `DOT_PRODUCT_DISTANCE` and shard count.
3. Deploy the index to an IndexEndpoint for online serving.
4. Replace `VectorStore.search` calls with `endpoint.find_neighbors()`.
5. Use `index.upsert_datapoints()` for streaming updates.
---
## Query: `How does the system handle peak load?`
### Strategy A — Raw Vector Search
| Rank | Score | Text |
|---|---|---|
| 1 | 0.3504 | Retrieval-Augmented Generation grounds LLM responses in external documents fetched at inference time. This reduces hallu... |
| 2 | 0.3045 | Sentence Transformers produce dense bi-encoder embeddings suitable for semantic search. The all-MiniLM-L6-v2 model balan... |
| 3 | 0.2961 | RAGAS evaluates RAG pipelines across four metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall... |

### Strategy B — Query Expansion + Dedup + Rerank
| Rank | Score | Text |
|---|---|---|
| 1 | 0.3504 | Retrieval-Augmented Generation grounds LLM responses in external documents fetched at inference time. This reduces hallu... |
| 2 | 0.3045 | Sentence Transformers produce dense bi-encoder embeddings suitable for semantic search. The all-MiniLM-L6-v2 model balan... |
| 3 | 0.2961 | RAGAS evaluates RAG pipelines across four metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall... |
