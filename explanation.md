### Why Cosine Similarity Is Preferred

Your embeddings are unit-normalized, so **cosine similarity** is the best retrieval metric because it compares vector direction (semantic meaning) rather than magnitude.

\cos(\theta)=\frac{A\cdot B}{|A||B|}

Euclidean distance is magnitude-sensitive and can bias retrieval toward longer documents. Since normalized vectors have unit length, cosine similarity becomes equivalent to dot product:

A\cdot B = \cos(\theta) \quad \text{when } |A|=|B|=1

That is why systems like FAISS and Vector Search commonly use:

* Cosine similarity
* Inner Product (dot product)

for transformer embeddings.

# Why This Matches Your Architecture

Your `VectorStore.search()` likely uses:

* FAISS `IndexFlatIP`
* or numpy dot products

because:

* embeddings are 384-dimensional
* normalized
* semantic retrieval oriented

This aligns perfectly with:

* Sentence Transformers
* Vertex AI embeddings
* OpenAI embeddings
* Gemini embeddings

---

### Production Migration Plan

The current prototype can be migrated to a production-grade RAG system with the following architecture and improvements:

---

## Embedding & Retrieval

* Use **Vertex AI Embeddings** for:

  * document embeddings
  * query embeddings

* Store vectors in **Weaviate Vector DB** for scalable semantic retrieval.

* Use:

  * **300–600 token chunks**
  * **50–100 token overlap**

to improve retrieval granularity and context continuity.

---

## Search Improvements

Enable **Hybrid Search**:

* dense vector search for semantic similarity
* keyword/BM25 search for exact term matching

This improves:

* recall
* precision
* acronym and technical keyword retrieval

Use **metadata filtering** dynamically based on query context:

* document type
* source
* timestamp
* tenant/project

---

## Reranking

Use **all-MiniLM-L6-v2 / all-mini-macro-l6** as the reranker model after retrieval.

Pipeline:

1. retrieve top-K candidates
2. rerank semantically
3. return best contexts to the LLM

This improves answer relevance and reduces noisy chunks.

---

## Ingestion & Indexing Pipeline

Build an asynchronous ingestion pipeline:

* Use **PyMuPDF** for PDF/document text extraction.
* Use **AWS Lambda** for scalable ingestion.
* Configure:

  * reserved concurrency
  * batch size of 5 documents per Lambda
* Use **AWS SQS Standard Queues** for asynchronous queue management, triggered automatically by new document uploads to the S3 bucket. 
Pipeline:
```text
Upload → Extract → Chunk → Embed → Store in Weaviate
```

This enables scalable and fault-tolerant indexing.

---

## Evaluation Layer

Use **RAGAS** and retrieval metrics to continuously evaluate system quality.

Metrics:

* Faithfulness
* Context Recall
* Precision@K
* MRR (Mean Reciprocal Rank)
* Latency
* Cost per query

This provides both retrieval-quality and operational monitoring.

---

## Orchestration & State Management

Use **LangGraph** for:

* flow orchestration
* query routing
* retry handling
* state management
* multi-step retrieval pipelines

This keeps the RAG workflow modular, observable, and production-ready.

---

## Final Production Flow

```text
Document Upload
    ↓
PyMuPDF Extraction
    ↓
Chunking + Overlap
    ↓
Vertex AI Embeddings
    ↓
Weaviate Vector Store
    ↓
Hybrid Retrieval
    ↓
Metadata Filtering
    ↓
Reranking
    ↓
LLM Generation
    ↓
RAGAS Evaluation
```