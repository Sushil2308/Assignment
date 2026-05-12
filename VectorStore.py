import numpy as np
import faiss


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict] = []

    def add(self, texts: list[str], embeddings: list[list[float]], metadata: list[dict] = None):
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for i, text in enumerate(texts):
            self.documents.append({
                "text": text,
                "metadata": metadata[i] if metadata else {},
            })

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(score),
            })
        return results