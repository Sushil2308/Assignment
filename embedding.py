import hashlib
import math
from unittest.mock import MagicMock

DIMENSION = 384


def _text_to_vector(text: str) -> list[float]:
    vec = [0.0] * DIMENSION
    for i, char in enumerate(text):
        bucket = (ord(char) * 31 + i) % DIMENSION
        vec[bucket] += 1.0
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(4):
            vec[(h >> (j * 8)) % DIMENSION] += 0.5
    magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / magnitude for v in vec]


def get_mock_vertex_embedding_model():
    mock_model = MagicMock()
    mock_model.model_name = "textembedding-gecko@003"

    def get_embeddings(texts: list[str]):
        results = []
        for text in texts:
            instance = MagicMock()
            instance.values = _text_to_vector(text)
            results.append(instance)
        return results

    mock_model.get_embeddings.side_effect = get_embeddings
    return mock_model


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    results = model.get_embeddings(texts)
    return [r.values for r in results]