import sys
import types
import pytest
from unittest.mock import MagicMock, patch

def _install_fake_vertexai():
    vertexai = types.ModuleType("vertexai")
    language_models = types.ModuleType("vertexai.language_models")
    generative_models = types.ModuleType("vertexai.generative_models")

    class TextEmbeddingModel:
        @staticmethod
        def from_pretrained(name):
            m = MagicMock()
            m.get_embeddings.side_effect = lambda texts: [
                MagicMock(values=[0.1] * 384) for _ in texts
            ]
            return m

    class GenerativeModel:
        def __init__(self, name):
            self.name = name

        def generate_content(self, prompt):
            r = MagicMock()
            r.text = "expanded query text"
            return r

    language_models.TextEmbeddingModel = TextEmbeddingModel
    generative_models.GenerativeModel = GenerativeModel
    vertexai.language_models = language_models
    vertexai.generative_models = generative_models

    sys.modules["vertexai"] = vertexai
    sys.modules["vertexai.language_models"] = language_models
    sys.modules["vertexai.generative_models"] = generative_models


_install_fake_vertexai()

from main import RAGEngine, build_graph
from VectorStore import VectorStore
from embedding import embed_texts


class TestVectorStore:
    def test_add_and_search_returns_results(self):
        store = VectorStore(dimension=384)
        vecs = [[0.1] * 384, [0.9] * 384]
        store.add(["doc one", "doc two"], vecs)
        results = store.search([0.1] * 384, top_k=1)
        assert len(results) == 1
        assert "text" in results[0]
        assert "score" in results[0]

    def test_search_respects_top_k(self):
        store = VectorStore(dimension=384)
        vecs = [[float(i) / 10] * 384 for i in range(1, 6)]
        texts = [f"doc {i}" for i in range(5)]
        store.add(texts, vecs)
        results = store.search([0.5] * 384, top_k=3)
        assert len(results) == 3

    def test_scores_are_descending(self):
        store = VectorStore(dimension=384)
        vecs = [[0.1] * 384, [0.5] * 384, [0.9] * 384]
        store.add(["a", "b", "c"], vecs)
        results = store.search([0.9] * 384, top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestEmbeddingModel:
    def test_mock_model_returns_correct_count(self):
        from embedding import get_mock_vertex_embedding_model
        model = get_mock_vertex_embedding_model()
        texts = ["hello", "world", "foo"]
        results = model.get_embeddings(texts)
        assert len(results) == 3

    def test_mock_model_vectors_are_floats(self):
        from embedding import get_mock_vertex_embedding_model
        model = get_mock_vertex_embedding_model()
        [result] = model.get_embeddings(["test"])
        assert all(isinstance(v, float) for v in result.values)

    def test_embed_texts_helper(self):
        from embedding import get_mock_vertex_embedding_model, embed_texts
        model = get_mock_vertex_embedding_model()
        embeddings = embed_texts(model, ["a", "b"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384


class TestRAGEngine:
    @pytest.fixture
    def engine(self):
        return RAGEngine()

    def test_strategy_a_returns_top_3(self, engine):
        results = engine.strategy_a("peak load", top_k=3)
        assert len(results) == 3

    def test_strategy_b_returns_top_3(self, engine):
        results = engine.strategy_b("peak load", top_k=3)
        assert len(results) == 3

    def test_strategy_a_results_have_required_keys(self, engine):
        results = engine.strategy_a("similarity metric")
        for r in results:
            assert "text" in r
            assert "score" in r

    def test_strategy_b_results_have_rerank_score(self, engine):
        results = engine.strategy_b("retrieval augmented generation")
        for r in results:
            assert "rerank_score" in r

    def test_generative_model_is_mocked(self, engine):
        assert engine.generative_model.model_name == "gemini-pro"

    def test_embedding_model_is_mocked(self, engine):
        assert engine.embedding_model.model_name == "textembedding-gecko@003"

    def test_strategy_b_scores_descending(self, engine):
        results = engine.strategy_b("peak load")
        scores = [r.get("rerank_score", r["score"]) for r in results]
        assert scores == sorted(scores, reverse=True)


class TestLangGraph:
    @pytest.fixture
    def graph_and_engine(self):
        engine = RAGEngine()
        graph = build_graph(engine)
        return graph, engine

    def test_graph_produces_final_top3(self, graph_and_engine):
        graph, _ = graph_and_engine
        state = graph.invoke({"query": "peak load", "results_a": [], "results_b": []})
        assert "final" in state
        assert len(state["final"]) == 3

    def test_graph_deduplicates_results(self, graph_and_engine):
        graph, _ = graph_and_engine
        state = graph.invoke({"query": "similarity metric", "results_a": [], "results_b": []})
        texts = [r["text"] for r in state["final"]]
        assert len(texts) == len(set(texts))

    def test_graph_final_has_scores(self, graph_and_engine):
        graph, _ = graph_and_engine
        state = graph.invoke({"query": "retrieval", "results_a": [], "results_b": []})
        for doc in state["final"]:
            assert "rerank_score" in doc or "score" in doc

    def test_graph_state_contains_merged(self, graph_and_engine):
        graph, _ = graph_and_engine
        state = graph.invoke({"query": "peak load", "results_a": [], "results_b": []})
        assert "merged" in state