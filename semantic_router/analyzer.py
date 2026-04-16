from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_router.config import EMBEDDING_MODEL

_COMPLEXITY_ANCHORS: dict[str, list[str]] = {
    "easy": ["What is 2 plus 2?", "What is the capital of France?", "Convert 100 Fahrenheit to Celsius.", "What color is the sky?", "Translate hello to Spanish."],
    "medium": ["Write a Python function to reverse a linked list.", "Explain how TCP handshake works.", "Summarise the key points of this article.", "What are the pros and cons of microservices?", "Describe the water cycle."],
    "hard": ["Design a distributed consensus algorithm tolerant to Byzantine faults.", "Prove that the square root of 2 is irrational.", "Architect a fault-tolerant recommendation system serving 10M users.", "Analyse the geopolitical implications of the semiconductor supply chain.", "Implement a lock-free concurrent hash map from scratch."],
}

_DOMAIN_ANCHORS: dict[str, list[str]] = {
    "code":      ["Write Python code to", "Fix this bug in my program", "Implement this algorithm", "Debug the following function", "Refactor this class"],
    "math":      ["Solve this equation", "Calculate the integral of", "Prove this mathematical theorem", "Find the derivative of", "What is the probability of"],
    "creative":  ["Write a short story about", "Compose a poem on", "Create a marketing tagline for", "Write a song lyric", "Generate a creative description of"],
    "factual":   ["What is the definition of", "Who invented", "When did this historical event happen", "What country is", "How many"],
    "reasoning": ["Compare and contrast", "What are the implications of", "Analyse the argument that", "Evaluate the trade-offs between", "Why does"],
}

_COMPLEXITY_TOKEN_THRESHOLDS = {"easy": 20, "medium": 80}


@dataclass
class QueryMetadata:
    complexity: str
    domain: str
    embedding: np.ndarray
    complexity_score: float
    domain_score: float


class SemanticAnalyzer:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._complexity_embeddings: dict[str, np.ndarray] = {}
        self._domain_embeddings: dict[str, np.ndarray] = {}

    def load(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        for label, phrases in _COMPLEXITY_ANCHORS.items():
            vecs = self._model.encode(phrases, normalize_embeddings=True)
            self._complexity_embeddings[label] = vecs.mean(axis=0)
        for label, phrases in _DOMAIN_ANCHORS.items():
            vecs = self._model.encode(phrases, normalize_embeddings=True)
            self._domain_embeddings[label] = vecs.mean(axis=0)

    def _flatten_messages(self, messages: list[dict]) -> str:
        return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

    def analyze(self, messages: list[dict]) -> QueryMetadata:
        assert self._model is not None, "Call load() before analyze()"
        text = self._flatten_messages(messages)
        embedding = self._model.encode(text, normalize_embeddings=True)
        sem_scores = {label: float(np.dot(embedding, anchor)) for label, anchor in self._complexity_embeddings.items()}
        token_count = len(text.split())
        heuristic = "easy" if token_count <= _COMPLEXITY_TOKEN_THRESHOLDS["easy"] else "medium" if token_count <= _COMPLEXITY_TOKEN_THRESHOLDS["medium"] else "hard"
        heuristic_bonus = {k: 0.0 for k in sem_scores}
        heuristic_bonus[heuristic] = 1.0
        combined = {k: 0.7 * sem_scores[k] + 0.3 * heuristic_bonus[k] for k in sem_scores}
        complexity = max(combined, key=combined.__getitem__)
        domain_scores = {label: float(np.dot(embedding, anchor)) for label, anchor in self._domain_embeddings.items()}
        domain = max(domain_scores, key=domain_scores.__getitem__)
        return QueryMetadata(complexity=complexity, domain=domain, embedding=embedding,
                             complexity_score=combined[complexity], domain_score=domain_scores[domain])
