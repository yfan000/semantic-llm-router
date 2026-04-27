from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_router.config import EMBEDDING_MODEL

# Single shared thread pool for CPU-bound embedding work.
# Prevents GIL contention when many async requests arrive concurrently.
_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")

_HARD_MARKERS = [
    r"\bprove\b", r"\bderive\b", r"\btheorem\b", r"\blemma\b", r"\baxiom\b",
    r"\boptimize\b", r"\bminimize\b", r"\bmaximize\b",
    r"\bdistributed\b", r"\bscalable\b", r"\bfault.tolerant\b", r"\bhigh.availability\b",
    r"\bconsensus\b", r"\bbyzantine\b", r"\bconcurrent\b", r"\block.free\b",
    r"\bmicroservice", r"\barchitect\b",
    r"\bfrom scratch\b", r"\bimplement\b.*\balgorithm\b", r"\bdesign\b.*\bsystem\b",
    r"\bcomplex(ity)?\b", r"\badvanced\b", r"\bin.depth\b",
    r"\bintegral\b", r"\bderivative\b", r"\bdifferential equation\b",
    r"\beigenvalue\b", r"\blinear algebra\b", r"\bconverg",
    r"\bconcurrency\b", r"\bthread.safe\b", r"\basync\b.*\bawait\b",
    r"\btime complexity\b", r"\bspace complexity\b", r"\bbig.o\b",
]

_EASY_MARKERS = [
    r"\bwhat is\b", r"\bwho is\b", r"\bwho was\b", r"\bwhen (was|did|is)\b",
    r"\bwhere is\b", r"\bwhich (country|city|planet|element)\b",
    r"\bhow many\b", r"\bhow much\b", r"\bdefine\b", r"\blist\b",
    r"\bname\b.*\b(few|some|three|five)\b", r"\bwhat (color|year|language)\b",
    r"\bcapital (of|city)\b", r"\btranslate\b", r"\bconvert\b",
    r"\bsimple\b", r"\bbasic\b", r"\bquick\b",
]

_MEDIUM_MARKERS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bsummarise\b", r"\bsummarize\b",
    r"\bcompare\b", r"\bcontrast\b", r"\bpros and cons\b", r"\btrade.off\b",
    r"\bhow does\b", r"\bhow do\b", r"\bwrite a (function|class|script)\b",
    r"\bstep.by.step\b", r"\bwhat are the (steps|stages|phases)\b",
    r"\banalyse\b", r"\banalyze\b", r"\bevaluate\b",
]


def _keyword_complexity(text: str) -> dict[str, float]:
    t = text.lower()
    hard_hits   = sum(1 for p in _HARD_MARKERS   if re.search(p, t))
    easy_hits   = sum(1 for p in _EASY_MARKERS   if re.search(p, t))
    medium_hits = sum(1 for p in _MEDIUM_MARKERS if re.search(p, t))
    total = hard_hits + easy_hits + medium_hits
    if total == 0:
        return {"easy": 0.33, "medium": 0.33, "hard": 0.33}
    return {"easy": easy_hits/total, "medium": medium_hits/total, "hard": hard_hits/total}


def _token_complexity(text: str) -> dict[str, float]:
    n = len(text.split())
    if n <= 12: return {"easy": 0.70, "medium": 0.20, "hard": 0.10}
    if n <= 35: return {"easy": 0.20, "medium": 0.55, "hard": 0.25}
    return {"easy": 0.10, "medium": 0.30, "hard": 0.60}


_COMPLEXITY_ANCHORS: dict[str, list[str]] = {
    "easy": [
        "What is 2 plus 2?", "What is the capital of France?",
        "Convert 100 Fahrenheit to Celsius.", "What color is the sky?",
        "Translate hello to Spanish.", "Who invented the telephone?",
        "What year was the Eiffel Tower built?", "Define photosynthesis.",
        "How many days are in a week?", "What is the chemical symbol for gold?",
    ],
    "medium": [
        "Write a Python function to reverse a linked list.",
        "Explain how TCP handshake works.",
        "Summarise the key points of this article.",
        "What are the pros and cons of microservices?",
        "Describe the water cycle.", "Compare SQL and NoSQL databases.",
        "How does garbage collection work in Python?",
        "Write a function to check if a string is a palindrome.",
        "Explain the difference between supervised and unsupervised learning.",
        "What are the stages of the software development lifecycle?",
    ],
    "hard": [
        "Design a distributed consensus algorithm tolerant to Byzantine faults.",
        "Prove that the square root of 2 is irrational.",
        "Architect a fault-tolerant recommendation system serving 10M users.",
        "Analyse the geopolitical implications of the semiconductor supply chain.",
        "Implement a lock-free concurrent hash map from scratch.",
        "Derive the gradient descent update rule from first principles.",
        "Design a system that handles 1 million concurrent websocket connections.",
        "Prove the time complexity of the Floyd-Warshall algorithm.",
        "Implement a distributed rate limiter with Redis consistent under network partition.",
        "Explain the CAP theorem and its implications for distributed database design.",
    ],
}

_DOMAIN_ANCHORS: dict[str, list[str]] = {
    "code": [
        "Write Python code to", "Fix this bug in my program",
        "Implement this algorithm", "Debug the following function",
        "Refactor this class", "Write a script that",
        "How do I parse JSON in Python?", "What is wrong with this code?",
    ],
    "math": [
        "Solve this equation", "Calculate the integral of",
        "Prove this mathematical theorem", "Find the derivative of",
        "What is the probability of", "Compute the eigenvalues of",
        "Simplify this expression", "What is the sum of the series",
    ],
    "creative": [
        "Write a short story about", "Compose a poem on",
        "Create a marketing tagline for", "Write a song lyric",
        "Generate a creative description of", "Write a dialogue between",
        "Imagine a world where",
    ],
    "factual": [
        "What is the definition of", "Who invented",
        "When did this historical event happen", "What country is",
        "How many", "What is the capital of", "Who was the first", "What year did",
    ],
    "reasoning": [
        "Compare and contrast", "What are the implications of",
        "Analyse the argument that", "Evaluate the trade-offs between",
        "Why does", "What would happen if", "Argue for and against",
        "What are the pros and cons of",
    ],
}

_W_KEYWORD  = 0.50
_W_TOKEN    = 0.15
_W_SEMANTIC = 0.35


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

    def _encode_sync(self, text: str) -> np.ndarray:
        """CPU-bound encode — always called via thread pool, never inline."""
        return self._model.encode(text, normalize_embeddings=True)

    async def analyze_async(self, messages: list[dict]) -> QueryMetadata:
        """Offloads encode() to thread pool so the asyncio event loop is never
        blocked. With concurrency=50, the sync version serialized all requests
        on the GIL adding ~1900ms overhead."""
        assert self._model is not None, "Call load() before analyze_async()"
        text = self._flatten_messages(messages)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(_THREAD_POOL, self._encode_sync, text)
        return self._classify(text, embedding)

    def analyze(self, messages: list[dict]) -> QueryMetadata:
        """Sync fallback — used in calibration and tests."""
        assert self._model is not None, "Call load() before analyze()"
        text = self._flatten_messages(messages)
        return self._classify(text, self._encode_sync(text))

    def _classify(self, text: str, embedding: np.ndarray) -> QueryMetadata:
        sem = {
            label: float(np.dot(embedding, anchor))
            for label, anchor in self._complexity_embeddings.items()
        }
        kw  = _keyword_complexity(text)
        tok = _token_complexity(text)
        combined = {
            k: _W_SEMANTIC * sem[k] + _W_KEYWORD * kw[k] + _W_TOKEN * tok[k]
            for k in sem
        }
        complexity       = max(combined, key=combined.__getitem__)
        complexity_score = combined[complexity]

        domain_scores = {
            label: float(np.dot(embedding, anchor))
            for label, anchor in self._domain_embeddings.items()
        }
        domain       = max(domain_scores, key=domain_scores.__getitem__)
        domain_score = domain_scores[domain]

        return QueryMetadata(
            complexity=complexity, domain=domain, embedding=embedding,
            complexity_score=complexity_score, domain_score=domain_score,
        )
