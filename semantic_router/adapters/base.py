from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from semantic_router.schemas import BidRequest, BidResponse


class ModelAdapter(ABC):
    def __init__(
        self,
        model_id: str,
        base_url: str,
        efficiency_tokens_per_joule: float,
        max_concurrent_requests: int,
        input_rate_usd_per_token: float,
        output_rate_usd_per_token: float,
        accuracy_priors: dict[str, float] | None = None,
        model_name: str = "",
    ) -> None:
        self.model_id = model_id
        # model_name is sent to the vLLM /v1/chat/completions endpoint.
        # Defaults to model_id but must match the HuggingFace repo name
        # that vLLM was launched with (e.g. "Qwen/Qwen2.5-7B-Instruct").
        self.model_name = model_name or model_id
        self.base_url = base_url.rstrip("/")
        self.efficiency_tokens_per_joule = efficiency_tokens_per_joule
        self.max_concurrent_requests = max_concurrent_requests
        self.input_rate = input_rate_usd_per_token
        self.output_rate = output_rate_usd_per_token
        self.accuracy_priors: dict[str, float] = accuracy_priors or {}

        # Router-side in-flight counter.
        # Tracks requests dispatched but not yet completed — updated
        # immediately at dispatch time, before vLLM's Prometheus metrics
        # catch up. Prevents thundering-herd overbidding when many
        # concurrent requests all see the same stale low-load snapshot.
        self._in_flight: int = 0
        self._lock = asyncio.Lock()

    @property
    def router_queue_pressure(self) -> float:
        """Fraction of capacity already claimed by in-flight router requests."""
        return min(self._in_flight / max(self.max_concurrent_requests, 1), 1.0)

    async def _increment_in_flight(self) -> None:
        async with self._lock:
            self._in_flight += 1

    async def _decrement_in_flight(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def get_accuracy_prior(self, domain: str, complexity: str) -> float:
        from semantic_router.config import DEFAULT_ACCURACY_PRIOR
        return self.accuracy_priors.get(f"{domain}:{complexity}", DEFAULT_ACCURACY_PRIOR)

    def _estimate_output_tokens(self, domain: str, complexity: str) -> int:
        TABLE: dict[tuple[str, str], int] = {
            ("factual",   "easy"):   80,
            ("factual",   "medium"): 200,
            ("factual",   "hard"):   350,
            ("math",      "easy"):   120,
            ("math",      "medium"): 280,
            ("math",      "hard"):   450,
            ("code",      "easy"):   150,
            ("code",      "medium"): 350,
            ("code",      "hard"):   650,
            ("creative",  "easy"):   250,
            ("creative",  "medium"): 500,
            ("creative",  "hard"):   800,
            ("reasoning", "easy"):   180,
            ("reasoning", "medium"): 380,
            ("reasoning", "hard"):   600,
        }
        return TABLE.get((domain, complexity), 300)

    @abstractmethod
    async def get_load(self) -> float:
        """Return load signal in [0, 1]: max(kv_cache_pressure, queue_pressure)."""

    @abstractmethod
    async def bid(self, request: BidRequest) -> BidResponse:
        """Return a committed bid for this request."""

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> dict:
        """Run inference and return an OpenAI-compatible response dict."""
