from __future__ import annotations
import asyncio
import time
import httpx
from semantic_router.adapters.base import ModelAdapter
from semantic_router.schemas import BidRequest, BidResponse


_METRICS_CACHE: dict[str, tuple[float, dict]] = {}   # base_url -> (timestamp, metrics)
_METRICS_LOCKS: dict[str, asyncio.Lock] = {}          # one lock per base_url
_CACHE_TTL = 1.0


def _get_lock(base_url: str) -> asyncio.Lock:
    if base_url not in _METRICS_LOCKS:
        _METRICS_LOCKS[base_url] = asyncio.Lock()
    return _METRICS_LOCKS[base_url]


def _load_multiplier(load: float) -> float:
    if load < 0.50: return 1.0
    if load < 0.75: return 1.25
    if load < 0.90: return 1.6
    return 2.0


class VLLMAdapter(ModelAdapter):
    async def _scrape_metrics(self) -> dict[str, float]:
        now = time.monotonic()

        # Fast path: cache hit (no lock needed)
        cached = _METRICS_CACHE.get(self.base_url)
        if cached and now - cached[0] < _CACHE_TTL:
            return cached[1]

        # Slow path: only ONE coroutine scrapes at a time.
        # Others wait on the lock then get the cached result (double-check pattern).
        # Without this, 100 concurrent requests cause 100 simultaneous /metrics calls
        # which timeout within the 200ms bid window -> all bids fail -> 503.
        async with _get_lock(self.base_url):
            cached = _METRICS_CACHE.get(self.base_url)
            if cached and now - cached[0] < _CACHE_TTL:
                return cached[1]

            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.base_url}/metrics")
                    resp.raise_for_status()
                metrics: dict[str, float] = {}
                for line in resp.text.splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            metrics[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
                _METRICS_CACHE[self.base_url] = (time.monotonic(), metrics)
                return metrics
            except Exception:
                # Return last known good metrics rather than failing the bid
                old = _METRICS_CACHE.get(self.base_url)
                return old[1] if old else {}

    async def get_load(self) -> float:
        try:
            m = await self._scrape_metrics()
            cache_pressure = m.get("vllm:gpu_cache_usage_perc", 0.0)
            waiting        = m.get("vllm:num_requests_waiting", 0.0)
            vllm_queue     = min(waiting / max(self.max_concurrent_requests, 1), 1.0)
            combined_queue = max(vllm_queue, self.router_queue_pressure)
            return max(cache_pressure, combined_queue)
        except Exception:
            return max(0.3, self.router_queue_pressure)

    async def bid(self, request: BidRequest) -> BidResponse:
        load = await self.get_load()
        mult = _load_multiplier(load)
        inp  = sum(len(m.get("content", "").split()) * 1.3 for m in request.messages)
        out  = self._estimate_output_tokens(request.domain, request.complexity)
        cost = (inp * self.input_rate + out * self.output_rate) * mult
        effective_tok_per_s = max(1000.0 * (1.0 - load * 0.8), 1.0)
        return BidResponse(
            model_id=self.model_id,
            estimated_cost_usd=cost,
            estimated_latency_ms=int(out / effective_tok_per_s * 1000),
            estimated_accuracy=self.get_accuracy_prior(request.domain, request.complexity),
            estimated_energy_j=out / self.efficiency_tokens_per_joule,
            efficiency_tokens_per_joule=self.efficiency_tokens_per_joule,
            current_load=load,
        )

    async def complete(self, messages: list[dict], **kwargs) -> dict:
        await self._increment_in_flight()
        try:
            payload = {"model": self.model_id, "messages": messages, **kwargs}
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )
                resp.raise_for_status()
                return resp.json()
        finally:
            await self._decrement_in_flight()
