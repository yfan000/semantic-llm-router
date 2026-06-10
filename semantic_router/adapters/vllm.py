from __future__ import annotations
import asyncio
import time
import httpx
from semantic_router.adapters.base import ModelAdapter
from semantic_router.schemas import BidRequest, BidResponse


_METRICS_CACHE: dict[str, tuple[float, dict]] = {}   # base_url -> (timestamp, metrics)
_METRICS_LOCKS: dict[str, asyncio.Lock] = {}          # one lock per base_url -- prevents stampede
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cached raw metric values from the last get_load() call.
        # Used by bid() to compute decomposed latency without a second scrape.
        self._last_waiting: float = 0.0
        self._last_running: float = 1.0

    async def _scrape_metrics(self) -> dict[str, float]:
        now = time.monotonic()

        # Fast path: cache hit -- no lock needed
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
            running        = m.get("vllm:num_requests_running",  1.0)
            # Cache raw values for bid() decomposed latency -- avoids a second scrape.
            self._last_waiting = waiting
            self._last_running  = max(running, 1.0)
            vllm_queue     = min(waiting / max(self.max_concurrent_requests, 1), 1.0)
            combined_queue = max(vllm_queue, self.router_queue_pressure)
            return max(cache_pressure, combined_queue)
        except Exception:
            self._last_waiting = 0.0
            self._last_running  = 1.0
            return max(0.3, self.router_queue_pressure)

    async def bid(self, request: BidRequest) -> BidResponse:
        load = await self.get_load()   # also populates _last_waiting / _last_running
        mult = _load_multiplier(load)

        # Input token count (word-split approximation)
        prompt_tokens = sum(len(m.get("content", "").split()) * 1.3
                            for m in request.messages)

        # Output token count -- per-model per-category observed average when available,
        # falls back to static domain:complexity table (see base._estimate_output_tokens)
        out = self._estimate_output_tokens(request.domain, request.complexity)

        cost = (prompt_tokens * self.input_rate + out * self.output_rate) * mult

        # -- Decomposed latency -----------------------------------------------
        waiting = self._last_waiting
        running  = self._last_running   # already clamped to >= 1.0 in get_load()

        # Queue time: num_waiting / requests_per_second
        # requests_per_second = num_running / avg_latency_s  (continuous batching
        # completes num_running requests roughly every avg_latency interval)
        # → queue_ms = num_waiting × avg_latency_ms / num_running
        #
        # Using avg_latency_ms (EMA of actual response time) avoids assuming all
        # waiting requests have the same token count as the current request.
        # Falls back to formula estimate on cold start (no history yet).
        avg_lat = (self._reputation.get_avg_latency_ms(self.model_id)
                   if self._reputation is not None else None)
        per_req_ms = avg_lat if avg_lat is not None else (out / self.decode_tokens_per_sec * 1000)
        queue_ms = waiting * per_req_ms / running

        # Prefill time: GPU processes all input tokens in parallel (compute-bound,
        # much faster per token than decode).
        prefill_ms = prompt_tokens / self.prefill_tokens_per_sec * 1000

        # Decode time: output tokens generated one-by-one (memory-bandwidth-bound).
        # Divide throughput by the number of concurrently running requests because
        # continuous batching shares GPU bandwidth -- each request gets 1/N of capacity.
        effective_decode_tps = self.decode_tokens_per_sec / running
        decode_ms = out / effective_decode_tps * 1000

        estimated_latency_ms = int(queue_ms + prefill_ms + decode_ms)

        return BidResponse(
            model_id=self.model_id,
            estimated_cost_usd=cost,
            estimated_latency_ms=estimated_latency_ms,
            estimated_accuracy=self.get_accuracy_prior(request.domain, request.complexity),
            estimated_energy_j=out / self.efficiency_tokens_per_joule,
            efficiency_tokens_per_joule=self.efficiency_tokens_per_joule,
            current_load=load,
        )

    async def complete(self, messages: list[dict], **kwargs) -> dict:
        await self._increment_in_flight()
        try:
            # Use self.model_name (the HuggingFace repo name vLLM was launched with),
            # NOT self.model_id (the router's friendly alias).
            payload = {"model": self.model_name, "messages": messages, **kwargs}
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )
                resp.raise_for_status()
                return resp.json()
        finally:
            await self._decrement_in_flight()
