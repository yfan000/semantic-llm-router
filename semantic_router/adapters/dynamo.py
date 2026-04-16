from __future__ import annotations
import time, httpx
from semantic_router.adapters.base import ModelAdapter
from semantic_router.schemas import BidRequest, BidResponse

_CACHE: dict[str, tuple[float, dict]] = {}
_TTL = 1.0

def _mult(load: float) -> float:
    if load < .50: return 1.0
    if load < .75: return 1.25
    if load < .90: return 1.6
    return 2.0

class DynamoAdapter(ModelAdapter):
    async def _scrape(self) -> dict[str, float]:
        now = time.monotonic()
        if (c := _CACHE.get(self.base_url)) and now - c[0] < _TTL: return c[1]
        async with httpx.AsyncClient(timeout=2.0) as cl:
            r = await cl.get(f"{self.base_url}/v1/scheduler/stats"); r.raise_for_status(); data = r.json()
        m = {"kv_cache_utilization": float(data.get("kv_cache_utilization", 0.0)),
             "prefill_queue_depth":  float(data.get("prefill_queue_depth", 0))}
        _CACHE[self.base_url] = (now, m); return m

    async def get_load(self) -> float:
        try:
            m = await self._scrape()
            return max(m["kv_cache_utilization"], min(m["prefill_queue_depth"] / max(self.max_concurrent_requests,1), 1.0))
        except: return 0.5

    async def bid(self, request: BidRequest) -> BidResponse:
        load = await self.get_load()
        inp = sum(len(m.get("content","").split()) * 1.3 for m in request.messages)
        out = self._estimate_output_tokens(request.domain, request.complexity)
        return BidResponse(
            model_id=self.model_id,
            estimated_cost_usd=(inp * self.input_rate + out * self.output_rate) * _mult(load),
            estimated_latency_ms=int(out / max(1000.0*(1-load*0.8), 1) * 1000),
            estimated_accuracy=self.get_accuracy_prior(request.domain, request.complexity),
            estimated_energy_j=out / self.efficiency_tokens_per_joule,
            efficiency_tokens_per_joule=self.efficiency_tokens_per_joule,
            current_load=load,
        )

    async def complete(self, messages: list[dict], **kw) -> dict:
        async with httpx.AsyncClient(timeout=120.0) as cl:
            r = await cl.post(f"{self.base_url}/v1/responses", json={"model": self.model_id, "messages": messages, **kw})
            r.raise_for_status(); data = r.json()
        return {"id": data.get("id",""), "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": data.get("output",[{}])[0].get("content","")}, "finish_reason": data.get("stop_reason","stop")}],
                "usage": data.get("usage",{})}
