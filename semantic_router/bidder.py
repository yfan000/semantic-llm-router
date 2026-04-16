from __future__ import annotations
import asyncio, logging
from semantic_router.adapters.base import ModelAdapter
from semantic_router.schemas import BidRequest, BidResponse
from semantic_router.config import BID_TIMEOUT_MS

log = logging.getLogger(__name__)


async def _safe_bid(adapter: ModelAdapter, request: BidRequest) -> BidResponse | None:
    try:
        return await asyncio.wait_for(adapter.bid(request), timeout=BID_TIMEOUT_MS / 1000)
    except asyncio.TimeoutError:
        log.debug("Bid timeout from %s", adapter.model_id)
    except Exception as e:
        log.debug("Bid error from %s: %s", adapter.model_id, e)
    return None


async def collect_bids(adapters: list[ModelAdapter], request: BidRequest) -> list[BidResponse]:
    results = await asyncio.gather(*(_safe_bid(a, request) for a in adapters))
    return [r for r in results if r is not None]
