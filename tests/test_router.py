"""
test_router.py — Pytest suite for the semantic LLM router.

Runs entirely in-process with mocked adapters — no GPU or real vLLM needed.

Install test deps:
    pip install pytest pytest-asyncio httpx

Run all tests:
    cd output
    pytest tests/test_router.py -v

Run a specific test:
    pytest tests/test_router.py::TestLatencyPenalty::test_chronic_overbidder_penalised -v
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from semantic_router.schemas import (
    BidRequest, BidResponse, UserPreference, UserBudget, RouterMode,
)
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.selector import select, NoEligibleModelError
from semantic_router.user_registry import UserRegistry
from semantic_router.registry import ModelRegistry, ModelConfig
from semantic_router.adapters.base import ModelAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bid(model_id, cost=0.001, latency_ms=500, accuracy=0.80, energy_j=10.0, load=0.3):
    return BidResponse(model_id=model_id, estimated_cost_usd=cost,
                       estimated_latency_ms=latency_ms, estimated_accuracy=accuracy,
                       estimated_energy_j=energy_j, efficiency_tokens_per_joule=30.0, current_load=load)

def make_tracker(tmp_path): return ReputationTracker(path=os.path.join(tmp_path, "rep.json"))
def make_pref(**kw):
    d = dict(cost_weight=0.25, latency_weight=0.25, accuracy_weight=0.25, energy_weight=0.25)
    d.update(kw); return UserPreference(**d)


# ---------------------------------------------------------------------------
# 1. Selector
# ---------------------------------------------------------------------------

class TestSelector:
    def test_hard_latency_filter(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        winner = select([make_bid("slow", latency_ms=800), make_bid("fast", latency_ms=200)],
                        make_pref(max_latency_ms=300), tracker, "factual", "easy")
        assert winner.model_id == "fast"

    def test_hard_accuracy_filter(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        winner = select([make_bid("low", accuracy=0.60), make_bid("high", accuracy=0.90)],
                        make_pref(min_accuracy=0.85), tracker, "math", "hard")
        assert winner.model_id == "high"

    def test_raises_when_no_bids(self, tmp_path):
        with pytest.raises(NoEligibleModelError):
            select([], make_pref(), make_tracker(str(tmp_path)), "code", "easy")

    def test_raises_when_all_filtered(self, tmp_path):
        with pytest.raises(NoEligibleModelError):
            select([make_bid("a", accuracy=0.70), make_bid("b", accuracy=0.80)],
                   make_pref(min_accuracy=0.99), make_tracker(str(tmp_path)), "code", "easy")

    def test_cost_mode_prefers_cheap(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        pref = UserPreference(mode=RouterMode.COST, cost_weight=0.55, latency_weight=0.15,
                              accuracy_weight=0.20, energy_weight=0.10)
        winner = select([make_bid("cheap", cost=0.001, accuracy=0.70, latency_ms=600),
                         make_bid("pricey", cost=0.010, accuracy=0.95, latency_ms=200)],
                        pref, tracker, "factual", "easy")
        assert winner.model_id == "cheap"

    def test_eco_mode_prefers_efficient(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        pref = UserPreference(mode=RouterMode.ECO, accuracy_weight=0.25, cost_weight=0.15,
                              latency_weight=0.20, energy_weight=0.40)
        winner = select([make_bid("small", energy_j=5.0, cost=0.002, accuracy=0.70),
                         make_bid("large", energy_j=50.0, cost=0.001, accuracy=0.90)],
                        pref, tracker, "creative", "easy")
        assert winner.model_id == "small"

    def test_accuracy_mode_prefers_accurate(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        pref = UserPreference(mode=RouterMode.ACCURACY, accuracy_weight=0.70, cost_weight=0.15,
                              latency_weight=0.10, energy_weight=0.05)
        winner = select([make_bid("small", accuracy=0.65, cost=0.001),
                         make_bid("large", accuracy=0.92, cost=0.010)],
                        pref, tracker, "reasoning", "hard")
        assert winner.model_id == "large"


# ---------------------------------------------------------------------------
# 2. Latency penalty
# ---------------------------------------------------------------------------

class TestLatencyPenalty:
    def test_honest_bidder_no_penalty(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(20):
            tracker.record_latency("a", bid_latency_ms=300, actual_latency_ms=310)
        assert tracker.get_latency_reliability("a") > 0.95
        assert tracker.get_penalty_multiplier("a") < 1.06

    def test_chronic_overbidder_penalised(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(40):
            tracker.record_latency("a", bid_latency_ms=300, actual_latency_ms=900)
        assert tracker.get_latency_reliability("a") < 0.5
        assert tracker.get_penalty_multiplier("a") > 2.0

    def test_penalty_inflates_effective_cost_in_selector(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(40):
            tracker.record_latency("a", bid_latency_ms=200, actual_latency_ms=800)
        pref = make_pref(cost_weight=0.6, latency_weight=0.2, accuracy_weight=0.1, energy_weight=0.1)
        winner = select([make_bid("a", cost=0.001, latency_ms=200),
                         make_bid("b", cost=0.003, latency_ms=400)],
                        pref, tracker, "factual", "easy")
        assert winner.model_id == "b"

    def test_reliability_recovers_after_improvement(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(20):
            tracker.record_latency("a", bid_latency_ms=200, actual_latency_ms=600)
        bad = tracker.get_latency_reliability("a")
        for _ in range(40):
            tracker.record_latency("a", bid_latency_ms=200, actual_latency_ms=205)
        assert tracker.get_latency_reliability("a") > bad


# ---------------------------------------------------------------------------
# 3. Accuracy penalty
# ---------------------------------------------------------------------------

class TestAccuracyPenalty:
    def test_honest_bidder_no_discount(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(20):
            tracker.record_accuracy_bid("a", "math", "easy", bid_accuracy=0.80, judge_score=0.78)
        assert tracker.get_accuracy_discount("a", "math", "easy") > 0.95

    def test_chronic_overbidder_discounted(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(40):
            tracker.record_accuracy_bid("a", "math", "hard", bid_accuracy=0.90, judge_score=0.40)
        assert tracker.get_accuracy_discount("a", "math", "hard") < 0.6

    def test_overbidder_loses_in_selector(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for _ in range(40):
            tracker.record_accuracy_bid("a", "reasoning", "hard", bid_accuracy=0.92, judge_score=0.45)
        pref = make_pref(accuracy_weight=0.6, cost_weight=0.15, latency_weight=0.15, energy_weight=0.1)
        winner = select([make_bid("a", accuracy=0.92, cost=0.001),
                         make_bid("b", accuracy=0.75, cost=0.002)],
                        pref, tracker, "reasoning", "hard")
        assert winner.model_id == "b"


# ---------------------------------------------------------------------------
# 4. Domain floor
# ---------------------------------------------------------------------------

class TestDomainFloor:
    def test_floor_none_before_data(self, tmp_path):
        assert make_tracker(str(tmp_path)).get_domain_floor("a", "math") is None

    def test_floor_is_min_over_complexities(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for c, v in [("easy", 0.90), ("medium", 0.70), ("hard", 0.40)]:
            tracker.update_accuracy_prior("a", "math", c, v)
        assert abs(tracker.get_domain_floor("a", "math") - 0.40) < 0.05

    def test_live_floor_overrides_calibration(self, tmp_path):
        tracker = make_tracker(str(tmp_path))
        for c, v in [("easy", 0.85), ("medium", 0.60), ("hard", 0.45)]:
            tracker.update_accuracy_prior("a", "math", c, v)
        adapter = MagicMock(); adapter.model_id = "a"
        reg = ModelRegistry()
        reg.register(ModelConfig(adapter=adapter, domains=["math"],
                                 min_accuracy_capability={"math": 0.80, "_default": 0.80}))
        # Without tracker: calibration floor 0.80 → eligible for min=0.75
        assert len(reg.get_eligible("math", 0.75, reputation=None)) == 1
        # With tracker: live floor 0.45 → NOT eligible for min=0.75
        assert len(reg.get_eligible("math", 0.75, reputation=tracker)) == 0


# ---------------------------------------------------------------------------
# 5. Budget
# ---------------------------------------------------------------------------

class TestBudget:
    def setup_reg(self, tmp_path):
        return UserRegistry(path=os.path.join(str(tmp_path), "prefs.json"))

    def test_budget_deducted(self, tmp_path):
        reg = self.setup_reg(tmp_path)
        reg.set_budget("alice", UserBudget(remaining_tokens=1000, remaining_energy_j=500.0))
        reg.deduct_budget("alice", 100, 50.0)
        b = reg.get_budget("alice")
        assert b.remaining_tokens == 900
        assert abs(b.remaining_energy_j - 450.0) < 0.01

    def test_token_exhausted_raises_402(self, tmp_path):
        from semantic_router.user_registry import BudgetExhaustedError
        reg = self.setup_reg(tmp_path)
        reg.set_budget("bob", UserBudget(remaining_tokens=50, remaining_energy_j=9999.0))
        with pytest.raises(BudgetExhaustedError):
            reg.check_budget("bob", 100, 1.0)

    def test_energy_exhausted_raises_402(self, tmp_path):
        from semantic_router.user_registry import BudgetExhaustedError
        reg = self.setup_reg(tmp_path)
        reg.set_budget("carol", UserBudget(remaining_tokens=99999, remaining_energy_j=5.0))
        with pytest.raises(BudgetExhaustedError):
            reg.check_budget("carol", 10, 100.0)

    def test_budget_not_negative(self, tmp_path):
        reg = self.setup_reg(tmp_path)
        reg.set_budget("dave", UserBudget(remaining_tokens=10, remaining_energy_j=1.0))
        reg.deduct_budget("dave", 9999, 9999.0)
        b = reg.get_budget("dave")
        assert b.remaining_tokens == 0 and b.remaining_energy_j == 0.0


# ---------------------------------------------------------------------------
# 6. Preference resolution
# ---------------------------------------------------------------------------

class TestPreferenceResolution:
    def test_eco_mode_expands(self, tmp_path):
        from semantic_router.schemas import RequestSLA
        from semantic_router.config import MODE_PRESETS
        reg = UserRegistry(path=os.path.join(str(tmp_path), "p.json"))
        pref = reg.resolve_preference(None, RequestSLA(mode=RouterMode.ECO))
        assert abs(pref.energy_weight - MODE_PRESETS[RouterMode.ECO]["energy_weight"]) < 1e-6

    def test_per_request_overrides_profile(self, tmp_path):
        from semantic_router.schemas import RequestSLA
        reg = UserRegistry(path=os.path.join(str(tmp_path), "p.json"))
        reg.set_preference("alice", UserPreference(mode=RouterMode.COST, cost_weight=0.55,
                                                   latency_weight=0.15, accuracy_weight=0.20, energy_weight=0.10))
        pref = reg.resolve_preference("alice", RequestSLA(user_id="alice", latency_weight=0.50))
        assert abs(pref.latency_weight - 0.50) < 0.01

    def test_weights_sum_to_one(self, tmp_path):
        from semantic_router.schemas import RequestSLA
        reg = UserRegistry(path=os.path.join(str(tmp_path), "p.json"))
        sla = RequestSLA(cost_weight=0.8, latency_weight=0.8, accuracy_weight=0.8, energy_weight=0.8)
        pref = reg.resolve_preference(None, sla)
        total = pref.cost_weight + pref.latency_weight + pref.accuracy_weight + pref.energy_weight
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 7. Semantic analyzer
# ---------------------------------------------------------------------------

class TestAnalyzer:
    @pytest.fixture(scope="class")
    def analyzer(self):
        from semantic_router.analyzer import SemanticAnalyzer
        a = SemanticAnalyzer(); a.load(); return a

    def test_code_classified(self, analyzer):
        assert analyzer.analyze([{"role": "user", "content": "Write a Python function to sort a list."}]).domain == "code"

    def test_math_classified(self, analyzer):
        assert analyzer.analyze([{"role": "user", "content": "What is the integral of x squared?"}]).domain == "math"

    def test_easy_classified(self, analyzer):
        assert analyzer.analyze([{"role": "user", "content": "What is 2 plus 2?"}]).complexity == "easy"

    def test_hard_classified(self, analyzer):
        meta = analyzer.analyze([{"role": "user", "content": "Design a distributed fault-tolerant database with Byzantine fault tolerance."}])
        assert meta.complexity == "hard"

    def test_embedding_shape(self, analyzer):
        assert analyzer.analyze([{"role": "user", "content": "Hello"}]).embedding.shape == (384,)


# ---------------------------------------------------------------------------
# 8. Full routing flow (in-process, mocked adapters)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFullRoutingFlow:
    @pytest.fixture
    async def router(self, tmp_path):
        import semantic_router.main as m
        m.user_reg   = UserRegistry(path=str(tmp_path / "prefs.json"))
        m.reputation = ReputationTracker(path=str(tmp_path / "rep.json"))
        m.registry   = ModelRegistry()
        m.sampler    = MagicMock(); m.sampler.enqueue = MagicMock()

        def mock_adapter(model_id, cost, lat, acc, energy):
            a = MagicMock(spec=ModelAdapter)
            a.model_id = model_id
            a.efficiency_tokens_per_joule = 10.0
            a.bid = AsyncMock(return_value=make_bid(model_id, cost=cost, latency_ms=lat, accuracy=acc, energy_j=energy))
            a.complete = AsyncMock(return_value={
                "id": "t1", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hi from {model_id}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
            })
            return a

        for adapter, cap in [
            (mock_adapter("cheap-8b",   0.001, 300, 0.70, 5),  {"_default": 0.65}),
            (mock_adapter("accurate-70b", 0.010, 700, 0.92, 50), {"_default": 0.88}),
        ]:
            m.registry.register(ModelConfig(adapter=adapter,
                domains=["factual","code","math","reasoning","creative"],
                min_accuracy_capability=cap))

        async with AsyncClient(transport=ASGITransport(app=m.app), base_url="http://test") as client:
            yield client, m

    async def test_cheap_wins_with_loose_sla(self, router):
        client, _ = router
        r = await client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "extra_body": {"router": {"max_latency_ms": 5000, "min_accuracy": 0.60}},
        })
        assert r.status_code == 200
        assert r.headers["x-router-model"] == "cheap-8b"

    async def test_accurate_wins_with_strict_sla(self, router):
        client, _ = router
        r = await client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Prove the Pythagorean theorem."}],
            "extra_body": {"router": {"min_accuracy": 0.88}},
        })
        assert r.status_code == 200
        assert r.headers["x-router-model"] == "accurate-70b"

    async def test_503_when_impossible_sla(self, router):
        client, _ = router
        r = await client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "anything"}],
            "extra_body": {"router": {"min_accuracy": 0.999}},
        })
        assert r.status_code == 503

    async def test_402_when_budget_exhausted(self, router):
        client, module = router
        module.user_reg.set_budget("alice", UserBudget(remaining_tokens=1, remaining_energy_j=0.001))
        r = await client.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {"router": {"user_id": "alice"}},
        })
        assert r.status_code == 402

    async def test_router_headers_present(self, router):
        client, _ = router
        r = await client.post("/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "2+2"}],
        })
        assert r.status_code == 200
        for h in ("x-router-model", "x-router-charged-usd", "x-router-energy-j",
                  "x-router-bid-latency-ms", "x-router-actual-latency-ms"):
            assert h in r.headers

    async def test_user_preference_endpoints(self, router):
        client, _ = router
        r = await client.post("/users/bob/preference", json={
            "mode": "eco", "cost_weight": 0.15, "latency_weight": 0.20,
            "accuracy_weight": 0.25, "energy_weight": 0.40})
        assert r.status_code == 201
        assert (await client.get("/users/bob/preference")).json()["mode"] == "eco"

    async def test_budget_endpoints(self, router):
        client, _ = router
        r = await client.post("/users/carol/budget",
                              json={"remaining_tokens": 500000, "remaining_energy_j": 18000.0})
        assert r.status_code == 201
        assert (await client.get("/users/carol/budget")).json()["remaining_tokens"] == 500000
