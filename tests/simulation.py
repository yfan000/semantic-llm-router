"""
simulation.py — Load balancing simulation for the semantic LLM router.

Runs the real router logic (bidding, selection, reputation, preferences)
against simulated backends that maintain dynamic KV-cache load state.
No GPUs, no real HTTP — pure asyncio in-process simulation.

Usage:
    cd output
    python tests/simulation.py                        # default scenario
    python tests/simulation.py --scenario spike       # sudden burst
    python tests/simulation.py --scenario sla         # strict accuracy SLA
    python tests/simulation.py --scenario eco         # all eco-mode users
    python tests/simulation.py --scenario mixed_users # cost/eco/accuracy mix
    python tests/simulation.py --requests 500 --concurrency 20
"""
from __future__ import annotations
import argparse, asyncio, os, random, sys, time, tempfile
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from semantic_router.schemas import BidRequest, BidResponse, UserPreference, RouterMode
from semantic_router.registry import ModelRegistry, ModelConfig
from semantic_router.reputation_tracker import ReputationTracker
from semantic_router.selector import select, NoEligibleModelError


# ── Load multiplier ────────────────────────────────────────────────────────────────

def _load_multiplier(load: float) -> float:
    if load < 0.50: return 1.0
    if load < 0.75: return 1.25
    if load < 0.90: return 1.6
    return 2.0


OUTPUT_TOKENS = {
    ("factual","easy"):80,   ("factual","medium"):200,   ("factual","hard"):350,
    ("math","easy"):120,     ("math","medium"):280,       ("math","hard"):450,
    ("code","easy"):150,     ("code","medium"):350,       ("code","hard"):650,
    ("creative","easy"):250, ("creative","medium"):500,   ("creative","hard"):800,
    ("reasoning","easy"):180,("reasoning","medium"):380,  ("reasoning","hard"):600,
}


# ── Simulated backend ─────────────────────────────────────────────────────────────

@dataclass
class BackendConfig:
    model_id: str
    base_latency_ms: float
    input_rate: float
    output_rate: float
    efficiency_tok_per_j: float
    accuracy_by_domain: dict
    max_concurrent: int

    def accuracy(self, domain: str) -> float:
        return self.accuracy_by_domain.get(domain, self.accuracy_by_domain.get("_default", 0.70))

    def capability(self) -> dict:
        return {**self.accuracy_by_domain, "_default": min(self.accuracy_by_domain.values())}


class SimulatedBackend:
    """
    Stateful simulated backend. Tracks in-flight requests so load reflects
    real concurrency — the auction naturally shifts traffic away when overloaded.
    """
    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.model_id
        self.efficiency_tokens_per_joule = cfg.efficiency_tok_per_j
        self._active = 0
        self._lock = asyncio.Lock()
        self.requests_served = 0
        self.latencies_ms: list[float] = []
        self.costs: list[float] = []

    @property
    def load(self) -> float:
        return min(self._active / max(self.cfg.max_concurrent, 1), 1.0)

    async def bid(self, request: BidRequest) -> BidResponse:
        load = self.load
        mult = _load_multiplier(load)
        inp  = sum(len(m.get("content","").split()) * 1.3 for m in request.messages)
        out  = OUTPUT_TOKENS.get((request.domain, request.complexity), 300)
        cost = (inp * self.cfg.input_rate + out * self.cfg.output_rate) * mult
        return BidResponse(
            model_id=self.model_id,
            estimated_cost_usd=cost,
            estimated_latency_ms=int(self.cfg.base_latency_ms * (1.0 + load * 1.5)),
            estimated_accuracy=self.cfg.accuracy(request.domain),
            estimated_energy_j=out / self.cfg.efficiency_tok_per_j,
            efficiency_tokens_per_joule=self.cfg.efficiency_tok_per_j,
            current_load=load,
        )

    async def complete(self, messages: list[dict], **kwargs) -> tuple[dict, float, float]:
        async with self._lock:
            self._active += 1
        t0 = time.monotonic()
        try:
            actual_lat = self.cfg.base_latency_ms * (1.0 + self.load * 1.5)
            await asyncio.sleep(actual_lat / 1000)
            out = 50
            energy_j = out / self.cfg.efficiency_tok_per_j
            resp = {"choices": [{"message": {"role": "assistant", "content": f"[{self.model_id}] ok"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": out}}
            return resp, (time.monotonic() - t0) * 1000, energy_j
        finally:
            async with self._lock:
                self._active -= 1


# ── Request generator ───────────────────────────────────────────────────────────────

SAMPLE_QUERIES = {
    ("factual","easy"):    "What is the capital of France?",
    ("factual","hard"):    "Explain the geopolitical implications of the Bretton Woods collapse.",
    ("math","easy"):       "What is 15% of 240?",
    ("math","hard"):       "Prove that the sum of squares of two odd numbers cannot be a perfect square.",
    ("code","easy"):       "Write a Python function to add two numbers.",
    ("code","hard"):       "Implement a lock-free concurrent hash map in Python.",
    ("reasoning","medium"):"Compare and contrast microservices vs monolithic architecture.",
    ("creative","medium"): "Write a short story about a robot learning to cook.",
}

def random_query():
    domain, complexity = random.choice(list(SAMPLE_QUERIES.keys()))
    return [{"role": "user", "content": SAMPLE_QUERIES[(domain, complexity)]}], domain, complexity


def make_preference(scenario: str, user_idx: int) -> UserPreference:
    if scenario == "eco":
        return UserPreference(mode=RouterMode.ECO, accuracy_weight=0.25, cost_weight=0.15,
                              latency_weight=0.20, energy_weight=0.40)
    if scenario == "sla":
        return UserPreference(mode=RouterMode.ACCURACY, accuracy_weight=0.70, cost_weight=0.15,
                              latency_weight=0.10, energy_weight=0.05, min_accuracy=0.80)
    if scenario == "mixed_users":
        presets = [
            dict(mode=RouterMode.COST,     cost_weight=0.55, latency_weight=0.15, accuracy_weight=0.20, energy_weight=0.10),
            dict(mode=RouterMode.ECO,      accuracy_weight=0.25, cost_weight=0.15, latency_weight=0.20, energy_weight=0.40),
            dict(mode=RouterMode.ACCURACY, accuracy_weight=0.70, cost_weight=0.15, latency_weight=0.10, energy_weight=0.05),
        ]
        return UserPreference(**presets[user_idx % 3])
    return UserPreference(cost_weight=0.25, latency_weight=0.25, accuracy_weight=0.25, energy_weight=0.25)


# ── Single request ─────────────────────────────────────────────────────────────────

@dataclass
class Result:
    winner: str
    actual_latency_ms: float
    bid_latency_ms: int
    charged_usd: float
    energy_j: float
    domain: str
    complexity: str
    sla_violated: bool = False


async def simulate_request(req_id, backends, registry, reputation, scenario) -> Result | None:
    messages, domain, complexity = random_query()
    pref = make_preference(scenario, req_id)
    eligible = registry.get_eligible(domain, pref.min_accuracy, reputation)
    if not eligible:
        return Result("", 0, 0, 0, 0, domain, complexity, sla_violated=True)

    bid_req = BidRequest(messages=messages, complexity=complexity, domain=domain,
                         query_embedding=[], preference=pref)

    async def safe_bid(b):
        try: return await asyncio.wait_for(b.bid(bid_req), timeout=0.2)
        except: return None

    bids = [b for b in await asyncio.gather(*[safe_bid(b) for b in eligible]) if b]
    if not bids: return None

    try:
        winning_bid = select(bids, pref, reputation, domain, complexity)
    except NoEligibleModelError:
        return Result("", 0, 0, 0, 0, domain, complexity, sla_violated=True)

    backend = backends[winning_bid.model_id]
    _, actual_lat, energy_j = await backend.complete(messages)
    reputation.record_latency(winning_bid.model_id, winning_bid.estimated_latency_ms, int(actual_lat))
    backend.requests_served += 1
    backend.latencies_ms.append(actual_lat)
    backend.costs.append(winning_bid.estimated_cost_usd)

    return Result(winning_bid.model_id, actual_lat, winning_bid.estimated_latency_ms,
                  winning_bid.estimated_cost_usd, energy_j, domain, complexity)


# ── Load sampler ───────────────────────────────────────────────────────────────────

async def sample_loads(backends, samples, interval, stop):
    while not stop.is_set():
        samples.append({b.model_id: b.load for b in backends.values()})
        await asyncio.sleep(interval)


# ── ASCII helpers ─────────────────────────────────────────────────────────────────

def bar(f, w=30): return "█" * int(f*w) + "░" * (w - int(f*w))
def p50(v): return sorted(v)[len(v)//2] if v else 0
def p95(v): return sorted(v)[int(len(v)*.95)] if v else 0


# ── Main simulation ─────────────────────────────────────────────────────────────────

async def run(n_requests, concurrency, scenario, arrival):
    W = 60
    print(f"\n{'='*W}")
    print(f"  Semantic Router Load Balancing Simulation")
    print(f"{'='*W}")
    print(f"  Scenario: {scenario}  |  Requests: {n_requests}  |  Concurrency: {concurrency}  |  Arrival: {arrival}")
    print(f"{'='*W}\n")

    cfgs = [
        BackendConfig("llama-3.1-8b",  200, 2e-7, 4e-7, 26.7,
                      {"factual":0.82,"math":0.50,"code":0.75,"reasoning":0.65,"creative":0.72,"_default":0.50}, 32),
        BackendConfig("llama-3.1-13b", 400, 6e-7, 1.2e-6, 10.0,
                      {"factual":0.88,"math":0.68,"code":0.82,"reasoning":0.75,"creative":0.80,"_default":0.68}, 20),
        BackendConfig("llama-3.1-70b", 800, 1e-6, 2e-6,  2.1,
                      {"factual":0.95,"math":0.88,"code":0.91,"reasoning":0.90,"creative":0.89,"_default":0.88}, 8),
    ]
    backends = {c.model_id: SimulatedBackend(c) for c in cfgs}

    print(f"  {'Model':<22} {'Base latency':>14} {'Eff (tok/J)':>12} {'Max concurrent':>16}")
    print(f"  {'-'*66}")
    for c in cfgs:
        print(f"  {c.model_id:<22} {c.base_latency_ms:>12}ms  {c.efficiency_tok_per_j:>10.1f}   {c.max_concurrent:>14}")

    tmp = tempfile.mkdtemp()
    reputation = ReputationTracker(path=os.path.join(tmp, "rep.json"))
    registry   = ModelRegistry()
    for c in cfgs:
        registry.register(ModelConfig(adapter=backends[c.model_id], domains=list(c.accuracy_by_domain.keys()),
                                      min_accuracy_capability=c.capability()))

    load_samples: list[dict] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_loads(backends, load_samples, 2.0, stop))

    results: list[Result] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0_wall = time.monotonic()

    async def run_req(i):
        nonlocal done
        if arrival == "spike":
            if i > n_requests * 0.2: await asyncio.sleep(random.uniform(0, 0.3))
        elif arrival == "ramp":
            await asyncio.sleep(i * 0.08)
        else:
            await asyncio.sleep(random.expovariate(max(concurrency / 2, 1)))
        async with sem:
            r = await simulate_request(i, backends, registry, reputation, scenario)
            if r: results.append(r)
            done += 1
            if done % max(n_requests // 20, 1) == 0:
                p = done / n_requests
                print(f"\r  Progress: [{bar(p, 40)}] {done}/{n_requests}", end="", flush=True)

    print("")
    await asyncio.gather(*[run_req(i) for i in range(n_requests)])
    wall = time.monotonic() - t0_wall
    stop.set(); sampler.cancel()
    print(f"\r  Progress: [{bar(1.0, 40)}] {n_requests}/{n_requests}  ({wall:.1f}s)\n")

    valid = [r for r in results if not r.sla_violated]
    sla_v = sum(1 for r in results if r.sla_violated)
    total = len(results)
    if not valid:
        print("  No valid results."); return

    # Routing distribution
    print(f"{'='*W}")
    print("  ROUTING DISTRIBUTION")
    print(f"{'='*W}")
    counts = {c.model_id: 0 for c in cfgs}
    for r in valid: counts[r.winner] = counts.get(r.winner, 0) + 1
    for mid, cnt in counts.items():
        f = cnt / max(total, 1)
        print(f"  {mid:<22} {bar(f)}  {100*cnt//max(total,1):3}%  ({cnt})")

    # Load over time
    print(f"\n{'='*W}")
    print("  LOAD OVER TIME  (last 12 samples @ 2s intervals)")
    print(f"{'='*W}")
    tail = load_samples[-12:] if len(load_samples) >= 12 else load_samples
    for c in cfgs:
        vals = "  ".join(f"{s.get(c.model_id,0):.2f}" for s in tail)
        peak = max((s.get(c.model_id,0) for s in load_samples), default=0)
        avg_ = mean(s.get(c.model_id,0) for s in load_samples) if load_samples else 0
        print(f"  {c.model_id:<22} {vals}")
        print(f"  {'':22} peak={peak:.2f}  avg={avg_:.2f}")

    # Latency percentiles
    print(f"\n{'='*W}")
    print("  LATENCY PERCENTILES (actual ms)")
    print(f"{'='*W}")
    for c in cfgs:
        b = backends[c.model_id]
        if b.latencies_ms:
            print(f"  {c.model_id:<22} P50={p50(b.latencies_ms):.0f}ms  P95={p95(b.latencies_ms):.0f}ms  mean={mean(b.latencies_ms):.0f}ms")
        else:
            print(f"  {c.model_id:<22} (no requests)")

    # Cost & energy
    total_cost   = sum(r.charged_usd for r in valid)
    total_energy = sum(r.energy_j for r in valid)
    always_70_cost   = sum(OUTPUT_TOKENS.get((r.domain,r.complexity),300) * cfgs[-1].output_rate for r in valid)
    always_70_energy = sum(OUTPUT_TOKENS.get((r.domain,r.complexity),300) / cfgs[-1].efficiency_tok_per_j for r in valid)

    print(f"\n{'='*W}")
    print("  COST & ENERGY SUMMARY")
    print(f"{'='*W}")
    print(f"  Total cost:      ${total_cost:.4f}  (vs always-70b: ${always_70_cost:.4f})")
    if always_70_cost > 0:
        print(f"  Cost savings:    {(1-total_cost/always_70_cost)*100:.1f}% cheaper than always routing to 70B")
    print(f"  Total energy:    {total_energy:.1f} J  (vs always-70b: {always_70_energy:.1f} J)")
    if always_70_energy > 0:
        print(f"  Energy savings:  {(1-total_energy/always_70_energy)*100:.1f}% less energy")

    # Load balancing quality
    print(f"\n{'='*W}")
    print("  LOAD BALANCING QUALITY")
    print(f"{'='*W}")
    dyn_events = sum(1 for s in load_samples for v in s.values() if v >= 0.75)
    print(f"  Dynamic pricing events (any backend load ≥0.75): {dyn_events}")
    print(f"  SLA violations (no eligible model):             {sla_v}/{total}")
    print(f"  Throughput:                                      {len(valid)/wall:.1f} req/s")
    print(f"  Wall time:                                       {wall:.1f}s")

    # Routing by domain
    print(f"\n{'='*W}")
    print("  ROUTING BY DOMAIN  (shows which model handles each domain)")
    print(f"{'='*W}")
    for domain in sorted({r.domain for r in valid}):
        dreqs = [r for r in valid if r.domain == domain]
        dc = {}
        for r in dreqs: dc[r.winner] = dc.get(r.winner, 0) + 1
        parts = "  ".join(f"{m.split('.')[-1]}:{c}" for m, c in sorted(dc.items(), key=lambda x: -x[1]))
        print(f"  {domain:<14} ({len(dreqs):3} reqs)  {parts}")

    print(f"\n{'='*W}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests",    type=int, default=200)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--scenario",    choices=["default","spike","sla","eco","mixed_users"], default="default")
    p.add_argument("--arrival",     choices=["constant","spike","ramp"], default="constant")
    args = p.parse_args()
    asyncio.run(run(args.requests, args.concurrency, args.scenario,
                    "spike" if args.scenario == "spike" else args.arrival))


if __name__ == "__main__":
    main()
