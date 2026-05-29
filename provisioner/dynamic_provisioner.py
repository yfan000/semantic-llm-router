"""
dynamic_provisioner.py -- Dynamic model provisioning for Sophia.

Monitors router metrics and vLLM queue depth, then automatically spins up
or shuts down models based on latency, accuracy, and traffic conditions.

Spin-up triggers:
  - Queue depth > QUEUE_DEPTH_THRESHOLD: scale-out (throughput problem)
      1. Same model replica (horizontal scale) -- preferred
      2. Smaller/similar model (load balance) -- fallback
      NEVER spin up a larger model for queue overload (larger = slower per request)
  - Accuracy < ACCURACY_THRESHOLD: quality upgrade (accuracy problem)
      Spin up next higher-tier model; keep current for easy requests
  - No model for a domain: bootstrap

Spin-down triggers:
  - Model idle < MIN_REQUESTS_TO_STAY in IDLE_WINDOW_S and domains covered by others

Usage:
    # Start provisioner with initial models
    python provisioner/dynamic_provisioner.py \\
        --router-url http://localhost:8080 \\
        --initial-models qwen-7b,deepseek-r1-7b

    # No initial models -- provisioner bootstraps qwen-7b automatically
    python provisioner/dynamic_provisioner.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from collections import deque

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLL_INTERVAL_S       = 30
IDLE_WINDOW_S         = 300
MIN_REQUESTS_TO_STAY  = 5
QUEUE_DEPTH_THRESHOLD = 10
ACCURACY_THRESHOLD    = 0.65
STARTUP_WAIT_S        = 300
COOLDOWN_S            = 120

HF_HOME = "/eagle/UIC-HPC/yuping/hf_cache"

LATENCY_SLO_MS = {
    "factual:easy":    1000, "factual:medium":   2000, "factual:hard":   4000,
    "math:easy":       1000, "math:medium":      3000, "math:hard":      6000,
    "code:easy":       1500, "code:medium":      4000, "code:hard":      8000,
    "reasoning:easy":  1000, "reasoning:medium": 3000, "reasoning:hard": 6000,
    "creative:easy":   1500, "creative:medium":  5000, "creative:hard":  8000,
}

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    model_id:    str
    model_name:  str
    gpus_needed: int
    port:        int
    domains:     list[str]
    accuracy_tier: int    # 1=small/fast, 2=medium, 3=large/accurate
    min_accuracy_capability: dict[str, float] = field(default_factory=dict)
    efficiency_tokens_per_joule: float = 5.0


MODEL_CATALOG: dict[str, ModelSpec] = {
    "qwen-7b": ModelSpec(
        model_id="qwen-7b", model_name="Qwen/Qwen2.5-7B-Instruct",
        gpus_needed=1, port=8000,
        domains=["factual", "creative", "reasoning"], accuracy_tier=1,
        min_accuracy_capability={"factual": 0.70, "creative": 0.70, "reasoning": 0.68},
        efficiency_tokens_per_joule=13.0,
    ),
    "qwen-14b": ModelSpec(
        model_id="qwen-14b", model_name="Qwen/Qwen2.5-14B-Instruct",
        gpus_needed=2, port=8001,
        domains=["factual", "reasoning", "creative"], accuracy_tier=2,
        min_accuracy_capability={"factual": 0.80, "reasoning": 0.78, "creative": 0.78},
        efficiency_tokens_per_joule=8.0,
    ),
    "deepseek-r1-7b": ModelSpec(
        model_id="deepseek-r1-7b", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gpus_needed=1, port=8002,
        domains=["math", "reasoning"], accuracy_tier=2,
        min_accuracy_capability={"math": 0.82, "reasoning": 0.80},
        efficiency_tokens_per_joule=13.0,
    ),
    "coder-32b": ModelSpec(
        model_id="coder-32b", model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        gpus_needed=4, port=8003,
        domains=["code", "math", "reasoning"], accuracy_tier=3,
        min_accuracy_capability={"code": 0.90, "math": 0.88, "reasoning": 0.88},
        efficiency_tokens_per_joule=4.0,
    ),
    "qwen-32b": ModelSpec(
        model_id="qwen-32b", model_name="Qwen/Qwen2.5-32B-Instruct",
        gpus_needed=4, port=8004,
        domains=["factual", "reasoning", "creative", "math"], accuracy_tier=3,
        min_accuracy_capability={"_default": 0.85},
        efficiency_tokens_per_joule=4.0,
    ),
}

# UPGRADE_PATH: accuracy problem -> spin up higher-tier model
# Ordered by accuracy_tier ascending (try smallest upgrade first)
UPGRADE_PATH: dict[str, list[str]] = {
    "factual":   ["qwen-7b", "qwen-14b", "qwen-32b"],
    "reasoning": ["qwen-7b", "qwen-14b", "deepseek-r1-7b", "qwen-32b"],
    "math":      ["deepseek-r1-7b", "coder-32b"],
    "code":      ["coder-32b"],
    "creative":  ["qwen-7b", "qwen-14b", "qwen-32b"],
}

# THROUGHPUT_PATH: queue overload -> add capacity with smallest model first
# Ordered by gpus_needed ascending (fewest GPUs = fastest to scale, cheapest)
# IMPORTANT: do NOT add larger models here -- they process requests SLOWER
THROUGHPUT_PATH: dict[str, list[str]] = {
    "factual":   ["qwen-7b", "qwen-14b", "qwen-32b"],
    "reasoning": ["qwen-7b", "deepseek-r1-7b", "qwen-14b", "qwen-32b"],
    "math":      ["deepseek-r1-7b", "coder-32b"],
    "code":      ["coder-32b"],
    "creative":  ["qwen-7b", "qwen-14b", "qwen-32b"],
}

# ---------------------------------------------------------------------------
# GPU Pool
# ---------------------------------------------------------------------------

class GPUPool:
    def __init__(self, total_gpus: int = 8):
        self._free: list[int] = list(range(total_gpus))
        self._allocated: dict[str, list[int]] = {}

    def allocate(self, model_id: str, n: int) -> list[int] | None:
        if len(self._free) < n:
            return None
        gpus = self._free[:n]
        self._free = self._free[n:]
        self._allocated[model_id] = gpus
        return gpus

    def release(self, model_id: str) -> None:
        gpus = self._allocated.pop(model_id, [])
        self._free = sorted(self._free + gpus)
        log.info("GPU pool: released %s from %s | free=%s", gpus, model_id, self._free)

    def free_count(self) -> int:
        return len(self._free)

    def __repr__(self) -> str:
        return f"GPUPool(free={self._free}, allocated={self._allocated})"

# ---------------------------------------------------------------------------
# Running model state
# ---------------------------------------------------------------------------

@dataclass
class RunningModel:
    spec:          ModelSpec
    pid:           int
    gpus:          list[int]
    started_at:    float
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_request(self) -> None:
        self.request_times.append(time.monotonic())

    def requests_in_window(self, window_s: float) -> int:
        cutoff = time.monotonic() - window_s
        return sum(1 for t in self.request_times if t > cutoff)

# ---------------------------------------------------------------------------
# Dynamic Provisioner
# ---------------------------------------------------------------------------

class DynamicProvisioner:
    def __init__(
        self,
        router_url:    str,
        total_gpus:    int   = 8,
        poll_interval: float = POLL_INTERVAL_S,
        priors_path:   str   = "results/priors.json",
    ):
        self.router_url        = router_url.rstrip("/")
        self.gpu_pool          = GPUPool(total_gpus)
        self.poll_interval     = poll_interval
        self.priors_path       = priors_path
        self.running: dict[str, RunningModel] = {}
        self._last_action_time = 0.0
        self._env = {
            **os.environ,
            "HF_HOME": HF_HOME,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "CXX": "g++", "CC": "gcc",
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
            "LD_LIBRARY_PATH": (
                "/soft/compilers/openmpi/5.0.10/lib:"
                "/soft/libraries/ucx/1.20.0/lib:"
                + os.environ.get("LD_LIBRARY_PATH", "")
            ),
        }

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.router_url}{path}")
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(f"{self.router_url}{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def _delete(self, path: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.delete(f"{self.router_url}{path}")

    async def _vllm_queue(self, base_url: str) -> int:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base_url}/metrics")
                for line in r.text.splitlines():
                    if "vllm:num_requests_waiting" in line and not line.startswith("#"):
                        return int(float(line.split()[-1]))
        except Exception:
            pass
        return 0

    async def _reputation(self) -> dict:
        try:
            models = await self._get("/v1/models")
            result = {}
            for m in models.get("data", []):
                try:
                    result[m["id"]] = await self._get(f"/router/{m['id']}/reputation")
                except Exception:
                    pass
            return result
        except Exception:
            return {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        remaining = COOLDOWN_S - (time.monotonic() - self._last_action_time)
        if remaining > 0:
            log.info("Cooldown: %ds remaining", int(remaining))
            return True
        return False

    def _load_priors(self) -> dict:
        import json
        try:
            with open(self.priors_path) as f:
                return json.load(f)
        except Exception:
            return {}

    async def spin_up(self, model_id: str, reason: str) -> bool:
        if model_id in self.running:
            return True
        if self._in_cooldown():
            return False

        spec = MODEL_CATALOG.get(model_id)
        if not spec:
            log.error("Unknown model: %s", model_id)
            return False

        gpus = self.gpu_pool.allocate(model_id, spec.gpus_needed)
        if gpus is None:
            log.warning("Not enough GPUs for %s (need %d, free %d)",
                        model_id, spec.gpus_needed, self.gpu_pool.free_count())
            return False

        gpu_str     = ",".join(str(g) for g in gpus)
        master_port = 29500 + (spec.port - 8000)
        log_path    = os.path.expanduser(f"~/vllm_logs/vllm_{model_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log.info(">>> SPIN UP %s [%s] gpus=%s port=%d", model_id, reason, gpus, spec.port)
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                ["vllm", "serve", spec.model_name,
                 "--tensor-parallel-size", str(spec.gpus_needed),
                 "--port", str(spec.port)],
                env={**self._env, "CUDA_VISIBLE_DEVICES": gpu_str, "MASTER_PORT": str(master_port)},
                stdout=lf, stderr=lf,
            )

        self.running[model_id] = RunningModel(
            spec=spec, pid=proc.pid, gpus=gpus, started_at=time.monotonic()
        )
        self._last_action_time = time.monotonic()

        # Poll until ready (max STARTUP_WAIT_S)
        base_url = f"http://localhost:{spec.port}"
        deadline = time.monotonic() + STARTUP_WAIT_S
        log.info("  Waiting for %s (up to %ds)...", model_id, STARTUP_WAIT_S)
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            try:
                async with httpx.AsyncClient(timeout=3.0) as c:
                    if (await c.get(f"{base_url}/health")).status_code == 200:
                        log.info("  %s ready!", model_id)
                        break
            except Exception:
                pass
        else:
            log.error("  %s startup timeout", model_id)
            await self.spin_down(model_id, "startup_timeout")
            return False

        # Register with router
        priors = self._load_priors().get(model_id, {})
        try:
            await self._post("/router/register", {
                "model_id": spec.model_id, "model_name": spec.model_name,
                "backend": "vllm", "base_url": base_url,
                "domains": spec.domains,
                "min_accuracy_capability": spec.min_accuracy_capability,
                "accuracy_priors": priors,
                "efficiency_tokens_per_joule": spec.efficiency_tokens_per_joule,
                "skip_calibration": True,
            })
            log.info("  Registered %s with router", model_id)
        except Exception as e:
            log.error("  Registration failed: %s", e)
        return True

    async def spin_down(self, model_id: str, reason: str) -> bool:
        rm = self.running.get(model_id)
        if not rm:
            return False
        if reason != "startup_timeout" and self._in_cooldown():
            return False

        log.info(">>> SPIN DOWN %s [%s]", model_id, reason)
        try:
            await self._delete(f"/router/{model_id}")
        except Exception as e:
            log.warning("  Deregister failed: %s", e)

        try:
            os.kill(rm.pid, signal.SIGTERM)
            await asyncio.sleep(5)
            try:
                os.kill(rm.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

        self.gpu_pool.release(model_id)
        del self.running[model_id]
        self._last_action_time = time.monotonic()
        log.info("  %s stopped. Free GPUs: %d", model_id, self.gpu_pool.free_count())
        return True

    # ── Provisioning decisions ────────────────────────────────────────────────

    async def _try_scale_out(self, overloaded_id: str, queue_depth: int) -> bool:
        """Queue overload = THROUGHPUT problem. Need more capacity, not more accuracy.

        Decision priority:
          1. Same model replica (horizontal scale)
             - Identical latency profile, most efficient use of GPU
             - Picked when: free GPUs >= model.gpus_needed
          2. Smaller/similar-tier model for same domain (load balance)
             - Offloads some traffic to a parallel model
             - Picked when: no GPUs for replica, but smaller model fits
          3. NEVER larger model — larger models are SLOWER per request,
             adding a 32B when a 7B is overloaded makes the queue WORSE

        Example: deepseek-r1-7b (1 GPU, tier=2) queue=12
          Free GPUs=3 → Option 1: deepseek-r1-7b-replica (1 GPU) ← chosen
          Free GPUs=0 → Option 2: qwen-7b (1 GPU, tier=1) if any free next poll
          Both fail   → log warning, wait for next poll
        """
        spec = MODEL_CATALOG.get(overloaded_id)
        if not spec:
            return False

        # Option 1: horizontal scale — same model, new port (port + 10)
        replica_id = f"{overloaded_id}-replica"
        if replica_id not in MODEL_CATALOG:
            MODEL_CATALOG[replica_id] = ModelSpec(
                model_id=replica_id,
                model_name=spec.model_name,
                gpus_needed=spec.gpus_needed,
                port=spec.port + 10,
                domains=spec.domains,
                accuracy_tier=spec.accuracy_tier,
                min_accuracy_capability=spec.min_accuracy_capability,
                efficiency_tokens_per_joule=spec.efficiency_tokens_per_joule,
            )

        if replica_id not in self.running and self.gpu_pool.free_count() >= spec.gpus_needed:
            log.info("  Scale-out: horizontal replica of %s (queue=%d)", overloaded_id, queue_depth)
            return await self.spin_up(
                replica_id,
                reason=f"horizontal_scale:{overloaded_id}:queue={queue_depth}",
            )

        # Option 2: load balance — smallest available model for same domain
        # Use THROUGHPUT_PATH (ordered fewest GPUs first, same/lower tier only)
        for domain in spec.domains:
            for candidate_id in THROUGHPUT_PATH.get(domain, []):
                if candidate_id in (overloaded_id, replica_id):
                    continue
                if candidate_id in self.running:
                    continue
                candidate = MODEL_CATALOG[candidate_id]
                if candidate.accuracy_tier > spec.accuracy_tier:
                    continue  # skip larger models — slower per request
                if self.gpu_pool.free_count() < candidate.gpus_needed:
                    continue
                log.info("  Scale-out: load-balance with %s for domain=%s", candidate_id, domain)
                return await self.spin_up(
                    candidate_id,
                    reason=f"load_balance:{overloaded_id}:queue={queue_depth}",
                )

        log.warning("  Scale-out: no options (GPUs full or no suitable model)")
        return False

    async def _try_quality_upgrade(self, domain: str, reason: str, current_tier: int) -> bool:
        """Low accuracy = QUALITY problem. Need a smarter model, not more copies.

        Spins up the NEXT higher-tier model for the domain.
        The current model keeps running for easy/fast requests.
        The new model handles complex requests where accuracy matters.

        Example: qwen-7b (tier=1) factual accuracy=0.62 < 0.65
          UPGRADE_PATH factual: [qwen-7b, qwen-14b, qwen-32b]
          → qwen-7b: tier=1, skip (not higher than current_tier=1)
          → qwen-14b: tier=2 > 1, not running, has GPUs → spin up ✓
        """
        for model_id in UPGRADE_PATH.get(domain, []):
            spec = MODEL_CATALOG[model_id]
            if spec.accuracy_tier <= current_tier:
                continue  # must be strictly higher tier
            if model_id in self.running:
                continue
            if self.gpu_pool.free_count() < spec.gpus_needed:
                log.warning("  Quality upgrade: want %s but %d GPUs free (need %d)",
                            model_id, self.gpu_pool.free_count(), spec.gpus_needed)
                continue
            log.info("  Quality upgrade: %s (tier %d→%d) for domain=%s",
                     model_id, current_tier, spec.accuracy_tier, domain)
            return await self.spin_up(model_id, reason=reason)
        return False

    # ── Main policy loop ──────────────────────────────────────────────────────

    async def evaluate_and_act(self) -> None:
        reputation = await self._reputation()

        for model_id, rm in list(self.running.items()):
            spec     = rm.spec
            base_url = f"http://localhost:{spec.port}"

            # Spin-down: idle
            reqs = rm.requests_in_window(IDLE_WINDOW_S)
            if reqs < MIN_REQUESTS_TO_STAY:
                covered = all(
                    any(domain in MODEL_CATALOG[other].domains
                        for other in self.running if other != model_id)
                    for domain in spec.domains
                )
                if covered:
                    log.info("TRIGGER idle: %s (%d reqs/%ds, covered)", model_id, reqs, IDLE_WINDOW_S)
                    await self.spin_down(model_id, "idle")
                    continue

            # Spin-up: queue overload (throughput problem)
            q = await self._vllm_queue(base_url)
            if q > QUEUE_DEPTH_THRESHOLD:
                log.info("TRIGGER queue_overload: %s has %d waiting", model_id, q)
                await self._try_scale_out(model_id, q)

            # Spin-up: low accuracy (quality problem)
            rep    = reputation.get(model_id, {})
            priors = rep.get("accuracy_priors", {})
            for key, acc in priors.items():
                if acc < ACCURACY_THRESHOLD:
                    domain = key.split(":")[0]
                    log.info("TRIGGER low_accuracy: %s %s=%.3f", model_id, key, acc)
                    await self._try_quality_upgrade(
                        domain,
                        reason=f"low_accuracy:{model_id}:{key}",
                        current_tier=spec.accuracy_tier,
                    )

        # Bootstrap: nothing running
        if not self.running:
            log.info("TRIGGER bootstrap")
            await self.spin_up("qwen-7b", "bootstrap")

    # ── Run loop ──────────────────────────────────────────────────────────────

    async def run(self, initial_models: list[str] | None = None) -> None:
        log.info("Dynamic Provisioner started")
        log.info("GPU pool: %s | Poll: %ds | Cooldown: %ds",
                 self.gpu_pool, int(self.poll_interval), COOLDOWN_S)

        for model_id in (initial_models or []):
            if model_id in MODEL_CATALOG:
                await self.spin_up(model_id, "initial")
            else:
                log.warning("Unknown initial model: %s", model_id)

        while True:
            try:
                running_str = ", ".join(
                    f"{mid}(GPU{rm.gpus})" for mid, rm in self.running.items()
                ) or "none"
                log.info("Running: %s | Free GPUs: %d", running_str, self.gpu_pool.free_count())
                await self.evaluate_and_act()
            except Exception as e:
                log.error("Provisioner error: %s", e)
            await asyncio.sleep(self.poll_interval)

    async def shutdown(self) -> None:
        log.info("Shutting down")
        for model_id in list(self.running.keys()):
            await self.spin_down(model_id, "shutdown")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic vLLM provisioner for Sophia")
    parser.add_argument("--router-url",     default="http://localhost:8080")
    parser.add_argument("--total-gpus",     type=int,   default=8)
    parser.add_argument("--poll-interval",  type=float, default=POLL_INTERVAL_S)
    parser.add_argument("--priors-path",    default="results/priors.json")
    parser.add_argument("--initial-models", default="",
                        help="Comma-separated model IDs to start immediately")
    args = parser.parse_args()

    initial = [m.strip() for m in args.initial_models.split(",") if m.strip()]
    provisioner = DynamicProvisioner(
        router_url=args.router_url,
        total_gpus=args.total_gpus,
        poll_interval=args.poll_interval,
        priors_path=args.priors_path,
    )

    async def _run():
        try:
            await provisioner.run(initial_models=initial)
        except KeyboardInterrupt:
            pass
        finally:
            await provisioner.shutdown()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
