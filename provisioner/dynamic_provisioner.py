"""
dynamic_provisioner.py -- Dynamic model provisioning for Sophia.

Monitors router metrics and vLLM queue depth, then spins up/down models
based on accuracy and traffic signals. Decisions are fully data-driven from
results/priors.json -- no hardcoded upgrade/downgrade paths.

Low accuracy (domain:complexity < ACCURACY_THRESHOLD):
  Case 1: better model not running -> spin it up
  Case 2: better model already running -> refresh its router priors
          (stale priors cause router to route to the wrong model)
  Case 3: no better model exists -> log accuracy ceiling

Queue overload (vLLM waiting > QUEUE_DEPTH_THRESHOLD):
  Accuracy HIGH (>= HIGH_ACCURACY_THRESHOLD):
    -> Downgrade: find smallest model above MIN_ACCEPTABLE_ACCURACY
       (trade excess quality for throughput)
  Accuracy borderline:
    -> Horizontal scale: same model replica (preserve accuracy)

Spin-down: model idle < MIN_REQUESTS_TO_STAY in IDLE_WINDOW_S and domains covered.

Usage:
    python provisioner/dynamic_provisioner.py \\
        --router-url http://localhost:8080 \\
        --initial-models qwen-7b,deepseek-r1-7b
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

MIN_ACCEPTABLE_ACCURACY: float = 0.60   # floor: never use any model below this
HIGH_ACCURACY_THRESHOLD: float = 0.85   # above this, safe to downgrade for speed

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
    accuracy_tier: int
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

    async def _router_get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{self.router_url}{path}")
            r.raise_for_status()
            return r.json()

    async def _router_post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(f"{self.router_url}{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def _router_delete(self, path: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.delete(f"{self.router_url}{path}")

    async def _vllm_queue_depth(self, base_url: str) -> int:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base_url}/metrics")
                for line in r.text.splitlines():
                    if "vllm:num_requests_waiting" in line and not line.startswith("#"):
                        return int(float(line.split()[-1]))
        except Exception:
            pass
        return 0

    async def _get_router_reputation(self) -> dict:
        try:
            models = await self._router_get("/v1/models")
            result = {}
            for m in models.get("data", []):
                try:
                    result[m["id"]] = await self._router_get(f"/router/{m['id']}/reputation")
                except Exception:
                    pass
            return result
        except Exception:
            return {}

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
                env={**self._env, "CUDA_VISIBLE_DEVICES": gpu_str,
                     "MASTER_PORT": str(master_port)},
                stdout=lf, stderr=lf,
            )

        self.running[model_id] = RunningModel(
            spec=spec, pid=proc.pid, gpus=gpus, started_at=time.monotonic()
        )
        self._last_action_time = time.monotonic()

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

        priors = self._load_priors().get(model_id, {})
        try:
            await self._router_post("/router/register", {
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
            await self._router_delete(f"/router/{model_id}")
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

    # ── General data-driven provisioning decisions ────────────────────────────

    def _get_prior(self, model_id: str, domain: str, complexity: str,
                   all_priors: dict) -> float | None:
        priors = all_priors.get(model_id, {})
        return priors.get(f"{domain}:{complexity}") or priors.get(domain)

    def _candidates_not_running(self) -> list[ModelSpec]:
        return [
            spec for mid, spec in MODEL_CATALOG.items()
            if mid not in self.running and not mid.endswith("-replica")
        ]

    async def _refresh_router_priors(self, model_id: str, all_priors: dict) -> None:
        """Re-register a running model with updated accuracy priors.

        Called when a better model is already running but the router may have
        stale priors causing it to route to the worse model. Re-registering
        refreshes the bid accuracy without restarting the model.
        """
        rm = self.running.get(model_id)
        if not rm:
            return
        spec     = rm.spec
        priors   = all_priors.get(model_id, {})
        base_url = f"http://localhost:{spec.port}"
        try:
            await self._router_post("/router/register", {
                "model_id": spec.model_id, "model_name": spec.model_name,
                "backend": "vllm", "base_url": base_url,
                "domains": spec.domains,
                "min_accuracy_capability": spec.min_accuracy_capability,
                "accuracy_priors": priors,
                "efficiency_tokens_per_joule": spec.efficiency_tokens_per_joule,
                "skip_calibration": True,
            })
            log.info("  Refreshed router priors for %s", model_id)
        except Exception as e:
            log.warning("  Failed to refresh priors for %s: %s", model_id, e)

    async def _try_quality_upgrade(
        self,
        domain: str,
        complexity: str,
        current_accuracy: float,
        reason: str,
        all_priors: dict,
    ) -> bool:
        """Low accuracy in domain:complexity. Three cases:

        Case 1 -- Better model NOT running, fits in GPU budget:
          Spin it up. Router starts routing hard requests to it.

        Case 2 -- Better model IS already running:
          Router should use it, but may have stale accuracy priors.
          Re-register it with updated priors so bid selection reflects
          its true quality. This is a soft fix without spinning anything up.

        Case 3 -- No better model exists (accuracy ceiling):
          All models covering this domain are at or below current accuracy,
          or no GPU budget available. Log warning so operator can add a
          stronger model to MODEL_CATALOG.

        Selection for Cases 1 & 2:
          - Collect all models (running + not) covering domain with higher accuracy
          - Sort: best accuracy first, fewest GPUs as tiebreaker
          - Case 1 wins if any not-running candidate exists (prefers spin-up
            over refresh since fresh model may have better priors too)
        """
        better_running:     list[tuple[ModelSpec, float]] = []
        better_not_running: list[tuple[ModelSpec, float]] = []

        for mid, spec in MODEL_CATALOG.items():
            if mid.endswith("-replica"):
                continue
            if domain not in spec.domains:
                continue
            acc = self._get_prior(mid, domain, complexity, all_priors)
            if acc is None or acc <= current_accuracy or acc < MIN_ACCEPTABLE_ACCURACY:
                continue
            if mid in self.running:
                better_running.append((spec, acc))
            elif self.gpu_pool.free_count() >= spec.gpus_needed:
                better_not_running.append((spec, acc))

        better_running.sort(key=lambda x: (-x[1], x[0].gpus_needed))
        better_not_running.sort(key=lambda x: (-x[1], x[0].gpus_needed))

        # Case 1: spin up the best not-running model
        if better_not_running:
            best_spec, best_acc = better_not_running[0]
            log.info("  [Case 1 spin-up] %s acc=%.3f > %.3f for %s:%s",
                     best_spec.model_id, best_acc, current_accuracy, domain, complexity)
            return await self.spin_up(best_spec.model_id, reason=reason)

        # Case 2: better model already running -- refresh its router priors
        if better_running:
            best_spec, best_acc = better_running[0]
            log.warning(
                "  [Case 2 refresh] %s acc=%.3f > %.3f for %s:%s already running "
                "-- refreshing router priors (stale priors may cause wrong routing)",
                best_spec.model_id, best_acc, current_accuracy, domain, complexity
            )
            await self._refresh_router_priors(best_spec.model_id, all_priors)
            return True

        # Case 3: accuracy ceiling
        log.warning(
            "  [Case 3 ceiling] no model beats %.3f for %s:%s "
            "-- add a stronger model to MODEL_CATALOG",
            current_accuracy, domain, complexity
        )
        return False

    async def _try_scale_out(
        self,
        overloaded_id: str,
        queue_depth:   int,
        domain:        str,
        all_priors:    dict,
    ) -> bool:
        """Queue overload. Strategy chosen by current accuracy level.

        Strategy A (Downgrade) -- accuracy is HIGH >= HIGH_ACCURACY_THRESHOLD:
          We're over-provisioned on quality. Find the smallest model (fewest GPUs)
          covering the domain above MIN_ACCEPTABLE_ACCURACY floor.
          Smallest = fastest per request = drains queue most effectively.

        Strategy B (Horizontal scale) -- accuracy is borderline:
          Can't safely downgrade. Spin up same model on different port/GPUs.
          Doubles throughput with identical accuracy.
        """
        spec = MODEL_CATALOG.get(overloaded_id)
        if not spec:
            return False

        domain_priors = {
            k: v for k, v in all_priors.get(overloaded_id, {}).items()
            if k.startswith(domain + ":")
        }
        current_acc   = min(domain_priors.values()) if domain_priors else None
        accuracy_high = current_acc is not None and current_acc >= HIGH_ACCURACY_THRESHOLD

        if accuracy_high:
            candidates: list[tuple[ModelSpec, float]] = []
            for candidate in self._candidates_not_running():
                if domain not in candidate.domains:
                    continue
                if candidate.gpus_needed >= spec.gpus_needed:
                    continue
                if self.gpu_pool.free_count() < candidate.gpus_needed:
                    continue
                acc = self._get_prior(candidate.model_id, domain, "medium", all_priors)
                if acc is None or acc < MIN_ACCEPTABLE_ACCURACY:
                    continue
                candidates.append((candidate, acc))

            if candidates:
                candidates.sort(key=lambda x: (x[0].gpus_needed, -x[1]))
                best, best_acc = candidates[0]
                log.info("  [Strategy A downgrade] %s (%d GPUs, acc=%.3f) "
                         "current=%.3f is high -> trade quality for speed",
                         best.model_id, best.gpus_needed, best_acc, current_acc)
                return await self.spin_up(
                    best.model_id,
                    reason=f"downgrade:{overloaded_id}:q={queue_depth}"
                )
            log.info("  Strategy A: no smaller model, trying B")

        replica_id = f"{overloaded_id}-replica"
        if replica_id not in MODEL_CATALOG:
            MODEL_CATALOG[replica_id] = ModelSpec(
                model_id=replica_id, model_name=spec.model_name,
                gpus_needed=spec.gpus_needed, port=spec.port + 10,
                domains=spec.domains, accuracy_tier=spec.accuracy_tier,
                min_accuracy_capability=spec.min_accuracy_capability,
                efficiency_tokens_per_joule=spec.efficiency_tokens_per_joule,
            )

        if replica_id not in self.running and self.gpu_pool.free_count() >= spec.gpus_needed:
            log.info("  [Strategy B replica] %s queue=%d acc=%.3f",
                     overloaded_id, queue_depth, current_acc or 0)
            return await self.spin_up(
                replica_id, reason=f"horizontal_scale:{overloaded_id}:q={queue_depth}"
            )

        log.warning("  Scale-out: no options (GPUs full)")
        return False

    async def evaluate_and_act(self) -> None:
        reputation = await self._get_router_reputation()
        all_priors = self._load_priors()

        for model_id, rm in list(self.running.items()):
            spec     = rm.spec
            base_url = f"http://localhost:{spec.port}"

            reqs_in_window = rm.requests_in_window(IDLE_WINDOW_S)
            if reqs_in_window < MIN_REQUESTS_TO_STAY:
                covered = all(
                    any(other_id != model_id and
                        domain in MODEL_CATALOG[other_id].domains
                        for other_id in self.running)
                    for domain in spec.domains
                )
                if covered:
                    log.info("TRIGGER idle: %s %d reqs/%ds covered",
                             model_id, reqs_in_window, IDLE_WINDOW_S)
                    await self.spin_down(model_id, reason="idle")
                    continue

            queue_depth = await self._vllm_queue_depth(base_url)
            if queue_depth > QUEUE_DEPTH_THRESHOLD:
                log.info("TRIGGER queue_overload: %s has %d waiting", model_id, queue_depth)
                for domain in spec.domains:
                    if await self._try_scale_out(model_id, queue_depth, domain, all_priors):
                        break

            rep    = reputation.get(model_id, {})
            priors = rep.get("accuracy_priors", {})
            for key, acc in priors.items():
                if acc < ACCURACY_THRESHOLD:
                    parts      = key.split(":")
                    domain     = parts[0]
                    complexity = parts[1] if len(parts) > 1 else "medium"
                    log.info("TRIGGER low_accuracy: %s %s=%.3f", model_id, key, acc)
                    await self._try_quality_upgrade(
                        domain=domain, complexity=complexity,
                        current_accuracy=acc,
                        reason=f"low_accuracy:{model_id}:{key}",
                        all_priors=all_priors,
                    )

        if not self.running:
            log.info("TRIGGER bootstrap")
            await self.spin_up("qwen-7b", reason="bootstrap")

    async def run(self, initial_models: list[str] | None = None) -> None:
        log.info("Dynamic Provisioner started")
        log.info("GPU pool: %s | Poll: %ds | Cooldown: %ds",
                 self.gpu_pool, int(self.poll_interval), COOLDOWN_S)

        for model_id in (initial_models or []):
            if model_id in MODEL_CATALOG:
                await self.spin_up(model_id, reason="initial")
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
            await self.spin_down(model_id, reason="shutdown")


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
