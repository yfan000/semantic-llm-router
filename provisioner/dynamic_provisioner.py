"""
dynamic_provisioner.py -- Dynamic model provisioning for Sophia.

Monitors router metrics and vLLM queue depth, then spins up/down models
based on accuracy and traffic signals. Decisions are fully data-driven from
results/priors.json -- no hardcoded upgrade/downgrade paths.

Low accuracy (domain:complexity < ACCURACY_THRESHOLD):
  Case 1: better model not running -> spin up MINIMUM SUFFICIENT model
  Case 2: better model already running -> refresh its router priors
  Case 3: no better model exists -> log accuracy ceiling

Overload detection (_vllm_overloaded) -- THREE signals:
  1. num_requests_waiting > QUEUE_DEPTH_THRESHOLD  (classic waiting queue)
  2. num_requests_running > RUNNING_THRESHOLD       (continuous batch full)
  3. kv_cache_usage_perc > KV_CACHE_THRESHOLD      (GPU memory pressure)
  vLLM continuous batching keeps waiting=0 even under heavy load.
  running and kv_cache are better indicators of actual overload.

Process health check: HTTP GET /health (not PID signal).
TTCA-aware spin-up gate: only spin up if E[TTCA] improvement justifies GPU cost.

Usage:
    python provisioner/dynamic_provisioner.py \\
        --router-url http://localhost:8080 \\
        --router-mode ttca \\
        --initial-models qwen-7b
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
QUEUE_DEPTH_THRESHOLD = 10      # num_requests_waiting trigger
RUNNING_THRESHOLD     = 20      # num_requests_running trigger (continuous batching)
KV_CACHE_THRESHOLD    = 0.70    # kv_cache_usage_perc trigger (0-1)
ACCURACY_THRESHOLD    = 0.65
STARTUP_WAIT_S        = 300
COOLDOWN_S            = 120

MIN_ACCEPTABLE_ACCURACY: float = 0.60
HIGH_ACCURACY_THRESHOLD: float = 0.85
ROUTER_MODE:             str   = "accuracy"
TTCA_TARGET_MS:    float | None = 3000.0
TTCA_MIN_IMPROVEMENT:    float  = 0.20

ESTIMATED_TOKENS_PER_SEC: dict[int, float] = {
    1: 2500.0, 2: 1800.0, 4: 900.0, 8: 500.0,
}

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
    expected_tokens_per_sec: float = 1000.0


MODEL_CATALOG: dict[str, ModelSpec] = {
    "qwen-7b": ModelSpec(
        model_id="qwen-7b", model_name="Qwen/Qwen2.5-7B-Instruct",
        gpus_needed=1, port=8000,
        domains=["factual", "creative", "reasoning"], accuracy_tier=1,
        min_accuracy_capability={"factual": 0.70, "creative": 0.70, "reasoning": 0.68},
        efficiency_tokens_per_joule=13.0, expected_tokens_per_sec=2800.0,
    ),
    "qwen-14b": ModelSpec(
        model_id="qwen-14b", model_name="Qwen/Qwen2.5-14B-Instruct",
        gpus_needed=2, port=8001,
        domains=["factual", "reasoning", "creative"], accuracy_tier=2,
        min_accuracy_capability={"factual": 0.80, "reasoning": 0.78, "creative": 0.78},
        efficiency_tokens_per_joule=8.0, expected_tokens_per_sec=1600.0,
    ),
    "deepseek-r1-7b": ModelSpec(
        model_id="deepseek-r1-7b", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gpus_needed=1, port=8002,
        domains=["math", "reasoning"], accuracy_tier=2,
        min_accuracy_capability={"math": 0.82, "reasoning": 0.80},
        efficiency_tokens_per_joule=13.0, expected_tokens_per_sec=1200.0,
    ),
    "coder-32b": ModelSpec(
        model_id="coder-32b", model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        gpus_needed=4, port=8003,
        domains=["code", "math", "reasoning"], accuracy_tier=3,
        min_accuracy_capability={"code": 0.90, "math": 0.88, "reasoning": 0.88},
        efficiency_tokens_per_joule=4.0, expected_tokens_per_sec=800.0,
    ),
    "qwen-32b": ModelSpec(
        model_id="qwen-32b", model_name="Qwen/Qwen2.5-32B-Instruct",
        gpus_needed=4, port=8004,
        domains=["factual", "reasoning", "creative", "math"], accuracy_tier=3,
        min_accuracy_capability={"_default": 0.85},
        efficiency_tokens_per_joule=4.0, expected_tokens_per_sec=750.0,
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
    spec:            ModelSpec
    pid:             int
    gpus:            list[int]
    started_at:      float
    request_times:   deque = field(default_factory=lambda: deque(maxlen=1000))
    latency_samples: list  = field(default_factory=list)

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

    async def _vllm_overloaded(self, base_url: str) -> tuple[bool, str]:
        """Check if vLLM is overloaded using three signals.

        vLLM continuous batching keeps num_requests_waiting=0 even under heavy
        load because all requests go to "running" state immediately.
        We check three signals to detect real overload:

          1. num_requests_waiting > QUEUE_DEPTH_THRESHOLD  (classic queue)
          2. num_requests_running > RUNNING_THRESHOLD      (batch full)
          3. kv_cache_usage_perc  > KV_CACHE_THRESHOLD    (GPU memory full)
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base_url}/metrics")
            metrics: dict[str, float] = {}
            for line in r.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                key = line.split("{")[0].split()[0]
                try:
                    metrics[key] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass

            waiting  = metrics.get("vllm:num_requests_waiting", 0.0)
            running  = metrics.get("vllm:num_requests_running",  0.0)
            kv_cache = metrics.get("vllm:kv_cache_usage_perc",   0.0)

            log.debug("  metrics: waiting=%.0f running=%.0f kv=%.2f",
                      waiting, running, kv_cache)

            if waiting > QUEUE_DEPTH_THRESHOLD:
                return True, f"waiting={waiting:.0f}>{QUEUE_DEPTH_THRESHOLD}"
            if running > RUNNING_THRESHOLD:
                return True, f"running={running:.0f}>{RUNNING_THRESHOLD}"
            if kv_cache > KV_CACHE_THRESHOLD:
                return True, f"kv_cache={kv_cache:.2f}>{KV_CACHE_THRESHOLD}"
        except Exception:
            pass
        return False, ""

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

    async def _check_dead_processes(self) -> None:
        for model_id, rm in list(self.running.items()):
            base_url = f"http://localhost:{rm.spec.port}"
            alive = False
            try:
                async with httpx.AsyncClient(timeout=5.0) as c:
                    alive = (await c.get(f"{base_url}/health")).status_code == 200
            except Exception:
                alive = False
            if not alive:
                log.warning("DETECTED dead: %s -- cleaning up", model_id)
                try:
                    await self._router_delete(f"/router/{model_id}")
                except Exception:
                    pass
                try:
                    os.kill(rm.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.gpu_pool.release(model_id)
                del self.running[model_id]
                log.warning("  Removed %s. Free GPUs: %d", model_id, self.gpu_pool.free_count())

    def _estimate_latency_ms(self, model_id: str, output_tokens: int = 300) -> float:
        import json
        try:
            rep_path = os.path.expanduser("~/semantic-llm-router/model_reputation.json")
            if os.path.exists(rep_path):
                with open(rep_path) as f:
                    rep = json.load(f)
                hist = rep.get(model_id, {}).get("avg_latency_ms")
                if hist is not None:
                    return float(hist)
        except Exception:
            pass
        spec = MODEL_CATALOG.get(model_id)
        if spec and spec.expected_tokens_per_sec > 0:
            return output_tokens / spec.expected_tokens_per_sec * 1000
        gpus = spec.gpus_needed if spec else 1
        return output_tokens / ESTIMATED_TOKENS_PER_SEC.get(gpus, 1000.0) * 1000

    def _compute_effective_ttca(self, model_id: str, all_priors: dict) -> float | None:
        rm = self.running.get(model_id)
        if not rm or len(rm.latency_samples) < 10:
            return None
        samples    = rm.latency_samples
        total_lat  = sum(lat for lat, _ in samples)
        resolved_1 = sum(1 for _, att in samples if att == 1)
        res_rate   = resolved_1 / len(samples)
        return float("inf") if res_rate < 0.01 else (total_lat / len(samples)) / res_rate

    def _estimate_ttca_with_candidate(
        self, current_models: list[tuple[float, float]],
        candidate_lat: float, candidate_acc: float,
    ) -> float:
        all_m = sorted(current_models + [(candidate_lat, candidate_acc)],
                       key=lambda x: x[0] / max(x[1], 0.01))
        e, pw, cl = 0.0, 1.0, 0.0
        for lat, acc in all_m:
            cl += lat; e += pw * acc * cl; pw *= (1.0 - acc)
        return e + pw * cl

    def _should_spin_up_for_ttca(
        self, current_e_ttca: float | None,
        candidate_lat: float, candidate_acc: float,
        current_models: list[tuple[float, float]],
    ) -> bool:
        if TTCA_TARGET_MS is None or current_e_ttca is None:
            return True
        if current_e_ttca <= TTCA_TARGET_MS:
            return False
        new_e = self._estimate_ttca_with_candidate(current_models, candidate_lat, candidate_acc)
        return (current_e_ttca - new_e) / current_e_ttca >= TTCA_MIN_IMPROVEMENT

    def _get_prior(self, model_id: str, domain: str, complexity: str,
                   all_priors: dict) -> float | None:
        priors = all_priors.get(model_id, {})
        return priors.get(f"{domain}:{complexity}") or priors.get(domain)

    def _candidates_not_running(self) -> list[ModelSpec]:
        return [spec for mid, spec in MODEL_CATALOG.items()
                if mid not in self.running and not mid.endswith("-replica")]

    async def _refresh_router_priors(self, model_id: str, all_priors: dict) -> None:
        rm = self.running.get(model_id)
        if not rm:
            return
        spec, priors = rm.spec, all_priors.get(model_id, {})
        try:
            await self._router_post("/router/register", {
                "model_id": spec.model_id, "model_name": spec.model_name,
                "backend": "vllm", "base_url": f"http://localhost:{spec.port}",
                "domains": spec.domains,
                "min_accuracy_capability": spec.min_accuracy_capability,
                "accuracy_priors": priors,
                "efficiency_tokens_per_joule": spec.efficiency_tokens_per_joule,
                "skip_calibration": True,
            })
        except Exception as e:
            log.warning("  Refresh failed: %s", e)

    async def _try_quality_upgrade(
        self, domain: str, complexity: str, current_accuracy: float,
        reason: str, all_priors: dict,
    ) -> bool:
        better_running:     list[tuple[ModelSpec, float]] = []
        better_not_running: list[tuple[ModelSpec, float]] = []

        for mid, spec in MODEL_CATALOG.items():
            if mid.endswith("-replica") or domain not in spec.domains:
                continue
            acc = self._get_prior(mid, domain, complexity, all_priors)
            if acc is None or acc <= current_accuracy or acc < MIN_ACCEPTABLE_ACCURACY:
                continue
            if mid in self.running:
                better_running.append((spec, acc))
            elif self.gpu_pool.free_count() >= spec.gpus_needed:
                better_not_running.append((spec, acc))

        def _sort_key(item: tuple[ModelSpec, float]) -> tuple:
            spec, acc = item
            if ROUTER_MODE == "ttca":
                return (self._estimate_latency_ms(spec.model_id) / max(acc, 0.01), spec.gpus_needed)
            elif ROUTER_MODE == "cost":
                return (spec.gpus_needed, -acc)
            crosses = acc >= ACCURACY_THRESHOLD
            return (0, spec.gpus_needed, -acc) if crosses else (1, -acc, spec.gpus_needed)

        better_running.sort(key=_sort_key)
        better_not_running.sort(key=_sort_key)

        if better_not_running:
            best_spec, best_acc = better_not_running[0]
            current_models = [
                (self._estimate_latency_ms(mid), a)
                for mid, rm2 in self.running.items()
                if (a := self._get_prior(mid, domain, complexity, all_priors))
                and a >= MIN_ACCEPTABLE_ACCURACY and domain in rm2.spec.domains
            ]
            cand_lat       = self._estimate_latency_ms(best_spec.model_id)
            current_e_ttca = next(
                (t for mid in self.running
                 if (t := self._compute_effective_ttca(mid, all_priors)) is not None), None
            )
            if self._should_spin_up_for_ttca(current_e_ttca, cand_lat, best_acc, current_models):
                phase = "phase1" if best_acc >= ACCURACY_THRESHOLD else "phase2"
                log.info("  [Case 1 %s] %s acc=%.3f for %s:%s", phase, best_spec.model_id,
                         best_acc, domain, complexity)
                return await self.spin_up(best_spec.model_id, reason=reason)

        if better_running:
            best_spec, best_acc = better_running[0]
            log.warning("  [Case 2 refresh] %s acc=%.3f for %s:%s",
                        best_spec.model_id, best_acc, domain, complexity)
            await self._refresh_router_priors(best_spec.model_id, all_priors)
            return True

        log.warning("  [Case 3 ceiling] no model beats %.3f for %s:%s",
                    current_accuracy, domain, complexity)
        return False

    async def _try_scale_out(
        self, overloaded_id: str, queue_depth: int, domain: str, all_priors: dict
    ) -> bool:
        spec = MODEL_CATALOG.get(overloaded_id)
        if not spec:
            return False

        domain_priors = {k: v for k, v in all_priors.get(overloaded_id, {}).items()
                         if k.startswith(domain + ":")}
        current_acc   = min(domain_priors.values()) if domain_priors else None
        accuracy_high = current_acc is not None and current_acc >= HIGH_ACCURACY_THRESHOLD

        if accuracy_high:
            candidates = [
                (c, a) for c in self._candidates_not_running()
                if domain in c.domains and c.gpus_needed < spec.gpus_needed
                and self.gpu_pool.free_count() >= c.gpus_needed
                and (a := self._get_prior(c.model_id, domain, "medium", all_priors))
                and a >= MIN_ACCEPTABLE_ACCURACY
            ]
            if candidates:
                candidates.sort(key=lambda x: (x[0].gpus_needed, -x[1]))
                best, best_acc = candidates[0]
                log.info("  [Downgrade] %s (%d GPU)", best.model_id, best.gpus_needed)
                return await self.spin_up(best.model_id,
                                          reason=f"downgrade:{overloaded_id}:q={queue_depth}")

        replica_id = f"{overloaded_id}-replica"
        if replica_id not in MODEL_CATALOG:
            MODEL_CATALOG[replica_id] = ModelSpec(
                model_id=replica_id, model_name=spec.model_name,
                gpus_needed=spec.gpus_needed, port=spec.port + 10,
                domains=spec.domains, accuracy_tier=spec.accuracy_tier,
                min_accuracy_capability=spec.min_accuracy_capability,
                efficiency_tokens_per_joule=spec.efficiency_tokens_per_joule,
                expected_tokens_per_sec=spec.expected_tokens_per_sec,
            )

        if replica_id not in self.running and self.gpu_pool.free_count() >= spec.gpus_needed:
            log.info("  [Replica] %s", overloaded_id)
            return await self.spin_up(replica_id,
                                      reason=f"horizontal_scale:{overloaded_id}:q={queue_depth}")

        log.warning("  Scale-out: no options (GPUs full)")
        return False

    async def evaluate_and_act(self) -> None:
        await self._check_dead_processes()
        reputation = await self._get_router_reputation()
        all_priors = self._load_priors()

        for model_id, rm in list(self.running.items()):
            spec, base_url = rm.spec, f"http://localhost:{rm.spec.port}"

            reqs = rm.requests_in_window(IDLE_WINDOW_S)
            if reqs < MIN_REQUESTS_TO_STAY:
                covered = all(
                    any(oid != model_id and domain in MODEL_CATALOG[oid].domains
                        for oid in self.running)
                    for domain in spec.domains
                )
                if covered:
                    log.info("TRIGGER idle: %s %d/%ds", model_id, reqs, IDLE_WINDOW_S)
                    await self.spin_down(model_id, reason="idle")
                    continue

            overloaded, reason = await self._vllm_overloaded(base_url)
            if overloaded:
                log.info("TRIGGER overload: %s [%s]", model_id, reason)
                for domain in spec.domains:
                    if await self._try_scale_out(model_id, 11, domain, all_priors):
                        break

            rep = reputation.get(model_id, {})
            for key, acc in rep.get("accuracy_priors", {}).items():
                if acc < ACCURACY_THRESHOLD:
                    parts = key.split(":")
                    log.info("TRIGGER accuracy: %s %s=%.3f", model_id, key, acc)
                    await self._try_quality_upgrade(
                        domain=parts[0],
                        complexity=parts[1] if len(parts) > 1 else "medium",
                        current_accuracy=acc,
                        reason=f"low_accuracy:{model_id}:{key}",
                        all_priors=all_priors,
                    )

        if not self.running:
            log.info("TRIGGER bootstrap")
            await self.spin_up("qwen-7b", reason="bootstrap")

    async def run(self, initial_models: list[str] | None = None) -> None:
        log.info("Provisioner started | mode=%s | TTCA target=%s",
                 ROUTER_MODE, f"{TTCA_TARGET_MS:.0f}ms" if TTCA_TARGET_MS else "disabled")
        log.info("GPU pool: %s | Poll: %ds | Cooldown: %ds",
                 self.gpu_pool, int(self.poll_interval), COOLDOWN_S)

        for model_id in (initial_models or []):
            if model_id in MODEL_CATALOG:
                await self.spin_up(model_id, reason="initial")
            else:
                log.warning("Unknown: %s", model_id)

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
    global ROUTER_MODE, TTCA_TARGET_MS, TTCA_MIN_IMPROVEMENT
    parser = argparse.ArgumentParser(description="Dynamic vLLM provisioner for Sophia")
    parser.add_argument("--router-url",       default="http://localhost:8080")
    parser.add_argument("--total-gpus",       type=int,   default=8)
    parser.add_argument("--poll-interval",    type=float, default=POLL_INTERVAL_S)
    parser.add_argument("--priors-path",      default="results/priors.json")
    parser.add_argument("--router-mode",      default="accuracy",
                        choices=["accuracy", "ttca", "cost"])
    parser.add_argument("--ttca-target-ms",   type=float, default=3000.0)
    parser.add_argument("--ttca-min-improve", type=float, default=0.20)
    parser.add_argument("--initial-models",   default="")
    args = parser.parse_args()

    ROUTER_MODE          = args.router_mode
    TTCA_TARGET_MS       = args.ttca_target_ms if args.ttca_target_ms > 0 else None
    TTCA_MIN_IMPROVEMENT = args.ttca_min_improve

    initial = [m.strip() for m in args.initial_models.split(",") if m.strip()]
    provisioner = DynamicProvisioner(
        router_url=args.router_url, total_gpus=args.total_gpus,
        poll_interval=args.poll_interval, priors_path=args.priors_path,
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
