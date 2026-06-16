"""
dynamic_provisioner.py -- Dynamic model provisioning for Sophia.

Monitors router metrics and vLLM queue depth, then automatically spins up
or shuts down models based on latency, accuracy, and traffic conditions.

Spin-up triggers:
  - Latency P90 > 2x SLO for any domain
  - Accuracy below threshold for a domain (need better model)
  - vLLM queue depth > QUEUE_DEPTH_THRESHOLD (overloaded)
  - Requests arrive for domain with no registered model

Spin-down triggers:
  - Model idle (< MIN_REQUESTS_TO_STAY) in last IDLE_WINDOW_S seconds
  - All domains covered by a better model already running

GPU pool: tracks which GPUs are free on this node.

Usage:
    # Start with a minimal set (or none) and let provisioner add more
    python provisioner/dynamic_provisioner.py \
        --router-url http://localhost:8080 \
        --poll-interval 30

    # Or start with initial models already running
    python provisioner/dynamic_provisioner.py \
        --router-url http://localhost:8080 \
        --initial-models qwen-7b,qwen-14b
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
from collections import defaultdict, deque
from typing import Optional

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

POLL_INTERVAL_S       = 30      # seconds between each metric check
IDLE_WINDOW_S         = 300     # seconds of inactivity before spin-down
MIN_REQUESTS_TO_STAY  = 5       # minimum requests in idle window to keep model
QUEUE_DEPTH_THRESHOLD = 10      # vLLM waiting requests before scale-up
RUNNING_THRESHOLD     = 20      # vLLM running requests before scale-up
#                               # With continuous batching, num_requests_waiting stays 0
#                               # but num_requests_running grows → use this as overload signal
KV_CACHE_THRESHOLD    = 0.70    # KV cache usage (0-1) before scale-up
LATENCY_SLO_MULTIPLIER = 2.0   # spin up if P90 > this * SLO
ACCURACY_THRESHOLD    = 0.65   # spin up better model if accuracy below this
STARTUP_WAIT_S        = 600     # max seconds to wait for a model to be ready (32B needs ~5-8 min)
COOLDOWN_S            = 120     # seconds between spin-up/down actions (avoid thrashing)

HF_HOME = "/eagle/UIC-HPC/yuping/hf_cache"

# SLO targets per domain (ms) — same as config.py
LATENCY_SLO_MS = {
    "factual:easy":    1000, "factual:medium":   2000, "factual:hard":   4000,
    "math:easy":       1000, "math:medium":      3000, "math:hard":      6000,
    "code:easy":       1500, "code:medium":      4000, "code:hard":      8000,
    "reasoning:easy":  1000, "reasoning:medium": 3000, "reasoning:hard": 6000,
    "creative:easy":   1500, "creative:medium":  5000, "creative:hard":  8000,
}

# ---------------------------------------------------------------------------
# Model catalog — all available models and their requirements
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
    # Expected output tokens/sec on Sophia A100s — used for TTCA latency estimation
    # when the model has never been run (no historical data in reputation tracker).
    # Set from benchmarks or prior runs. More accurate than GPU-count lookup.
    expected_tokens_per_sec: float = 1000.0
    # Cap context length (tokens). Required for large models on fewer GPUs to keep
    # KV cache within VRAM. 0 = let vLLM choose its default.
    max_model_len: int = 0
    # Extra flags passed verbatim to `vllm serve` (e.g. --trust-remote-code).
    extra_vllm_args: list = field(default_factory=list)


MODEL_CATALOG: dict[str, ModelSpec] = {
    # ── Node 1 (8 GPUs total: 1+1+2+2+2) ──────────────────────────────────────────
    "qwen-7b": ModelSpec(
        model_id="qwen-7b", model_name="Qwen/Qwen2.5-7B-Instruct",
        gpus_needed=1, port=8000,
        domains=["factual", "creative", "reasoning"], accuracy_tier=1,
        min_accuracy_capability={"factual": 0.70, "creative": 0.70, "reasoning": 0.68},
        efficiency_tokens_per_joule=13.0,
        expected_tokens_per_sec=2800.0,
    ),
    "deepseek-r1-7b": ModelSpec(
        model_id="deepseek-r1-7b", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        gpus_needed=1, port=8001,
        domains=["math", "reasoning", "code"], accuracy_tier=2,
        min_accuracy_capability={"math": 0.82, "reasoning": 0.80, "code": 0.75},
        efficiency_tokens_per_joule=13.0,
        expected_tokens_per_sec=1200.0,
    ),
    "qwen3-coder-30b": ModelSpec(
        model_id="qwen3-coder-30b", model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        gpus_needed=2, port=8002,
        domains=["code", "math", "reasoning"], accuracy_tier=3,
        min_accuracy_capability={"code": 0.91, "math": 0.88, "reasoning": 0.87},
        efficiency_tokens_per_joule=12.0,  # MoE 3.3B active ≈ 3× more efficient than dense 32B
        expected_tokens_per_sec=2500.0,    # 3.3B active params → ~3.5× faster than dense 32B
        max_model_len=8192,               # 60GB weights on 2×40GB; 20GB left for KV cache
        extra_vllm_args=["--trust-remote-code"],
    ),
    "gemma-3-27b": ModelSpec(
        model_id="gemma-3-27b", model_name="google/gemma-3-27b-it",
        gpus_needed=2, port=8003,
        domains=["factual", "reasoning", "creative", "math", "code"], accuracy_tier=3,
        min_accuracy_capability={"_default": 0.83},
        efficiency_tokens_per_joule=5.0,
        expected_tokens_per_sec=1000.0,   # ~1000 tok/s on 2 A100s (54GB weights)
        max_model_len=4096,               # 54GB weights leaves ~26GB KV cache on 2×40GB
        extra_vllm_args=["--trust-remote-code"],
    ),
    "deepseek-r1-14b": ModelSpec(
        model_id="deepseek-r1-14b", model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        gpus_needed=2, port=8004,
        domains=["math", "reasoning", "code"], accuracy_tier=3,
        min_accuracy_capability={"math": 0.90, "reasoning": 0.88, "code": 0.82},
        efficiency_tokens_per_joule=6.0,
        expected_tokens_per_sec=900.0,    # 14B with reasoning, 28GB weights → fast on 2×40GB
        max_model_len=8192,
    ),
    # ── Node 2 (8 GPUs) ────────────────────────────────────────────────────────
    "llama4-scout": ModelSpec(
        model_id="llama4-scout", model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        gpus_needed=8, port=8005,
        domains=["factual", "reasoning", "creative", "math", "code"], accuracy_tier=4,
        min_accuracy_capability={"_default": 0.88},
        efficiency_tokens_per_joule=3.0,
        expected_tokens_per_sec=500.0,    # MoE 17B active on 8 A100s
        max_model_len=8192,
        extra_vllm_args=["--trust-remote-code"],
    ),
}

# UPGRADE_PATH: ordered by tier ascending (fast/cheap → accurate/expensive)
UPGRADE_PATH: dict[str, list[str]] = {
    "factual":   ["qwen-7b", "gemma-3-27b", "llama4-scout"],
    "reasoning": ["qwen-7b", "deepseek-r1-7b", "deepseek-r1-14b", "gemma-3-27b", "llama4-scout"],
    "math":      ["deepseek-r1-7b", "deepseek-r1-14b", "qwen3-coder-30b", "llama4-scout"],
    "code":      ["deepseek-r1-7b", "qwen3-coder-30b", "deepseek-r1-14b", "llama4-scout"],
    "creative":  ["qwen-7b", "gemma-3-27b", "llama4-scout"],
}

# Minimum acceptable accuracy for any model serving a domain.
# Models below this are never selected, even for throughput relief.
MIN_ACCEPTABLE_ACCURACY: float = 0.60

# If current accuracy is above this, we can afford to downgrade to a
# smaller/faster model for queue overload without hurting users.
HIGH_ACCURACY_THRESHOLD: float = 0.85

# Router mode: provisioner picks candidates differently per mode.
#   accuracy → spin up MOST ACCURATE model above floor
#   ttca     → spin up FASTEST model above floor (lowest lat/acc ratio)
#   cost     → spin up SMALLEST model (fewest GPUs) above floor
ROUTER_MODE: str = "accuracy"

# Estimated tokens/sec per GPU count — used to estimate latency for
# models not yet running (no measured history available yet).
ESTIMATED_TOKENS_PER_SEC: dict[int, float] = {
    1: 2500.0,   # 7B on 1 A100
    2: 1800.0,   # 14B on 2 A100s
    4:  900.0,   # 32B on 4 A100s
    8:  500.0,   # 70B on 8 A100s
}

# ---------------------------------------------------------------------------
# TTCA-aware spin-up gate
# ---------------------------------------------------------------------------
# Spinning up a new model always has a cost (GPUs, startup time, energy).
# The gate ensures we only spin up when the TTCA improvement is worth it.
#
# Decision rule:
#   1. Compute current E[TTCA] from live request history
#   2. Estimate new E[TTCA] if the candidate model is added
#   3. Only spin up if:
#        current_E_TTCA > TTCA_TARGET_MS           (TTCA is bad)
#        AND improvement >= TTCA_MIN_IMPROVEMENT   (candidate helps enough)
#
# E[TTCA] formula (cascade model, no replacement):
#   E[TTCA] = L1/p1 * p1 + (L1+L2)/p2*(1-p1) + ...
#   Simplified: E[TTCA] ≈ avg_first_attempt_latency / resolution_rate
#
# Set TTCA_TARGET_MS = None to disable gating (always spin up on trigger).

TTCA_TARGET_MS:      float | None = 3000.0  # spin up if E[TTCA] > 3s
TTCA_MIN_IMPROVEMENT: float       = 0.20   # candidate must reduce E[TTCA] by >= 20%

# ---------------------------------------------------------------------------
# GPU Pool
# ---------------------------------------------------------------------------

class GPUPool:
    def __init__(self, total_gpus: int = 8):
        self._total   = total_gpus
        self._free    = list(range(total_gpus))    # [0,1,2,...,7]
        self._allocated: dict[str, list[int]] = {}  # model_id -> gpu_ids

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
        log.info("GPU pool: released %s from %s, free=%s", gpus, model_id, self._free)

    def free_count(self) -> int:
        return len(self._free)

    def allocated_to(self, model_id: str) -> list[int]:
        return self._allocated.get(model_id, [])

    def __repr__(self) -> str:
        return f"GPUPool(free={self._free}, allocated={self._allocated})"

# ---------------------------------------------------------------------------
# Running model state
# ---------------------------------------------------------------------------

@dataclass
class RunningModel:
    spec:       ModelSpec
    pid:        int
    gpus:       list[int]
    started_at: float
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
        router_url: str,
        total_gpus: int = 8,
        poll_interval: float = POLL_INTERVAL_S,
        priors_path: str = "results/priors.json",
        node_host: str = "localhost",
    ):
        self.router_url     = router_url.rstrip("/")
        self.gpu_pool       = GPUPool(total_gpus)
        self.poll_interval  = poll_interval
        self.priors_path    = priors_path
        self.node_host      = node_host   # hostname/IP reported to router for model base_url
        self.running: dict[str, RunningModel] = {}
        self._last_action_time: float = 0.0
        self._env = {
            **os.environ,
            "HF_HOME": HF_HOME,
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "CXX": "g++",
            "CC": "gcc",
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
            "LD_LIBRARY_PATH": (
                "/soft/compilers/openmpi/5.0.10/lib:"
                "/soft/libraries/ucx/1.20.0/lib:"
                + os.environ.get("LD_LIBRARY_PATH", "")
            ),
        }

    # ── Router helpers ────────────────────────────────────────────────────────────────────

    async def _router_get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.get(f"{self.router_url}{path}")
            r.raise_for_status()
            return r.json()

    async def _router_post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            r = await client.post(f"{self.router_url}{path}", json=body)
            r.raise_for_status()
            return r.json()

    async def _router_delete(self, path: str) -> None:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            await client.delete(f"{self.router_url}{path}")

    async def _vllm_overloaded(self, base_url: str) -> tuple[bool, str]:
        """Check if vLLM is overloaded using multiple signals.

        Returns (is_overloaded, reason_string).

        vLLM continuous batching means num_requests_waiting stays 0 even under
        heavy load — all requests go into "running" state immediately.
        We therefore check three signals:
          1. num_requests_waiting > QUEUE_DEPTH_THRESHOLD  (classic queue)
          2. num_requests_running > RUNNING_THRESHOLD      (continuous batch full)
          3. kv_cache_usage_perc  > KV_CACHE_THRESHOLD    (GPU memory pressure)
        """
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                r = await client.get(f"{base_url}/metrics")
            metrics: dict[str, float] = {}
            for line in r.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # strip label selectors: metric{label="v"} value
                    key = parts[0].split("{")[0]
                    try:
                        metrics[key] = float(parts[-1])
                    except ValueError:
                        pass

            waiting  = metrics.get("vllm:num_requests_waiting", 0.0)
            running  = metrics.get("vllm:num_requests_running",  0.0)
            kv_cache = metrics.get("vllm:kv_cache_usage_perc",   0.0)

            if waiting > QUEUE_DEPTH_THRESHOLD:
                return True, f"waiting={waiting:.0f} > {QUEUE_DEPTH_THRESHOLD}"
            if running > RUNNING_THRESHOLD:
                return True, f"running={running:.0f} > {RUNNING_THRESHOLD}"
            if kv_cache > KV_CACHE_THRESHOLD:
                return True, f"kv_cache={kv_cache:.2f} > {KV_CACHE_THRESHOLD}"
        except Exception:
            pass
        return False, ""

    async def _get_router_reputation(self) -> dict:
        """Returns {model_id: {latency_reliability, accuracy_priors}}."""
        try:
            models = await self._router_get("/v1/models")
            result = {}
            for m in models.get("data", []):
                mid = m["id"]
                try:
                    rep = await self._router_get(f"/router/{mid}/reputation")
                    result[mid] = rep
                except Exception:
                    pass
            return result
        except Exception:
            return {}

    # ── Spin up / down ────────────────────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        return time.monotonic() - self._last_action_time < COOLDOWN_S

    def _load_priors(self) -> dict:
        import json
        try:
            with open(self.priors_path) as f:
                return json.load(f)
        except Exception:
            return {}

    async def spin_up(self, model_id: str, reason: str) -> bool:
        if model_id in self.running:
            log.info("spin_up skipped: %s already running", model_id)
            return True
        if reason != "initial" and self._in_cooldown():
            log.info("spin_up skipped: in cooldown (%ds remaining)",
                     int(COOLDOWN_S - (time.monotonic() - self._last_action_time)))
            return False

        spec = MODEL_CATALOG.get(model_id)
        if not spec:
            log.error("spin_up: unknown model %s", model_id)
            return False

        gpus = self.gpu_pool.allocate(model_id, spec.gpus_needed)
        if gpus is None:
            log.warning("spin_up: not enough free GPUs for %s (need %d, free %d)",
                        model_id, spec.gpus_needed, self.gpu_pool.free_count())
            return False

        gpu_str = ",".join(str(g) for g in gpus)
        master_port = 29500 + spec.port - 8000  # unique per model

        cmd = [
            "vllm", "serve", spec.model_name,
            "--tensor-parallel-size", str(spec.gpus_needed),
            "--port", str(spec.port),
        ]
        if spec.max_model_len > 0:
            cmd += ["--max-model-len", str(spec.max_model_len)]
        if spec.extra_vllm_args:
            cmd += spec.extra_vllm_args
        env = {**self._env, "CUDA_VISIBLE_DEVICES": gpu_str, "MASTER_PORT": str(master_port)}

        log.info("SPIN UP  %s  reason=%s  gpus=%s  port=%d", model_id, reason, gpus, spec.port)
        log_path = os.path.expanduser(f"~/vllm_logs/vllm_{model_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf)

        self.running[model_id] = RunningModel(
            spec=spec, pid=proc.pid, gpus=gpus, started_at=time.monotonic()
        )
        self._last_action_time = time.monotonic()

        # localhost for health checks (always local); node_host for router registration
        local_url = f"http://localhost:{spec.port}"
        base_url  = f"http://{self.node_host}:{spec.port}"
        deadline = time.monotonic() + STARTUP_WAIT_S
        log.info("  Waiting for %s to be ready (up to %ds)...", model_id, STARTUP_WAIT_S)
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            try:
                async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                    r = await client.get(f"{local_url}/health")
                    if r.status_code == 200:
                        log.info("  %s is ready!", model_id)
                        break
            except Exception:
                pass
        else:
            log.error("  %s failed to start within %ds — spinning down", model_id, STARTUP_WAIT_S)
            await self.spin_down(model_id, reason="startup_timeout")
            return False

        # Register with router
        priors = self._load_priors()
        model_priors = priors.get(model_id, {})
        payload = {
            "model_id":   spec.model_id,
            "model_name": spec.model_name,
            "backend":    "vllm",
            "base_url":   base_url,
            "domains":    spec.domains,
            "min_accuracy_capability": spec.min_accuracy_capability,
            "accuracy_priors":         model_priors,
            "efficiency_tokens_per_joule": spec.efficiency_tokens_per_joule,
            "skip_calibration": True,
        }
        try:
            await self._router_post("/router/register", payload)
            log.info("  Registered %s with router", model_id)
        except Exception as e:
            log.error("  Failed to register %s: %s", model_id, e)

        return True

    async def spin_down(self, model_id: str, reason: str) -> bool:
        rm = self.running.get(model_id)
        if not rm:
            return False
        if self._in_cooldown():
            log.info("spin_down skipped: in cooldown")
            return False

        log.info("SPIN DOWN %s  reason=%s", model_id, reason)

        # Deregister from router
        try:
            await self._router_delete(f"/router/{model_id}")
            log.info("  Deregistered %s from router", model_id)
        except Exception as e:
            log.warning("  Failed to deregister %s: %s", model_id, e)

        # Kill the vLLM process
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

    # ── Provisioning policy ───────────────────────────────────────────────────────────────

    async def _check_dead_processes(self) -> None:
        """Detect models that are dead or unresponsive.

        Uses HTTP /health check rather than PID check because:
        - vLLM spawns multiple child processes; pkill kills workers but the
          parent PID (stored in rm.pid) may still be alive as a zombie
        - os.kill(pid, 0) sees the zombie as alive → bootstrap never fires
        - HTTP health check is definitive: if the model can't respond, it's dead
        """
        for model_id, rm in list(self.running.items()):
            base_url = f"http://localhost:{rm.spec.port}"
            alive = False
            try:
                async with httpx.AsyncClient(timeout=5.0, trust_env=False) as c:
                    r = await c.get(f"{base_url}/health")
                    alive = (r.status_code == 200)
            except Exception:
                alive = False

            if not alive:
                log.warning("DETECTED dead/unresponsive: %s (port=%d) -- cleaning up",
                            model_id, rm.spec.port)
                try:
                    await self._router_delete(f"/router/{model_id}")
                except Exception:
                    pass
                # Kill any remaining zombie processes
                try:
                    os.kill(rm.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.gpu_pool.release(model_id)
                del self.running[model_id]
                log.warning("  Removed %s from fleet. Free GPUs: %d",
                            model_id, self.gpu_pool.free_count())

    async def evaluate_and_act(self) -> None:
        """Main decision loop: collect metrics, decide, act."""
        # First: detect any externally-killed processes
        await self._check_dead_processes()

        reputation  = await self._get_router_reputation()
        all_priors  = self._load_priors()

        for model_id, rm in list(self.running.items()):
            spec     = rm.spec
            base_url = f"http://localhost:{spec.port}"

            # --- Spin-down: idle ---
            reqs_in_window = rm.requests_in_window(IDLE_WINDOW_S)
            if reqs_in_window < MIN_REQUESTS_TO_STAY:
                covered = all(
                    any(other_id != model_id and
                        domain in MODEL_CATALOG[other_id].domains
                        for other_id in self.running)
                    for domain in spec.domains
                )
                if covered:
                    log.info("TRIGGER idle: %s only %d reqs in %ds, domains covered",
                             model_id, reqs_in_window, IDLE_WINDOW_S)
                    await self.spin_down(model_id, reason="idle")
                    continue

            # --- Spin-up: queue overload (throughput problem) ---
            # Strategy A (downgrade): if accuracy is very high, trade quality for speed
            # Strategy B (replica):   if accuracy is borderline, horizontal scale
            overloaded, reason = await self._vllm_overloaded(base_url)
            if overloaded:
                log.info("TRIGGER overload: %s [%s]", model_id, reason)
                for domain in spec.domains:
                    if await self._try_scale_out(model_id, 11, domain, all_priors):
                        break

            # --- Spin-up: low accuracy (quality problem) ---
            # General search: find any model that beats current accuracy in that category
            rep    = reputation.get(model_id, {})
            priors = rep.get("accuracy_priors", {})
            for key, acc in priors.items():
                if acc < ACCURACY_THRESHOLD:
                    parts = key.split(":")
                    domain     = parts[0]
                    complexity = parts[1] if len(parts) > 1 else "medium"
                    log.info("TRIGGER low_accuracy: %s %s=%.3f < %.2f",
                             model_id, key, acc, ACCURACY_THRESHOLD)
                    await self._try_quality_upgrade(
                        domain=domain,
                        complexity=complexity,
                        current_accuracy=acc,
                        reason=f"low_accuracy:{model_id}:{key}",
                        all_priors=all_priors,
                    )

        # --- Spin-up: no model for a domain appearing in traffic ---
        registered = set(self.running.keys())
        all_covered_domains = set()
        for mid in registered:
            all_covered_domains.update(MODEL_CATALOG[mid].domains)

        # Check router health for any domain hints
        try:
            health = await self._router_get("/router/health")
            registered_count = health.get("registered_models", 0)
            if registered_count == 0 and len(self.running) == 0:
                # Nothing running — spin up a baseline
                log.info("TRIGGER bootstrap: no models running, spinning up qwen-7b")
                await self.spin_up("qwen-7b", reason="bootstrap")
        except Exception:
            pass

    # ── General data-driven provisioning decisions ─────────────────────────────────────────

    def _estimate_latency_ms(self, model_id: str, output_tokens: int = 300) -> float:
        """Estimate per-request latency for a model, using best available source.

        Priority:
          1. Historical avg_latency_ms from model_reputation.json
             (most accurate — actual measured latency on this hardware)
          2. Per-model expected_tokens_per_sec from MODEL_CATALOG
             (better than GPU count — accounts for model architecture differences)
          3. GPU-count-based fallback from ESTIMATED_TOKENS_PER_SEC
             (roughest estimate — only if model spec is missing throughput)

        Why this matters for TTCA:
          Two 1-GPU models (qwen-7b vs deepseek-r1-7b) have very different latencies
          because deepseek uses chain-of-thought reasoning which generates more tokens.
          GPU count alone cannot capture this difference.
        """
        # Source 1: historical data (best)
        try:
            import json
            rep_path = os.path.expanduser("~/semantic-llm-router/model_reputation.json")
            if os.path.exists(rep_path):
                with open(rep_path) as f:
                    rep = json.load(f)
                hist = rep.get(model_id, {}).get("avg_latency_ms")
                if hist is not None:
                    return float(hist)
        except Exception:
            pass

        # Source 2: per-model catalog estimate (better)
        spec = MODEL_CATALOG.get(model_id)
        if spec and spec.expected_tokens_per_sec > 0:
            return output_tokens / spec.expected_tokens_per_sec * 1000

        # Source 3: GPU count fallback (roughest)
        gpus = spec.gpus_needed if spec else 1
        tps  = ESTIMATED_TOKENS_PER_SEC.get(gpus, 1000.0)
        return output_tokens / tps * 1000

    def _get_prior(self, model_id: str, domain: str, complexity: str,
                   all_priors: dict) -> float | None:
        """Look up accuracy of any model (running or not) from priors file."""
        priors = all_priors.get(model_id, {})
        return priors.get(f"{domain}:{complexity}") or priors.get(domain)

    def _candidates_not_running(self) -> list[ModelSpec]:
        return [
            spec for mid, spec in MODEL_CATALOG.items()
            if mid not in self.running and not mid.endswith("-replica")
        ]

    def _compute_effective_ttca(self, model_id: str, all_priors: dict) -> float | None:
        """Estimate E[TTCA] for a running model from its request history.

        E[TTCA] ≈ avg_latency_first_attempt / resolution_rate

        Where resolution_rate = fraction of requests answered correctly on
        the first attempt (no retry). Collected from the X-Router-Attempt
        and actual_latency_ms headers stored in RunningModel.request_times.

        Returns None if insufficient data (< 10 samples).
        """
        rm = self.running.get(model_id)
        if not rm or not hasattr(rm, "latency_samples"):
            return None
        samples = getattr(rm, "latency_samples", [])  # list of (latency_ms, attempts)
        if len(samples) < 10:
            return None
        total_lat   = sum(lat for lat, _ in samples)
        resolved_1  = sum(1 for _, att in samples if att == 1)
        avg_lat     = total_lat / len(samples)
        res_rate    = resolved_1 / len(samples)
        if res_rate < 0.01:
            return float("inf")
        return avg_lat / res_rate

    def _estimate_ttca_with_candidate(
        self,
        current_models: list[tuple[float, float]],  # [(lat_ms, accuracy), ...]
        candidate_lat:  float,
        candidate_acc:  float,
    ) -> float:
        """Estimate E[TTCA] if a candidate model is added to the pool.

        Uses the cascade formula: models tried in lat/acc order (TTCA order).
        E[TTCA] = sum over positions i of:
            P(first i-1 wrong) * (sum latencies 1..i) * P(model i correct)

        Args:
            current_models: [(latency_ms, accuracy)] of currently running models
                            sorted by lat/acc ascending (TTCA order)
            candidate_lat:  estimated latency of new model
            candidate_acc:  estimated accuracy of new model
        """
        # Add candidate and re-sort by lat/acc
        all_models = current_models + [(candidate_lat, candidate_acc)]
        all_models.sort(key=lambda x: x[0] / max(x[1], 0.01))

        e_ttca         = 0.0
        p_all_wrong    = 1.0  # probability that all previous models were wrong
        cumulative_lat = 0.0

        for lat, acc in all_models:
            cumulative_lat += lat
            # Expected TTCA contribution: probability we reach this model * latency so far * P(correct)
            e_ttca      += p_all_wrong * acc * cumulative_lat
            p_all_wrong *= (1.0 - acc)

        # If all models fail (unlikely), add penalty
        e_ttca += p_all_wrong * cumulative_lat
        return e_ttca

    def _should_spin_up_for_ttca(
        self,
        current_e_ttca: float | None,
        candidate_lat:  float,
        candidate_acc:  float,
        current_models: list[tuple[float, float]],
    ) -> bool:
        """Gate: should we spin up this candidate based on TTCA analysis?

        Only spin up if:
          1. Current E[TTCA] exceeds TTCA_TARGET_MS  (we need to improve)
          2. Candidate reduces E[TTCA] by >= TTCA_MIN_IMPROVEMENT  (it helps enough)

        If TTCA_TARGET_MS is None, gate is disabled (always spin up on trigger).

        Example:
          current E[TTCA] = 4500ms > 3000ms target  -> condition 1 met
          new E[TTCA] with candidate = 3200ms
          improvement = (4500-3200)/4500 = 28.9% >= 20% min -> condition 2 met
          -> spin up ✓

          current E[TTCA] = 1800ms < 3000ms target  -> condition 1 NOT met
          -> skip spin-up, TTCA is already acceptable ✗
        """
        if TTCA_TARGET_MS is None:
            return True  # gate disabled

        if current_e_ttca is None:
            return True  # no data yet, spin up conservatively

        if current_e_ttca <= TTCA_TARGET_MS:
            log.info("  TTCA gate: current E[TTCA]=%.0fms <= target %.0fms — skip spin-up",
                     current_e_ttca, TTCA_TARGET_MS)
            return False

        # Estimate improvement
        new_e_ttca   = self._estimate_ttca_with_candidate(current_models, candidate_lat, candidate_acc)
        improvement  = (current_e_ttca - new_e_ttca) / current_e_ttca

        if improvement < TTCA_MIN_IMPROVEMENT:
            log.info("  TTCA gate: improvement=%.1f%% < %.0f%% min "
                     "(%.0fms -> %.0fms) — marginal gain, skip spin-up",
                     improvement * 100, TTCA_MIN_IMPROVEMENT * 100,
                     current_e_ttca, new_e_ttca)
            return False

        log.info("  TTCA gate: improvement=%.1f%% (%.0fms -> %.0fms) >= %.0f%% min — spin up",
                 improvement * 100, current_e_ttca, new_e_ttca, TTCA_MIN_IMPROVEMENT * 100)
        return True

    async def _refresh_router_priors(self, model_id: str, all_priors: dict) -> None:
        """Re-register a running model with updated accuracy priors.

        Called when a better model is already running but the router may not
        know its updated accuracy. Re-registering refreshes the priors used
        for bid selection without restarting the model.
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
        """Low accuracy in domain:complexity. Three possible outcomes:

        Case 1 — Better model is NOT running and fits in GPU budget:
          Spin it up. The router will start routing hard requests to it.

        Case 2 — Better model IS already running:
          The router should already use it, but may have stale priors.
          Re-register it with updated accuracy priors so the router's
          bid selection reflects its true quality.
          Also check if the low-accuracy model is confusing the router by
          bidding on the same domain — log a warning for investigation.

        Case 3 — No better model exists (ceiling reached):
          All models covering this domain are below current accuracy or
          have no GPU budget. Log that we've hit the accuracy ceiling.

        Algorithm:
          1. Collect all models (running + not) that cover the domain
          2. Filter: must have HIGHER accuracy than current, above floor
          3. If best is NOT running → spin it up  (Case 1)
             If best IS running    → refresh its router priors  (Case 2)
             If none found         → log ceiling reached  (Case 3)
        """
        # Gather ALL models covering the domain (running or not)
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
            else:
                if self.gpu_pool.free_count() >= spec.gpus_needed:
                    better_not_running.append((spec, acc))

        # Sort candidates by ROUTER_MODE priority:
        #   accuracy → most accurate first (maximize quality)
        #   ttca     → lowest lat/acc ratio first (fastest above floor)
        #   cost     → fewest GPUs first (cheapest above floor)
        #   accuracy → TWO-PHASE:
        #     Phase 1: cheapest model that crosses ACCURACY_THRESHOLD (just good enough)
        #     Phase 2: if no Phase 1 candidate, most accurate above floor (best effort)
        #   Rationale: don't spin up a 32B model if a 14B fixes the problem.
        #   A 14B with acc=0.69 solves the threshold (0.65) just as well as
        #   a 32B with acc=0.88 — and costs half the GPUs.
        def _candidate_sort_key(item: tuple[ModelSpec, float]) -> tuple:
            spec, acc = item
            if ROUTER_MODE == "ttca":
                # Use best available latency estimate (3 sources, in priority order):
                #   1. Historical avg_latency_ms from reputation tracker (most accurate)
                #   2. Per-model expected_tokens_per_sec from MODEL_CATALOG
                #   3. GPU-count-based ESTIMATED_TOKENS_PER_SEC (roughest fallback)
                est_lat_ms = self._estimate_latency_ms(spec.model_id)
                return (est_lat_ms / max(acc, 0.01), spec.gpus_needed)
            elif ROUTER_MODE == "cost":
                return (spec.gpus_needed, -acc)
            else:
                # accuracy mode: minimum sufficient first
                # Models that cross the threshold are sorted by fewest GPUs (cheapest fix).
                # Models below threshold but above floor are sorted by highest accuracy
                # (best effort when no model can fully fix it).
                crosses_threshold = acc >= ACCURACY_THRESHOLD
                if crosses_threshold:
                    return (0, spec.gpus_needed, -acc)   # group 0: cheapest sufficient
                else:
                    return (1, -acc, spec.gpus_needed)   # group 1: best effort

        better_running.sort(key=_candidate_sort_key)
        better_not_running.sort(key=_candidate_sort_key)

        # Case 1: spin up the best not-running model — if TTCA gate passes
        if better_not_running:
            best_spec, best_acc = better_not_running[0]

            # Build current model pool latency/accuracy for TTCA estimation
            current_models: list[tuple[float, float]] = []
            for mid, rm2 in self.running.items():
                a = self._get_prior(mid, domain, complexity, all_priors)
                if a and a >= MIN_ACCEPTABLE_ACCURACY and domain in rm2.spec.domains:
                    current_models.append((self._estimate_latency_ms(mid), a))

            cand_lat = self._estimate_latency_ms(best_spec.model_id)

            # Use live E[TTCA] if available, else None (gate uses conservative default)
            current_e_ttca = next(
                (self._compute_effective_ttca(mid, all_priors)
                 for mid in self.running
                 if self._compute_effective_ttca(mid, all_priors) is not None),
                None
            )

            if self._should_spin_up_for_ttca(
                current_e_ttca, cand_lat, best_acc, current_models
            ):
                log.info("  [Case 1 spin-up] %s acc=%.3f for %s:%s (mode=%s)",
                         best_spec.model_id, best_acc, domain, complexity, ROUTER_MODE)
                return await self.spin_up(best_spec.model_id, reason=reason)

        # Case 2: a better model is already running — refresh its router priors
        if better_running:
            best_spec, best_acc = better_running[0]
            log.warning(
                "  Quality upgrade [Case 2 already running]: %s acc=%.3f > %.3f for %s:%s"
                " — refreshing router priors so routing reflects its quality",
                best_spec.model_id, best_acc, current_accuracy, domain, complexity
            )
            await self._refresh_router_priors(best_spec.model_id, all_priors)
            return True

        # Case 3: accuracy ceiling reached
        log.warning(
            "  Quality upgrade [Case 3 ceiling]: no model beats %.3f for %s:%s"
            " — accuracy ceiling reached, consider adding a stronger model to MODEL_CATALOG",
            current_accuracy, domain, complexity
        )
        return await self.spin_up(best_spec.model_id, reason=reason)

    async def _try_scale_out(
        self,
        overloaded_id: str,
        queue_depth:   int,
        domain:        str,
        all_priors:    dict,
    ) -> bool:
        """Queue overload. Two strategies based on current accuracy:

        Strategy A — Downgrade (accuracy is HIGH):
          If current accuracy >= HIGH_ACCURACY_THRESHOLD, we are over-provisioned
          on quality. Users get 90%+ accuracy but wait in queue. A smaller, faster
          model (fewer GPUs) processes requests faster while staying above the
          MIN_ACCEPTABLE_ACCURACY floor. Queue drains; accuracy stays acceptable.

          Selection: fewest GPUs first (fastest), accuracy floor enforced,
                     must be SMALLER than current model.

        Strategy B — Horizontal scale (accuracy is borderline):
          Accuracy is not high enough to safely downgrade. Spin up the same
          model on a different port/GPUs to double throughput identically.

        Fallback: if neither is possible, log warning and wait for next poll.

        Example: deepseek-r1-7b math queue=12
          accuracy=0.89 >= 0.85 → Strategy A (downgrade)
            qwen-7b: math:medium=0.83 >= 0.60 floor, 1 GPU (< deepseek 1 GPU) — no, same size
            qwen-7b IS 1 GPU, deepseek IS 1 GPU — not smaller, skip
            No smaller found → fallback to Strategy B (replica)

          If deepseek were 4 GPUs and accuracy=0.89:
            qwen-7b: 1 GPU < 4 GPUs, math:medium=0.83 >= 0.60 → downgrade ✓
            deepseek queue: all requests go to qwen-7b for easy/medium
                           hard requests still routed to deepseek by router
        """
        spec = MODEL_CATALOG.get(overloaded_id)
        if not spec:
            return False

        # Measure worst-case accuracy across this domain for the overloaded model
        domain_priors = {
            k: v for k, v in all_priors.get(overloaded_id, {}).items()
            if k.startswith(domain + ":")
        }
        current_acc = min(domain_priors.values()) if domain_priors else None
        accuracy_is_high = (current_acc is not None and
                            current_acc >= HIGH_ACCURACY_THRESHOLD)

        if accuracy_is_high:
            # Strategy A: downgrade to smaller/faster model
            candidates: list[tuple[ModelSpec, float]] = []
            for candidate in self._candidates_not_running():
                if domain not in candidate.domains:
                    continue
                if candidate.gpus_needed >= spec.gpus_needed:
                    continue  # must be smaller (fewer GPUs = faster throughput)
                if self.gpu_pool.free_count() < candidate.gpus_needed:
                    continue
                acc = self._get_prior(candidate.model_id, domain, "medium", all_priors)
                if acc is None or acc < MIN_ACCEPTABLE_ACCURACY:
                    continue
                candidates.append((candidate, acc))

            if candidates:
                # Fewest GPUs first (maximum throughput gain), accuracy as tiebreaker
                candidates.sort(key=lambda x: (x[0].gpus_needed, -x[1]))
                best, best_acc = candidates[0]
                log.info("  Strategy A downgrade: %s (%d GPUs, acc=%.3f) for %s "
                         "(current=%.3f is high → trade quality for speed)",
                         best.model_id, best.gpus_needed, best_acc, domain, current_acc)
                return await self.spin_up(
                    best.model_id,
                    reason=f"downgrade:{overloaded_id}:q={queue_depth}:acc={best_acc:.2f}"
                )
            log.info("  Strategy A: no smaller model available, trying horizontal scale")

        # Strategy B: horizontal scale — same model, new port
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
            log.info("  Strategy B replica: %s (queue=%d, acc=%.3f → scale horizontally)",
                     overloaded_id, queue_depth, current_acc or 0)
            return await self.spin_up(
                replica_id,
                reason=f"horizontal_scale:{overloaded_id}:q={queue_depth}"
            )

        log.warning("  Scale-out: no options (GPUs full)")
        return False

    # ── Main loop ─────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        initial_models: list[str] | None = None,
        static: bool = False,
    ) -> None:
        log.info("Dynamic Provisioner starting")
        log.info("GPU pool: %s", self.gpu_pool)
        log.info("Router:   %s", self.router_url)
        log.info("Poll interval: %ds", int(self.poll_interval))
        if static:
            log.info("Mode: STATIC — auto-scaling disabled, models fixed after startup")

        for model_id in (initial_models or []):
            if model_id in MODEL_CATALOG:
                await self.spin_up(model_id, reason="initial")
            else:
                log.warning("Unknown initial model: %s", model_id)

        # Reset cooldown after initial models are ready so startup spin-up
        # does not block reactive scale-out during the first poll window.
        self._last_action_time = 0.0
        log.info("Initial models ready. Cooldown reset -- reactive scaling enabled.")

        # Register all cold (non-running) models with the router so it can
        # score them in the TTCA amortized spin-up formula.
        await self._cold_register_all()

        while True:
            try:
                log.info("--- Poll: %d models running, %d GPUs free ---",
                         len(self.running), self.gpu_pool.free_count())
                # Always drain the router spin-up queue first (router-triggered spin-ups
                # take priority over the provisioner's own reactive decisions).
                await self._process_spin_up_queue()
                if static:
                    # Static mode: only check for dead processes, no scale decisions.
                    await self._check_dead_processes()
                else:
                    await self.evaluate_and_act()
            except Exception as e:
                log.error("Provisioner loop error: %s", e)
            await asyncio.sleep(self.poll_interval)

    async def _cold_register_all(self) -> None:
        """Register all MODEL_CATALOG entries that are not currently running with the
        router as 'cold' models.  The router uses these specs to score cold models
        alongside running ones and request proactive spin-up when worthwhile."""
        priors = self._load_priors()
        for model_id, spec in MODEL_CATALOG.items():
            if model_id in self.running or model_id.endswith("-replica"):
                continue
            payload = {
                "model_id":            model_id,
                "domains":             spec.domains,
                "accuracy_priors":     priors.get(model_id, {}),
                "estimated_latency_ms": self._estimate_latency_ms(model_id),
                # Use half of STARTUP_WAIT_S as the expected spin-up time (optimistic).
                "estimated_spin_up_ms": STARTUP_WAIT_S * 500,
            }
            try:
                await self._router_post("/router/cold-register", payload)
                log.info("Cold-registered %s with router", model_id)
            except Exception as e:
                log.warning("Failed to cold-register %s: %s", model_id, e)

    async def _process_spin_up_queue(self) -> None:
        """Drain the router's spin-up queue and spin up each requested model.
        The router enqueues model IDs when their amortized TTCA score beats the
        running fleet — this allows the router to proactively trigger spin-ups
        based on live traffic patterns rather than waiting for the provisioner's
        reactive poll cycle."""
        try:
            resp = await self._router_get("/router/spin-up-queue")
            for model_id in resp.get("spin_up", []):
                if model_id in self.running:
                    log.debug("Spin-up queue: %s already running", model_id)
                    continue
                if model_id not in MODEL_CATALOG:
                    log.warning("Spin-up queue: unknown model %s — skipping", model_id)
                    continue
                log.info("Router requested spin-up of %s — processing", model_id)
                await self.spin_up(model_id, reason="router_requested")
        except Exception as e:
            log.debug("Could not poll spin-up queue: %s", e)

    async def shutdown(self) -> None:
        log.info("Shutting down — stopping all models")
        for model_id in list(self.running.keys()):
            await self.spin_down(model_id, reason="shutdown")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic vLLM provisioner")
    parser.add_argument("--router-url",      default="http://localhost:8080")
    parser.add_argument("--total-gpus",      type=int, default=8)
    parser.add_argument("--poll-interval",   type=float, default=POLL_INTERVAL_S)
    parser.add_argument("--priors-path",     default="results/priors.json")
    parser.add_argument("--router-mode",     default="accuracy",
                        choices=["accuracy", "ttca", "cost"],
                        help="Router mode: controls which model is spun up for quality upgrade. "
                             "accuracy=most accurate, ttca=fastest above floor, cost=fewest GPUs")
    parser.add_argument("--initial-models",  default="",
                        help="Comma-separated list of model IDs to start immediately")
    parser.add_argument("--static", action="store_true",
                        help="Disable auto-scaling: spin up initial models and only route, "
                             "no further spin-up or spin-down decisions")
    parser.add_argument("--node-host", default="localhost",
                        help="Hostname or IP of this node, used as base_url when registering "
                             "models with the router. Set to the node's hostname when running "
                             "a second provisioner on a different node (e.g. sophia-gpu-06).")
    args = parser.parse_args()

    global ROUTER_MODE
    ROUTER_MODE = args.router_mode
    log.info("Provisioner router mode: %s", ROUTER_MODE)

    initial = [m.strip() for m in args.initial_models.split(",") if m.strip()]

    provisioner = DynamicProvisioner(
        router_url=args.router_url,
        total_gpus=args.total_gpus,
        poll_interval=args.poll_interval,
        priors_path=args.priors_path,
        node_host=args.node_host,
    )

    async def _run():
        try:
            await provisioner.run(initial_models=initial, static=args.static)
        except KeyboardInterrupt:
            pass
        finally:
            await provisioner.shutdown()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
