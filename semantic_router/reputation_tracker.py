from __future__ import annotations
import json
import os
import tempfile
import time
from collections import deque
from semantic_router.config import (
    LATENCY_EMA_ALPHA, LATENCY_GRACE_RATIO,
    ACCURACY_EMA_ALPHA, DEFAULT_ACCURACY_PRIOR,
    ACCURACY_BID_EMA_ALPHA, ACCURACY_BID_GRACE_RATIO,
    MODEL_REPUTATION_PATH,
)


class ReputationTracker:
    def __init__(self, path: str = MODEL_REPUTATION_PATH) -> None:
        self._path = path
        # {model_id: {"latency_reliability": float, "sample_count": int,
        #              "accuracy_priors": {"domain:complexity": float}}}
        self._data: dict[str, dict] = {}
        self._load()
        # Sliding-window log of routed requests: (timestamp, domain, complexity).
        # Used to count how many recent requests a cold model would serve better,
        # which determines its amortized spin-up cost in the TTCA score.
        # In-memory only (not persisted) — resets on router restart.
        self._request_log: deque = deque()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        dir_ = os.path.dirname(self._path) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(self._data, f, indent=2)
            tmp = f.name
        os.replace(tmp, self._path)

    def _ensure(self, model_id: str) -> dict:
        if model_id not in self._data:
            self._data[model_id] = {
                "latency_reliability": 1.0,
                "sample_count": 0,
                "accuracy_priors": {},
                # Bid reliability per "domain:complexity" — EMA of judge/bid ratio
                # 1.0 = honest bidder, <1.0 = consistently overbids accuracy
                "accuracy_bid_reliability": {},
                # EMA of actual measured latency (ms) — global across all categories.
                # None until first request completes.
                "avg_latency_ms": None,
                # EMA of actual measured latency per "domain:complexity" — more
                # accurate for TTCA scoring because latency varies by category.
                "avg_latency_ms_per_cat": {},
                # EMA of actual output token count per "domain:complexity".
                # Reasoning models (deepseek-r1) produce far more tokens than
                # instruction models for the same prompt, so the static global
                # table underestimates their latency. This corrects that.
                "avg_output_tokens": {},
            }
        else:
            # Backfill fields for models loaded from older JSON snapshots
            rep = self._data[model_id]
            rep.setdefault("avg_latency_ms_per_cat", {})
            rep.setdefault("avg_output_tokens", {})
        return self._data[model_id]

    # ── Latency reliability ──────────────────────────────────────────────────

    def record_latency(
        self, model_id: str, bid_latency_ms: int, actual_latency_ms: int
    ) -> None:
        rep = self._ensure(model_id)
        overrun = actual_latency_ms / max(bid_latency_ms, 1)
        ratio = 1.0 if overrun <= LATENCY_GRACE_RATIO else min(1.0 / overrun, 1.0)
        rep["latency_reliability"] = (
            (1 - LATENCY_EMA_ALPHA) * rep["latency_reliability"]
            + LATENCY_EMA_ALPHA * ratio
        )
        # Track EMA of actual latency for TTCA scoring.
        # First sample initialises directly; subsequent samples use EMA.
        old = rep.get("avg_latency_ms")
        rep["avg_latency_ms"] = (
            float(actual_latency_ms) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * actual_latency_ms
        )
        rep["sample_count"] += 1
        self._save()

    def get_avg_latency_ms(self, model_id: str) -> float | None:
        """Return global EMA of actual measured latency, or None if no data yet."""
        return self._data.get(model_id, {}).get("avg_latency_ms")

    def record_latency_per_cat(
        self, model_id: str, domain: str, complexity: str, actual_latency_ms: int
    ) -> None:
        """Record per-category latency EMA (more accurate for TTCA than global EMA)."""
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["avg_latency_ms_per_cat"].get(key)
        rep["avg_latency_ms_per_cat"][key] = (
            float(actual_latency_ms) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * actual_latency_ms
        )
        self._save()

    def get_avg_latency_ms_per_cat(
        self, model_id: str, domain: str, complexity: str
    ) -> float | None:
        """Return per-category latency EMA, or None if no data yet for this category."""
        key = f"{domain}:{complexity}"
        return self._data.get(model_id, {}).get("avg_latency_ms_per_cat", {}).get(key)

    def record_output_tokens(
        self, model_id: str, domain: str, complexity: str, tokens: int
    ) -> None:
        """Record actual completion token count for a (model, domain:complexity) pair."""
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["avg_output_tokens"].get(key)
        rep["avg_output_tokens"][key] = (
            float(tokens) if old is None
            else (1 - LATENCY_EMA_ALPHA) * old + LATENCY_EMA_ALPHA * tokens
        )
        self._save()

    def get_avg_output_tokens(
        self, model_id: str, domain: str, complexity: str
    ) -> float | None:
        """Return observed avg output token count, or None if no data yet."""
        key = f"{domain}:{complexity}"
        return self._data.get(model_id, {}).get("avg_output_tokens", {}).get(key)

    def get_penalty_multiplier(self, model_id: str) -> float:
        rep = self._data.get(model_id, {})
        reliability = rep.get("latency_reliability", 1.0)
        return 1.0 / max(reliability, 0.1)  # cap at 10× penalty

    def get_latency_reliability(self, model_id: str) -> float:
        return self._data.get(model_id, {}).get("latency_reliability", 1.0)

    # ── Accuracy priors ──────────────────────────────────────────────────────

    def get_accuracy_prior(self, model_id: str, domain: str, complexity: str) -> float:
        key = f"{domain}:{complexity}"
        rep = self._data.get(model_id, {})
        return rep.get("accuracy_priors", {}).get(key, DEFAULT_ACCURACY_PRIOR)

    def update_accuracy_prior(
        self, model_id: str, domain: str, complexity: str, judge_score: float
    ) -> None:
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        old = rep["accuracy_priors"].get(key, DEFAULT_ACCURACY_PRIOR)
        rep["accuracy_priors"][key] = (
            (1 - ACCURACY_EMA_ALPHA) * old + ACCURACY_EMA_ALPHA * judge_score
        )
        self._save()

    def seed_if_absent(
        self, model_id: str, domain: str, complexity: str, score: float
    ) -> bool:
        """Set accuracy prior only if this cell has no existing value.

        Used at model-registration time to seed calibration data without
        overwriting values already populated by warmup_from_eval_matrix().
        Returns True if the value was written, False if it already existed.
        """
        self._ensure(model_id)
        key = f"{domain}:{complexity}"
        if key in self._data[model_id]["accuracy_priors"]:
            return False
        self._data[model_id]["accuracy_priors"][key] = score
        self._save()
        return True

    def warmup_from_eval_matrix(self, csv_path: str) -> int:
        """Pre-seed accuracy priors from eval_matrix.csv before live traffic begins.

        Computes mean is_correct per (model_id, domain, complexity) cell and
        writes the result directly into accuracy_priors, overwriting any stale
        values from a previous cold run. Returns the number of cells seeded.
        """
        import csv
        from collections import defaultdict
        scores: defaultdict = defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "200") != "200":
                    continue
                is_correct = row.get("is_correct", "").lower() in ("true", "1", "yes")
                scores[(row["model_id"], row["domain"], row["complexity"])].append(
                    1.0 if is_correct else 0.0
                )
        count = 0
        for (model_id, domain, complexity), vals in scores.items():
            self._ensure(model_id)
            self._data[model_id]["accuracy_priors"][f"{domain}:{complexity}"] = (
                sum(vals) / len(vals)
            )
            count += 1
        if count:
            self._save()
        return count

    def record_accuracy_bid(
        self,
        model_id: str,
        domain: str,
        complexity: str,
        bid_accuracy: float,
        judge_score: float,
    ) -> None:
        """
        Record how well the model's accuracy bid matched the judge's score.
        ratio = min(judge_score / bid_accuracy, 1.0)
          - 1.0: delivered what was promised (or better)
          - <1.0: overbid — claimed higher accuracy than was judged
        Uses a grace ratio so small discrepancies don't penalise.
        """
        rep = self._ensure(model_id)
        key = f"{domain}:{complexity}"
        ratio = min(judge_score / max(bid_accuracy, 1e-6), 1.0)
        # Apply grace: if ratio >= grace threshold treat as honest
        ratio = 1.0 if ratio >= ACCURACY_BID_GRACE_RATIO else ratio
        old = rep["accuracy_bid_reliability"].get(key, 1.0)
        rep["accuracy_bid_reliability"][key] = (
            (1 - ACCURACY_BID_EMA_ALPHA) * old + ACCURACY_BID_EMA_ALPHA * ratio
        )
        self._save()

    def get_accuracy_discount(self, model_id: str, domain: str, complexity: str) -> float:
        """
        Return a discount factor [0.1, 1.0] to apply to bid.estimated_accuracy in
        the selector. Honest bidders get 1.0 (no discount). Chronic overbidders
        get < 1.0, making their accuracy appear lower in the scoring formula.
        """
        rep = self._data.get(model_id, {})
        key = f"{domain}:{complexity}"
        reliability = rep.get("accuracy_bid_reliability", {}).get(key, 1.0)
        return max(reliability, 0.1)  # cap at 90% discount

    def get_domain_floor(self, model_id: str, domain: str) -> float | None:
        """
        Return the minimum accuracy prior across all complexity levels for a domain.
        This is the live production floor — replaces the static calibration value
        once enough production samples exist (EMA needs ~20 samples to stabilize).
        Returns None if no production data exists yet for this domain.
        """
        rep = self._data.get(model_id, {})
        priors = rep.get("accuracy_priors", {})
        domain_scores = [v for k, v in priors.items() if k.startswith(f"{domain}:")]
        return min(domain_scores) if domain_scores else None

    # ── Request-rate tracking (for cold-model amortized spin-up cost) ──────────

    def record_request_arrival(self, domain: str, complexity: str) -> None:
        """Call at the start of every routed request to maintain the sliding window."""
        now = time.monotonic()
        self._request_log.append((now, domain, complexity))
        # Prune entries older than 10s (generous buffer beyond the 5s window)
        cutoff = now - 10.0
        while self._request_log and self._request_log[0][0] < cutoff:
            self._request_log.popleft()

    def count_requests_needing_model(
        self,
        cold_model_id: str,
        cold_accuracy_priors: dict[str, float],
        running_model_ids: list[str],
        window_s: float = 5.0,
    ) -> int:
        """Return count of requests in the last window_s seconds where cold_model_id
        has higher accuracy than ALL currently running models.

        Used to compute amortized spin-up cost:
            spin_up_cost_ms = T_spin_up_ms / max(count, 1)
        High count → low per-request cost → worth spinning up.
        """
        cutoff = time.monotonic() - window_s
        count = 0
        for ts, domain, complexity in self._request_log:
            if ts < cutoff:
                continue
            key = f"{domain}:{complexity}"
            cold_acc = cold_accuracy_priors.get(key, 0.0)
            best_running_acc = max(
                (self._data.get(rid, {}).get("accuracy_priors", {}).get(key, 0.0)
                 for rid in running_model_ids),
                default=0.0,
            )
            if cold_acc > best_running_acc:
                count += 1
        return count

    def get_all(self) -> dict:
        return dict(self._data)
