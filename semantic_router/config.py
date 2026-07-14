from semantic_router.schemas import RouterMode, UserPreference

MODE_PRESETS: dict[RouterMode, dict] = {
    RouterMode.ACCURACY: dict(
        accuracy_weight=0.90, cost_weight=0.05,
        latency_weight=0.03, energy_weight=0.02,
    ),
    RouterMode.ECO: dict(
        accuracy_weight=0.25, cost_weight=0.15,
        latency_weight=0.20, energy_weight=0.40,
    ),
    RouterMode.COST: dict(
        accuracy_weight=0.20, cost_weight=0.55,
        latency_weight=0.15, energy_weight=0.10,
    ),
    # TTCA mode uses latency/accuracy scoring in selector.py (not a weighted preset)
    # — no entry needed here since rank_bids() handles it directly.
}

DEFAULT_PREFERENCE = UserPreference()  # equal weights, no hard constraints

# Bidding
BID_TIMEOUT_MS: int = 500
SAMPLE_RATE: float = 0.20          # fraction of requests sent to accuracy judge (increased from 0.05 for faster prior updates)
JUDGE_BATCH_SIZE: int = 10         # max items per judge batch
JUDGE_BATCH_WINDOW_S: float = 2.0  # seconds to wait before flushing a partial batch
JUDGE_QUEUE_MAX: int = 1000        # drop samples when queue is full

# Latency reputation
LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10      # up to 10% overrun treated as accurate
# TTCA scoring: score = accuracy / (latency^TTCA_ALPHA × cost^TTCA_COST_BETA)  (higher = better)
# TTCA_ALPHA: 0.0 = pure accuracy  |  1.0 = classic TTCA  |  2.0 = quadratic latency penalty
# TTCA_COST_BETA: 0.0 = cost excluded (default, backward compatible)  |  1.0 = cost penalized equally to latency
TTCA_ALPHA: float = 1.0
TTCA_COST_BETA: float = 0.0

# Per-domain latency exponents for TTCA scoring.
# Lower = more accuracy-focused (latency penalized less), higher = speed-focused.
# Falls back to TTCA_ALPHA for domains not listed here.
# Patched per-experiment via sed in run_experiment.sh:
#   TTCA_ALPHA_FACTUAL=0.0 bash scripts/submit.sh   (fully accuracy-focused factual)
#   TTCA_ALPHA_FACTUAL=1.0 bash scripts/submit.sh   (revert to classic TTCA)
TTCA_ALPHA_FACTUAL:   float = 0.1   # factual:  below crossover — lets Gemma-3-27B beat Qwen-7B on easy
TTCA_ALPHA_MATH:      float = 0.7   # math:     moderate — reasoning speed matters
TTCA_ALPHA_CODE:      float = 1.0   # code:     classic TTCA — compile+test latency matters
TTCA_ALPHA_REASONING: float = 0.4   # reasoning: reduced from 0.7 to close reasoning:easy accuracy gap

# Accuracy prior (judge-scored, updates eligibility floor)
ACCURACY_EMA_ALPHA: float = 0.10  # increased from 0.05 for faster adaptation to observed accuracy
DEFAULT_ACCURACY_PRIOR: float = 0.70

# Accuracy bid reliability (penalises models that overbid estimated_accuracy)
ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90  # bid within 10% of judge score → no penalty

# Calibration spot-check disabled — requires an external OpenAI/Anthropic API key
# and causes 500 errors if OPENAI_API_KEY routes through an Anthropic proxy.
CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

# LLM judge — uses registered vLLM endpoints via HTTP (no separate model download).
# Map: model_id being judged → (judge_base_url, judge_model_id) to call.
# Cross-family pairing avoids self-enhancement bias.
# Leave empty {} to disable LLM judging (only deterministic math/code scoring).
# Example:
#   JUDGE_ENDPOINTS = {
#       "qwen2.5-1.5b": ("http://localhost:8002", "phi-3.5-mini"),
#       "phi-3.5-mini":  ("http://localhost:8001", "qwen2.5-1.5b"),
#   }
JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

# Benchmark ground-truth lookup for objective online accuracy scoring.
# Point to the JSON file produced by build_dataset.py (must include ground_truth field).
# When a query matches a known benchmark item, its response is scored against
# the ground truth deterministically — no LLM judge needed for that item.
# Set to None to disable benchmark lookup (fall back to LLM judge only).
BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

# Per-request latency SLO (ms) by domain × complexity.
# Applied automatically after semantic classification if the user has not
# set max_latency_ms explicitly. Override per-request via extra_body.router.max_latency_ms.
# Tighter SLOs for easy tasks force fast models (7B) to be selected,
# reducing TTCA by avoiding large model overhead on simple requests.
LATENCY_SLO_MS: dict[tuple[str, str], int] = {
    # Calibrated to Sophia A100 observed P95 latencies + 10% buffer (target <5% violation).
    # Previous values (1–8s) were for faster hardware and caused 80%+ violation.
    ("factual",   "easy"):    8_000,
    ("factual",   "medium"): 15_000,
    ("factual",   "hard"):   20_000,
    ("math",      "easy"):   14_000,
    ("math",      "medium"): 12_000,
    ("math",      "hard"):   50_000,
    ("code",      "easy"):   14_000,
    ("code",      "medium"): 16_000,
    ("code",      "hard"):   22_000,
    ("reasoning", "easy"):   28_000,
    ("reasoning", "medium"): 20_000,
    ("reasoning", "hard"):    5_000,
    ("creative",  "easy"):   14_000,
    ("creative",  "medium"): 20_000,
    ("creative",  "hard"):   25_000,
}

# Persistence
USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

# Embedding model for semantic classifier
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
