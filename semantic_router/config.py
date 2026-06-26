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
    # -- no entry needed here since rank_bids() handles it directly.
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
# TTCA scoring: score = accuracy / (latency^TTCA_ALPHA x cost^TTCA_COST_BETA)  (higher = better)
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
TTCA_ALPHA_FACTUAL:   float = 0.3   # factual:  accuracy-focused (MMLU-Pro rewards large models)
TTCA_ALPHA_MATH:      float = 0.7   # math:     moderate -- reasoning speed matters
TTCA_ALPHA_CODE:      float = 1.0   # code:     classic TTCA -- compile+test latency matters
TTCA_ALPHA_REASONING: float = 0.7   # reasoning: moderate balance

# Accuracy prior (judge-scored, updates eligibility floor)
ACCURACY_EMA_ALPHA: float = 0.10  # increased from 0.05 for faster adaptation to observed accuracy
DEFAULT_ACCURACY_PRIOR: float = 0.70

# Accuracy bid reliability (penalises models that overbid estimated_accuracy)
ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90  # bid within 10% of judge score -> no penalty

# Calibration spot-check disabled -- requires an external OpenAI/Anthropic API key
# and causes 500 errors if OPENAI_API_KEY routes through an Anthropic proxy.
CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

# LLM judge -- uses registered vLLM endpoints via HTTP (no separate model download).
JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

# Benchmark ground-truth lookup for objective online accuracy scoring.
BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

# Per-request latency SLO (ms) by domain x complexity.
LATENCY_SLO_MS: dict[tuple[str, str], int] = {
    ("factual",   "easy"):   1000,
    ("factual",   "medium"): 2000,
    ("factual",   "hard"):   4000,
    ("math",      "easy"):   1000,
    ("math",      "medium"): 3000,
    ("math",      "hard"):   6000,
    ("code",      "easy"):   1500,
    ("code",      "medium"): 4000,
    ("code",      "hard"):   8000,
    ("reasoning", "easy"):   1000,
    ("reasoning", "medium"): 3000,
    ("reasoning", "hard"):   6000,
    ("creative",  "easy"):   1500,
    ("creative",  "medium"): 5000,
    ("creative",  "hard"):   8000,
}

# Persistence
USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

# Embedding model for semantic classifier
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
