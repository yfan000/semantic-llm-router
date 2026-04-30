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
}

DEFAULT_PREFERENCE = UserPreference()  # equal weights, no hard constraints

# Bidding
BID_TIMEOUT_MS: int = 500
SAMPLE_RATE: float = 0.05          # fraction of requests sent to accuracy judge
JUDGE_BATCH_SIZE: int = 10         # max items per judge batch
JUDGE_BATCH_WINDOW_S: float = 2.0  # seconds to wait before flushing a partial batch
JUDGE_QUEUE_MAX: int = 1000        # drop samples when queue is full

# Latency reputation
LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10      # up to 10% overrun treated as accurate

# Accuracy prior (judge-scored, updates eligibility floor)
ACCURACY_EMA_ALPHA: float = 0.05
DEFAULT_ACCURACY_PRIOR: float = 0.70

# Accuracy bid reliability (penalises models that overbid estimated_accuracy)
ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90  # bid within 10% of judge score -> no penalty

# Calibration spot-check disabled
CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

# LLM judge endpoints (empty = disabled)
JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

# Benchmark ground-truth lookup for online accuracy scoring
BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

# Per-request latency SLO (ms) by domain x complexity.
# Applied automatically after semantic classification if the user has not
# set max_latency_ms explicitly. Override per-request via extra_body.router.max_latency_ms.
LATENCY_SLO_MS: dict[tuple[str, str], int] = {
    ("factual",   "easy"):   1000,
    ("factual",   "medium"): 2000,
    ("factual",   "hard"):   4000,
    ("math",      "easy"):   1500,
    ("math",      "medium"): 3000,
    ("math",      "hard"):   6000,
    ("code",      "easy"):   2000,
    ("code",      "medium"): 4000,
    ("code",      "hard"):   8000,
    ("reasoning", "easy"):   1500,
    ("reasoning", "medium"): 3000,
    ("reasoning", "hard"):   6000,
    ("creative",  "easy"):   2000,
    ("creative",  "medium"): 5000,
    ("creative",  "hard"):   8000,
}

# Persistence
USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

# Embedding model for semantic classifier
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
