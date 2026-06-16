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
SAMPLE_RATE: float = 0.20
JUDGE_BATCH_SIZE: int = 10
JUDGE_BATCH_WINDOW_S: float = 2.0
JUDGE_QUEUE_MAX: int = 1000

# Latency reputation
LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10
# TTCA scoring: score = accuracy / (latency^TTCA_ALPHA x cost^TTCA_COST_BETA)  (higher = better)
# TTCA_ALPHA: 0.0 = pure accuracy  |  1.0 = classic TTCA  |  2.0 = quadratic latency penalty
# TTCA_COST_BETA: 0.0 = cost excluded (default, backward compatible)  |  1.0 = cost penalized equally to latency
TTCA_ALPHA: float = 1.0
TTCA_COST_BETA: float = 0.0

# Accuracy prior (judge-scored, updates eligibility floor)
ACCURACY_EMA_ALPHA: float = 0.10
DEFAULT_ACCURACY_PRIOR: float = 0.70

# Accuracy bid reliability
ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90

# Calibration spot-check disabled
CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

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

USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
