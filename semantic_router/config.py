from semantic_router.schemas import RouterMode, UserPreference

MODE_PRESETS: dict[RouterMode, dict] = {
    RouterMode.ACCURACY: dict(
        accuracy_weight=0.70, cost_weight=0.15,
        latency_weight=0.10, energy_weight=0.05,
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

DEFAULT_PREFERENCE = UserPreference()

# Bidding
BID_TIMEOUT_MS: int = 200
SAMPLE_RATE: float = 0.05
JUDGE_BATCH_SIZE: int = 10
JUDGE_BATCH_WINDOW_S: float = 2.0
JUDGE_QUEUE_MAX: int = 1000

# Latency reputation
LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10

# Accuracy prior
ACCURACY_EMA_ALPHA: float = 0.05
DEFAULT_ACCURACY_PRIOR: float = 0.70

# Accuracy bid reliability
ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90

# Calibration spot-check disabled.
# Requires an external API key and causes 500 errors if OPENAI_API_KEY
# routes through an Anthropic proxy (triggers "Streaming is required" error).
CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

# LLM judge — uses registered vLLM endpoints via HTTP.
# Map: model_id being judged -> (judge_base_url, judge_model_id).
# Cross-family pairing avoids self-enhancement bias.
# Example:
#   JUDGE_ENDPOINTS = {
#       "qwen2.5-1.5b": ("http://localhost:8002", "phi-3.5-mini"),
#       "phi-3.5-mini":  ("http://localhost:8001", "qwen2.5-1.5b"),
#   }
JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

# Benchmark ground-truth lookup for objective online accuracy scoring.
BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

# Persistence
USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

# Embedding model for semantic classifier
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
