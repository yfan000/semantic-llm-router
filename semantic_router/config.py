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

DEFAULT_PREFERENCE = UserPreference()

BID_TIMEOUT_MS: int = 500
SAMPLE_RATE: float = 0.05
JUDGE_BATCH_SIZE: int = 10
JUDGE_BATCH_WINDOW_S: float = 2.0
JUDGE_QUEUE_MAX: int = 1000

LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10

ACCURACY_EMA_ALPHA: float = 0.05
DEFAULT_ACCURACY_PRIOR: float = 0.70

ACCURACY_BID_EMA_ALPHA: float = 0.05
ACCURACY_BID_GRACE_RATIO: float = 0.90

CALIBRATION_RATE: float = 0.0
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

JUDGE_ENDPOINTS: dict[str, tuple[str, str]] = {}

BENCHMARK_PATH: str | None = "datasets/hf_1000.json"

USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
