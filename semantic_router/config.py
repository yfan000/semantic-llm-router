from semantic_router.schemas import RouterMode, UserPreference

MODE_PRESETS: dict[RouterMode, dict] = {
    RouterMode.ACCURACY: dict(accuracy_weight=0.70, cost_weight=0.15, latency_weight=0.10, energy_weight=0.05),
    RouterMode.ECO:      dict(accuracy_weight=0.25, cost_weight=0.15, latency_weight=0.20, energy_weight=0.40),
    RouterMode.COST:     dict(accuracy_weight=0.20, cost_weight=0.55, latency_weight=0.15, energy_weight=0.10),
}

DEFAULT_PREFERENCE = UserPreference()

BID_TIMEOUT_MS: int = 200
SAMPLE_RATE: float = 0.05
JUDGE_BATCH_SIZE: int = 10
JUDGE_BATCH_WINDOW_S: float = 2.0
JUDGE_QUEUE_MAX: int = 1000

LATENCY_EMA_ALPHA: float = 0.05
LATENCY_GRACE_RATIO: float = 1.10

ACCURACY_EMA_ALPHA: float = 0.05
DEFAULT_ACCURACY_PRIOR: float = 0.70

CALIBRATION_RATE: float = 0.01
CALIBRATION_DRIFT_THRESHOLD: float = 0.10

USER_PREFS_PATH: str = "user_preferences.json"
MODEL_REPUTATION_PATH: str = "model_reputation.json"

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
