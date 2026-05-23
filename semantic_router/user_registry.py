from __future__ import annotations
import json
import os
import tempfile
from fastapi import HTTPException
from semantic_router.schemas import RouterMode, UserPreference, UserBudget, RequestSLA
from semantic_router.config import MODE_PRESETS, DEFAULT_PREFERENCE, USER_PREFS_PATH


class BudgetExhaustedError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=402, detail=detail)


class UserRegistry:
    def __init__(self, path: str = USER_PREFS_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

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

    # -- Preference ------------------------------------------------------------

    def set_preference(self, user_id: str, pref: UserPreference) -> None:
        self._data.setdefault(user_id, {})["preference"] = pref.model_dump()
        self._save()

    def get_preference_raw(self, user_id: str) -> UserPreference:
        entry = self._data.get(user_id, {})
        if "preference" not in entry:
            return DEFAULT_PREFERENCE.model_copy()
        return UserPreference(**entry["preference"])

    def resolve_preference(self, user_id: str | None, sla: RequestSLA) -> UserPreference:
        base = self.get_preference_raw(user_id) if user_id else DEFAULT_PREFERENCE.model_copy()

        # Expand mode preset (request-level mode overrides stored mode)
        mode = sla.mode or base.mode
        if mode and mode != RouterMode.CUSTOM:
            if mode in MODE_PRESETS:
                # Apply preset weights (accuracy, cost, eco, ...)
                preset = MODE_PRESETS[mode]
                base = UserPreference(mode=mode, **preset,
                                      max_latency_ms=base.max_latency_ms,
                                      min_accuracy=base.min_accuracy)
            else:
                # Mode has no weight preset (e.g. TTCA uses lat/acc scoring in selector)
                # Keep existing weights, just record the mode so rank_bids() can detect it.
                base = base.model_copy(update={"mode": mode})

        # Apply per-request field overrides
        overrides: dict = {}
        for field in ("cost_weight", "latency_weight", "accuracy_weight", "energy_weight",
                      "max_latency_ms", "min_accuracy"):
            val = getattr(sla, field, None)
            if val is not None:
                overrides[field] = val

        if overrides:
            data = base.model_dump()
            data.update(overrides)
            base = UserPreference(**data)

        return base

    def delete_preference(self, user_id: str) -> None:
        self._data.get(user_id, {}).pop("preference", None)
        self._save()

    # -- Budget ----------------------------------------------------------------

    def set_budget(self, user_id: str, budget: UserBudget) -> None:
        self._data.setdefault(user_id, {})["budget"] = budget.model_dump()
        self._save()

    def get_budget(self, user_id: str) -> UserBudget:
        entry = self._data.get(user_id, {})
        if "budget" not in entry:
            return UserBudget()
        return UserBudget(**entry["budget"])

    def check_budget(
        self, user_id: str, estimated_tokens: int, estimated_energy_j: float
    ) -> None:
        budget = self.get_budget(user_id)
        if budget.remaining_tokens < estimated_tokens:
            raise BudgetExhaustedError(
                f"Token budget exhausted: {budget.remaining_tokens} remaining, "
                f"{estimated_tokens} estimated"
            )
        if budget.remaining_energy_j < estimated_energy_j:
            raise BudgetExhaustedError(
                f"Energy budget exhausted: {budget.remaining_energy_j:.1f} J remaining, "
                f"{estimated_energy_j:.1f} J estimated"
            )

    def deduct_budget(
        self, user_id: str, actual_tokens: int, actual_energy_j: float
    ) -> None:
        budget = self.get_budget(user_id)
        budget.remaining_tokens = max(0, budget.remaining_tokens - actual_tokens)
        budget.remaining_energy_j = max(0.0, budget.remaining_energy_j - actual_energy_j)
        self.set_budget(user_id, budget)
