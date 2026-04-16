from __future__ import annotations
import json, os, tempfile
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
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def _save(self) -> None:
        dir_ = os.path.dirname(self._path) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
            json.dump(self._data, f, indent=2)
            tmp = f.name
        os.replace(tmp, self._path)

    def set_preference(self, user_id: str, pref: UserPreference) -> None:
        self._data.setdefault(user_id, {})["preference"] = pref.model_dump()
        self._save()

    def get_preference_raw(self, user_id: str) -> UserPreference:
        entry = self._data.get(user_id, {})
        return UserPreference(**entry["preference"]) if "preference" in entry else DEFAULT_PREFERENCE.model_copy()

    def resolve_preference(self, user_id: str | None, sla: RequestSLA) -> UserPreference:
        base = self.get_preference_raw(user_id) if user_id else DEFAULT_PREFERENCE.model_copy()
        mode = sla.mode or base.mode
        if mode and mode != RouterMode.CUSTOM:
            preset = MODE_PRESETS[mode]
            base = UserPreference(mode=mode, **preset, max_latency_ms=base.max_latency_ms, min_accuracy=base.min_accuracy)
        overrides = {f: getattr(sla, f) for f in ("cost_weight", "latency_weight", "accuracy_weight", "energy_weight", "max_latency_ms", "min_accuracy") if getattr(sla, f) is not None}
        if overrides:
            base = UserPreference(**{**base.model_dump(), **overrides})
        return base

    def delete_preference(self, user_id: str) -> None:
        self._data.get(user_id, {}).pop("preference", None)
        self._save()

    def set_budget(self, user_id: str, budget: UserBudget) -> None:
        self._data.setdefault(user_id, {})["budget"] = budget.model_dump()
        self._save()

    def get_budget(self, user_id: str) -> UserBudget:
        entry = self._data.get(user_id, {})
        return UserBudget(**entry["budget"]) if "budget" in entry else UserBudget()

    def check_budget(self, user_id: str, estimated_tokens: int, estimated_energy_j: float) -> None:
        b = self.get_budget(user_id)
        if b.remaining_tokens < estimated_tokens:
            raise BudgetExhaustedError(f"Token budget exhausted: {b.remaining_tokens} remaining, {estimated_tokens} estimated")
        if b.remaining_energy_j < estimated_energy_j:
            raise BudgetExhaustedError(f"Energy budget exhausted: {b.remaining_energy_j:.1f} J remaining, {estimated_energy_j:.1f} J estimated")

    def deduct_budget(self, user_id: str, actual_tokens: int, actual_energy_j: float) -> None:
        b = self.get_budget(user_id)
        b.remaining_tokens = max(0, b.remaining_tokens - actual_tokens)
        b.remaining_energy_j = max(0.0, b.remaining_energy_j - actual_energy_j)
        self.set_budget(user_id, b)
