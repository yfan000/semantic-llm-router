from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, model_validator


class RouterMode(str, Enum):
    ACCURACY = "accuracy"
    ECO = "eco"
    COST = "cost"
    CUSTOM = "custom"


class UserPreference(BaseModel):
    mode: RouterMode = RouterMode.CUSTOM
    cost_weight: float = 0.25
    latency_weight: float = 0.25
    accuracy_weight: float = 0.25
    energy_weight: float = 0.25
    max_latency_ms: Optional[int] = None
    min_accuracy: Optional[float] = None

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "UserPreference":
        total = self.cost_weight + self.latency_weight + self.accuracy_weight + self.energy_weight
        if abs(total - 1.0) > 1e-6:
            self.cost_weight /= total
            self.latency_weight /= total
            self.accuracy_weight /= total
            self.energy_weight /= total
        return self


class UserBudget(BaseModel):
    remaining_tokens: int = 1_000_000
    remaining_energy_j: float = 36_000.0


class RequestSLA(BaseModel):
    user_id: Optional[str] = None
    mode: Optional[RouterMode] = None
    cost_weight: Optional[float] = None
    latency_weight: Optional[float] = None
    accuracy_weight: Optional[float] = None
    energy_weight: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_accuracy: Optional[float] = None


class BidRequest(BaseModel):
    messages: list[dict]
    complexity: str
    domain: str
    query_embedding: list[float]
    preference: UserPreference


class BidResponse(BaseModel):
    model_id: str
    estimated_cost_usd: float
    estimated_latency_ms: int
    estimated_accuracy: float
    estimated_energy_j: float
    efficiency_tokens_per_joule: float
    current_load: float


class ModelReputation(BaseModel):
    model_id: str
    latency_reliability: float = 1.0
    sample_count: int = 0
    accuracy_priors: dict[str, float] = {}
