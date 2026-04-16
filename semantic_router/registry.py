from __future__ import annotations
from dataclasses import dataclass
from semantic_router.adapters.base import ModelAdapter


@dataclass
class ModelConfig:
    adapter: ModelAdapter
    domains: list[str]
    min_accuracy_capability: float


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        self._models[config.adapter.model_id] = config

    def deregister(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def get_eligible(self, domain: str, min_accuracy: float | None) -> list[ModelAdapter]:
        return [
            cfg.adapter for cfg in self._models.values()
            if (domain in cfg.domains or "general" in cfg.domains)
            and (min_accuracy is None or cfg.min_accuracy_capability >= min_accuracy)
        ]

    def list_all(self) -> list[str]:
        return list(self._models.keys())

    def get_adapter(self, model_id: str) -> ModelAdapter | None:
        cfg = self._models.get(model_id)
        return cfg.adapter if cfg else None
