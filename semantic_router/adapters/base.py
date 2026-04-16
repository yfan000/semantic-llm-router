from __future__ import annotations
from abc import ABC, abstractmethod
from semantic_router.schemas import BidRequest, BidResponse


class ModelAdapter(ABC):
    def __init__(self, model_id: str, base_url: str, efficiency_tokens_per_joule: float,
                 max_concurrent_requests: int, input_rate_usd_per_token: float,
                 output_rate_usd_per_token: float, accuracy_priors: dict[str, float] | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.efficiency_tokens_per_joule = efficiency_tokens_per_joule
        self.max_concurrent_requests = max_concurrent_requests
        self.input_rate = input_rate_usd_per_token
        self.output_rate = output_rate_usd_per_token
        self.accuracy_priors: dict[str, float] = accuracy_priors or {}

    def get_accuracy_prior(self, domain: str, complexity: str) -> float:
        from semantic_router.config import DEFAULT_ACCURACY_PRIOR
        return self.accuracy_priors.get(f"{domain}:{complexity}", DEFAULT_ACCURACY_PRIOR)

    def _estimate_output_tokens(self, domain: str, complexity: str) -> int:
        TABLE = {("factual","easy"):80,("factual","medium"):200,("factual","hard"):350,
                 ("math","easy"):120,("math","medium"):280,("math","hard"):450,
                 ("code","easy"):150,("code","medium"):350,("code","hard"):650,
                 ("creative","easy"):250,("creative","medium"):500,("creative","hard"):800,
                 ("reasoning","easy"):180,("reasoning","medium"):380,("reasoning","hard"):600}
        return TABLE.get((domain, complexity), 300)

    @abstractmethod
    async def get_load(self) -> float: ...
    @abstractmethod
    async def bid(self, request: BidRequest) -> BidResponse: ...
    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> dict: ...
