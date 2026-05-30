from __future__ import annotations
from abc import ABC, abstractmethod
from ..features.extractor import FeatureVector


class MetaController(ABC):
    window_size: int = 30

    @abstractmethod
    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str: ...

    def update(self, pipeline_name: str, reward: float, features: FeatureVector | None = None):
        pass
