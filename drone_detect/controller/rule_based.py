from __future__ import annotations
from .base import MetaController
from ..features.extractor import FeatureVector


class RuleBasedController(MetaController):
    window_size = 30

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        f = feature_history[-1]

        def _available(name: str) -> str:
            return name if name in pipeline_names else pipeline_names[0]

        # Low-light: dim and low contrast
        if f.mean_intensity < 60 and f.intensity_std < 25:
            return _available("clahe_pipeline")

        # High motion: prioritise speed
        if f.optical_flow_magnitude > 8.0:
            return _available("fast_baseline")

        # Small objects or dense scene
        if f.small_object_ratio > 0.5 or f.edge_density > 0.15:
            return _available("tiled")

        # Poor detection health: drop to higher-res model
        if f.mean_confidence < 0.35 and f.detection_count < 3:
            return _available("high_res")

        return _available("fast_baseline")
