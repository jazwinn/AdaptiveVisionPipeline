"""
No-op controller: always selects the first pipeline (plain object detection).

Use this as a clean baseline to compare against adaptive controllers.
No feature analysis, no model inference, zero pipeline switches.
"""
from __future__ import annotations

from .base import MetaController
from ..features.extractor import FeatureVector


class NoneController(MetaController):
    """
    Passes all frames through the first pipeline unchanged.

    No adaptive logic, no pipeline switches.  The first pipeline is always
    ``fast_baseline`` (PipelineA / YOLOv8n), giving a plain object-detection
    baseline with zero overhead from controller decision-making.
    """

    window_size: int = 30

    def select_pipeline(
        self,
        feature_history: list[FeatureVector],
        pipeline_names: list[str],
    ) -> str:
        return pipeline_names[0]

    # update() intentionally not overridden — base-class no-op is sufficient.
