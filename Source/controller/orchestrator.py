from __future__ import annotations
from collections import deque
from ..core.pipeline import DetectionPipeline, Detection
from ..core.frame_reader import Frame
from ..core.config import RuntimeConfig
from ..features.extractor import FeatureVector
from .base import MetaController


class PipelineOrchestrator:
    def __init__(
        self,
        controller: MetaController,
        pipelines: list[DetectionPipeline],
        window_size: int = 30,
        runtime_config: RuntimeConfig | None = None,
    ):
        self.controller = controller
        self.pipelines: dict[str, DetectionPipeline] = {p.name: p for p in pipelines}
        self.window_size = window_size
        self.runtime_config = runtime_config or RuntimeConfig()
        self.feature_buffer: deque[FeatureVector] = deque(maxlen=window_size)
        self.current_pipeline_name: str = pipelines[0].name
        self.frame_count = 0
        self.switch_log: list[tuple[int, str]] = []

    def process(self, frame: Frame, features: FeatureVector) -> tuple[list[Detection], dict]:
        self.feature_buffer.append(features)
        self.frame_count += 1

        if self.frame_count % self.window_size == 0 and len(self.feature_buffer) > 0:
            new_name = self.controller.select_pipeline(
                list(self.feature_buffer),
                list(self.pipelines.keys()),
            )
            if self.runtime_config.mode == "realtime":
                new_name = self._clamp_to_budget(new_name)
            if new_name != self.current_pipeline_name:
                self.switch_log.append((self.frame_count, new_name))
            self.current_pipeline_name = new_name

        pipeline = self.pipelines[self.current_pipeline_name]
        dets, meta = pipeline.run(frame)
        meta["selected_pipeline"] = self.current_pipeline_name
        return dets, meta

    def _clamp_to_budget(self, preferred_name: str) -> str:
        max_cost = self.runtime_config.max_pipeline_cost
        if self.pipelines[preferred_name].cost_estimate <= max_cost:
            return preferred_name
        # Fall back to cheapest pipeline within budget
        candidates = [
            (p.cost_estimate, name)
            for name, p in self.pipelines.items()
            if p.cost_estimate <= max_cost
        ]
        if not candidates:
            return min(self.pipelines, key=lambda n: self.pipelines[n].cost_estimate)
        return min(candidates)[1]

    def count_switches(self) -> int:
        return len(self.switch_log)
