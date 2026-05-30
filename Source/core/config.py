from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TilingConfig:
    tile_size: int = 640
    overlap: float = 0.2
    min_tile_confidence: float = 0.25


@dataclass
class RuntimeConfig:
    mode: Literal["realtime", "offline"] = "offline"
    target_fps: float = 15.0
    allow_frame_skip: bool = True
    max_pipeline_cost: float = 999.0


@dataclass
class RunConfig:
    source_path: str
    controller_type: str
    pipeline_names: list[str]
    window_size: int
    notes: str = ""
