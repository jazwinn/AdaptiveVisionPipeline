from __future__ import annotations
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from ..features.extractor import FeatureVector

# Absolute default path — project root / replay_buffer.jsonl.
# This is resolved from __file__ so it is the same regardless of
# the working directory when the GUI or a training script is launched.
_DEFAULT_REPLAY_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "replay_buffer.jsonl"
)


class ReplayBuffer:
    def __init__(self, path: str = _DEFAULT_REPLAY_PATH, max_size: int = 10_000):
        self.path = path
        self.max_size = max_size

    def append(self, features: FeatureVector, pipeline: str, reward: float):
        entry = {
            "timestamp": time.time(),
            "features": asdict(features),
            "pipeline": pipeline,
            "reward": reward,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries[-self.max_size:]

    def __len__(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)

    def __bool__(self) -> bool:
        # Always truthy — an empty/new buffer is still a valid active buffer.
        # Without this, `if replay:` evaluates False when the file doesn't exist
        # yet (because __len__ returns 0), silently blocking all writes.
        return True
