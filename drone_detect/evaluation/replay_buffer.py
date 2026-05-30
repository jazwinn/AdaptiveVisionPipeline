from __future__ import annotations
import json
import os
import time
from dataclasses import asdict
from ..features.extractor import FeatureVector


class ReplayBuffer:
    def __init__(self, path: str = "replay_buffer.jsonl", max_size: int = 10_000):
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
