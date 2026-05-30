from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional
import cv2
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _detect_source_type(source: str) -> Literal["video", "image", "directory"]:
    p = Path(source)
    if p.is_dir():
        return "directory"
    if p.suffix.lower() in _IMAGE_EXTS:
        return "image"
    return "video"


@dataclass
class Frame:
    index: int
    timestamp_ms: float
    image: np.ndarray
    source_path: str


class FrameReader:
    def __init__(self, source: str, max_fps: Optional[float] = None):
        self.source = source
        self.max_fps = max_fps
        self.source_type = _detect_source_type(source)

        if self.source_type == "image":
            img = cv2.imread(source)
            if img is None:
                raise RuntimeError(f"Cannot read image: {source}")
            self._single_image = img
            h, w = img.shape[:2]
            self._fps = 1.0
            self._frame_count = 1
            self._width = w
            self._height = h
            self.cap = None

        elif self.source_type == "directory":
            p = Path(source)
            paths = sorted(
                fp for fp in p.iterdir()
                if fp.suffix.lower() in _IMAGE_EXTS
            )
            if not paths:
                raise RuntimeError(
                    f"No images found in directory: {source}  "
                    f"(supported extensions: {sorted(_IMAGE_EXTS)})"
                )
            self._image_paths = paths
            probe = cv2.imread(str(paths[0]))
            if probe is None:
                raise RuntimeError(f"Cannot read first image: {paths[0]}")
            h, w = probe.shape[:2]
            self._fps = 1.0
            self._frame_count = len(paths)
            self._width = w
            self._height = h
            self.cap = None

        else:  # video
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def fps(self) -> float:
        if self.source_type == "video":
            return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return self._fps

    @property
    def frame_count(self) -> int:
        if self.source_type == "video":
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._frame_count

    @property
    def width(self) -> int:
        if self.source_type == "video":
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self.source_type == "video":
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    # ------------------------------------------------------------------ #
    # Iteration                                                            #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Frame]:
        if self.source_type == "image":
            yield Frame(index=0, timestamp_ms=0.0, image=self._single_image, source_path=self.source)
            return

        if self.source_type == "directory":
            for i, path in enumerate(self._image_paths):
                img = cv2.imread(str(path))
                if img is None:
                    continue
                yield Frame(index=i, timestamp_ms=float(i * 1000), image=img, source_path=str(path))
            return

        # video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        frame_interval = (1.0 / self.max_fps) if self.max_fps else 0.0
        last_ts = -frame_interval

        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                break
            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            if self.max_fps and (ts / 1000.0 - last_ts) < frame_interval:
                idx += 1
                continue

            last_ts = ts / 1000.0
            yield Frame(index=idx, timestamp_ms=ts, image=image, source_path=self.source)
            idx += 1

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
