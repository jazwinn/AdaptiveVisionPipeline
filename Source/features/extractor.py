from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from ..core.pipeline import Detection


@dataclass
class FeatureVector:
    laplacian_variance: float
    fft_blur_score: float
    mean_intensity: float
    intensity_std: float
    underexposed_ratio: float
    overexposed_ratio: float
    optical_flow_magnitude: float
    frame_displacement: float
    mean_confidence: float
    detection_count: int
    small_object_ratio: float
    edge_density: float
    entropy: float


def _blur_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _fft_blur_score(gray: np.ndarray) -> float:
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    # mean magnitude outside central low-freq region (radius = 60)
    mask = np.ones_like(magnitude, dtype=bool)
    y_idx, x_idx = np.ogrid[:h, :w]
    mask[(y_idx - cy) ** 2 + (x_idx - cx) ** 2 < 60 ** 2] = False
    return float(magnitude[mask].mean())


def _motion_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = float(mag.mean())
    # frame displacement: mean shift of corner region
    h, w = curr_gray.shape
    corners = flow[[0, -1, 0, -1], [0, 0, -1, -1]]
    displacement = float(np.linalg.norm(corners.mean(axis=0)))
    return magnitude, displacement


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.mean()) / 255.0


def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _detection_stats(dets: list[Detection]) -> tuple[float, int, float]:
    if not dets:
        return 0.0, 0, 0.0
    confidences = [d.confidence for d in dets]
    mean_conf = float(np.mean(confidences))
    small = 0
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        if (x2 - x1) < 32 and (y2 - y1) < 32:
            small += 1
    small_ratio = small / len(dets)
    return mean_conf, len(dets), small_ratio


class FeatureExtractor:
    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None

    def extract(self, frame_bgr: np.ndarray, detections: list[Detection]) -> FeatureVector:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        lap_var = _blur_laplacian(gray)
        fft_score = _fft_blur_score(gray)

        mean_i = float(gray.mean())
        std_i = float(gray.std())
        under = float((gray < 30).mean())
        over = float((gray > 225).mean())

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            flow_mag, displacement = _motion_flow(self._prev_gray, gray)
        else:
            flow_mag, displacement = 0.0, 0.0

        self._prev_gray = gray.copy()

        edge_d = _edge_density(gray)
        ent = _entropy(gray)
        mean_conf, det_count, small_ratio = _detection_stats(detections)

        return FeatureVector(
            laplacian_variance=lap_var,
            fft_blur_score=fft_score,
            mean_intensity=mean_i,
            intensity_std=std_i,
            underexposed_ratio=under,
            overexposed_ratio=over,
            optical_flow_magnitude=flow_mag,
            frame_displacement=displacement,
            mean_confidence=mean_conf,
            detection_count=det_count,
            small_object_ratio=small_ratio,
            edge_density=edge_d,
            entropy=ent,
        )

    def reset(self):
        self._prev_gray = None
