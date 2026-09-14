"""detect(image) -> list[Detection].

Thin wrapper over the trained YOLO model. Deliberately thin: everything that turns a
detection into an alarm lives in alarm.py, because that is the part with the engineering
value and the part that needs to be tunable without touching the model.

Classes are the model's own: 0 = fire, 1 = smoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Where the detection meets the ground/equipment - better for ROI tests than
        the centroid, since a plume's centroid drifts far from its source as it rises."""
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, y2)


class Detector:
    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = self.model.names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """One frame in, detections out. `conf` here is the floor; per-camera thresholds
        are applied later in the alarm layer so a single model serves every camera."""
        res = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        out = []
        if res.boxes is None:
            return out
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(
                Detection(
                    cls=self.names[int(box.cls.item())],
                    conf=float(box.conf.item()),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out
