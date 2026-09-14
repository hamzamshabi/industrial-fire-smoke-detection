"""Detections in, alarms out.

A detector alone is not a system. In a refinery the flare burns 24/7 and the cooling
towers vent continuously, so a model that fires on every flame-shaped thing produces an
alarm every frame forever and gets switched off in week one. This layer is what makes the
output usable, and it is deliberately deterministic - a steady plume cannot survive the
growth check regardless of how confident the model is about it.

Four mechanisms, applied in order:

  ROI mask       drop detections inside per-camera flare/vent polygons
  threshold      per-camera confidence floor, so one noisy camera can be tuned alone
  N-of-M         require detection in N of the last M frames - kills single-frame flicker
  growth check   require the box to expand across the window - a real fire grows,
                 a flare and a steam plume do not

then a cooldown so one incident does not produce a hundred notifications.

The two-class model has no `flare` or `steam` class, so geometry and behaviour carry the
whole burden of nuisance rejection here. ROI masks handle a flare that stays put; the
growth check handles plumes that wander out of the mask when the wind moves them.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Sequence

import cv2
import numpy as np

from detector import Detection

Polygon = Sequence[tuple[float, float]]


@dataclass
class CameraConfig:
    """Per-camera tuning. Zones differ - a tank farm camera and a cooling tower camera
    should not share a confidence floor."""

    camera_id: str
    zone: str = "unknown"
    roi_masks: list[Polygon] = field(default_factory=list)

    conf_threshold: float = 0.35
    alarm_classes: tuple[str, ...] = ("fire", "smoke")

    # Per-class overrides of conf_threshold. The confusion matrix says `fire` is both the
    # weaker class (67.8% recall vs smoke's 80.2%) and the source of ~69% of all false
    # positives, so it earns a higher bar than smoke rather than sharing one floor.
    # In a refinery the fire false positives are flare tips, sun glare off steel and
    # plant lighting - exactly what the ROI masks also target.
    class_thresholds: dict[str, float] = field(default_factory=lambda: {"fire": 0.45})

    n_of: int = 5          # need N detections
    m_window: int = 10     # within the last M frames
    require_growth: bool = True
    growth_ratio: float = 1.15   # last third vs first third of the window
    cooldown_s: float = 60.0

    # Free-text, carried through to the alert. The model reports `fire`; the config
    # supplies what is likely burning, with no extra annotation.
    fuel: str = "unknown"
    note: str = ""


@dataclass
class AlarmEvent:
    camera_id: str
    zone: str
    cls: str
    timestamp: float
    confidence: float
    box: tuple[float, float, float, float]
    frames_seen: int
    growth: float
    fuel: str = "unknown"
    note: str = ""


@dataclass
class _Observation:
    timestamp: float
    detections: list[Detection]


class AlarmStateMachine:
    """One instance per camera. Feed it every frame; it returns an AlarmEvent or None."""

    def __init__(self, config: CameraConfig) -> None:
        self.cfg = config
        self._window: deque[_Observation] = deque(maxlen=config.m_window)
        self._last_alarm: dict[str, float] = {}
        self._masks = [np.array(p, dtype=np.int32) for p in config.roi_masks]

    # ---- filtering -------------------------------------------------------------

    def _in_mask(self, det: Detection) -> bool:
        """True when the detection originates inside an excluded zone.

        Tested at the bottom-centre rather than the centroid: a plume rising from a
        masked flare tip has a centroid well above the mask within a second or two,
        while its base stays put.
        """
        x, y = det.bottom_center
        return any(cv2.pointPolygonTest(m, (float(x), float(y)), False) >= 0 for m in self._masks)

    def _threshold(self, cls: str) -> float:
        return self.cfg.class_thresholds.get(cls, self.cfg.conf_threshold)

    def _keep(self, det: Detection) -> bool:
        return (
            det.cls in self.cfg.alarm_classes
            and det.conf >= self._threshold(det.cls)
            and not self._in_mask(det)
        )

    # ---- growth ----------------------------------------------------------------

    def _growth(self, cls: str) -> float:
        """Ratio of mean box area in the last third of the window to the first third.

        Tracks the largest box of the class per frame rather than matching identities -
        a prototype simplification that holds while there is one event in view. With two
        simultaneous fires in one frame this would need real tracking.
        """
        areas = [
            max((d.area for d in obs.detections if d.cls == cls), default=None)
            for obs in self._window
        ]
        areas = [a for a in areas if a is not None]
        if len(areas) < 4:
            return 1.0

        cut = max(1, len(areas) // 3)
        early, late = areas[:cut], areas[-cut:]
        if not early or mean(early) <= 0:
            return 1.0
        return mean(late) / mean(early)

    # ---- main ------------------------------------------------------------------

    def update(
        self, detections: Iterable[Detection], timestamp: float | None = None
    ) -> AlarmEvent | None:
        ts = time.time() if timestamp is None else timestamp
        kept = [d for d in detections if self._keep(d)]
        self._window.append(_Observation(ts, kept))

        for cls in self.cfg.alarm_classes:
            frames_seen = sum(1 for obs in self._window if any(d.cls == cls for d in obs.detections))
            if frames_seen < self.cfg.n_of:
                continue

            last = self._last_alarm.get(cls)
            if last is not None and ts - last < self.cfg.cooldown_s:
                continue

            growth = self._growth(cls)
            if self.cfg.require_growth and growth < self.cfg.growth_ratio:
                continue

            best = max(
                (d for obs in self._window for d in obs.detections if d.cls == cls),
                key=lambda d: d.conf,
            )
            self._last_alarm[cls] = ts
            return AlarmEvent(
                camera_id=self.cfg.camera_id,
                zone=self.cfg.zone,
                cls=cls,
                timestamp=ts,
                confidence=best.conf,
                box=best.xyxy,
                frames_seen=frames_seen,
                growth=growth,
                fuel=self.cfg.fuel,
                note=self.cfg.note,
            )
        return None

    def reset(self) -> None:
        self._window.clear()
        self._last_alarm.clear()
