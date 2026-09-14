"""Behaviour tests for the alarm layer. Pure logic: no model, no weights, no GPU.

    python -m pytest tests
    python tests/test_alarm.py        # same tests without pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alarm import AlarmStateMachine, CameraConfig  # noqa: E402
from detector import Detection  # noqa: E402

DT = 0.5  # seconds per frame, i.e. 2 fps


def det(cls: str, size: float, conf: float = 0.9, cx: float = 320, bottom: float = 400) -> Detection:
    """Square box of side `size` standing on (cx, bottom)."""
    return Detection(cls, conf, (cx - size / 2, bottom - size, cx + size / 2, bottom))


def run(cfg: CameraConfig, frames) -> list[tuple[float, str]]:
    """Feed frames (each a list of detections) at DT spacing; return (t, class) per alarm."""
    machine = AlarmStateMachine(cfg)
    alarms = []
    for i, dets in enumerate(frames):
        event = machine.update(dets, timestamp=i * DT)
        if event:
            alarms.append((event.timestamp, event.cls))
    return alarms


def growing(cls: str, n: int, rate: float = 1.05, conf: float = 0.9, **kw):
    return [[det(cls, 40 * rate**i, conf, **kw)] for i in range(n)]


def test_growing_fire_alarms_on_the_fifth_frame():
    # 5-of-10 is met on frame 5 (t = 2.0 s); area has grown 1.05^8 = 1.48x > 1.15.
    assert run(CameraConfig("cam"), growing("fire", 10)) == [(2.0, "fire")]


def test_steady_flare_never_alarms():
    # Seen in every frame at high confidence, but its box does not grow.
    assert run(CameraConfig("cam"), [[det("fire", 40)] for _ in range(60)]) == []


def test_single_frame_flicker_never_alarms():
    # A growing fire seen only every third frame: never 5 hits in any 10-frame window.
    frames = [[det("fire", 40 * 1.05**i)] if i % 3 == 0 else [] for i in range(60)]
    assert run(CameraConfig("cam"), frames) == []


def test_roi_mask_uses_the_base_of_the_box():
    # Mask over the base of the box: suppressed.
    base_mask = [(300, 380), (340, 380), (340, 420), (300, 420)]
    assert run(CameraConfig("cam", roi_masks=[base_mask]), growing("fire", 20)) == []

    # Mask over everything above the base (the plume that rises out of a masked flare):
    # the box centroid is inside it but the base is not, so the alarm still fires.
    sky_mask = [(0, 0), (640, 0), (640, 390), (0, 390)]
    assert run(CameraConfig("cam", roi_masks=[sky_mask]), growing("fire", 10)) == [(2.0, "fire")]


def test_fire_needs_a_higher_confidence_than_smoke():
    # 0.40 is under the fire floor (0.45) but over the default floor smoke uses (0.35).
    assert run(CameraConfig("cam"), growing("fire", 20, conf=0.40)) == []
    assert run(CameraConfig("cam"), growing("smoke", 20, conf=0.40)) == [(2.0, "smoke")]


def test_cooldown_holds_repeat_alarms_for_60_seconds():
    alarms = run(CameraConfig("cam"), growing("smoke", 200, rate=1.02))
    assert alarms == [(2.0, "smoke"), (62.0, "smoke")]


def test_growth_check_can_be_disabled():
    cfg = CameraConfig("cam", require_growth=False)
    assert run(cfg, [[det("fire", 40)] for _ in range(10)]) == [(2.0, "fire")]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
