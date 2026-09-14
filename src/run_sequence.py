"""Run the detector + alarm layer over a folder of frames, or a video.

    python src/run_sequence.py data/raw/csb/frames/hewMRcrOI0s --weights best.pt --fps 2
    python src/run_sequence.py data/raw/csb/frames/hewMRcrOI0s --weights best.pt --fps 2 --annotate out/

Frame folders produced by data/extract_frames.py sort into playback order by filename,
and --fps tells the state machine how much wall-clock time each step represents so the
cooldown means seconds rather than frames.

Prints a per-frame trace and every alarm raised, then a summary. Detections that were
suppressed - by ROI mask, by threshold, by N-of-M, by the growth check - are counted
separately, because when tuning you need to know *which* mechanism rejected something.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alarm import AlarmStateMachine, CameraConfig  # noqa: E402
from detector import Detector  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def load_camera(path: Path | None, camera_id: str) -> CameraConfig:
    if path is None:
        return CameraConfig(camera_id=camera_id)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    entry = cfg.get(camera_id) or cfg.get("default") or {}
    entry.setdefault("camera_id", camera_id)
    entry["roi_masks"] = [[tuple(p) for p in poly] for poly in entry.get("roi_masks", [])]
    return CameraConfig(**entry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="folder of frames, or a video file")
    # run-44 is the 640-trained model. It beats the 1024 model on every confusion-matrix
    # cell (fire recall 67.8 vs 66.4, smoke 80.2 vs 78.7, macro recall 74.0 vs 72.5) and
    # is ~2.5x faster. Keep --imgsz matched to the resolution the weights were trained at.
    ap.add_argument("--weights", required=True, help="trained YOLO fire/smoke weights (.pt), not distributed")
    ap.add_argument("--fps", type=float, default=2.0, help="playback rate of the source")
    ap.add_argument("--conf", type=float, default=0.25, help="model floor, before per-camera threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cameras", type=Path, default=None, help="JSON of camera configs")
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--annotate", type=Path, default=None, help="write annotated frames here")
    ap.add_argument("--csv", type=Path, default=None, help="write the per-frame trace here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_dir():
        frames = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not frames:
            raise SystemExit(f"no images in {src}")
    else:
        frames = None

    camera_id = args.camera_id or src.stem
    cfg = load_camera(args.cameras, camera_id)
    det = Detector(args.weights, conf=args.conf, imgsz=args.imgsz)
    machine = AlarmStateMachine(cfg)

    print(f"source     {src}")
    print(f"camera     {cfg.camera_id}  zone={cfg.zone}")
    print(f"rules      conf>={cfg.conf_threshold}  {cfg.n_of}-of-{cfg.m_window}  "
          f"growth>={cfg.growth_ratio if cfg.require_growth else 'off'}  "
          f"cooldown={cfg.cooldown_s}s  masks={len(cfg.roi_masks)}")
    print()

    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)

    rows, alarms = [], []
    raw_total = kept_total = 0
    step = 1.0 / args.fps

    source = frames if frames is not None else _video_frames(src)
    for i, item in enumerate(source):
        ts = i * step
        image = cv2.imread(str(item)) if isinstance(item, Path) else item
        if image is None:
            continue

        detections = det.detect(image)
        kept = [d for d in detections if machine._keep(d)]  # same predicate the machine uses
        raw_total += len(detections)
        kept_total += len(kept)

        event = machine.update(detections, timestamp=ts)
        if event:
            alarms.append(event)
            print(f"  ALARM  t={ts:7.1f}s  {event.cls:<5} conf={event.confidence:.2f}  "
                  f"seen={event.frames_seen}/{cfg.m_window}  growth={event.growth:.2f}")

        rows.append(
            {
                "frame": item.name if isinstance(item, Path) else i,
                "t_s": round(ts, 3),
                "raw": len(detections),
                "kept": len(kept),
                "suppressed": len(detections) - len(kept),
                "classes": "|".join(sorted({d.cls for d in kept})),
                "max_conf": round(max((d.conf for d in kept), default=0.0), 3),
                "alarm": event.cls if event else "",
            }
        )

        if args.annotate:
            _draw(image, detections, kept, cfg, bool(event))
            name = item.name if isinstance(item, Path) else f"frame_{i:06d}.jpg"
            cv2.imwrite(str(args.annotate / name), image)

        if not args.quiet and i % 50 == 0:
            print(f"  {i:>5}  t={ts:7.1f}s  raw={len(detections)} kept={len(kept)}")

    n = len(rows)
    frames_with_kept = sum(1 for r in rows if r["kept"])
    print(f"\n{n} frames, {n * step:.0f}s of footage")
    print(f"  raw detections         {raw_total}")
    print(f"  survived filtering     {kept_total}   (suppressed {raw_total - kept_total})")
    print(f"  frames with detection  {frames_with_kept} ({frames_with_kept / n:.1%})")
    print(f"  ALARMS RAISED          {len(alarms)}")
    if n * step > 0:
        print(f"  alarm rate             {len(alarms) / (n * step) * 3600:.1f} per hour")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ntrace -> {args.csv}")


def _video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def _draw(image, detections, kept, cfg, alarming: bool) -> None:
    for poly in cfg.roi_masks:
        import numpy as np

        cv2.polylines(image, [np.array(poly, dtype=np.int32)], True, (255, 200, 0), 2)

    keep_ids = {id(d) for d in kept}
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        live = id(d) in keep_ids
        colour = (0, 0, 255) if live else (128, 128, 128)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2 if live else 1)
        label = f"{d.cls} {d.conf:.2f}" + ("" if live else " [suppressed]")
        cv2.putText(image, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)

    if alarming:
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(image, "ALARM", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
