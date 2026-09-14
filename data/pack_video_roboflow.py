"""Pre-annotate the deduped video pool with run-58 and package it for Roboflow.

    python pack_video_roboflow.py --weights run58.pt [--conf 0.25]

Separate package from refinery_pool_train.zip - different intake, different model, and a
much larger frame count. Same layout the D-Fire zips use:

    train/images/*.jpg
    train/labels/*.txt        empty file = background image, no boxes
    data.yaml

These are MODEL PREDICTIONS, not ground truth. They exist so annotators correct boxes
instead of drawing them from scratch.

CLASS ORDER: run-58 emits 0 = fire, 1 = smoke. D-Fire - and every other zip in
data/roboflow - uses 0 = smoke, 1 = fire. Predictions are remapped to D-Fire order here so
the classes do not swap when packages land in the same Roboflow project.
"""

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "raw" / "video_pool"
OUT = ROOT / "data" / "roboflow"

REMAP = {0: 1, 1: 0}          # run-58 fire 0->1, smoke 1->0


def data_yaml(n_images: int, conf: float) -> str:
    return f"""train: train/images

nc: 2
names: ['smoke', 'fire']

# Video pool: {n_images} deduplicated frames from 139 YouTube clips sampled at 1 fps -
# refinery, chemical plant, warehouse and industrial fires, plus flare and steam negatives.
# Class order matches D-Fire: 0 = smoke, 1 = fire.
#
# LABELS ARE MODEL PREDICTIONS, NOT GROUND TRUTH. Produced by yolo26m-run-58 at
# conf={conf} as a starting point for annotation. Every frame needs a human pass.
# Empty label files are frames the model found nothing in - background images, not
# unannotated ones.
#
# NO per-frame content review was done at this scale: expect news studio shots, reporter
# stand-ups, titles and identifiable people to be present. Only whole clips were cut.
#
# manifest.csv (at the zip root) carries source video, YouTube id, timestamp and a `domain`
# column - `industrial` vs `structure` / `vehicle` / `steam` / `artificial_smoke` - so
# wrong-domain footage can be filtered out rather than silently trained on.
#
# LICENCE: YouTube standard licence. Training use only - do not redistribute the frames.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="trained fire/smoke weights (.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="video_pool_train")
    args = ap.parse_args()

    from ultralytics import YOLO

    images = sorted(POOL.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {POOL}")

    model = YOLO(str(args.weights))
    print(f"{args.weights.name}  names={model.names}  ->  remapped to ['smoke', 'fire']")
    print(f"{len(images)} images, conf={args.conf}, imgsz={args.imgsz}\n")

    labels: dict[str, str] = {}
    counts = Counter()
    empty = 0

    for i in range(0, len(images), 32):
        chunk = images[i:i + 32]
        for path, res in zip(chunk, model.predict(chunk, conf=args.conf, imgsz=args.imgsz,
                                                  verbose=False)):
            lines = []
            for box in res.boxes:
                cls = REMAP[int(box.cls)]
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts["smoke" if cls == 0 else "fire"] += 1
            labels[path.name] = "\n".join(lines) + ("\n" if lines else "")
            empty += not lines
        if (i // 32) % 20 == 0:
            print(f"  {min(i + 32, len(images))}/{len(images)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for path in images:
            z.write(path, f"train/images/{path.name}")
            z.writestr(f"train/labels/{path.stem}.txt", labels[path.name])
        z.writestr("data.yaml", data_yaml(len(images), args.conf))
        z.write(POOL / "manifest.csv", "manifest.csv")

    print(f"\n{len(images)} images, {counts['smoke']} smoke + {counts['fire']} fire boxes, "
          f"{empty} frames with no detection")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    manifest = {r["frame_file"]: r for r in csv.DictReader(
        open(POOL / "manifest.csv", encoding="utf-8"))}
    per = {}
    for path in images:
        key = manifest.get(path.name, {}).get("domain", "?")
        s = per.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += bool(labels[path.name].strip())
    print(f"\n{'domain':18s} {'imgs':>6s} {'detected':>9s}")
    for k, (n, d) in sorted(per.items()):
        print(f"{k:18s} {n:6d} {d:9d}")


if __name__ == "__main__":
    main()
