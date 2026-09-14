"""Pre-annotate the reviewed refinery pool with a trained model and package it for Roboflow.

    python pack_refinery_roboflow.py --weights run44.pt [--conf 0.25]

Runs yolo26m-run-44 over data/raw/csb/train_pool, writes YOLO-format labels and packs the
same layout the D-Fire zips use:

    train/images/*.jpg
    train/labels/*.txt        empty file = background image, no boxes
    data.yaml

These are MODEL PREDICTIONS, not ground truth. They exist so annotators correct boxes
instead of drawing them from scratch. Every frame needs a human pass - the model has never
seen refinery footage, which is the whole reason this set exists.

CLASS ORDER: run-44 emits 0 = fire, 1 = smoke. D-Fire - and therefore every other zip in
data/roboflow - uses 0 = smoke, 1 = fire. Predictions are remapped to D-Fire order here so
the classes do not swap when both land in the same Roboflow project.
"""

import argparse
import csv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "raw" / "csb" / "train_pool"
OUT = ROOT / "data" / "roboflow"

# run-44 index -> D-Fire index
REMAP = {0: 1, 1: 0}          # fire 0->1, smoke 1->0


def data_yaml(n_images: int, conf: float) -> str:
    return f"""train: train/images

nc: 2
names: ['smoke', 'fire']

# Refinery pool: {n_images} reviewed frames from the US CSB video archive and Pexels stock.
# Class order matches D-Fire: 0 = smoke, 1 = fire.
#
# LABELS ARE MODEL PREDICTIONS, NOT GROUND TRUTH. Produced by yolo26m-run-44 at
# conf={conf}, purely as a starting point for annotation. The model was trained on D-Fire
# and FASDD and has never seen petrochemical plant footage, so expect it to miss thin
# plumes and to fire on flares. Every frame needs a human pass.
#
# Empty label files are frames the model found nothing in - they are background images,
# not unannotated ones.
#
# Provenance, licence and a per-frame review verdict for every image are in
# data/raw/csb/train_pool/manifest.csv. Two licences are mixed: CSB frames are US federal
# work (public domain), Pexels frames are Pexels licence.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True, help="trained fire/smoke weights (.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640, help="run-44 was trained at 640")
    ap.add_argument("--name", default="refinery_pool_train")
    args = ap.parse_args()

    from ultralytics import YOLO

    images = sorted(POOL.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {POOL}")

    model = YOLO(str(args.weights))
    print(f"{args.weights.name}  names={model.names}  ->  remapped to ['smoke', 'fire']")
    print(f"{len(images)} images, conf={args.conf}, imgsz={args.imgsz}\n")

    labels: dict[str, str] = {}
    counts = {"smoke": 0, "fire": 0}
    empty = 0

    for i in range(0, len(images), 16):
        chunk = images[i:i + 16]
        for path, res in zip(chunk, model.predict(chunk, conf=args.conf, imgsz=args.imgsz,
                                                  verbose=False)):
            lines = []
            for box in res.boxes:
                cls = REMAP[int(box.cls)]
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts["smoke" if cls == 0 else "fire"] += 1
            labels[path.stem] = "\n".join(lines) + ("\n" if lines else "")
            empty += not lines
        print(f"  {min(i + 16, len(images))}/{len(images)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for path in images:
            z.write(path, f"train/images/{path.name}")
            z.writestr(f"train/labels/{path.stem}.txt", labels[path.stem])
        z.writestr("data.yaml", data_yaml(len(images), args.conf))
        z.write(POOL / "manifest.csv", "manifest.csv")

    print(f"\n{len(images)} images, {counts['smoke']} smoke + {counts['fire']} fire boxes, "
          f"{empty} frames with no detection")
    print(f"-> {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    # per-source hit rate, so it is obvious where the model is blind
    manifest = {r["frame_file"]: r for r in csv.DictReader(
        open(POOL / "manifest.csv", encoding="utf-8"))}
    per: dict[str, list[int]] = {}
    for path in images:
        r = manifest.get(path.name, {})
        key = f"{r.get('youtube_id', '?')} [{r.get('label_hint', '?')}]"
        s = per.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += bool(labels[path.stem].strip())
    print(f"\n{'source [label_hint]':46s} {'imgs':>5s} {'detected':>9s}")
    for k, (n, d) in sorted(per.items()):
        print(f"{k:46s} {n:5d} {d:9d}")


if __name__ == "__main__":
    main()
