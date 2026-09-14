"""Unpack the D-Fire parquet shards into a standard YOLO layout on disk.

    raw/dfire/images/{train,test}/<name>.jpg
    raw/dfire/labels/{train,test}/<name>.txt      (empty file = background, no boxes)

Class ids follow D-Fire: 0 = smoke, 1 = fire.

Also verifies the mirror against the figures published in the D-Fire repo:
21,527 images, 26,557 boxes (11,865 smoke / 14,692 fire). A mismatch means the
HuggingFace mirror is not a faithful copy and the canonical OneDrive zip should
be used instead.
"""

import glob
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent / "raw" / "dfire"
CLASS_NAMES = {0: "smoke", 1: "fire"}
OFFICIAL = {"images": 21527, "smoke": 11865, "fire": 14692}


def export(split: str, shards: list[str]) -> Counter:
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                stem = Path(row["filename"]).stem
                (img_dir / f"{stem}.jpg").write_bytes(row["image"]["bytes"])

                label = (row["label"] or "").strip()
                (lbl_dir / f"{stem}.txt").write_text(
                    label + "\n" if label else "", encoding="utf-8"
                )

                counts["images"] += 1
                for line in label.splitlines():
                    if line.strip():
                        counts[CLASS_NAMES[int(line.split()[0])]] += 1
        print(f"  {Path(shard).name}: {counts['images']} images so far", flush=True)
    return counts


def main() -> None:
    totals = Counter()
    for split, pattern in (("train", "data/train-*.parquet"), ("test", "data/test-*.parquet")):
        shards = sorted(glob.glob(str(ROOT / pattern)))
        print(f"{split}: {len(shards)} shards")
        c = export(split, shards)
        print(f"{split}: {c['images']} images, {c['smoke']} smoke, {c['fire']} fire boxes\n")
        totals.update(c)

    print("=== verification against the published D-Fire figures")
    ok = True
    for key, expected in OFFICIAL.items():
        got = totals[key]
        flag = "OK " if got == expected else "MISMATCH"
        ok &= got == expected
        print(f"  {flag} {key:>7}: got {got:,}  expected {expected:,}")
    print("\nmirror is faithful" if ok else "\nmirror DIFFERS from the published figures")


if __name__ == "__main__":
    main()
