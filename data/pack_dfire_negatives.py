"""Reduce the three D-Fire packages to the background images only.

The training platform's YOLO export already holds every
labelled D-Fire image - 11,689 of them - but dropped all 9,838 background images at
import, because empty label files were treated as "unannotated". Re-uploading the
labelled ones would be pure duplication, so these packages carry only what is missing.

Source of truth for "already uploaded" is the export itself: its filenames are
Roboflow-renamed (`WEB07925_jpg.rf.<hash>.jpg`), so the original D-Fire stem is
recovered from the prefix and subtracted from each split.

The remainder is asserted to be exactly the set of images with empty label files, so a
mismatch fails loudly rather than quietly shipping the wrong images.

Each package keeps its empty `.txt` label file per image. That file is what marks an
image as *annotated with zero objects* rather than *awaiting annotation* - dropping it
is what caused the original problem.

    train  7,051      valid  782      test  2,005      total  9,838

    python pack_dfire_negatives.py path/to/platform_export.zip
"""

import re
import sys
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent
ROOT = DATA / "raw" / "dfire"
OUT = DATA / "roboflow"

RENAMED = re.compile(r"^(.+?)_jpg\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", re.I)

# The project already has its classes from the export, in that order. These packages
# contain no boxes at all, so nothing references an index - matching the existing order
# just avoids the importer reordering or duplicating classes.
DATA_YAML = """{split}: {split}/images

nc: 2
names: ['fire', 'smoke']

# D-Fire background images only - no boxes by design.
# Every .txt here is intentionally empty: these are confirmed negatives (sunsets,
# lights, reflections), not images awaiting annotation. On import, keep images
# without annotations, or the entire package is discarded.
"""


def exported_stems(export: Path) -> set[str]:
    with zipfile.ZipFile(export) as z:
        stems = set()
        for name in z.namelist():
            m = RENAMED.match(Path(name).name)
            if m:
                stems.add(m.group(1))
    return stems


def split_stems(list_file: Path) -> list[str]:
    return sorted(Path(p).stem for p in list_file.read_text(encoding="utf-8").split())


def is_background(stem: str, source_split: str) -> bool:
    label = ROOT / "labels" / source_split / f"{stem}.txt"
    return not label.exists() or not label.read_text(encoding="utf-8").strip()


def pack(split: str, stems: list[str], source_split: str, already: set[str]) -> int:
    remaining = [s for s in stems if s not in already]

    backgrounds = {s for s in stems if is_background(s, source_split)}
    unexpected = set(remaining) ^ backgrounds
    if unexpected:
        raise SystemExit(
            f"{split}: leftover set does not match the background set "
            f"({len(unexpected)} differ, e.g. {sorted(unexpected)[:3]}) - refusing to write"
        )

    out = OUT / f"dfire_{split}.zip"
    src_img = ROOT / "images" / source_split
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for s in remaining:
            z.write(src_img / f"{s}.jpg", f"{split}/images/{s}.jpg")
            z.writestr(f"{split}/labels/{s}.txt", "")
        z.writestr("data.yaml", DATA_YAML.format(split=split))

    print(
        f"{split:<6} {len(stems):>6} in package -> {len(already & set(stems)):>6} already "
        f"uploaded, {len(remaining):>5} kept   ({out.stat().st_size / 1e9:.2f} GB)"
    )
    return len(remaining)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python pack_dfire_negatives.py path/to/platform_export.zip")
    already = exported_stems(Path(sys.argv[1]))
    print(f"export holds {len(already):,} labelled images\n")

    splits = ROOT / "splits"
    jobs = [
        ("train", split_stems(splits / "train.txt"), "train"),
        ("valid", split_stems(splits / "val.txt"), "train"),
        ("test", sorted(p.stem for p in (ROOT / "images" / "test").glob("*.jpg")), "test"),
    ]

    total = sum(pack(split, stems, source, already) for split, stems, source in jobs)
    print(f"\n{total:,} background images packed (expected 9,838)")
    assert total == 9838, f"expected 9,838, packed {total}"
    print(f"packages -> {OUT}")


if __name__ == "__main__":
    main()
