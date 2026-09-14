"""Carve a validation split out of the D-Fire training set without leaking video frames.

D-Fire ships train/test only. Selecting a val set at random would put near-duplicate
frames of the same video on both sides: AoF images are consecutive frames, and adjacent
indices routinely differ by almost nothing. Model selection against such a val set is
optimistic, and the baseline it produces is the number every later ablation is compared
against.

So group first, then split:

1. Hash every training image (dHash, 64-bit).
2. Walk each filename prefix in index order. Consecutive indices whose hashes are within
   HAMMING_THRESHOLD join the same group - one group is one shot. A gap in the index
   sequence, or a large hash jump, starts a new group.
3. Assign whole groups to train or val, stratified by the group's class signature
   (fire / smoke / both / background) so both sides keep D-Fire's class balance.

No group is ever divided, so no frame in val has a near-duplicate in train.

Writes splits/{train,val}.txt (one image path per line, YOLO-style) and splits/groups.csv
recording the group id, prefix, index range, size and class signature of every group.

    python split_dfire_val.py                 # 10% val, seed 0
    python split_dfire_val.py --val-frac 0.15
"""

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "raw" / "dfire"
IMAGES = ROOT / "images" / "train"
LABELS = ROOT / "labels" / "train"
OUT = ROOT / "splits"

HAMMING_THRESHOLD = 10  # <= this between adjacent indices means "same shot"
NAME_RE = re.compile(r"^([A-Za-z_]+)(\d+)$")
CLASS_NAMES = {0: "smoke", 1: "fire"}


def dhash(path: Path) -> int:
    """64-bit difference hash. draft() lets libjpeg decode at reduced size - much faster."""
    with Image.open(path) as im:
        im.draft("L", (32, 32))
        im = im.convert("L").resize((9, 8), Image.BILINEAR)
        px = list(im.tobytes())
    bits = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            bits = (bits << 1) | (px[base + col] > px[base + col + 1])
    return bits


def signature(stem: str) -> str:
    """Class signature of one image: fire / smoke / both / background."""
    p = LABELS / f"{stem}.txt"
    classes = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                classes.add(CLASS_NAMES[int(line.split()[0])])
    if not classes:
        return "background"
    return "both" if len(classes) == 2 else classes.pop()


def build_groups(stems: list[str], hashes: dict[str, int]) -> list[dict]:
    """Group consecutive, visually similar indices within each prefix into shots."""
    by_prefix: dict[str, list[int]] = defaultdict(list)
    for s in stems:
        m = NAME_RE.match(s)
        by_prefix[m.group(1)].append(int(m.group(2)))

    groups, gid = [], 0
    for prefix, indices in sorted(by_prefix.items()):
        indices.sort()
        width = len(NAME_RE.match(stems[0]).group(2))
        current = [indices[0]]
        for prev, cur in zip(indices, indices[1:]):
            a = f"{prefix}{prev:0{width}d}"
            b = f"{prefix}{cur:0{width}d}"
            contiguous = cur == prev + 1
            similar = contiguous and bin(hashes[a] ^ hashes[b]).count("1") <= HAMMING_THRESHOLD
            if similar:
                current.append(cur)
            else:
                groups.append((gid, prefix, width, current))
                gid += 1
                current = [cur]
        groups.append((gid, prefix, width, current))
        gid += 1

    out = []
    for gid, prefix, width, idxs in groups:
        members = [f"{prefix}{i:0{width}d}" for i in idxs]
        sigs = Counter(signature(m) for m in members)
        out.append(
            {
                "group_id": gid,
                "prefix": prefix,
                "first_index": idxs[0],
                "last_index": idxs[-1],
                "size": len(members),
                "signature": sigs.most_common(1)[0][0],
                "members": members,
            }
        )
    return out


def assign(groups: list[dict], val_frac: float, seed: int) -> set[int]:
    """Pick whole groups for val, stratified by signature, closest to the target size.

    Fills each stratum from a shuffled list but only takes a group that still fits the
    remaining budget. Taking greedily and stopping once the target is passed lets one
    large group overshoot a small stratum badly - `fire` is only ~900 images, so a single
    249-frame group would hand val 2.5x its share and skew the class balance it exists
    to preserve.
    """
    rng = random.Random(seed)
    by_sig: dict[str, list[dict]] = defaultdict(list)
    for g in groups:
        by_sig[g["signature"]].append(g)

    val_ids: set[int] = set()
    for sig, gs in sorted(by_sig.items()):
        gs = sorted(gs, key=lambda g: g["group_id"])
        rng.shuffle(gs)
        remaining = val_frac * sum(g["size"] for g in gs)
        for g in gs:
            if remaining <= 0:
                break
            if g["size"] <= remaining:
                val_ids.add(g["group_id"])
                remaining -= g["size"]
    return val_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    stems = sorted(p.stem for p in IMAGES.glob("*.jpg"))
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "hashes.tsv"

    hashes: dict[str, int] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            stem, value = line.split("\t")
            hashes[stem] = int(value)
        hashes = {s: h for s, h in hashes.items() if s in set(stems)}

    missing = [s for s in stems if s not in hashes]
    if missing:
        print(f"hashing {len(missing):,} training images ...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            hashes.update(zip(missing, pool.map(lambda s: dhash(IMAGES / f"{s}.jpg"), missing)))
        cache.write_text(
            "\n".join(f"{s}\t{hashes[s]}" for s in stems) + "\n", encoding="utf-8"
        )
    else:
        print(f"reusing cached hashes for {len(stems):,} images", flush=True)

    groups = build_groups(stems, hashes)
    multi = [g for g in groups if g["size"] > 1]
    print(f"{len(groups):,} groups ({len(multi):,} multi-frame, largest {max(g['size'] for g in groups)})")
    for prefix in sorted({g["prefix"] for g in groups}):
        gs = [g for g in groups if g["prefix"] == prefix]
        imgs = sum(g["size"] for g in gs)
        print(f"   {prefix:<14} {imgs:>6} images in {len(gs):>6} groups  (mean {imgs / len(gs):.1f} frames/group)")

    val_ids = assign(groups, args.val_frac, args.seed)
    train_g = [g for g in groups if g["group_id"] not in val_ids]
    val_g = [g for g in groups if g["group_id"] in val_ids]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, gs in (("train", train_g), ("val", val_g)):
        lines = sorted(f"images/train/{m}.jpg" for g in gs for m in g["members"])
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(OUT / "groups.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["group_id", "prefix", "first_index", "last_index", "size", "signature", "split"]
        )
        w.writeheader()
        for g in groups:
            row = {k: g[k] for k in w.fieldnames if k != "split"}
            row["split"] = "val" if g["group_id"] in val_ids else "train"
            w.writerow(row)

    n_train = sum(g["size"] for g in train_g)
    n_val = sum(g["size"] for g in val_g)
    print(f"\ntrain {n_train:,}  |  val {n_val:,}  ({n_val / (n_train + n_val):.1%})")
    print("\nclass signature balance:")
    tc, vc = Counter(), Counter()
    for g in train_g:
        tc[g["signature"]] += g["size"]
    for g in val_g:
        vc[g["signature"]] += g["size"]
    for sig in sorted(set(tc) | set(vc)):
        print(f"   {sig:<11} train {tc[sig] / n_train:6.1%}   val {vc[sig] / n_val:6.1%}")

    # No group spans both splits by construction; assert it anyway.
    assert not ({g["group_id"] for g in train_g} & val_ids)
    print(f"\nsplits -> {OUT}")


if __name__ == "__main__":
    main()
