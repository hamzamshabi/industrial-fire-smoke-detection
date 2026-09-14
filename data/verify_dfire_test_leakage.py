"""Verify the near-duplicate hits found by audit_dfire_test_leakage.py.

A dHash distance of <= 10 over 17,221 candidates is a *claim*, not a fact: dHash is only
64 bits of an 8x9 luminance gradient, so structurally similar but genuinely different
photos can collide, and taking a minimum over 17k candidates gives collisions many
chances to happen. Before the audit's percentage can be quoted, two things must hold.

1. CONTROL - what does the same statistic return for images that definitely do not
   leak? D-Fire test is scored against a FASDD sample it shares no provenance with. The
   near-duplicate rate there is the false-positive floor; the audit number only means
   something to the extent it clears that floor.

2. PIXEL CHECK - dHash agreement is confirmed against the actual pixels. Every flagged
   pair is re-scored as normalised RMSE over 64x64 grayscale, which sees detail dHash
   throws away. Pairs that are genuinely near-identical stay low; hash collisions
   between different scenes do not.

    python verify_dfire_test_leakage.py
"""

import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from audit_dfire_test_leakage import (
    HAMMING_THRESHOLD,
    POPCOUNT,
    ROOT,
    OUT,
    as_bytes,
    dhash,
    load_hashes,
    nearest,
)

RAW = Path(__file__).resolve().parent / "raw"
CONTROL_N = 4000       # comparable in size to the 17,221 train pool it stands in for
RMSE_SIMILAR = 0.20    # calibrated: 82% recall on known same-shot pairs, 0% on random unrelated pairs
SAMPLE = 800           # flagged pairs to pixel-check


def thumb(path: Path) -> np.ndarray:
    """64x64 grayscale, contrast-normalised - immune to the exposure drift between frames."""
    with Image.open(path) as im:
        im.draft("L", (64, 64))
        a = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def rmse(a: Path, b: Path) -> float:
    """Normalised RMSE in [0, ~2]. 0 = identical, ~1.4 = unrelated."""
    return float(np.sqrt(((thumb(a) - thumb(b)) ** 2).mean())) / 2


def control_pool() -> list[int]:
    """Hash a FASDD sample. Same domain (fire/smoke photos), unrelated provenance."""
    cache = OUT / "hashes_control_fasdd.tsv"
    if cache.exists():
        return [int(l.split("\t")[1]) for l in cache.read_text(encoding="utf-8").splitlines()]
    files = sorted(p for p in (RAW / "fasdd").rglob("*.jpg"))
    files = random.Random(0).sample(files, min(CONTROL_N, len(files)))
    print(f"hashing {len(files):,} FASDD control images ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(dhash, files))
    cache.write_text("\n".join(f"{p.stem}\t{v}" for p, v in zip(files, values)) + "\n", encoding="utf-8")
    return values


def main() -> None:
    train_h = load_hashes("train", OUT / "hashes.tsv")
    test_h = load_hashes("test", OUT / "hashes_test.tsv")
    train_stems, test_stems = list(train_h), list(test_h)
    train_arr = as_bytes(list(train_h.values()))
    test_arr = as_bytes(list(test_h.values()))

    dists, idx = nearest(test_arr, train_arr)
    real = (dists <= HAMMING_THRESHOLD).mean()

    # 1. Control -----------------------------------------------------------------
    ctrl = as_bytes(control_pool())
    # Score against an equally sized slice of train, so pool size cannot explain the gap.
    slice_n = len(ctrl)
    sub = np.array(random.Random(0).sample(range(len(train_arr)), slice_n))
    d_ctrl, _ = nearest(test_arr, ctrl)
    d_sub, _ = nearest(test_arr, train_arr[sub])

    print(f"\n{'pool':<40}{'n':>7}{'<= 10':>9}{'= 0':>9}")
    for name, d, n in (
        (f"D-Fire train (all)", dists, len(train_arr)),
        (f"D-Fire train (random {slice_n:,} subset)", d_sub, slice_n),
        (f"FASDD control (unrelated, {slice_n:,})", d_ctrl, slice_n),
    ):
        print(f"   {name:<37}{n:>7}{(d <= HAMMING_THRESHOLD).mean():>9.2%}{(d == 0).mean():>9.2%}")
    floor = (d_ctrl <= HAMMING_THRESHOLD).mean()
    print(f"\n   false-positive floor from the control: {floor:.2%}")
    print(f"   audit figure clears it by {real - floor:.2%} ({real:.2%} - {floor:.2%})")

    # 2. Pixel check --------------------------------------------------------------
    flagged = [i for i, d in enumerate(dists) if d <= HAMMING_THRESHOLD]
    check = random.Random(0).sample(flagged, min(SAMPLE, len(flagged)))
    print(f"\npixel-checking {len(check)} of {len(flagged):,} flagged pairs (normalised RMSE on 64x64) ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        scores = list(pool.map(
            lambda i: rmse(ROOT / "images" / "test" / f"{test_stems[i]}.jpg",
                           ROOT / "images" / "train" / f"{train_stems[idx[i]]}.jpg"),
            check,
        ))
    scores = np.array(scores)

    # Negative control for RMSE itself. Nearest-neighbour distance tops out at 20 here -
    # taking a minimum over 17,221 candidates never leaves anything far away - so the
    # control has to be random test/train pairs, not the far tail of the same statistic.
    rng = random.Random(1)
    pairs = [(rng.choice(test_stems), rng.choice(train_stems)) for _ in range(300)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        dscores = np.array(list(pool.map(
            lambda ab: rmse(ROOT / "images" / "test" / f"{ab[0]}.jpg",
                            ROOT / "images" / "train" / f"{ab[1]}.jpg"),
            pairs,
        )))

    print(f"   flagged pairs (dHash <= 10)   median RMSE {np.median(scores):.3f}   "
          f"{(scores < RMSE_SIMILAR).mean():.1%} below {RMSE_SIMILAR}")
    print(f"   random pairs (control)        median RMSE {np.median(dscores):.3f}   "
          f"{(dscores < RMSE_SIMILAR).mean():.1%} below {RMSE_SIMILAR}")
    print("\n   sensitivity of the corrected estimate to the RMSE threshold:")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(f"      thr {t:.2f}   confirmed {(scores < t).mean():>6.1%}   "
              f"-> leakage {real * (scores < t).mean():>6.2%}   "
              f"(control false-pos {(dscores < t).mean():.1%})")
    confirmed = (scores < RMSE_SIMILAR).mean()
    print(f"\n   pixel-confirmed share of flagged pairs: {confirmed:.1%}")
    print(f"   corrected leakage estimate: {real:.2%} x {confirmed:.1%} = {real * confirmed:.2%} of the test set")

    # 3. Per-prefix corrected -----------------------------------------------------
    by_prefix: dict[str, list[float]] = defaultdict(list)
    for i, s in zip(check, scores):
        by_prefix[re.match(r"^([A-Za-z_]+)", test_stems[i]).group(1)].append(s)
    print("\nconfirmation rate by prefix (of the pixel-checked sample)")
    for prefix in sorted(by_prefix):
        v = np.array(by_prefix[prefix])
        print(f"   {prefix:<14}{len(v):>5} checked   {(v < RMSE_SIMILAR).mean():>7.1%} confirmed")


if __name__ == "__main__":
    main()
