"""Audit D-Fire's official train/test boundary for near-duplicate leakage.

`split_dfire_val.py` established that D-Fire is heavily video-derived and that its
published 5-fold cross-validation split leaks ~40% near-duplicates. The reported figure
for this project is computed on D-Fire's *official* test split, so that boundary needs
the same scrutiny: if test images have near-twins in train, the headline number is
measuring memory too.

Method mirrors the val split exactly - dHash (64-bit) and HAMMING_THRESHOLD = 10 means
"near-duplicate" - so the percentage here is comparable to the 40% quoted for the folds.
The original fold-audit script no longer exists; this is a reconstruction using the same
hash family and threshold, not a byte-identical rerun.

Unlike the grouping pass, this compares every test image against every train image
(4,306 x 17,221 = 74M pairs), not just adjacent indices. Index adjacency catches split
video shots; a full scan also catches WEB scrape duplicates sitting at unrelated indices.

Two comparisons are run:
    test vs all 17,221 official train  -> is the authors' boundary sound?
    test vs the 15,500 in train.txt    -> was our own reported number contaminated?

    python audit_dfire_test_leakage.py
"""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "raw" / "dfire"
OUT = ROOT / "splits"
HAMMING_THRESHOLD = 10  # same definition of "near-duplicate" as split_dfire_val.py
NAME_RE = re.compile(r"^([A-Za-z_]+)(\d+)$")
BUCKETS = [0, 2, 5, 10]


def dhash(path: Path) -> int:
    """64-bit difference hash. Copied verbatim from split_dfire_val.py."""
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


def load_hashes(split: str, cache: Path, workers: int = 8) -> dict[str, int]:
    """Hash every image in a split, reusing the cache file when it already covers it."""
    stems = sorted(p.stem for p in (ROOT / "images" / split).glob("*.jpg"))
    hashes: dict[str, int] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            stem, value = line.split("\t")
            hashes[stem] = int(value)
        hashes = {s: h for s, h in hashes.items() if s in set(stems)}

    missing = [s for s in stems if s not in hashes]
    if missing:
        print(f"hashing {len(missing):,} {split} images ...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = (ROOT / "images" / split / f"{s}.jpg" for s in missing)
            hashes.update(zip(missing, pool.map(dhash, paths)))
        cache.write_text("\n".join(f"{s}\t{hashes[s]}" for s in stems) + "\n", encoding="utf-8")
    else:
        print(f"reusing cached hashes for {len(stems):,} {split} images", flush=True)
    return {s: hashes[s] for s in stems}


def as_bytes(values: list[int]) -> np.ndarray:
    """(n, 8) uint8 view of 64-bit hashes, so XOR + a popcount LUT can do the counting."""
    raw = b"".join(v.to_bytes(8, "big") for v in values)
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8)


POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def nearest(test: np.ndarray, train: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per test image: min Hamming distance to any train image, and which one."""
    best_d = np.empty(len(test), dtype=np.int16)
    best_i = np.empty(len(test), dtype=np.int64)
    for lo in range(0, len(test), chunk):
        hi = min(lo + chunk, len(test))
        # (chunk, n_train, 8) XOR -> popcount each byte -> sum to Hamming distance.
        dist = POPCOUNT[test[lo:hi, None, :] ^ train[None, :, :]].sum(axis=2)
        best_i[lo:hi] = dist.argmin(axis=1)
        best_d[lo:hi] = dist[np.arange(hi - lo), best_i[lo:hi]]
    return best_d, best_i


def histogram(dists: np.ndarray, label: str, n: int) -> None:
    print(f"\n{label}")
    prev = -1
    for b in BUCKETS:
        count = int(((dists > prev) & (dists <= b)).sum())
        cum = int((dists <= b).sum())
        name = f"= {b}" if b == 0 else f"{prev + 1}-{b}"
        print(f"   distance {name:<7} {count:>6}          cumulative <= {b:<3} {cum:>6}  ({cum / n:6.2%})")
        prev = b
    clean = int((dists > HAMMING_THRESHOLD).sum())
    print(f"   distance > {HAMMING_THRESHOLD:<5}  {clean:>6}          no near-duplicate in train  ({clean / n:6.2%})")


def main() -> None:
    train_h = load_hashes("train", OUT / "hashes.tsv")
    test_h = load_hashes("test", OUT / "hashes_test.tsv")

    train_stems, test_stems = list(train_h), list(test_h)
    train_arr, test_arr = as_bytes(list(train_h.values())), as_bytes(list(test_h.values()))
    n_test = len(test_stems)
    print(f"\ncomparing {n_test:,} test x {len(train_stems):,} train = {n_test * len(train_stems) / 1e6:.0f}M pairs")

    dists, idx = nearest(test_arr, train_arr)
    histogram(dists, f"A. test vs ALL {len(train_stems):,} official train images", n_test)

    # The same arrays masked to our own train.txt - the split the reported model saw.
    own = {Path(l).stem for l in (OUT / "train.txt").read_text(encoding="utf-8").split()}
    mask = np.array([s in own for s in train_stems])
    own_d, _ = nearest(test_arr, train_arr[mask])
    own_stems = [s for s, m in zip(train_stems, mask) if m]
    histogram(own_d, f"B. test vs our {len(own_stems):,} train.txt images (the reported model's training set)", n_test)

    print("\nper-prefix (comparison A, near-duplicates at <= 10)")
    by_prefix: dict[str, list[int]] = defaultdict(list)
    for s, d in zip(test_stems, dists):
        by_prefix[NAME_RE.match(s).group(1)].append(int(d))
    print(f"   {'prefix':<14}{'test n':>8}{'<= 10':>8}{'share':>9}{'= 0':>7}")
    for prefix in sorted(by_prefix):
        ds = np.array(by_prefix[prefix])
        near, exact = int((ds <= HAMMING_THRESHOLD).sum()), int((ds == 0).sum())
        print(f"   {prefix:<14}{len(ds):>8}{near:>8}{near / len(ds):>9.2%}{exact:>7}")

    print("\nboundary pairs (last train index vs first test index, per prefix)")
    tr_idx, te_idx = defaultdict(list), defaultdict(list)
    for s in train_stems:
        m = NAME_RE.match(s)
        tr_idx[m.group(1)].append((int(m.group(2)), s))
    for s in test_stems:
        m = NAME_RE.match(s)
        te_idx[m.group(1)].append((int(m.group(2)), s))
    for prefix in sorted(te_idx):
        last_i, last_s = max(tr_idx[prefix])
        first_i, first_s = min(te_idx[prefix])
        d = bin(train_h[last_s] ^ test_h[first_s]).count("1")
        gap = "adjacent" if first_i == last_i + 1 else f"gap of {first_i - last_i}"
        flag = "  <-- SAME SHOT" if d <= HAMMING_THRESHOLD else ""
        print(f"   {last_s} -> {first_s}   {gap:<12} distance {d:>2}{flag}")

    print("\ndegenerate hashes (near-uniform images that match each other spuriously)")
    for name, values in (("train", list(train_h.values())), ("test", list(test_h.values()))):
        pc = np.array([bin(v).count("1") for v in values])
        flat = int(((pc <= 2) | (pc >= 62)).sum())
        print(f"   {name:<6} {flat:>5} of {len(values):,} images have popcount <= 2 or >= 62  ({flat / len(values):.2%})")

    print("\nclosest cross-set pairs (comparison A) - eyeball these before quoting a number")
    for rank in np.argsort(dists)[:10]:
        print(f"   d={int(dists[rank]):>2}   test/{test_stems[rank]}.jpg   <->   train/{train_stems[idx[rank]]}.jpg")

    print(f"\nnear-duplicate = dHash distance <= {HAMMING_THRESHOLD}, the same threshold "
          f"split_dfire_val.py uses to decide two frames are one shot.")


if __name__ == "__main__":
    main()
