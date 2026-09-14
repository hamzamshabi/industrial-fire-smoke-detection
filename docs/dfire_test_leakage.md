# D-Fire's official train/test split leaks too — worse than its 5-fold split

Audited 2026-08-21. Scripts: `data/audit_dfire_test_leakage.py`, `data/verify_dfire_test_leakage.py`.

## Finding

**At least 51% of D-Fire's official 4,306-image test set has a near-duplicate in the
official training set.** The published 5-fold cross-validation split was already known to
leak ~40%; the train/test split most papers report on is worse.

Any mAP quoted on D-Fire's official test split — ours and everyone else's — is
substantially measuring recall of training frames.

## How the split was built, and why that failed

Per prefix, the split is a contiguous block of the index range:

| Prefix | Train | Test |
|---|---|---|
| AoF | 0–6722 | 6723–8383 |
| PublicDataset | 0–1054 | 1055–1335 |
| WEB | 0–9442 | 9443–11806 |

No interleaving, and the three boundary pairs are visually unrelated (dHash distance 31,
27, 43). A block split *looks* careful.

It fails because **the index order is not chronological**. Frames from one camera event
are scattered across the whole index range, so cutting the range in half cuts through
every event. Confirmed from the timestamps burned into the AoF frames:

| Test frame | Train frame | Camera | Apart |
|---|---|---|---|
| AoF08173 (17 Dec 2020 09:36:56) | AoF05907 (09:37:01) | Camera0003 | **5 s** |
| AoF07561 (02 Jan 2021 12:27:27) | AoF03292 (12:27:32) | Camera0002 | **5 s** |
| AoF07556 (02 Jan 2021 05:35:42) | AoF03272 (05:35:33) | Camera0001 | **9 s** |
| AoF08304 (18 Jun 2020 13:20:20) | AoF06414 (13:20:10) | Camera0002 | **10 s** |
| AoF07557 (02 Jan 2021 16:26:08) | AoF03276 (16:25:53) | Camera0001 | **15 s** |
| AoF08302 (22 Jun 2020 12:13:59) | AoF03796 (12:22:41) | Camera0002 | 9 min |

Index distance 1,900–4,500; wall-clock distance five seconds.

Side-by-side sheet of the pairs above: [`audit_pairs_exact.png`](audit_pairs_exact.png).

## Measurement

dHash (64-bit) and "near-duplicate = Hamming ≤ 10", identical to `split_dfire_val.py`, so
the number is comparable to the 40% quoted for the folds. Every test image scored against
every train image (74M pairs), not just adjacent indices.

Raw hit rate — test vs all 17,221 official train images:

| dHash distance | Images | Cumulative | Share |
|---|---|---|---|
| = 0 | 1,086 | 1,086 | 25.22% |
| 1–2 | 748 | 1,834 | 42.59% |
| 3–5 | 559 | 2,393 | 55.57% |
| 6–10 | 775 | **3,168** | **73.57%** |
| > 10 | 1,138 | — | 26.43% clean |

### Controls

73.57% cannot be quoted raw. Nearest-neighbour-over-17,221 saturates — the maximum
min-distance observed is 20, so the ≤10 cut sits mid-distribution and dHash alone
overcounts. Two controls bound the error.

**False-positive floor.** Same statistic against 4,000 FASDD images, which share the
domain but no provenance:

| Pool | n | ≤ 10 | = 0 |
|---|---|---|---|
| D-Fire train (all) | 17,221 | 73.57% | 25.22% |
| D-Fire train (random subset) | 4,000 | 63.91% | 13.01% |
| FASDD (unrelated) | 4,000 | 28.75% | 0.72% |

Pool size alone does not explain the gap: at equal n=4,000, D-Fire scores 63.91% against
the control's 28.75%.

**Pixel confirmation.** 800 flagged pairs re-scored as normalised RMSE on 64×64
contrast-normalised grayscale. Threshold 0.20 calibrated against 300 known same-shot
pairs (82% recall) and 300 random pairs (0% false positive).

| | median RMSE | below 0.20 |
|---|---|---|
| Flagged pairs (dHash ≤ 10) | 0.073 | 69.8% |
| Random pairs (control) | 0.642 | 1.0% |

73.57% × 69.8% = **51.32%**. Stable across the threshold: 42% at 0.10, 57% at 0.30.

**51% is a floor, not a point estimate.** Visual inspection of the decision margin
(RMSE 0.12–0.25) found every pair to be the same camera and same event, including pairs
*above* the 0.20 cut. The true figure is likely nearer the 57% that threshold 0.30 gives.
Two further reasons it is conservative: dHash misses duplicates that differ by crop or
zoom, and the audit only asks whether a *nearest* neighbour exists, not how many.

## Where it concentrates

| Prefix | Test n | dHash ≤ 10 | Pixel-confirmed | Reading |
|---|---|---|---|---|
| AoF | 1,661 | 98.07% | 93.7% | ~92% leaked — lookout-camera video, worst case |
| PublicDataset | 281 | 82.21% | 78.7% | ~65% leaked |
| WEB | 2,364 | 55.33% | 37.7% | ~21% leaked — scraped, some genuine re-uploads |

AoF is effectively unusable as held-out data. WEB is the most trustworthy third.

## Consequence for this project

Comparison B — test vs our own 15,500-image `train.txt` rather than the full official
train set — gives 72.55% raw, 23.71% at distance 0. Our grouped split does not fix this
and was never meant to: it removes leakage between *train and val*, while this is
contamination the authors baked into the train/test boundary before we touched it.

So the split situation is:

- The validation split (ours, grouped: [`data/splits/val.txt`](../data/splits/val.txt)) — clean. Model selection is sound.
- The official test split (theirs) — ~51%+ contaminated. **The reported figure is inflated.**

### What to do

1. **Keep reporting the official-test number** for comparability with published work, but
   state the contamination alongside it. Every competing number carries the same defect.
2. **Add a deduplicated test subset** as the honest figure — the 1,138 test images at
   dHash > 10 from the full train set, plus WEB-only breakouts. Small (26% of the test
   set) but with no *detected* near-duplicate — dHash misses crops and zooms, so treat it as a
   best-effort dedup rather than a guarantee.
3. **Do not cite AoF-only test metrics** under any circumstance.

## Reproducing

```
python data/audit_dfire_test_leakage.py      # the raw scan, ~1 min after hashes cache
python data/verify_dfire_test_leakage.py     # controls + pixel confirmation
```

Hashes cache to `data/raw/dfire/splits/hashes_test.tsv` and `hashes_control_fasdd.tsv`.

The original 5-fold audit script no longer exists, so the 40% fold figure is quoted from
my earlier notes rather than re-derived. This audit is a reconstruction using the same
hash family and threshold — the comparison holds, but it is not a byte-identical rerun.
If the fold audit was adjacency-based rather than a full pairwise scan, that 40% is itself
likely an undercount, for exactly the scattered-index reason found here. Either way the
direction of the comparison stands.
