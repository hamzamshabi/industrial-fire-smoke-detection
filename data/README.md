# Data

Scripts that fetch, verify, split and audit the training data. Everything they download
lands under `data/raw/` (git-ignored). No images from any dataset are stored in this repo
except the D-Fire audit sheet in `docs/` (CC0) and three CSB frames (public domain).

## Sources

| Source | Role | Licence | Script |
|---|---|---|---|
| [D-Fire](https://github.com/gaiasd/DFireDataset) | Base training set, 21,527 images | CC0 1.0 | `fetch_dfire.py`, `export_dfire.py` |
| [FASDD_CV](https://doi.org/10.57760/sciencedb.j00104.00103) | Hard negatives only (lights, sunsets, clouds) | CC BY 4.0 | `fetch_fasdd.py` |
| [RISE](https://github.com/CMU-CREATE-Lab/deep-smoke-machine) | Evaluation only: 7,477 normal-operation clips | CC0 | `fetch_rise.py` |
| [CSB video archive](https://archive.org/details/uscsb-video-archive-september-2025) | Refinery incident footage, incl. two fixed CCTV cameras | US public domain | `fetch_csb.py`, `review_csb.py`, `extract_frames.py` |
| Pexels stock clips | Refinery context and flare negatives | Pexels licence (no redistribution) | `fetch_stock.py` |

The YouTube incident clips used for the larger frame pool (`build_video_pool.py`,
`flag_people_intro.py`) are not listed or redistributed here; the scripts are included
because the filtering logic is reusable.

## Pipeline

```bash
python data/fetch_dfire.py           # 3.1 GB parquet from the HuggingFace mirror
python data/export_dfire.py          # unpack to YOLO layout, verify against published counts
python data/split_dfire_val.py       # leakage-free validation split (see below)
python data/audit_dfire_test_leakage.py    # official test split vs train, 74M pairs
python data/verify_dfire_test_leakage.py   # controls + pixel check for the audit

python data/fetch_csb.py --refinery-only
python data/review_csb.py --sheets   # contact sheets: mark real footage, skip CG and casualties
python data/review_csb.py --extract
python data/ocr_timestamps.py data/raw/csb/frames/hewMRcrOI0s   # read the CCTV clock
```

## D-Fire: verified, not assumed

The HuggingFace mirror was checked against every figure the D-Fire authors publish, and
the official split lists (`splits/official_{train,test}.txt`) are set-identical to what
the mirror produced:

| Category | Exported | Published |
|---|---|---|
| Only fire | 1,164 | 1,164 |
| Only smoke | 5,867 | 5,867 |
| Fire and smoke | 4,658 | 4,658 |
| None (empty label file) | 9,838 | 9,838 |
| Fire boxes | 14,692 | 14,692 |
| Smoke boxes | 11,865 | 11,865 |

That table also proves the class order is **`0 = smoke, 1 = fire`**: a reversed mapping
would swap the first two rows. FASDD, and the trained model, use the opposite order
(`0 = fire, 1 = smoke`). Nothing errors if you mix them up; every box silently changes
class. `pack_refinery_roboflow.py` and `pack_video_roboflow.py` remap model output to
D-Fire order and write it into `data.yaml`; `pack_dfire_negatives.py` has no boxes, so
its order is moot.

An empty label file is a deliberate background image, not a missing label. Those 9,838
images are 45.7% of D-Fire and are what teaches the model not to fire on sunsets and
streetlights (see the main README for what happened when an importer dropped them;
`pack_dfire_negatives.py` is the fix).

## Splits

`splits/official_train.txt`, `splits/official_test.txt`: D-Fire's own split, 17,221 / 4,306.

`splits/train.txt`, `splits/val.txt`, `splits/groups.csv`: our validation split, carved
from the official train set by `split_dfire_val.py`. D-Fire is largely video-derived, so
a random split would put near-identical frames on both sides. The script dHashes every
image, joins consecutive indices within 10 bits into one group (one shot), and assigns
whole groups, stratified by class signature:

```
train 15,500  |  val 1,721  (10.0%)
background  train 45.4%   val 45.4%
both        train 22.3%   val 22.3%
fire        train  5.2%   val  5.2%
smoke       train 27.1%   val 27.1%
```

D-Fire's published 5-fold cross-validation split does not do this: about **40%** of each
validation fold had a near-duplicate (dHash ≤ 10) in its training fold. Its official
train/test split is worse: **at least 51%** of test images have a near-duplicate in train,
confirmed with a control set and a pixel check. Full write-up:
[`docs/dfire_test_leakage.md`](../docs/dfire_test_leakage.md).

## CSB footage

101 videos, 23 refinery-related. Much of the archive is CG reconstruction, and animation
must not enter a training set as real footage. Titles don't separate them: an
"Animation of ..." video contained real news footage of the 2012 Richmond plume, and a
video called "About the CSB" held the best test frame in the archive. `review_csb.py`
builds contact sheets so a person marks the real spans. An automatic "is this CG" score
was tried and came out inverted on this corpus (reasons in the script's docstring).

This footage documents incidents in which people died. Frames showing casualties or
identifiable people were excluded during review.

The CSB clips are edited, not continuous CCTV: in the Chevron clip, 29 s of video covers
5 minutes of real time in one stretch. `ocr_timestamps.py` reads the burned-in camera
clock (524 of 560 frames parsed) so timing claims use real time, not frame index.
