"""Download the RISE dataset (Project RISE, CMU CREATE Lab).

Source   : https://github.com/CMU-CREATE-Lab/deep-smoke-machine  (paper: arXiv:2005.06111)
License  : CC0 for the data, BSD-3-Clause for their code.
Content  : 12,567 clips of industrial smoke emissions, 3 camera sites / 15 views.

Used for evaluation only - never for training. label == 0 clips are the
normal-operations subset the false-alarms-per-hour figure is measured on.

Label aggregation reproduces aggregate_label() from the authors' split_metadata.py:
researcher labels (label_state_admin) take precedence over citizen labels
(label_state); `weight` is their agreement confidence. Clips neither group
resolved get label -1 and should be excluded from any reported figure.

Writes raw/rise/videos/<file_name>.mp4 and raw/rise/manifest.csv.
"""

import argparse
import csv
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

METADATA_URL = (
    "https://raw.githubusercontent.com/CMU-CREATE-Lab/deep-smoke-machine/master/"
    "back-end/data/dataset/2020-02-24/metadata_02242020.json"
)
DEST = Path(__file__).resolve().parent / "raw" / "rise"
VIDEOS = DEST / "videos"

POS, NEG = 1, 0
CITIZEN = {23: (POS, 1.0), 19: (POS, 0.66), 16: (NEG, 1.0), 20: (NEG, 0.66)}
ADMIN_WEIGHT = {23: 1.0, 16: 0.5, 20: 0.66, 19: 0.83}


def aggregate_label(v: dict) -> tuple[int, float]:
    """Return (label, weight); (-1, 0.0) when neither researchers nor citizens resolved it."""
    admin, citizen = v["label_state_admin"], v["label_state"]

    if admin == 47:
        return POS, 1.0
    if admin == 32:
        return NEG, 1.0
    if admin == 23:
        return POS, ADMIN_WEIGHT.get(citizen, 0.75)
    if admin == 16:
        # Citizen agreement is mirrored: strong pos from citizens lowers confidence.
        mirror = {23: 0.5, 16: 1.0, 20: 0.83, 19: 0.66}
        return NEG, mirror.get(citizen, 0.75)
    return CITIZEN.get(citizen, (-1, 0.0))


def download(v: dict, resolution: str) -> tuple[dict, int]:
    url = v["url_root"].replace("/videos/180/", f"/videos/{resolution}/") + v["url_part"]
    out = VIDEOS / f"{v['file_name']}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return v, out.stat().st_size
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        fh.write(r.read())
    tmp.replace(out)
    return v, out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    # The 2020-02-24 metadata snapshot indexes the 180px clips only; the file names
    # themselves embed "180-180", so the 320px tree 404s for these paths. 180x180 is
    # low for detector evaluation - state that when reporting false alarms/hour.
    ap.add_argument("--resolution", default="180", choices=["180", "320"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    VIDEOS.mkdir(parents=True, exist_ok=True)
    print("fetching metadata ...", flush=True)
    with urllib.request.urlopen(METADATA_URL, timeout=180) as r:
        meta = json.load(r)
    (DEST / "metadata_02242020.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{len(meta)} clips at {args.resolution}px", flush=True)

    rows, done, failed = [], 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, v, args.resolution): v for v in meta}
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                _, size = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed.append((v["file_name"], repr(exc)))
                continue
            label, weight = aggregate_label(v)
            rows.append(
                {
                    "file": f"{v['file_name']}.mp4",
                    "url": v["url_root"].replace("/videos/180/", f"/videos/{args.resolution}/")
                    + v["url_part"],
                    "bytes": size,
                    "label": label,
                    "weight": weight,
                    "camera_id": v["camera_id"],
                    "view_id": v["view_id"],
                    "start_time": v["start_time"],
                    "date": v["url_part"].split("/")[0],
                }
            )
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(meta)}", flush=True)

    rows.sort(key=lambda r: r["file"])
    with open(DEST / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = sum(r["label"] == POS for r in rows)
    neg = sum(r["label"] == NEG for r in rows)
    unk = sum(r["label"] == -1 for r in rows)
    gb = sum(r["bytes"] for r in rows) / 1e9
    print(f"\n{len(rows)} clips, {gb:.2f} GB")
    print(f"  positive (smoke)        : {pos:,}")
    print(f"  negative (normal ops)   : {neg:,}   <- false-alarm denominator")
    print(f"  unresolved (exclude)    : {unk:,}")
    if failed:
        print(f"  FAILED: {len(failed)} (first 5: {failed[:5]})")
    print(f"manifest -> {DEST / 'manifest.csv'}")


if __name__ == "__main__":
    main()
