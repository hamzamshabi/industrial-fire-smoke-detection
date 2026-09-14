"""Read the burned-in CCTV clock from extracted frames and add it to the manifest.

    python data/ocr_timestamps.py data/raw/csb/frames/hewMRcrOI0s

Why this matters: the CSB clips are edited for presentation, not raw continuous CCTV.
In the Chevron footage 29 seconds of video covers 5 minutes of real time in one stretch
and 230 seconds of video covers 110 real seconds in another. Frame index therefore does
not map linearly to wall clock, so any time-to-detection or alarms-per-hour figure
computed from frame index is wrong.

The camera's own overlay is ground truth. This reads it, adds `cctv_datetime` and
`cctv_epoch_ms` to frames.csv, and reports where playback rate departs from real time so
you know which stretches are usable for timing claims.

Frames whose clock cannot be read keep an empty value rather than an interpolated guess -
a gap is recoverable, a fabricated timestamp is not.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2

# "[CH 5] 08-06-2012 18:33:09:218" - OCR mangles the brackets, so match only the parts
# that carry information. Whitespace is tolerated anywhere the OCR tends to insert it.
DATE_RE = re.compile(r"(\d{2})\s*-\s*(\d{2})\s*-\s*(\d{4})")
TIME_RE = re.compile(r"(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})\s*[:.]\s*(\d{1,3})")


EPOCH = datetime(1970, 1, 1)


def epoch_ms(dt: datetime) -> int:
    """datetime.timestamp() raises OSError on Windows for pre-1970 dates; subtracting
    a reference datetime avoids the platform call entirely."""
    return int((dt - EPOCH).total_seconds() * 1000)


def parse(text: str, last_date: tuple[int, int, int] | None = None) -> datetime | None:
    """Parse the overlay. The time is what matters; the date is often clipped by the
    crop, so a date read on an earlier frame carries forward rather than defaulting to
    something obviously wrong."""
    d, t = DATE_RE.search(text), TIME_RE.search(text)
    if not t:
        return None
    hh, mm, ss, ms = (int(g) for g in t.groups())
    if hh > 23 or mm > 59 or ss > 59:
        return None

    if d:
        mo, dy, yr = (int(g) for g in d.groups())
    elif last_date:
        yr, mo, dy = last_date
    else:
        yr, mo, dy = 1970, 1, 1

    try:
        return datetime(yr, mo, dy, hh, mm, ss, ms * 1000)
    except ValueError:
        return None


def read_clock(ocr, image, strip_h: float, strip_w: float, thresh: int) -> str:
    h, w = image.shape[:2]
    crop = image[0 : int(h * strip_h), 0 : int(w * strip_w)]
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    res, _ = ocr(bw)
    return " ".join(line[1] for line in res) if res else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--strip-height", type=float, default=0.075, help="fraction of frame height")
    ap.add_argument("--strip-width", type=float, default=0.62, help="fraction of frame width")
    ap.add_argument("--threshold", type=int, default=180, help="binarisation cutoff for white text")
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()

    manifest = args.folder / "frames.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8"))) if manifest.exists() else [
        {"frame_file": p.name} for p in sorted(args.folder.glob("*.jpg"))
    ]
    print(f"{len(rows)} frames in {args.folder}")

    parsed = 0
    last_date: tuple[int, int, int] | None = None
    for i, row in enumerate(rows):
        img = cv2.imread(str(args.folder / row["frame_file"]))
        if img is None:
            row["cctv_raw"] = row["cctv_datetime"] = row["cctv_epoch_ms"] = ""
            continue
        text = read_clock(ocr, img, args.strip_height, args.strip_width, args.threshold)
        dt = parse(text, last_date)
        if dt is not None and DATE_RE.search(text):
            last_date = (dt.year, dt.month, dt.day)
        row["cctv_raw"] = text
        row["cctv_datetime"] = dt.isoformat(sep=" ", timespec="milliseconds") if dt else ""
        row["cctv_epoch_ms"] = str(epoch_ms(dt)) if dt else ""
        parsed += dt is not None
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  parsed {parsed}", flush=True)

    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nclock read on {parsed}/{len(rows)} frames ({parsed / len(rows):.0%})")
    if parsed < 2:
        print("not enough timestamps to assess playback rate")
        return

    good = [r for r in rows if r["cctv_epoch_ms"]]
    t0, t1 = int(good[0]["cctv_epoch_ms"]), int(good[-1]["cctv_epoch_ms"])
    v0, v1 = int(good[0].get("timestamp_ms", 0)), int(good[-1].get("timestamp_ms", 0))
    print(f"real span   {(t1 - t0) / 1000:.1f}s   ({good[0]['cctv_datetime']} -> {good[-1]['cctv_datetime']})")
    print(f"video span  {(v1 - v0) / 1000:.1f}s")
    if v1 > v0:
        print(f"mean rate   {(t1 - t0) / (v1 - v0):.2f}x real time")

    # Where does the rate change? Uniform footage has a constant ratio.
    print("\nper-segment playback rate (10 segments):")
    seg = max(1, len(good) // 10)
    for i in range(0, len(good) - seg, seg):
        a, b = good[i], good[i + seg]
        dv = (int(b.get("timestamp_ms", 0)) - int(a.get("timestamp_ms", 0))) / 1000
        dr = (int(b["cctv_epoch_ms"]) - int(a["cctv_epoch_ms"])) / 1000
        rate = dr / dv if dv else 0
        flag = "  <- non-uniform" if rate and not (0.8 <= rate <= 1.25) else ""
        print(f"  video {dv:6.1f}s  real {dr:8.1f}s   {rate:6.2f}x{flag}")

    print(f"\nmanifest -> {manifest}")


if __name__ == "__main__":
    main()
