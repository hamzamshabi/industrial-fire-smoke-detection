"""Download curated refinery-context stock clips for run 3.

Why stock and not YouTube: Pexels and Pixabay both grant free use, permit modification,
and require no attribution. Neither licence restricts using the imagery to train a model -
their prohibitions are about *redistributing* the clips, trademark use, and depicting
people badly. Downloading a standard-licence YouTube video violates YouTube's terms, so
it is not a clean source for a dataset anyone else will see. YouTube's
Creative-Commons filter is a legitimate fallback, but every hit needs its licence line
checked by hand, so those go in BY_HAND below rather than being fetched automatically.

Caution: refinery footage shows company branding. Fine for training - the frames are not
redistributed - but keep branded plants out of anything published.

WHAT THIS IS FOR
----------------
run-44 does not under-detect fire and smoke; it over-detects. 69% of its false positives
are `fire`. So the payload of run 3 is *negatives that look like a refinery and are not
burning*: pipe racks, columns, tanks, sun glare off steel, night plant lighting, rust and
orange hardware, overcast sky.

WHAT THIS IS NOT FOR
--------------------
Anything plume-shaped. See QUARANTINE below - that distinction is the whole point of this
file and getting it wrong makes the model worse at the one case that matters.
"""

import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "raw" / "stock"
VIDEOS = DEST / "videos"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Pexels serves a stable no-auth endpoint per clip id.
PEXELS = "https://www.pexels.com/download/video/{id}/"

# (pexels_id, slug, bucket)
#
# bucket drives what happens at annotation time:
#   context  -> plant hardware with nothing burning. Annotate as background (empty label).
#               This is the payload.
#   flare    -> a lit flare is in frame. Annotate the flame truthfully as `fire`; the value
#               of these clips is the *rest* of the frame - glare, lit steel, dark sky.
#               Do not annotate a flare as background: teaching the detector that some
#               flames are not fire is an unbounded lesson. Site-specific flare suppression
#               lives in the ROI mask in alarm.py, which is where it can be audited.
#   night    -> plant lighting after dark, the other big `fire` false-positive source.
#
# Titles are the only evidence for these classifications - Pexels search returns a lot of
# mislabelled clips, so review each one before it enters the set. review_stock() below
# prints the checklist.
SOURCES = [
    (37151630, "aerial-industrial-oil-refinery", "context"),
    (12891229, "clear-sky-over-refinery", "context"),
    (10396407, "aerial-oil-refinery-plant", "context"),
    (29959444, "aerial-industrial-plant-urban", "context"),
    (12891231, "drone-shot-oil-refinery", "context"),
    (29959405, "aerial-refinery-by-the-sea", "context"),
    (12891228, "road-near-refinery", "context"),
    (38683845, "aerial-industrial-refinery-plant", "context"),
    (29959248, "aerial-coastal-industrial-complex", "context"),
    (30899591, "aerial-plant-and-storage-tanks", "context"),
    (34834880, "industrial-facility-from-boat-overcast", "context"),
    (38683852, "aerial-industrial-oil-refinery-2", "context"),
    (18692266, "industria-de-noche", "night"),
    (5760014, "flame-from-refinery-tower", "flare"),
    (26736801, "natural-gas-production-facility", "flare"),
    (12206553, "close-up-fire-in-pipe", "flare"),
    (11363492, "industrial-platform-on-sea", "flare"),
]

# Clips that a naive "refinery negatives" search returns and that must NOT be annotated as
# background. Both are stack/chimney emission plumes.
#
# The Chevron case - the one that motivated run 3 - is a white low-pressure hydrocarbon
# release detected in only 6 of 40 frames. To a detector that plume and a chimney plume are
# the same object: white, diffuse, continuous, against plant hardware. Label chimney plumes
# as background and you train away the sensitivity that case needs. Steam and stack vapour
# get rejected downstream by the ROI mask and the growth check, not by the detector.
QUARANTINE = [
    (10407687, "aerial-smoke-from-factory", "chimney plume - reads as smoke"),
    (10677831, "smoke-from-factory-chimneys", "chimney plume - reads as smoke"),
]

# Needs a manual licence check on the video page before use. YouTube renders the licence
# line in JS, so it cannot be verified by a scripted fetch - open each, scroll to the
# description, and confirm it reads "Creative Commons Attribution".
BY_HAND = """
YouTube, with the Creative Commons filter on (Filters -> Features -> Creative Commons):
    refinery flare stack
    petrochemical plant walkthrough
    cooling tower operation
    oil refinery night
    process plant pipe rack

Wikimedia Commons (CC-BY / CC0, licence stated on every file page):
    https://commons.wikimedia.org/wiki/Category:Oil_refineries
    https://commons.wikimedia.org/wiki/Category:Gas_flares

Already in the corpus and still the best fixed-camera source you have - CSB is US federal
government work, so public domain, and it is genuine CCTV rather than cinematography.
"""


def download(url: str, path: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as fh:
        total = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            total += len(chunk)
    return total


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if "--list" in sys.argv:
        for vid, slug, bucket in SOURCES:
            print(f"  {bucket:<8} {slug:<38} {PEXELS.format(id=vid)}")
        print("\nQUARANTINE - do not annotate as background:")
        for vid, slug, why in QUARANTINE:
            print(f"  {slug:<38} {why}")
        print(BY_HAND)
        return

    VIDEOS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (vid, slug, bucket) in enumerate(SOURCES, 1):
        path = VIDEOS / f"{bucket}_{slug}_{vid}.mp4"
        if path.exists():
            print(f"  {i:>2}/{len(SOURCES)}  have  {path.name}")
        else:
            print(f"  {i:>2}/{len(SOURCES)}  get   {path.name}", flush=True)
            try:
                download(PEXELS.format(id=vid), path)
            except Exception as exc:
                print(f"        FAILED {exc}")
                path.unlink(missing_ok=True)
                continue
        rows.append(
            {
                "file": path.name,
                "source": "pexels",
                "id": vid,
                "bucket": bucket,
                "url": f"https://www.pexels.com/video/{slug}-{vid}/",
                "licence": "Pexels licence - free use, modification allowed, no attribution",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "reviewed": "",  # set to y/n by hand after watching, see README
            }
        )

    manifest = DEST / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} clips -> {VIDEOS}")
    print(f"manifest -> {manifest}")
    print("\nNext: watch each clip, set `reviewed` in the manifest, then extract_frames.py")


if __name__ == "__main__":
    main()
