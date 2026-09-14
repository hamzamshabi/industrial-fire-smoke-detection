"""Download the US Chemical Safety Board (CSB) video archive from archive.org.

Source : https://archive.org/details/uscsb-video-archive-september-2025
Rights : work of the US federal government -> public domain. Frames AND annotations
         derived from these videos may be redistributed.

Two cautions:

1. Much of the archive is CG *animation*, not real footage. Animated segments must not
   enter the dataset as "real refinery frames" - they are a synthetic domain of their own.
   Titles beginning "Animation of ..." are flagged in the manifest as animation=True, but
   the safety videos also splice animation into otherwise-real footage, so per-clip review
   is still required.

2. This footage documents incidents in which people died. Exclude any frame showing
   casualties or identifiable victims.

Writes data/raw/csb/manifest.csv with one row per downloaded file (name, url, size,
archive.org sha1, whether the title reads as animation) so that every frame extracted
later can be traced back to its source.
"""

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

ITEM = "uscsb-video-archive-september-2025"
METADATA_URL = f"https://archive.org/metadata/{ITEM}"
DOWNLOAD_BASE = f"https://archive.org/download/{ITEM}"
DEST = Path(__file__).resolve().parent / "raw" / "csb"
VIDEOS = DEST / "videos"

# One file per title, preferring the most broadly decodable container.
FORMAT_PRIORITY = [".mp4", ".mkv", ".webm"]

# Titles matching this are refinery / petrochemical relevant.
REFINERY = re.compile(
    r"refin|petro|chevron|tesoro|exxon|husky|philadelphia energy|bp |texas city|"
    r"marathon|valero|phillips|citgo|motiva|torrance|anacortes|richmond|superior|"
    r"delaware city|flare|tank|hydro|caribbean|nustar|kuraray|ethylene|naphtha",
    re.I,
)
ANIMATION = re.compile(r"^animation of|animation$|\banimation\b", re.I)


def fetch_metadata() -> dict:
    with urllib.request.urlopen(METADATA_URL, timeout=120) as r:
        return json.load(r)


def pick_files(meta: dict, refinery_only: bool) -> list[dict]:
    by_title: dict[str, list[dict]] = {}
    for f in meta.get("files", []):
        name = f.get("name", "")
        ext = Path(name).suffix.lower()
        if ext not in FORMAT_PRIORITY:
            continue
        by_title.setdefault(name[: -len(ext)], []).append(f)

    chosen = []
    for title, variants in sorted(by_title.items()):
        if refinery_only and not REFINERY.search(title):
            continue
        variants.sort(key=lambda f: FORMAT_PRIORITY.index(Path(f["name"]).suffix.lower()))
        chosen.append(variants[0])
    return chosen


def download(f: dict) -> Path:
    out = VIDEOS / f["name"]
    expected = int(f.get("size", 0))
    if out.exists() and out.stat().st_size == expected:
        return out
    url = f"{DOWNLOAD_BASE}/{urllib.request.quote(f['name'])}"
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(out)
    return out


def main() -> None:
    refinery_only = "--refinery-only" in sys.argv
    VIDEOS.mkdir(parents=True, exist_ok=True)

    meta = fetch_metadata()
    files = pick_files(meta, refinery_only)
    total = sum(int(f.get("size", 0)) for f in files)
    print(f"{len(files)} videos, {total / 1e9:.1f} GB -> {VIDEOS}", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        title = f["name"].rsplit(".", 1)[0]
        print(f"[{i}/{len(files)}] {f['name']} ({int(f.get('size', 0)) / 1e6:.0f} MB)", flush=True)
        path = download(f)
        rows.append(
            {
                "file": f["name"],
                "url": f"{DOWNLOAD_BASE}/{urllib.request.quote(f['name'])}",
                "bytes": f.get("size", ""),
                "sha1": f.get("sha1", ""),
                "refinery_related": bool(REFINERY.search(title)),
                "animation": bool(ANIMATION.search(title)),
                "local_path": str(path.relative_to(DEST)),
            }
        )

    with open(DEST / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest -> {DEST / 'manifest.csv'}")


if __name__ == "__main__":
    main()
