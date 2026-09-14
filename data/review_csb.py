"""Two-pass extraction for the CSB safety videos: review first, extract second.

The long CSB videos are documentaries. They splice genuine incident footage - plant CCTV,
news helicopter, handheld, post-fire walkdowns - into CG animated reconstruction. The
animation must not enter the dataset: it is a synthetic domain of its own, and a detector
trained to find smoke in a rendered plume learns nothing about a real one. Only a human can
draw that line reliably, so this script does not try to.

    pass 1   python data/review_csb.py --sheets
             writes contact sheets to data/raw/csb/review/<id>/ and an empty
             real_segments.csv. Look at the sheets, write down the time ranges that are
             real footage.

    pass 2   python data/review_csb.py --extract
             reads real_segments.csv and extracts only those ranges, at full rate, into
             data/raw/csb/frames/<id>/ with the same provenance manifest extract_frames.py
             writes.

TWO THINGS TO EXCLUDE WHILE REVIEWING, not later:

  animation   obvious once you see it side by side on a sheet. Note that some videos open
              with real footage, cut to animation for the reconstruction, and return to
              real footage for the aftermath - so expect several disjoint ranges per video,
              not one.

  casualties  this footage documents incidents in which people died. Any frame showing
              casualties or identifiable victims is excluded, full stop. The review pass is
              the only point where that judgement gets made with eyes on every span.

There is deliberately no automatic animation score here. One was tried - mean residual
against a median filter, on the theory that CG lacks sensor grain and codec ringing - and
measured against known-CG and known-real CSB videos it came out *inverted*:

    BP animation      CG    1.697      Chevron CCTV      real  0.996
    Tesoro animation  CG    1.648      Silver Eagle      real  0.744
    Richmond anim.    CG    0.830      BP TC anniversary real  0.628

The premise was backwards for this corpus. These renders carry hard synthetic edges, dense
pipe-rack detail and text overlays at a decent bitrate; the real footage is 2005-2012 CCTV
and news video, heavily denoised and upscaled. Sorting by that column leads you to the
animation. Read the sheets.
"""

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary.

    OpenCV alone cannot do this job in useful time. Three of these videos are 1080p VP9
    .webm of 17-20 minutes; POS_MSEC seeking re-decodes from the preceding keyframe on
    every call and never finishes, and a plain sequential decode measured at ~34 minutes
    for one file. ffmpeg's fps filter does the same work in about a minute.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

ROOT = Path(__file__).resolve().parent / "raw" / "csb"
VIDEOS = ROOT / "videos"
REVIEW = ROOT / "review"
FRAMES = ROOT / "frames"
SEGMENTS = REVIEW / "real_segments.csv"

YT_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]")

# Every refinery-related video, INCLUDING the ones titled "Animation of ...".
#
# Those were excluded at first on the strength of their titles. That was wrong: the
# "Animation of Fire at Chevron's Richmond Refinery" video contains the real news footage
# of the 2012 Richmond plume, which run-44 calls smoke at 0.91. A title describes what the
# video is *for*, not what every frame contains - the reconstruction is animated, but the
# framing and aftermath material around it is not.
#
# Two videos have already been ruled out by review rather than by title:
#   c9JY3eT4cdM  Explosion at BP Refinery  - CG throughout, no real footage
#   _2UnKLm2Eag  Ethylene Oxide Explosion  - real, but no fire or smoke in any frame
TARGETS = [
    "Wake Up Call",
    "Transient Hazards",
    "Silent Killer",
    "Ethylene Oxide Explosion",
    "Sodium Hydrosulfide",
    "Explosion at BP Refinery",
    "BP Texas City 10 Year Anniversary",
    "Freedom Industries Tank Dismantling",
    # animation-titled, mined for their real segments
    "Animation of Fire at Chevron",
    "Animation of Explosion at Tesoro",
    "Animation of 2015 Explosion at ExxonMobil",
    "Animation of Fire at ExxonMobil",
    "Animation of 2018 Ethylene Release",
    "Animation of April 26, 2018",
    "Animation of Bayer CropScience",
    "CSB Interim Animation of Husky",
    "Preliminary Animation of Philadelphia",
    "Updated BP Texas City Animation",
    "BP Texas City Animation - French",
]

SAMPLE_S = 5.0          # one thumbnail per 5s of video
COLS, ROWS = 8, 5       # 40 thumbs per sheet
THUMB_W, THUMB_H = 320, 180
LABEL_H = 20


def hms(ms: float) -> str:
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def video_id(path: Path) -> str:
    m = YT_ID.search(path.stem)
    return m.group(1) if m else path.stem[:11]


def find_targets() -> list[Path]:
    out = []
    for p in sorted(VIDEOS.iterdir()):
        if p.suffix.lower() not in {".mp4", ".webm", ".mkv"}:
            continue
        if any(t.lower() in p.stem.lower() for t in TARGETS):
            out.append(p)
    return out


def make_sheets(video: Path) -> int:
    vid = video_id(video)
    out = REVIEW / vid
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  could not open {video.name}")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()

    # ffmpeg's fps filter emits one frame per SAMPLE_S starting at t=0, so output frame n
    # is at n * SAMPLE_S seconds - no need to ask it for timestamps.
    tmp = Path(tempfile.mkdtemp(prefix="csbsheet_"))
    try:
        subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-i", str(video),
             "-vf", f"fps=1/{SAMPLE_S},scale={THUMB_W}:{THUMB_H}",
             "-fps_mode", "passthrough", "-q:v", "3", str(tmp / "t_%05d.jpg")],
            check=True, capture_output=True,
        )
        files = sorted(tmp.glob("t_*.jpg"))
        thumbs, rows = [], []
        for n, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            t = n * SAMPLE_S
            thumbs.append((t, img))
            rows.append({"t_s": round(t, 1), "hms": hms(t * 1000)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not rows:
        print(f"  {vid}  no frames decoded, skipped")
        return 0

    per = COLS * ROWS
    n_sheets = 0
    for s in range(0, len(thumbs), per):
        chunk = thumbs[s : s + per]
        sheet = np.full((ROWS * (THUMB_H + LABEL_H), COLS * THUMB_W, 3), 30, np.uint8)
        for k, (ts, th) in enumerate(chunk):
            r, c = divmod(k, COLS)
            y = r * (THUMB_H + LABEL_H)
            x = c * THUMB_W
            sheet[y : y + THUMB_H, x : x + THUMB_W] = th
            cv2.putText(sheet, hms(ts * 1000)[:8], (x + 4, y + THUMB_H + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        idx = s // per
        cv2.imwrite(str(out / f"sheet_{idx:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        for k in range(len(chunk)):
            rows[s + k]["sheet"] = f"sheet_{idx:02d}.jpg"
            rows[s + k]["cell"] = k
        n_sheets += 1

    with open(out / "sheet_index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["t_s", "hms", "sheet", "cell"])
        w.writeheader()
        w.writerows(rows)

    print(f"  {vid}  {dur / 60:5.1f} min  {len(thumbs):>3} thumbs  {n_sheets} sheets")
    return n_sheets


def extract_segments() -> None:
    if not SEGMENTS.exists():
        raise SystemExit(f"no {SEGMENTS} - run --sheets first, then fill it in")

    # Strip comments before DictReader sees them - otherwise it takes the first comment
    # line as the header row and every real column name becomes data.
    lines = [ln for ln in SEGMENTS.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    segs = [r for r in csv.DictReader(lines) if (r.get("youtube_id") or "").strip()]
    if not segs:
        raise SystemExit(f"{SEGMENTS} has no rows filled in yet")

    by_video = {video_id(p): p for p in find_targets()}
    grouped: dict[str, list[dict]] = {}
    for s in segs:
        grouped.setdefault(s["youtube_id"].strip(), []).append(s)

    grand = 0
    for vid, rows in sorted(grouped.items()):
        video = by_video.get(vid)
        if video is None:
            print(f"  {vid}: no matching video, skipped")
            continue

        out = FRAMES / vid
        out.mkdir(parents=True, exist_ok=True)
        manifest = []

        for r in sorted(rows, key=lambda r: float(r["start_s"])):
            start, end = float(r["start_s"]), float(r["end_s"])
            fps_out = float(r.get("fps") or 2.0)

            # -ss before -i is the fast seek; -t bounds the span. The fps filter restarts
            # its clock at the seek point, so output frame n sits at start + n / fps_out.
            tmp = Path(tempfile.mkdtemp(prefix="csbcut_"))
            try:
                subprocess.run(
                    [ffmpeg_exe(), "-v", "error", "-ss", f"{start}", "-t", f"{end - start}",
                     "-i", str(video), "-vf", f"fps={fps_out}",
                     "-fps_mode", "passthrough", "-q:v", "2", str(tmp / "f_%05d.jpg")],
                    check=True, capture_output=True,
                )
                for n, f in enumerate(sorted(tmp.glob("f_*.jpg"))):
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    t_ms = (start + n / fps_out) * 1000.0
                    name = f"{vid}_{int(t_ms):07d}ms.jpg"
                    cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest.append(
                        {
                            "frame_file": name,
                            "source_video": video.name,
                            "youtube_id": vid,
                            "timestamp_ms": int(t_ms),
                            "timestamp_hms": hms(t_ms),
                            "source_frame_index": int(t_ms / 1000.0 * 30),  # nominal
                            "width": img.shape[1],
                            "height": img.shape[0],
                            "segment_note": r.get("note", ""),
                        }
                    )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        if manifest:
            with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
                w.writeheader()
                w.writerows(manifest)
        print(f"  {vid}  {len(rows)} segment(s)  {len(manifest)} frames -> {out}")
        grand += len(manifest)

    print(f"\n{grand} frames extracted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", action="store_true", help="pass 1: build contact sheets")
    ap.add_argument("--extract", action="store_true", help="pass 2: extract marked segments")
    args = ap.parse_args()

    if args.extract:
        extract_segments()
        return
    if not args.sheets:
        ap.error("pick --sheets or --extract")

    targets = find_targets()
    print(f"{len(targets)} videos to review\n")
    REVIEW.mkdir(parents=True, exist_ok=True)
    for v in targets:
        # Resume rather than restart. The three VP9 files take ~12 minutes each to decode,
        # so an interrupted run must not throw away the ones already built.
        if (REVIEW / video_id(v) / "sheet_00.jpg").exists():
            print(f"  {video_id(v)}  already built, skipped")
            continue
        make_sheets(v)

    if not SEGMENTS.exists():
        with open(SEGMENTS, "w", newline="", encoding="utf-8") as fh:
            fh.write("# One row per span of REAL footage. Delete nothing else.\n")
            fh.write("# fps: extraction rate for that span. 2 is the project default.\n")
            fh.write("# Exclude any span showing casualties or identifiable victims.\n")
            fh.write("youtube_id,start_s,end_s,fps,note\n")
    print(f"\nsheets -> {REVIEW}")
    print(f"now fill in -> {SEGMENTS}")
    print("then: python data/review_csb.py --extract")


if __name__ == "__main__":
    main()
