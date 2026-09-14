"""Extract frames from a video into a folder, with a provenance manifest.

    python extract_frames.py "<video>" --end 120 --fps 1

Every frame is recorded in frames.csv with its source video, YouTube id (parsed from the
CSB archive naming, `Title [id].mp4`), timestamp in ms and original frame index. That row
is what lets a frame be traced back to its source later - releasing annotations
depends on it, and reconstructing it after the fact from a folder of JPEGs is
miserable.

CSB footage is US federal work and therefore public domain, so its frames may be
redistributed. Frames pulled from YouTube may not be - only annotations plus the URL and
timestamp. Same script, different release rules, which is why the manifest records the
source either way.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2

YT_ID = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def hms(ms: float) -> str:
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=120.0, help="seconds")
    ap.add_argument("--fps", type=float, default=1.0, help="frames to extract per second")
    ap.add_argument("--out", default=None)
    ap.add_argument("--id", default=None,
                    help="clip id for frame names; defaults to the [id] in the filename. "
                         "Stock clips have no such bracket, so pass the source id here.")
    args = ap.parse_args()

    video = Path(args.video)
    m = YT_ID.search(video.stem)
    vid_id = args.id or (m.group(1) if m else video.stem[:11])

    out = Path(args.out) if args.out else video.parent.parent / "frames" / vid_id
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video.name}\n  {w}x{h}  {src_fps:.2f} fps  {total} frames  "
          f"{total / src_fps if src_fps else 0:.1f}s total")
    print(f"  extracting {args.start:.0f}-{args.end:.0f}s at {args.fps} fps -> {out}")

    step_ms = 1000.0 / args.fps
    rows = []
    t_ms = args.start * 1000.0
    end_ms = args.end * 1000.0

    while t_ms < end_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame = cap.read()
        if not ok:
            print(f"  stopped: no frame at {hms(t_ms)}")
            break
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        name = f"{vid_id}_{int(t_ms):07d}ms.jpg"
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "frame_file": name,
                "source_video": video.name,
                "youtube_id": vid_id,
                "timestamp_ms": int(t_ms),
                "timestamp_hms": hms(t_ms),
                "source_frame_index": idx,
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
        )
        t_ms += step_ms

    cap.release()

    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    size = sum(p.stat().st_size for p in out.glob("*.jpg"))
    print(f"\n{len(rows)} frames, {size / 1e6:.0f} MB")
    print(f"manifest -> {out / 'frames.csv'}")


if __name__ == "__main__":
    main()
