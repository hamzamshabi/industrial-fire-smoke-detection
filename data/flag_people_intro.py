"""Flag people shots and clip intros in video_pool/ so they can be cut without eyeballing 9,761 frames.

Per-frame visual review is not feasible at this scale, so two cheap automatic signals stand
in for it. Both are recorded as columns, not acted on - the flagged frames get reviewed on
contact sheets before anything is moved.

people:  Haar frontal + profile face detection. Anchors, reporter stand-ups and interviewees
         all face the camera and fill a good part of the frame, which is exactly what Haar
         is good at. MIN_FRAC ignores small faces so distant bystanders at a fire scene do
         not drag whole clips out - the target is talking heads, not crowds.

intro:   the first INTRO_SECS frames of each clip, where channel idents and title cards sit.
         Flagged for review rather than dropped outright: plenty of clips open straight on
         the fire, and those opening frames are often the clearest footage in the clip.
"""
import csv, os, sys

import cv2

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
POOL = os.path.join(ROOT, "video_pool")
OUT = os.path.join(ROOT, "video_pool", "flags.csv")

MIN_FRAC = 0.055        # min face height as a fraction of image height
INTRO_SECS = 3

FRONT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def faces(path):
    im = cv2.imread(path)
    if im is None:
        return 0, 0.0
    h, w = im.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        im = cv2.resize(im, (int(w * scale), int(h * scale)))
    g = cv2.equalizeHist(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    mn = int(im.shape[0] * MIN_FRAC)
    found = []
    for c in (FRONT, PROFILE):
        found += list(c.detectMultiScale(g, 1.15, 6, minSize=(mn, mn)))
    if not found:
        return 0, 0.0
    biggest = max(f[3] for f in found) / im.shape[0]
    return len(found), float(biggest)


def main():
    rows = list(csv.DictReader(open(os.path.join(POOL, "manifest.csv"), encoding="utf-8")))
    out = []
    for i, r in enumerate(rows, 1):
        n, frac = faces(os.path.join(POOL, r["frame_file"]))
        out.append({
            "frame_file": r["frame_file"], "youtube_id": r["youtube_id"],
            "second": r["second"], "faces": n, "face_frac": round(frac, 3),
            "intro": int(int(r["second"]) < INTRO_SECS),
        })
        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    people = [r for r in out if r["faces"]]
    intro = [r for r in out if r["intro"]]
    both = [r for r in out if r["faces"] and r["intro"]]
    print(f"\n{len(out)} frames")
    print(f"  faces detected : {len(people)}")
    print(f"  intro (<{INTRO_SECS}s)   : {len(intro)}")
    print(f"  union          : {len(people) + len(intro) - len(both)}")
    big = [r for r in people if r["face_frac"] >= 0.15]
    print(f"  large face (>=15% of height, i.e. a talking head): {len(big)}")


if __name__ == "__main__":
    main()
