"""Dedupe the 1 fps video_frames_1fps batch into data/raw/video_pool/.

Different brief from the CSB/stock passes: keep volume. So there is **no per-clip cap** here
and the dedup threshold is looser (0.02 vs 0.03). The cap existed because one 800-frame
locked-off shot was dominating a 245-frame pool; with 139 clips averaging 97 frames no
single clip can dominate, so redundancy is the only filter applied.

Per-frame content review is not attempted at this scale - 139 clips of broadcast footage
would need per-second passes, and inferred drop-ranges leaked three times on a single clip
last time. Only whole-clip decisions are made, from the folder name plus a sampled sheet.

Frames are renamed <youtube_id>_<n>.jpg: the source names are all `frame_000001.jpg` and
would collide across 139 folders, and the id carries provenance for free.

Watch out when editing: folder names contain [ ] : | Cyrillic and emoji, so use os.listdir,
never glob (which reads [...] as a character class), and reconfigure stdout to utf-8.
"""
import csv, os, re, shutil, sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
SRC = os.path.join(ROOT, "video_frames_1fps")
POOL = os.path.join(ROOT, "video_pool")
REJ = os.path.join(ROOT, "video_rejected")
THRESH = 0.02
YT = re.compile(r"\[([A-Za-z0-9_-]{11})\]")

# Whole clips dropped, with the reason recorded.
DROP = {
    "gc8qXTh6tTY": "CSB Philadelphia reconstruction - 100% CG animation. Established in the "
                   "earlier review (see csb/rejected/CG/gc8qXTh6tTY): rendered night "
                   "fireball, tank lofted through the air, CG city skyline.",
}

# Clips kept but tagged, because they are not petrochemical-plant footage. Wrong-domain fire
# data teaches wrong priors, so these are separable in one
# manifest column rather than silently mixed in or silently dropped.
DOMAIN = {
    "yWM7aGp1HRU": "structure", "oGK0DmTlS4Y": "structure", "g7NEOxhOvI0": "structure",
    "9S8wyKTkA_w": "vehicle", "lzV39v3xuRk": "vehicle",
    "L5qZgQDKwZA": "artificial_smoke",   # smoke-generator test rig, not combustion
    "gfk9kLep-Ds": "steam",              # stock footage of stack plumes, no fire
}


def sig(path):
    """64x64 grayscale of the content area, overlay strips trimmed. Same metric as before."""
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.uint8)
    rows = np.where(a.max(axis=1) > 24)[0]
    cols = np.where(a.max(axis=0) > 24)[0]
    if len(rows) and len(cols):
        im = im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
    w, h = im.size
    im = im.crop((int(w * 0.02), int(h * 0.11), int(w * 0.98), int(h * 0.89)))
    return np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0


def main():
    os.makedirs(POOL, exist_ok=True)
    dirs = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    rows, stats = [], []

    for i, d in enumerate(dirs, 1):
        vid = YT.search(d).group(1)
        files = sorted(f for f in os.listdir(os.path.join(SRC, d)) if f.lower().endswith(".jpg"))
        raw = len(files)

        if vid in DROP:
            dd = os.path.join(REJ, "CG", vid)
            os.makedirs(dd, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(SRC, d, f), os.path.join(dd, f"{vid}_{f[6:]}"))
            stats.append((d, vid, raw, 0, raw, "dropped whole clip"))
            print(f"[{i}/{len(dirs)}] {vid} {raw:5d} -> 0  DROPPED", flush=True)
            continue

        sigs = [sig(os.path.join(SRC, d, f)) for f in files]
        kept = []
        for j in range(len(files)):
            if all(float(np.abs(sigs[j] - sigs[k]).mean()) >= THRESH for k in kept):
                kept.append(j)
        keep = set(kept)

        for j, f in enumerate(files):
            name = f"{vid}_{f[6:]}"
            if j in keep:
                shutil.copy2(os.path.join(SRC, d, f), os.path.join(POOL, name))
                rows.append({
                    "frame_file": name, "youtube_id": vid, "source_dir": d,
                    "frame_index": int(f[6:12]), "second": int(f[6:12]) - 1,
                    "domain": DOMAIN.get(vid, "industrial"),
                    "licence": "YouTube standard licence - training use only, do not redistribute",
                })
            else:
                dd = os.path.join(REJ, "REDUNDANT", vid)
                os.makedirs(dd, exist_ok=True)
                shutil.copy2(os.path.join(SRC, d, f), os.path.join(dd, name))
        stats.append((d, vid, raw, len(kept), raw - len(kept), ""))
        print(f"[{i}/{len(dirs)}] {vid} {raw:5d} -> {len(kept):4d}", flush=True)

    with open(os.path.join(POOL, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REJ, "per_clip.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_dir", "youtube_id", "extracted", "kept", "dropped", "note"])
        w.writerows(stats)

    raw = sum(s[2] for s in stats)
    kept = sum(s[3] for s in stats)
    print(f"\nextracted {raw}  kept {kept}  dropped {raw - kept}")


if __name__ == "__main__":
    main()
