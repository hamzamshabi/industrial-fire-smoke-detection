"""Build the README figures from files already in this repo. No weights needed.

    python docs/make_figures.py

Inputs:
    results/*_trace.csv   per-frame traces written by src/run_sequence.py --csv
    docs/frames/*.jpg     three Silver Eagle frames written by src/run_sequence.py --annotate
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"

NAVY, BLUE, TEAL, AMBER, RED, GREY = "#1e2a44", "#3d6fb5", "#2f8f83", "#d98c2b", "#c0453b", "#b8bfcc"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def load(name: str) -> list[dict]:
    with open(RESULTS / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def first_alarm(rows: list[dict], cls: str) -> float | None:
    return next((float(r["t_s"]) for r in rows if r["alarm"] == cls), None)


silver = load("silver_eagle_trace.csv")
chevron = load("chevron_richmond_trace.csv")
first_kept = min(float(r["t_s"]) for r in silver if int(r["kept"]) > 0)
smoke_t, fire_t = first_alarm(silver, "smoke"), first_alarm(silver, "fire")
suppressed = sum(int(r["suppressed"]) for r in silver)


# ---------------------------------------------------------------- header
BG, CARD, MUTED = "#121419", "#1c2029", "#9aa0ab"
fig = plt.figure(figsize=(16, 8.2), dpi=100, facecolor=BG)
fig.text(0.03, 0.915, "Fire & Smoke Detection for Industrial CCTV", color="white",
         fontsize=27, fontweight="bold", va="center")
fig.text(0.03, 0.862, "Two-class YOLO26m detector behind an alarm state machine  ·  "
         "flare ROI mask, 5-of-10 persistence, growth check, 60 s cooldown",
         color=MUTED, fontsize=13.5, va="center")

panels = [
    ("silver_eagle_t06.5s.jpg", f"t = {smoke_t:.1f} s", "SMOKE ALARM CONFIRMED"),
    ("silver_eagle_t08.0s.jpg", f"t = {fire_t:.1f} s", "FIRE ALARM CONFIRMED"),
    ("silver_eagle_t13.0s.jpg", "t = 13.0 s", "EVENT FULLY DEVELOPED"),
]
left, right, gap = 0.03, 0.97, 0.0125
w = (right - left - 2 * gap) / 3
img_h = w * 16 * 0.75 / 8.2  # 4:3 frame, converted to figure-height units
for i, (name, t, label) in enumerate(panels):
    x = left + i * (w + gap)
    ax = fig.add_axes([x, 0.36, w, img_h])
    ax.imshow(Image.open(DOCS / "frames" / name), aspect="auto")
    ax.set_axis_off()
    fig.add_artist(Rectangle((x, 0.28), w, 0.075, transform=fig.transFigure, color=CARD))
    fig.text(x + 0.009, 0.335, t, color=AMBER, fontsize=15, fontweight="bold", va="center")
    fig.text(x + 0.009, 0.298, label, color="#c9ced6", fontsize=12, fontweight="bold", va="center")

cards = [
    ("73.1%", "mAP@50 on D-Fire, up from 65.2%"),
    (f"{smoke_t - first_kept:.1f} s", "first detection to confirmed alarm"),
    (f"{suppressed}", "below-threshold detections filtered,\nno alarm lost"),
    ("≥51%", "of D-Fire's official test images have\na near-duplicate in its train split"),
]
cw = (right - left - 3 * gap) / 4
for i, (big, small) in enumerate(cards):
    x = left + i * (cw + gap)
    fig.add_artist(Rectangle((x, 0.095), cw, 0.135, transform=fig.transFigure, color=CARD))
    fig.add_artist(Rectangle((x, 0.095), 0.0022, 0.135, transform=fig.transFigure, color=AMBER))
    fig.text(x + 0.012, 0.19, big, color="white", fontsize=22, fontweight="bold", va="center")
    fig.text(x + 0.012, 0.132, small, color=MUTED, fontsize=11.5, va="center", linespacing=1.3)

fig.text(0.03, 0.04, "Frames: public-domain U.S. Chemical Safety Board surveillance footage "
         "(Silver Eagle refinery, CAM 006). Boxes and alarms are the output of src/run_sequence.py.",
         color="#6b7280", fontsize=10.5, va="center")
fig.savefig(DOCS / "header.png", facecolor=BG)
plt.close(fig)
print("wrote header.png")


# ---------------------------------------------------------------- alarm trace
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#7a8296", "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.grid": True, "grid.color": "#e2e5eb",
    "axes.axisbelow": True,
})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.3), dpi=200,
                               gridspec_kw={"width_ratios": [1.7, 1.0]})

win = [r for r in silver if float(r["t_s"]) <= 30]
t = [float(r["t_s"]) for r in win]
axL.fill_between(t, [int(r["raw"]) for r in win], step="mid", color=GREY, label="Raw detections")
axL.step(t, [int(r["kept"]) for r in win], where="mid", color=BLUE, linewidth=1.3,
         label="Kept by ROI and threshold")
for ta, text, col, dx, ha in [(smoke_t, f"smoke alarm {smoke_t:.1f} s", RED, -0.35, "right"),
                              (fire_t, f"fire alarm {fire_t:.1f} s", AMBER, 0.35, "left")]:
    axL.axvline(ta, color=col, linewidth=1.5, linestyle="--")
    axL.annotate(text, (ta + dx, 5.7), color=col, fontsize=8.5, ha=ha, va="top")
axL.annotate("", xy=(smoke_t, 4.4), xytext=(first_kept, 4.4),
             arrowprops=dict(arrowstyle="<->", color=NAVY, linewidth=1.1))
axL.annotate(f"{smoke_t - first_kept:.1f} s latency", ((first_kept + smoke_t) / 2, 4.55),
             ha="center", fontsize=8.5)
axL.set(xlim=(0, 30), ylim=(0, 6), xlabel="Time in clip (s)", ylabel="Detections per frame")
axL.set_title("Silver Eagle: ignition on a fixed camera", loc="left", fontsize=10.5)
axL.legend(frameon=False, fontsize=8.5, loc="upper right")

names = [f"Silver Eagle\n{len(silver) / 2:.0f} s", f"Chevron Richmond\n{len(chevron) / 2:.0f} s"]
kept = [sum(int(r["kept"]) for r in rows) for rows in (silver, chevron)]
supp = [sum(int(r["suppressed"]) for r in rows) for rows in (silver, chevron)]
n_alarms = [sum(1 for r in rows if r["alarm"]) for rows in (silver, chevron)]
axR.bar(names, kept, 0.5, label="Passed the filters", color=BLUE)
axR.bar(names, supp, 0.5, bottom=kept, label="Below threshold", color=GREY)
for i, n in enumerate(n_alarms):
    total = kept[i] + supp[i]
    axR.annotate(f"{total} detections\n{n} alarms", (i, total), ha="center", va="bottom",
                 fontsize=8.5, xytext=(0, 3), textcoords="offset points")
axR.set(ylim=(0, 760), ylabel="Detections in the clip")
axR.set_title("What the filters remove", loc="left", fontsize=10.5)
axR.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(DOCS / "alarm_trace.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote alarm_trace.png")


# ---------------------------------------------------------------- pipeline
boxes = [
    ("Fixed plant\nCCTV", "one frame every\n0.5-1 s", "#eceef2", NAVY),
    ("YOLO26m\ndetector", "fire and smoke\nboxes at 640 px", "#e6ebf7", BLUE),
    ("ROI mask", "drops boxes whose\nbottom edge sits\non the flare stack", "#e4f2ef", TEAL),
    ("Per-class\nthreshold", "fire 0.45\nsmoke 0.35", "#e4f2ef", TEAL),
    ("N-of-M\npersistence", "5 hits in the\nlast 10 frames", "#e4f2ef", TEAL),
    ("Growth\ncheck", "recent box area\n15% above the\nstart of the window", "#e4f2ef", TEAL),
    ("Alarm +\ncooldown", "60 s per class:\none notification\nper incident", "#faeae7", RED),
]
W, H = 16.0, 2.25
fig, ax = plt.subplots(figsize=(W, H), dpi=150)
ax.set(xlim=(0, W), ylim=(0, H))
ax.axis("off")
margin, gap = 0.1, 0.36
bw = (W - 2 * margin - gap * (len(boxes) - 1)) / len(boxes)
for i, (title, sub, fill, edge) in enumerate(boxes):
    x = margin + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 0.12), bw, H - 0.24, boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    ax.text(x + bw / 2, H - 0.56, title, ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=edge, linespacing=1.15)
    ax.text(x + bw / 2, 0.72, sub, ha="center", va="center", fontsize=10, color="#4a5266", linespacing=1.3)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.04, H / 2), (x + bw + gap - 0.04, H / 2), arrowstyle="-|>",
                                     mutation_scale=14, color="#7a8296", linewidth=1.3))
fig.savefig(DOCS / "pipeline.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote pipeline.png")


# ---------------------------------------------------------------- D-Fire leakage pairs, two columns
pairs = Image.open(DOCS / "audit_pairs_exact.png").convert("RGB")
gray = pairs.convert("L")
pw, ph = pairs.size
blank = [sum(gray.getpixel((x, y)) > 245 for x in range(0, pw, 2)) > 0.97 * (pw // 2) for y in range(ph)]
blocks, start = [], None
for y, is_blank in enumerate(blank + [True]):
    if not is_blank and start is None:
        start = y
    elif is_blank and start is not None:
        blocks.append([start, y])
        start = None
# Each pair starts at its short "TEST ... TRAIN ..." label line.
labels = [b[0] for b in blocks if b[1] - b[0] < 20]
merged = [[y0, (labels[i + 1] if i + 1 < len(labels) else ph) - 6] for i, y0 in enumerate(labels)]
bh = max(b[1] - b[0] for b in merged)
gutter, rows_n = 36, (len(merged) + 1) // 2
grid = Image.new("RGB", (2 * pw + gutter, rows_n * bh + (rows_n - 1) * 14), "white")
for i, (y0, y1) in enumerate(merged):
    col, row = divmod(i, rows_n)
    grid.paste(pairs.crop((0, y0, pw, y1)), (col * (pw + gutter), row * (bh + 14)))
grid.save(DOCS / "audit_pairs_grid.png")
print("wrote audit_pairs_grid.png")
