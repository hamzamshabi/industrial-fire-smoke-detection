"""Download the D-Fire dataset (HuggingFace mirror badsaarow/d-fire).

Mirror preserves the official D-Fire split: 17,221 train / 4,306 test = 21,527 images.
Canonical source: https://github.com/gaiasd/DFireDataset (OneDrive zip, browser-only).
export_dfire.py verifies the mirror against the published counts.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "badsaarow/d-fire"
DEST = Path(__file__).resolve().parent / "raw" / "dfire"

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=str(DEST),
        allow_patterns=["data/*.parquet", "README.md", ".gitattributes"],
        max_workers=8,
    )
    print("done ->", path)
