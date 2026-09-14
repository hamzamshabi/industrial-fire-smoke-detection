"""Download FASDD_CV (Flame And Smoke Detection Dataset, computer-vision subset).

Paper   : https://essd.copernicus.org/preprints/essd-2023-73/  (Wang et al.)
Canonical: https://doi.org/10.57760/sciencedb.j00104.00103  (ScienceDB, browser-only)
License : CC BY 4.0

Published composition (ESSD Table 2), FASDD_CV:
    95,314 images = 56,115 positive + 39,199 negative
    73,297 flame objects, 53,080 smoke objects

Only FASDD_CV is fetched. FASDD_RS is satellite imagery - wrong sensor and scale.

FASDD is used for its **negatives only**: its 56k positives would outweigh D-Fire's
17k training images roughly 3:1 and impose wildfire priors (large, distant fire against
sky and treeline) on a detector meant for plant CCTV. Its negatives are objects confused with flame (lights, sunset
glow, reflective clothing) and with smoke (dark clouds, shadows, hair, impervious
surfaces) - note there is no steam and no flare, so this does nothing for the refinery
nuisance problem and only addresses generic false positives.

Fetched from the HuggingFace mirror because ScienceDB has no scriptable API. The
mirror's stated breakdown (95k total / 39.1k null) matches the published table.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "weslleyskah/fire_smoke_dataset_fasdd_cv"
FILENAME = "FASDD_CV.zip"
DEST = Path(__file__).resolve().parent / "raw" / "fasdd"

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=FILENAME,
        local_dir=str(DEST),
    )
    size = Path(path).stat().st_size
    print(f"done -> {path}  ({size / 1e9:.2f} GB)")
