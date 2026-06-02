import shutil
from pathlib import Path

# Paths
converted_dir = Path("datasets/NordTank_coco/converted/labels")
dataset_dir = Path("datasets/NordTank_coco")

# Move labels next to images for each split
for split in ["test","train", "valid"]:
    src = converted_dir / split  # convert_coco strips "instances_" prefix from JSON filename
    dst = dataset_dir / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        shutil.move(str(f), str(dst / f.name))