import json
from pathlib import Path

import yaml

# Read categories from your COCO JSON
with open("datasets/NordTank_coco/annotations/train.json") as f:
    coco = json.load(f)

# Build class names matching convert_coco output (category_id - 1)
categories = sorted(coco["categories"], key=lambda x: x["id"])
names = {cat["id"] - 1: cat["name"] for cat in categories}
# NOTE: convert_coco maps class IDs as category_id - 1, so category_id must
# start from 1. If your categories start from 0, add 1 to each ID first.

# Create dataset.yaml
dataset = {
    "path": str(Path("datasets/NordTank_coco").resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": names,
}

with open("datasets/NordTank_coco/dataset.yaml", "w") as f:
    yaml.dump(dataset, f, default_flow_style=False)