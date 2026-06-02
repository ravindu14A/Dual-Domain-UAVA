from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="./datasets/NordTank_coco/annotations",
    use_segments=False,   # True only if segmentation polygons exist
    use_keypoints=False,
    cls91to80=False,
    save_dir="./datasets/NordTank_coco/converted",
)