import os
import json

def load_split(split_dir):
    """
    split_dir = train  OU valid  OU test
    """

    json_path = os.path.join(split_dir, "_annotations.coco.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    # image_id -> filename
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # category_id -> class name
    cat_id_to_name = {
        cat["id"]: cat["name"]
        for cat in data["categories"]
        if cat["name"] in ["child", "adult", "elderly"]
    }

    X_paths = []
    y_labels = []

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if cat_id not in cat_id_to_name:
            continue

        img_path = os.path.join(split_dir, id_to_filename[img_id])
        label = cat_id_to_name[cat_id]

        X_paths.append(img_path)
        y_labels.append(label)

    return X_paths, y_labels
