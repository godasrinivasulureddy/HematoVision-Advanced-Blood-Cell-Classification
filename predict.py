"""
predict.py

Standalone helper you can run locally to get predictions on a folder or single image.
Usage:
    python predict.py /path/to/image.jpg
    python predict.py /path/to/folder_of_images/
"""

import sys
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_class_map(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            m = json.load(f)
            return {int(k): v for k, v in m.items()}
    return None


def preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)


def predict_image(model, class_map, img_path):
    x = preprocess(img_path)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    return class_map.get(idx, "Unknown"), float(np.max(preds))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_folder_path>")
        sys.exit(1)

    target = sys.argv[1]
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "Blood Cell.h5")
    map_path = os.path.join(base, "class_indices.json")

    if not os.path.exists(model_path):
        print("Model file Blood Cell.h5 not found in project folder.")
        sys.exit(1)

    model = load_model(model_path)
    class_map = load_class_map(map_path) or {
        0: "Eosinophil",
        1: "Lymphocyte",
        2: "Monocyte",
        3: "Neutrophil",
    }

    if os.path.isdir(target):
        files = [
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        files = [target]

    for f in files:
        if not os.path.exists(f):
            print("NOT FOUND:", f)
            continue
        label, conf = predict_image(model, class_map, f)
        print(f"{os.path.basename(f)} -> {label} ({conf*100:.2f}%)")
