import os
import shutil
from sklearn.model_selection import train_test_split

DATASET = "raw_dataset"  # input folder
OUTPUT = "dataset"  # structured output folder
SPLIT = 0.2  # 20% validation


def create_structure():
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    for sub in ["train", "val"]:
        for cls in os.listdir(DATASET):
            os.makedirs(os.path.join(OUTPUT, sub, cls), exist_ok=True)


def split_data():
    for cls in os.listdir(DATASET):
        cls_path = os.path.join(DATASET, cls)
        imgs = [f for f in os.listdir(cls_path)]

        train, val = train_test_split(imgs, test_size=SPLIT, random_state=42)

        for i in train:
            shutil.copy(
                os.path.join(cls_path, i), os.path.join(OUTPUT, "train", cls, i)
            )

        for i in val:
            shutil.copy(os.path.join(cls_path, i), os.path.join(OUTPUT, "val", cls, i))

    print("Dataset successfully split into train/val folders.")


if __name__ == "__main__":
    create_structure()
    split_data()
