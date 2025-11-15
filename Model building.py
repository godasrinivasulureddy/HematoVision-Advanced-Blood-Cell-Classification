"""
Model_building.py

Transfer-learning training script using MobileNetV2 backbone.
Adjust DATA_DIR to your dataset layout:

DATA_DIR/
    train/
        classA/
        classB/
        ...
    val/
        classA/
        classB/
        ...
"""

import os
import json

with open("history.json", "w") as f:
    json.dump(history.history, f)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

# ---------- CONFIG ----------
DATA_DIR = "dataset"  # replace with actual path or set before run
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_OUT = "Blood Cell.h5"
CLASS_MAP_OUT = "class_indices.json"
# ----------------------------

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        "Train/Val folders not found. Make sure DATA_DIR has train/ and val/ subfolders."
    )

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
).flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# Save class_indices mapping
with open(CLASS_MAP_OUT, "w") as f:
    json.dump({str(v): k for k, v in train_gen.class_indices.items()}, f)
print("Saved class indices to", CLASS_MAP_OUT)

# Build model (MobileNetV2 backbone)
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False  # freeze backbone initially

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks
)

# Optional: unfreeze last few layers and fine-tune
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_epochs = 10
history_f = model.fit(
    train_gen, validation_data=val_gen, epochs=fine_tune_epochs, callbacks=callbacks
)

# Save final model (best already saved by checkpoint)
if not os.path.exists(MODEL_OUT):
    model.save(MODEL_OUT)
print("Training finished. Model saved to", MODEL_OUT)
