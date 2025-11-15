import os
import json
import numpy as np
from flask import Flask, request, render_template, send_from_directory, abort
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "Blood Cell.h5")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_indices.json")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB limit

# Load model and class mapping once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)

if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r") as f:
        class_indices = json.load(f)
else:
    # fallback default (replace with your actual map if different)
    class_indices = {
        "0": "Eosinophil",
        "1": "Lymphocyte",
        "2": "Monocyte",
        "3": "Neutrophil",
    }

# invert mapping so numeric index -> label
idx_to_label = {int(k): v for k, v in class_indices.items()}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/")
def home():
    # You should have templates/home.html; fallback simple form
    try:
        return render_template("home.html")
    except:
        return """
        <h3>Upload image (form fallback)</h3>
        <form method="POST" action="/predict" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
        """


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return abort(400, "No file part in request")

    file = request.files["file"]
    if file.filename == "":
        return abort(400, "No selected file")

    if not allowed_file(file.filename):
        return abort(400, "File type not allowed. Allowed: png/jpg/jpeg")

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Preprocess and predict
    try:
        x = preprocess_image(save_path, target_size=(224, 224))
        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        label = idx_to_label.get(idx, "Unknown")
        confidence = float(np.max(preds) * 100)
    except Exception as e:
        # If something fails, remove file and return error
        try:
            os.remove(save_path)
        except:
            pass
        return abort(500, f"Prediction error: {str(e)}")

    # Render result page if templates exist
    try:
        return render_template(
            "result.html",
            label=label,
            confidence=round(confidence, 2),
            image_file=f"static/uploads/{filename}",
        )
    except:
        return {
            "label": label,
            "confidence": round(confidence, 2),
            "image_path": f"static/uploads/{filename}",
        }
    @app.route("/home")
    def home_page():
        return render_template("home.html")

    @app.route("/")
    def index():
        return render_template("index.html")



@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # For local testing only. In production use gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
