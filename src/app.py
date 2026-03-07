from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Enable CORS
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "malaria_model.h5")

model = load_model(MODEL_PATH)

@app.route("/")
def home():
    return "Malaria Detection API Running"


def predict_image(img):

    img = img.resize((224,224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "Parasitized (Malaria Detected)"
    else:
        result = "Uninfected (Healthy Cell)"

    return result, float(prediction)


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    img = Image.open(file.stream)

    result, confidence = predict_image(img)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })



if __name__ == "__main__":
    app.run(debug=True)