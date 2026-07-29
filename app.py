from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    # Save uploaded image
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)[0][0]

    # Swap labels (Cat = 1, Dog = 0)
    if prediction > 0.5:
        result = "🐱 Cat"
        confidence = prediction * 100
    else:
        result = "🐶 Dog"
        confidence = (1 - prediction) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence:.2f}",
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)