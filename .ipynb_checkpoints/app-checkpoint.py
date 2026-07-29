from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("cat_dog_model.keras")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    result = "🐶 Dog" if prediction[0][0] > 0.5 else "🐱 Cat"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)