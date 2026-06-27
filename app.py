from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Load the model using the absolute path
model = pickle.load(open(model_path, "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Find Your Crop Page
@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    nitrogen = float(request.form["nitrogen"])
    phosphorous = float(request.form["phosphorous"])
    potassium = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    features = np.array([
        [
            nitrogen,
            phosphorous,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]
    ])

    prediction = model.predict(features)

    return render_template(
        "findyourcrop.html",
        prediction_text=f"Recommended Crop: {prediction[0]}"
    )

if __name__ == "__main__":
    app.run(debug=True)