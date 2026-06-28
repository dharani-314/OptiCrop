from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ==========================
# Load Trained Model
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# About Page
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# Find Your Crop Page
# ==========================

@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")


# ==========================
# Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    nitrogen = float(request.form["nitrogen"])
    phosphorous = float(request.form["phosphorous"])
    potassium = float(request.form["potassium"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    features = np.array([[
        nitrogen,
        phosphorous,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    prediction = model.predict(features)

    return render_template(
        "findyourcrop.html",

        prediction_text=prediction[0],

        nitrogen=nitrogen,
        phosphorous=phosphorous,
        potassium=potassium,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall
    )


# ==========================
# Run Application
# ==========================

if __name__ == "__main__":
    app.run(debug=True)