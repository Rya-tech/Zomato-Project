
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained machine learning model
MODEL_PATH = "zomato_sentiment_model.joblib"

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        review = data.get("review", "").strip()
        cuisine = data.get("cuisine", "Unknown").strip()

        cost = data.get("cost", "")

        if not review:
            return jsonify({"error": "Please enter a review."}), 400

        cost = float(cost) if cost else float("nan")

        # Prepare input in the same format used during training
        input_data = pd.DataFrame([{
            "Review": review,
            "Cuisines": cuisine if cuisine else "Unknown",
            "Cost_numeric": cost
        }])

        # Predict sentiment
        prediction = model.predict(input_data)[0]

        return jsonify({
            "sentiment": prediction
        })

    except ValueError:
        return jsonify({"error": "Please enter a valid restaurant cost."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
