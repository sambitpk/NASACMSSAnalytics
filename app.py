# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:12:23 2024

@author: sambi
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("xgb_rul_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define API routes
@app.route('/')
def home():
    return "RUL Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request
        print(f"Incoming request: {request.data}")

        # Get JSON data from the request
        input_data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame(input_data)

        # Predict using the loaded model
        predictions = model.predict(input_df)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

print(f"Incoming request: {request.data}")
