3. Model Deployment with Flask API
Now we deploy the trained model using Flask:

# app.py
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the MLflow model
model = mlflow.sklearn.load_model('runs:/<run_id>/random_forest')  # Replace <run_id> with your actual run ID

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # Ensure input is 2D

        # Predict using the model
        prediction = model.predict(features)

        # Return prediction as a response
        return jsonify({'prediction': int(prediction[0])})
   
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)