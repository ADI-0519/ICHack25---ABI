import os
from flask import Flask, request, jsonify, send_from_directory
import openai
import numpy as np

# Regression models
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# Classification model
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, static_folder='static')

# Set your OpenAI API key here or via an environment variable named OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/run-model', methods=['POST'])
def run_model():
    data = request.get_json()
    model_type = data.get("model_type", "")
    model_name = data.get("model_name", "Unnamed Model")
    try:
        ratio = float(data.get("ratio", 0.7))
    except Exception as e:
        return jsonify({"error": "Invalid ratio: " + str(e)}), 400

    # Optionally receive hyperparameters
    hyperparams = data.get("hyperparams", {})

    # For demonstration, log hyperparameters.
    print(f"Received hyperparameters for model '{model_name}': {hyperparams}")
    
    # Call OpenAI to generate a code snippet (for demonstration)
    prompt = (f"Generate Python code to create and train a {model_type} model using scikit-learn "
              f"with model name '{model_name}'. The training data is provided in CSV format, with each "
              "row containing feature values and a target value (last column) for regression or classification. "
              "Include splitting of data based on a train:test ratio if needed. "
              f"Use the following hyperparameters: {hyperparams}.")
    try:
        openai_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5
        )
        generated_code = openai_response.choices[0].text.strip()
    except Exception as e:
        generated_code = f"Error calling OpenAI API: {str(e)}"
    
    # Get CSV data for training and prediction
    train_csv = data.get("train_csv_data")
    predict_csv = data.get("predict_csv_data")
    if not train_csv:
        return jsonify({"error": "Missing train_csv_data in the payload."}), 400
    if not predict_csv:
        return jsonify({"error": "Missing predict_csv_data in the payload."}), 400

    try:
        # Parse training CSV data.
        train_rows = train_csv.strip().splitlines()
        train_list = [list(map(float, row.split(','))) for row in train_rows if row.strip() != '']
        train_array = np.array(train_list)
    except Exception as e:
        return jsonify({"error": "Error parsing train_csv_data: " + str(e)}), 400

    try:
        # Parse prediction CSV data.
        predict_rows = predict_csv.strip().splitlines()
        predict_list = [list(map(float, row.split(','))) for row in predict_rows if row.strip() != '']
        X_predict = np.array(predict_list)
    except Exception as e:
        return jsonify({"error": "Error parsing predict_csv_data: " + str(e)}), 400

    predictions = None
    try:
        if model_type in ["linearRegression", "polynomialRegression", "sgdRegression"]:
            # Regression models.
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            if model_type == "linearRegression":
                model = LinearRegression()
            elif model_type == "polynomialRegression":
                model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            elif model_type == "sgdRegression":
                model = SGDRegressor(max_iter=1000, tol=1e-3)
            model.fit(X_train, y_train)
            predictions = model.predict(X_predict).tolist()
        elif model_type in ["logisticRegression", "randomForest"]:
            # Classification models.
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            if model_type == "logisticRegression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "randomForest":
                model = RandomForestClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_predict).tolist()
        elif model_type == "feedforwardNN":
            # Neural network case.
            # For demonstration, we simply return a dummy prediction.
            # In a real implementation, you would construct and train a neural network using a library like Keras or PyTorch,
            # using the provided train_csv_data, predict_csv_data, nn_architecture (an array of neuron counts), and hyperparams.
            nn_architecture = data.get("nn_architecture", [])
            print(f"Neural network architecture for model '{model_name}': {nn_architecture}")
            # Dummy implementation:
            predictions = ["nn_prediction_dummy"]
        else:
            return jsonify({"error": "Model type not supported."}), 400
    except Exception as e:
        return jsonify({"error": "Error training model or making predictions: " + str(e)}), 500
    
    return jsonify({
        "generated_code": generated_code,
        "predictions": predictions
    })

if __name__ == '__main__':
    app.run(debug=True)
