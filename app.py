import numpy as np
import pickle
import io
import base64
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------
model = pickle.load(open('model.pkl', 'rb'))

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    logging.info("Scaler loaded successfully")
except:
    scaler = None
    logging.warning("Scaler not found, using raw inputs")

# -------------------------------------------------
# FEATURE ORDER (VERY IMPORTANT)
# -------------------------------------------------
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

# -------------------------------------------------
# INPUT VALIDATION (UTILS)
# -------------------------------------------------
def validate_inputs(data):
    if not (1 <= data['age'] <= 120):
        return False, "Invalid Age"

    if not (50 <= data['trestbps'] <= 250):
        return False, "Invalid Resting Blood Pressure"

    if not (50 <= data['chol'] <= 600):
        return False, "Invalid Cholesterol Level"

    if not (60 <= data['thalach'] <= 220):
        return False, "Invalid Maximum Heart Rate"

    return True, None

# -------------------------------------------------
# PREDICTION LOGIC (PREDICTOR)
# -------------------------------------------------
def predict_heart_disease(input_data):
    values = [input_data[feature] for feature in FEATURE_ORDER]
    X = np.array(values).reshape(1, -1)

    if scaler:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk (Healthy)"

    # Confidence Graph
    plt.figure(figsize=(6, 4))
    plt.bar(['Healthy', 'Disease'], probabilities, color=['green', 'red'])
    plt.title('Prediction Confidence')
    plt.ylim(0, 1)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return result, graph_url

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {key: float(value) for key, value in request.form.items()}
        logging.info(f"Received input: {input_data}")

        valid, error = validate_inputs(input_data)
        if not valid:
            return render_template(
                'index.html',
                prediction_text=error,
                text_color="orange"
            )

        result, graph_url = predict_heart_disease(input_data)

        return render_template(
            'index.html',
            prediction_text=result,
            text_color="red" if "High" in result else "green",
            graph_url=graph_url
        )

    except Exception as e:
        logging.error(str(e))
        return render_template(
            'index.html',
            prediction_text="Internal Server Error",
            text_color="red"
        )

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
