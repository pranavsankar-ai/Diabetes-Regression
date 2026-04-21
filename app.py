from flask import Flask, render_template_string, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names from the diabetes dataset
FEATURES = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Progression Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 50px; }
        .container { max-width: 600px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; background-color: #e9ecef; font-weight: bold; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center mb-4">Diabetes Progression Predictor</h2>
        <form method="POST">
            <div class="row">
                {% for feature in features %}
                <div class="col-md-6 mb-3">
                    <label class="form-label text-capitalize">{{ feature }}</label>
                    <input type="number" step="any" name="{{ feature }}" class="form-control" required value="0">
                </div>
                {% endfor %}
            </div>
            <button type="submit" class="btn btn-primary w-100">Predict Progression</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            Predicted Diabetes Progression Score: {{ "%.2f"|format(prediction) }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features from form
            input_data = [float(request.form[f]) for f in FEATURES]
            # Scale and predict
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]
        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE, features=FEATURES, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
