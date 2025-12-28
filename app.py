from flask import Flask, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        bmi = float(request.form['bmi'])
        input_scaled = scaler.transform([[bmi]])
        prediction = model.predict(input_scaled)[0]
        return f'Predicted diabetes progression score: {prediction:.2f}'
    
    return '''
        <form method="post">
            BMI Offset Mean: <input name="bmi" type="text">
            <input type="submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
