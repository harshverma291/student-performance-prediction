from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_study = float(request.form['hours_study'])
        attendance = float(request.form['attendance'])
        previous_score = float(request.form['previous_score'])
        internet_access = int(request.form['internet_access'])
        extracurricular = int(request.form['extracurricular'])
        parent_education = int(request.form['parent_education'])

        features = np.array([[
            hours_study,
            attendance,
            previous_score,
            internet_access,
            extracurricular,
            parent_education
        ]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = 'Student is likely to PASS with good performance.'
        else:
            result = 'Student may FAIL or perform poorly.'

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)