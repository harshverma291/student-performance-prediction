from flask import Flask, render_template, request
import numpy as np
import joblib
from pymongo import MongoClient
import datetime

app = Flask(__name__)

# MongoDB Cloud Connection (Atlas String)
MONGO_URI = "mongodb+srv://hrverma438_db_user:Harshve730@cluster0.h0ao7g0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

# Database aur Collection setup
db = client['student_db']
collection = db['predictions']

# Load model
model = joblib.load('model.pkl')

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
# Dictionary containing the data to save
        prediction_data = {
            "study_hours": hours_study,
            "attendance": attendance,
            "previous_score": previous_score,
            "internet_access": internet_access,
            "extracurricular_activities": extracurricular,
            "parent_education": parent_education,
            "predicted_score": float(prediction[0]),
            "created_at": datetime.datetime.now()
        }
        
        # Inserting data into MongoDB
        collection.insert_one(prediction_data)

        if prediction[0] == 1:
            result = 'Student is likely to PASS with good performance.'
        else:
            result = 'Student may FAIL or perform poorly.'

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)