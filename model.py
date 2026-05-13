import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
Data = pd.read_csv('student_data.csv')

# Input features
X = Data.drop('final_result', axis=1)

# Output label
Y = Data['final_result']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, Y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, prediction)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model
joblib.dump(model, 'model.pkl')

print('Model saved successfully!')