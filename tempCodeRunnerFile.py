
from sklearn.metrics import accuracy_score

from model import X_test, model

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(X_test, prediction)

print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model