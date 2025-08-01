
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#Load model
model = joblib.load("models/sklearn_model.joblib")

#Load test data
data = fetch_california_housing()
_, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

#Predict
predictions = model.predict(X_test)
print("Sample predictions:", predictions[:5])

