import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#Load model
model = joblib.load("models/sklearn_model.joblib")

#Load data
data = fetch_california_housing()
_, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

#Predict
preds = model.predict(X_test)

#Evaluate
r2 = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)

#Show output
print("Sample predictions:", preds[:5])
print("R2 Score:", round(r2, 4))
print("MSE:", round(mse, 4))

