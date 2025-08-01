from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

#Prepare folder
os.makedirs("models", exist_ok=True)

#Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Original Model")
print("R² Score:", round(r2, 4))
print("MSE:", round(mse, 4))

#Save model
joblib.dump(model, "models/sklearn_model.joblib")

#File size
print("Model file size:", round(os.path.getsize("models/sklearn_model.joblib") / 1024, 2), "KB")

