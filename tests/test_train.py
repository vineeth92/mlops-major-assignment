import sys
import os
sys.path.append(r"C:\Users\Vineeth Varghese\mlops-major-assignment\src")

import joblib
from sklearn.linear_model import LinearRegression
import train

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_trained():
    model = joblib.load("models/sklearn_model.joblib")
    assert hasattr(model, "coef_")

def test_r2_threshold():
    assert train.model.score(train.X_test, train.y_test) > 0.5

