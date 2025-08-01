import os
import joblib
from sklearn.linear_model import LinearRegression

def test_model_exists():
    assert os.path.exists("models/sklearn_model.joblib")

def test_model_type():
    model = joblib.load("models/sklearn_model.joblib")
    assert isinstance(model, LinearRegression)

def test_model_has_coefficients():
    model = joblib.load("models/sklearn_model.joblib")
    assert hasattr(model, "coef_")

