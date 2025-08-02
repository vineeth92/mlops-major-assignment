import sys
import os
import joblib
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import train

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_model_trained():
    model = joblib.load("models/sklearn_model.joblib")
    assert hasattr(model, "coef_")

def test_r2_threshold():
    assert train.model.score(train.X_test, train.y_test) > 0.5

