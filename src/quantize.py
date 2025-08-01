
import joblib
import numpy as np
import torch

model = joblib.load("models/sklearn_model.joblib")
weights, bias = model.coef_, model.intercept_

#Save original
joblib.dump({"weights": weights, "bias": bias}, "models/unquant_params.joblib")

#Quantization
scale = 255 / (weights.max() - weights.min())
q_weights = np.round((weights - weights.min()) * scale).astype(np.uint8)
q_bias = np.round((bias - weights.min()) * scale).astype(np.uint8)

joblib.dump({"weights": q_weights, "bias": q_bias, "scale": scale, "min": weights.min()}, "models/quant_params.joblib")

#Dequantize and validate with PyTorch
dq_weights = (q_weights.astype(np.float32) / scale) + weights.min()
dq_bias = (q_bias.astype(np.float32) / scale) + weights.min()

model_torch = torch.nn.Linear(len(dq_weights), 1)
model_torch.weight.data = torch.tensor([dq_weights], dtype=torch.float32)
model_torch.bias.data = torch.tensor([dq_bias], dtype=torch.float32)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = fetch_california_housing()
_, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
predicted = model_torch(X_test_tensor).detach().numpy()

#Print R² scores
print("Original R²:", r2_score(y_test, model.predict(X_test)))
print("Quantized R²:", r2_score(y_test, predicted))

