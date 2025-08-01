import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

#Load sklearn model
model = joblib.load("models/sklearn_model.joblib")
weights = model.coef_
bias = model.intercept_

#Save unquantized parameters
joblib.dump({"weights": weights, "bias": bias}, "models/unquant_params.joblib")

#Quantization
scale = 255 / (weights.max() - weights.min())
zero_point = weights.min()
q_weights = np.round((weights - zero_point) * scale).astype(np.uint8)
q_bias = np.round((bias - zero_point) * scale).astype(np.uint8)

#Save quantized values
joblib.dump({
    "weights": q_weights,
    "bias": q_bias,
    "scale": scale,
    "zero_point": zero_point
}, "models/quant_params.joblib")

#Dequantize
dequant_weights = (q_weights.astype(np.float32) / scale) + zero_point
dequant_bias = (q_bias.astype(np.float32) / scale) + zero_point

#Torch model
class QuantizedLinearModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

#Prepare data
data = fetch_california_housing()
X = data.data
y = data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Load weights into torch model
model_torch = QuantizedLinearModel(in_features=X.shape[1])
with torch.no_grad():
    model_torch.linear.weight.data = torch.tensor([dequant_weights], dtype=torch.float32)
    model_torch.linear.bias.data = torch.tensor(dequant_bias, dtype=torch.float32)

#Predict and evaluate
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
predicted = model_torch(X_test_tensor).squeeze().detach().numpy()

r2 = r2_score(y_test, predicted)
mse = mean_squared_error(y_test, predicted)

print("Quantized Model")
print("R2 Score:", round(r2, 4))
print("MSE:", round(mse, 4))

#File size
print("Quantized file size:", round(os.path.getsize("models/quant_params.joblib") / 1024, 2), "KB")

