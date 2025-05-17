import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("edbo_itae_results.keras")

# Load the scalers
scaler_X = joblib.load("edbo_scaler_X.pkl")
scaler_y = joblib.load("edbo_scaler_y.pkl")

# Load dataset
df = pd.read_csv("edbo_itae_results.csv")

# Extract input (ITAE) and output (Kp, Ki)
X = df[["ITAE"]].values
y = df[["Kp", "Ki"]].values

# Normalize input (use same scaling as training)
X_scaled = scaler_X.transform(X)

# Predict using trained model
y_pred_scaled = model.predict(X_scaled)

# Inverse transform predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate performance
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot Actual vs Predicted values for Kp and Ki
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y[:, 0], y_pred[:, 0], alpha=0.6)
plt.xlabel("Actual Kp")
plt.ylabel("Predicted Kp")
plt.title("Actual vs Predicted Kp")
plt.plot([min(y[:, 0]), max(y[:, 0])], [min(y[:, 0]), max(y[:, 0])], color='red', linestyle='--')  # Ideal line

plt.subplot(1, 2, 2)
plt.scatter(y[:, 1], y_pred[:, 1], alpha=0.6)
plt.xlabel("Actual Ki")
plt.ylabel("Predicted Ki")
plt.title("Actual vs Predicted Ki")
plt.plot([min(y[:, 1]), max(y[:, 1])], [min(y[:, 1]), max(y[:, 1])], color='red', linestyle='--')  # Ideal line

plt.tight_layout()
plt.show()
