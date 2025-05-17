import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("grid_search_results.csv", on_bad_lines="skip")

 
# Include additional inputs
X = df[["ITAE", "Settling Time", "Steady State Error"]].values  # Now using 3 inputs
y = df[["Kp", "Ki"]].values  # Output remains the same

# Normalize inputs with StandardScaler, outputs with MinMaxScaler
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(scaler_X, "grid_search_scaler_X.pkl")
joblib.dump(scaler_y, "grid_search_scaler_y.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, activation='softplus')  # Ensures Kp, Ki are non-negative
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)

# Compile model
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["mae"])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Stop training when validation loss stops improving
    patience=3,         # Wait 3 epochs before stopping
    restore_best_weights=True  # Restore the best model weights
)



history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=128,  # Increased batch size for stability
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stopping]  
)
# Plot Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
# Save model
model.save("grid_search_results.keras")

#Evaluate model
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Performance metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)


# Plot actual vs predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y[:, 0], y_pred[:, 0], alpha=0.6)
plt.xlabel("Actual Kp")
plt.ylabel("Predicted Kp")
plt.title("Actual vs Predicted Kp")
plt.plot([min(y[:, 0]), max(y[:, 0])], [min(y[:, 0]), max(y[:, 0])], color='red', linestyle='--')

plt.subplot(1, 2, 2)
plt.scatter(y[:, 1], y_pred[:, 1], alpha=0.6)
plt.xlabel("Actual Ki")
plt.ylabel("Predicted Ki")
plt.title("Actual vs Predicted Ki")
plt.plot([min(y[:, 1]), max(y[:, 1])], [min(y[:, 1]), max(y[:, 1])], color='red', linestyle='--')

plt.tight_layout()
plt.show()

 
