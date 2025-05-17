import numpy as np
import joblib
import tensorflow as tf
import pandas as pd  
from scipy.integrate import simpson
from scipy.signal import lti, step
import os
import control_analysis as ca
MODEL_FOLDER = r"C:\Users\jonat"  # Update this with the correct path

# Load the trained model
try:
    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "grid_search_results.keras"))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the scalers
try:
    scaler_X = joblib.load(os.path.join(MODEL_FOLDER, "grid_search_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_FOLDER, "grid_search_scaler_y.pkl"))
except Exception as e:
    print(f"Error loading scalers: {e}")
    exit()
  

# **Generate Random ITAE Values for Testing**
num_tests = 1000
random_itae_values = np.random.uniform(0.03, 0.1, num_tests)  

# Apply correction to input ITAE before passing to model
corrected_input = (random_itae_values + 0.002) / 0.457  
itae_scaled = scaler_X.transform(corrected_input.reshape(-1, 1))

# **Predict Kp and Ki using the ANN**
predicted_scaled = model.predict(itae_scaled)

# **Inverse transform to get real Kp, Ki values**
predicted_values = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 2))
predicted_values = np.clip(predicted_values, 0, None)  

# **Compute ITAE using the predicted Kp, Ki**
ann_itae_values = np.array([ca.ITAE((Kp, Ki)) for Kp, Ki in predicted_values])

# **Compute errors (absolute and relative)**
absolute_errors = np.abs(ann_itae_values - random_itae_values)
relative_errors = absolute_errors / np.maximum(random_itae_values, 1e-6)  

# **Store results in a dataframe**
results_df = pd.DataFrame({
    "Random ITAE Input": random_itae_values,
    "Predicted Kp": predicted_values[:, 0],
    "Predicted Ki": predicted_values[:, 1],
    "ANN ITAE": ann_itae_values,
    "Absolute Error": absolute_errors,
    "Relative Error (%)": relative_errors * 100
})

# **Save results to a CSV file**
results_df.to_csv("ann_vs_true_itae_comparison_grid_search.csv", index=False)

 

import pandas as pd

 

# Display in console
print(results_df)
