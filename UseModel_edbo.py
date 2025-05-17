import joblib
import numpy as np
import tensorflow as tf
import control_analysis as ca
# Load the trained model
model = tf.keras.models.load_model("edbo_itae_results_improved.keras")


scaler_X = joblib.load("edbo_scaler_X.pkl")
scaler_y = joblib.load("edbo_scaler_y.pkl")

# Function to interactively predict Kp and Ki
def predict_kp_ki():
    while True:
        try:
            # Ask for user input
            itae = float(input("Enter ITAE value: "))
            settling_time = float(input("Enter Settling Time: "))
            steady_state_error = float(input("Enter Steady State Error: "))

            # Prepare input
            new_input = np.array([[itae, settling_time, steady_state_error]])
            new_input_scaled = scaler_X.transform(new_input)  # Normalize input

            # Predict
            predicted_scaled = model.predict(new_input_scaled)
            predicted = scaler_y.inverse_transform(predicted_scaled)  # Convert back to original scale

            # Print results
            print(f"\nPredicted Kp: {predicted[0][0]:.4f}")
            print(f"Predicted Ki: {predicted[0][1]:.4f}\n")
            print(f"Actual ITAE Ki: {ca.ITAE((predicted[0][0],predicted[0][1]))}")

        except ValueError:
            print("Invalid input! Please enter numeric values.")

        # Ask if user wants to continue
        again = input("Do you want to predict again? (yes/no): ").strip().lower()
        if again not in ["yes", "y"]:
            print("Exiting...")
            break

# Run the interactive prediction function
predict_kp_ki()
