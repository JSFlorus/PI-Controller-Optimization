import signal
import pickle
import os
import sys
from joblib import Parallel, delayed
import numpy as np
import control_analysis as ca
import pandas as pd
import joblib

# Checkpoint and pause files
CHECKPOINT_FILE = "checkpoint.pkl"
PAUSE_FILE = "pause.flag"
RESULTS_FILE = "grid_search_results.csv"
paused = False

def save_checkpoint(processed):
    """Saves checkpoint data to a file (excluding large results list)."""
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({"processed": processed}, f)

def load_checkpoint():
    """Loads checkpoint data from a file if available."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    return None

def check_pause():
    """Checks if the pause flag exists."""
    return os.path.exists(PAUSE_FILE)

def signal_handler(signum, frame):
    """Handles pause signals."""
    global paused
    paused = True
    print("\nPausing... Saving progress...")
    open(PAUSE_FILE, 'w').close()  # Create pause flag

def evaluate_params(Kp, Ki, bounds, itae_threshold=0.5, settling_time_threshold=1, steady_state_error_threshold=1):
    """Evaluates the performance metrics and stores unique valid results."""
    ITAE, overshoot, settling_time, steady_state_error = ca.get_performance_metrics((Kp, Ki))
    
    if ITAE < itae_threshold and settling_time < settling_time_threshold and steady_state_error < steady_state_error_threshold:
        return {
            "Kp": round(Kp, 8), 
            "Ki": round(Ki, 8),
            "ITAE": round(ITAE, 8), 
            "Settling Time": round(settling_time, 8), 
            "Steady State Error": round(steady_state_error, 8)
        }
    return None

import threading

csv_lock = threading.Lock()  # Ensure only one thread writes at a time

def save_results_to_csv(results):
    """Safely appends results to CSV, ensuring no duplicated headers."""
    if not results:
        return  # Skip empty data

    with csv_lock:  # Prevent parallel write conflicts
        file_exists = os.path.exists(RESULTS_FILE)
        
        df = pd.DataFrame(results)

        # Remove corrupted rows (e.g., incorrect column count)
        df = df.dropna()  # Drop any invalid rows
        df = df[df.columns[:5]]  # Ensure exactly 5 columns (avoiding extra header issues)

        df.to_csv(
            RESULTS_FILE,
            mode='a',
            index=False,
            header=not file_exists,  # Write header only if file doesn't exist
            float_format="%.8f",
            encoding="utf-8"
        )

def grid_search(bounds, step_size, itae_threshold=0.5, settling_time_threshold=1, steady_state_error_threshold=1, n_jobs=-1):
    """Performs parallelized grid search with checkpointing and optimized storage."""
    global paused

    # Generate parameter grid
    Kp_values = np.arange(bounds[0][0], bounds[0][1], step_size)
    Ki_values = np.arange(bounds[1][0], bounds[1][1], step_size)
    param_grid = [(Kp, Ki) for Kp in Kp_values for Ki in Ki_values]

    # Load checkpoint if available
    checkpoint = load_checkpoint()
    processed = checkpoint["processed"] if checkpoint else set()

    # Filter out already processed values
    param_grid = [(Kp, Ki) for Kp, Ki in param_grid if (Kp, Ki) not in processed]

    total_iterations = len(param_grid)
    processed_count = 0
    buffer_results = []  # Store temporary results before writing to CSV

    def process_param_pair(Kp, Ki):
        """Evaluates a parameter pair and returns result if valid."""
        if check_pause():
            raise KeyboardInterrupt  # Stop processing when paused
        return evaluate_params(Kp, Ki, bounds, itae_threshold, settling_time_threshold, steady_state_error_threshold)

    batch_size = 1000  # Process in chunks of 1000 for progress updates
    try:
        for i in range(0, len(param_grid), batch_size):
            batch = param_grid[i : i + batch_size]
            try:
                results_list = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(process_param_pair)(Kp, Ki) for Kp, Ki in batch
                )
            except KeyboardInterrupt:
                print("\nPause detected! Saving progress before exiting...")
                save_checkpoint(processed)
                save_results_to_csv(buffer_results)  # Save remaining results
                return

            # Filter valid results
            valid_results = [res for res in results_list if res is not None]

            # Add extra validation step
            valid_results = [r for r in valid_results if len(r) == 5]  # Ensure 5 columns only

            buffer_results.extend(valid_results)

            # Save periodically (Every 5000 entries)
            if len(buffer_results) >= 5000:
                save_results_to_csv(buffer_results)
                buffer_results.clear()
            processed_count += len(batch)
            save_checkpoint(processed)

            # Print progress
            print(f"Processed {processed_count}/{total_iterations} parameter sets...")

    except KeyboardInterrupt:
        print("\nManual Interrupt (Ctrl+C)! Saving progress before exiting...")
        save_checkpoint(processed)
        save_results_to_csv(buffer_results)
        return  # Gracefully exit

    # Final save to CSV (if buffer still has results)
    if buffer_results:
        save_results_to_csv(buffer_results)

# Run grid search
bounds = [(1e-6, 40), (1e-6, 40)]
step_size = 0.025

if os.path.exists(PAUSE_FILE):
    os.remove(PAUSE_FILE)

grid_search(bounds, step_size, n_jobs=-1)

print("Grid search completed. Results saved to grid_search_results.csv.")
