import numpy as np
import pandas as pd
import os
import time
from scipy.integrate import simpson
from scipy.signal import lti, step
import random
from joblib import Parallel, delayed
import control_analysis as ca
import warnings
import bisect

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.signal")

class EDBO_PI:
    def __init__(self, obj_func, dim, bounds, pop_size=50, max_iter=100):
        self.obj_func = obj_func  # Objective function
        self.dim = dim  # Number of parameters (Kp, Ki)
        self.bounds = np.array(bounds)
        self.pop_size = pop_size  # Number of dung beetles
        self.max_iter = max_iter  # Number of iterations

        # Initialize population
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (pop_size, dim))
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []
    def evaluate_fitness(self, index):
        """Evaluate fitness function in parallel."""
        fitness = self.obj_func(self.population[index])
        return index, fitness


    def optimize(self):
        for t in range(self.max_iter):
                
            results = Parallel(n_jobs=-1)(
            delayed(self.evaluate_fitness)(i) for i in range(self.pop_size)
            )
            for i, fitness in results:
                fitness = self.obj_func(self.population[i])
                
                # Update best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.population[i]
                # Merit-Oriented Search Mechanism (Guided movement toward better solutions)
                h = np.random.uniform(0,1)  # Increase range for more movement
                I = np.random.uniform(1,2)
                if len(self.history) >= 5:
                    top_kp_ki = [entry for entry in self.history[:5]]  # Get the top 5 solutions
                    x_s = np.array([top_kp_ki[np.random.randint(0, len(top_kp_ki))]["Kp"], 
                                    top_kp_ki[np.random.randint(0, len(top_kp_ki))]["Ki"]])
                else:
                    x_s = self.population[i]  # Fallback to the current individual

                merit_position = self.population[i] + h * (x_s - I * self.population[i]) 
                self.population[i] = np.clip(merit_position, self.bounds[:, 0], self.bounds[:, 1])
                
                # Sine Learning Factor for Balancing Exploration & Exploitation
                r_min, r_max = 0.1, 1.0  # Ensuring a valid range
                t_theta = np.tan(np.random.uniform(0, np.pi))  # Random angle θ
                r = r_min + (r_max - r_min) * np.sin(np.pi * t / self.max_iter)
                prev_position = self.population[i-1] if i > 0 else self.population[i]
                self.population[i] = r * self.population[i] + (1 - r) * t_theta * np.abs(self.population[i] - prev_position)
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])


                # Dynamic Spiral Search for Fine-Tuning
                q = np.exp(np.cos(np.pi * t / self.max_iter))
                l = np.random.uniform(-1, 1)
                spiral_move = q * np.cos(2 * np.pi * l) * np.random.uniform(-1, 1, self.dim)
                self.population[i] += spiral_move
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])


                # Adaptive t-Distribution Disturbance
                df = np.exp((t / self.max_iter) ** 2)  
                t_disturbance = np.random.standard_t(df=df, size=self.dim)  #
                if np.random.rand() < 0.5: 
                    self.population[i] += t_disturbance  
                else: 
                    self.population[i] = self.best_solution  + self.best_solution  * t_disturbance  
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])


                # Print progress
                Kp= self.population[i][0]
                Ki = self.population[i][1]
                ITAE, overshoot, settling_time, steady_state_error   = ca.get_performance_metrics((Kp,Ki))
                if( ITAE < 5):
                    self.history.append({
                    "Iteration": t + 1,
                    "Kp": float(f"{Kp:.5f}"),
                    "Ki": float(f"{Ki:.5f}"),
                    "ITAE": float(f"{ITAE:.5f}"),
                    "Percent Overshoot (%)": float(f"{overshoot:.5f}"),
                    })
                self.history.sort(key=lambda x: x["ITAE"])
            if (t+1) % 10 == 0:
                print(f"Iteration {t+1}, ITAE = {ITAE}, Solution = {self.population[i]}, Over Shoot  = {overshoot}")

        print(f"Best ITAE = {self.best_fitness}, Best Solution = {self.best_solution}, Over Shoot  = {overshoot}")
        return self.best_solution, self.best_fitness, self.history
max = 40
step_size = 10
bounds = [(1e-6, max), (1e-6, max-step_size)] 
 

bound_generator = ca.BoundGenerator(bounds, step_size)  
b = bound_generator.bound_combinations_ranges  

num_jobs = -1  # Use all CPU cores
def run_edbo():
    results = []
    try:
        optimizer = EDBO_PI(lambda params: ca.ITAE(params), dim=2, bounds=bounds, pop_size=30, max_iter=100)
        best_pi_params, best_fitness, history = optimizer.optimize()
        df = pd.DataFrame(history)

        # Ensure we return a valid DataFrame
        if not df.empty:
            results.append(df)

    except Exception as e:
        print(f"Error in run_edbo: {e}")  # Debugging output
        return []  # Ensure we return a list, not a string

    return results


 
def edbo_optimization_run(kp, kp_upper, ki, ki_upper, num_iterations=2):
    """
    Runs the EDBO_PI optimization for a given Kp and Ki bound combination.
    """
    results = []
    bounds = [(kp, kp_upper), (ki, ki_upper)]

    for i in range(num_iterations):
        try:
            
            optimizer = EDBO_PI(lambda params: ca.ITAE(params), dim=2, bounds=bounds, pop_size=30, max_iter=100)
            best_pi_params, best_fitness, history = optimizer.optimize()
            results.append(pd.DataFrame(history))
            
        except TypeError as e:
            print(f"Error encountered during optimization: {e}")
            break
        except ValueError as e:
            print(f"ValueError encountered: {e}")
            break

    return results


def run_opt():
    optimization_results = Parallel(n_jobs=num_jobs)(
        delayed(edbo_optimization_run)(kp, kp_upper, ki, ki_upper, num_iterations=2)
        for kp, kp_upper, ki, ki_upper in b 
    )
    optimization_results = run_edbo()

    # Flatten results and filter out empty DataFrames
    filtered_results = [df for sublist in optimization_results for df in sublist if not df.empty]

    # Ensure there's data before concatenating
    if filtered_results:
        df_final = pd.concat(filtered_results, ignore_index=True)
    else:
        df_final = pd.DataFrame(columns=["Iteration", "Kp", "Ki", "ITAE", "Percent Overshoot (%)"])  # Empty DataFrame
    csv_path = "edbo_itae_results.csv"
    df_final.to_csv(csv_path, index=False)
    time.sleep(2)
    print(f"Optimization results saved to {csv_path}")
    return os.startfile(csv_path)

# df_final =  run_opt



optimization_results = run_edbo()

# Ensure `optimization_results` is a list of DataFrames
if isinstance(optimization_results, list) and all(isinstance(df, pd.DataFrame) for df in optimization_results):
    filtered_results = pd.concat(optimization_results, ignore_index=True)
else:
    filtered_results = pd.DataFrame()  # Default empty DataFrame

# Save or print the results
if not filtered_results.empty:
    # Save to a CSV file
    filtered_results.to_csv("EDBO_Optimization_Results.csv", index=False)
    
    # Print a preview of the first few rows
    print(filtered_results.head())
else:
    print("No valid results found!")

 
 