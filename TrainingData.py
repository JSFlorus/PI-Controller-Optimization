import numpy as np
import pandas as pd
import os
import time
from scipy.integrate import simpson
from scipy.signal import lti, step
import random
from joblib import Parallel, delayed

# Fixed Derivative Gain
fixed_Kd = 0  # Keeping Kd constant

# Objective function: Combination of ISE and ITAE
# You can adjust weights to prioritize one over the other
weight_ITAE = 1  # Weight for ITAE
weight_ISE = 0   # Weight for ISE

def simulate_response(Kp, Ki):
    """
    Simulates the closed-loop step response for a simple PI-controlled first-order system.
    """
    num = [Kp, Ki]  # PI Controller Transfer Function Numerator (Kp + Ki/s)
    den = [1, Kp, Ki]  # Simple First-order System
    system = lti(num, den)
    t, y = step(system)
    e = 1 - y  # Error signal (assuming unit step input)
    return t, e

def ITAE(params):
    """Computes Integral of Time-weighted Absolute Error (ITAE)."""
    Kp, Ki = params
    t, e = simulate_response(Kp, Ki)
    return simpson(t * np.abs(e), t)  # Integration using Simpson's rule

def ISE(params):
    """Computes Integral of Squared Error (ISE)."""
    Kp, Ki = params
    t, e = simulate_response(Kp, Ki)
    return simpson(e**2, t)  # Integration using Simpson's rule

def objective_function(params):
    return weight_ITAE * ITAE(params) + weight_ISE * ISE(params)

class EDBO_PI:
    def __init__(self, obj_func, dim, bounds, pop_size=5, max_iter=10):
        self.obj_func = obj_func  # Objective function
        self.dim = dim  # Number of parameters (Kp, Ki)
        self.bounds = np.array(bounds)
        self.pop_size = pop_size  # Number of dung beetles
        self.max_iter = max_iter  # Number of iterations

        # Initialize population
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (pop_size, dim))
        self.best_solution = None
        self.best_fitness = float("inf")

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                fitness = self.obj_func(self.population[i])
                
                # Update best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = self.population[i]

                # Merit-Oriented Search Mechanism (Guided movement toward better solutions)
                h = np.random.uniform(0, 1)
                merit_position = self.best_solution + h * (self.best_solution - self.population[i])
                self.population[i] = np.clip(merit_position, self.bounds[:, 0], self.bounds[:, 1])

                # Sine Learning Factor for Balancing Exploration & Exploitation
                r = np.sin(2*np.pi * t / self.max_iter)
                self.population[i] = r * self.population[i] + (1 - r) * np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

                # Dynamic Spiral Search for Fine-Tuning
                q = np.exp(np.cos(np.pi * t / self.max_iter))
                spiral_move = q * np.random.uniform(-2, 2, self.dim)
                self.population[i] += spiral_move

                # Adaptive t-Distribution Disturbance
                if np.random.rand() < 0.7:
                    self.population[i] += np.random.standard_t(df=3, size=self.dim)

                # Ensure values remain within bounds
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])

            # Print progress
            if (t+1) % 3 == 0:
                print(f"Iteration {t+1}, Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness
    

def grid_search(bounds, step_size):
    """
    Performs a grid search over the specified bounds with the given step size.
    """
    Kp_values = np.arange(bounds[0][0], bounds[0][1], step_size)
    Ki_values = np.arange(bounds[1][0], bounds[1][1], step_size)
    
    best_params = None
    best_fitness = float("inf")
    
    results = []
    
    for Kp in Kp_values:
        for Ki in Ki_values:
            fitness = objective_function([Kp, Ki])
             
            if fitness < 0.3:
                results.append({"Kp": Kp, "Ki": Ki, "Fitness": fitness})
                best_fitness = fitness
                best_params = [Kp, Ki]
    
    return best_params, best_fitness, results
def parallel_grid_search(bounds, step_size):
    """
    Performs a parallelized grid search over the specified bounds with the given step size.
    """
    Kp_values = np.arange(bounds[0][0], bounds[0][1], step_size)
    Ki_values = np.arange(bounds[1][0], bounds[1][1], step_size)

    # Generate all possible (Kp, Ki) pairs
    bound_combinations = [(Kp, Ki) for Kp in Kp_values for Ki in Ki_values]

    # Run optimization in parallel
    num_cores = -1  # Use all available CPU cores
    results = Parallel(n_jobs=num_cores)(
        delayed(evaluate_grid_point)(Kp, Ki) for Kp, Ki in bound_combinations
    )

    # Filter out None results (if fitness is not < 0.3)
    filtered_results = [res for res in results if res is not None]

    # Find the best parameters
    if filtered_results:
        best_result = min(filtered_results, key=lambda x: x["ITAE"])
        best_params = [best_result["Kp"], best_result["Ki"]]
        best_fitness = best_result["ITAE"]
    else:
        best_params, best_fitness = None, float("inf")

    return best_params, best_fitness, filtered_results
def evaluate_grid_point(Kp, Ki):
    """
    Evaluates a single (Kp, Ki) point in the grid search.
    """
    fitness = objective_function([Kp, Ki])
    if fitness < 5:
        return {"Kp": Kp, "Ki": Ki, "ITAE": fitness}
    return None  # Exclude non-optimal results


bounds = [(1e-6, 40), (1e-6, 40)]  # Kp and Ki bounds
step_size =  0.025 # Step size for the grid search

# Run the optimized grid search
best_params, best_fitness, results = parallel_grid_search(bounds, step_size)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv("grid_search_results.csv", index=False)

# Print the best result
print(f"Best Parameters: {best_params}, Best Fitness: {best_fitness}")
print(results_df.head())  # Display first few results
 
 

# import pandas as pd
# import numpy as np
# from joblib import Parallel, delayed
# range_step = 1
# def edbo_optimization_run(kp, kp_upper, ki, ki_upper, num_iterations=1):
#     """
#     Runs the EDBO_PI optimization for a given Kp and Ki bound combination.
#     """
#     results = []
#     bounds = [(kp, kp_upper), (ki, ki_upper)]
    
#     for i in range(num_iterations):
#         try:
#             optimizer = EDBO_PI(objective_function, dim=2, bounds=bounds, pop_size=10, max_iter=3)
#             best_pi_params, best_fitness = optimizer.optimize()

#             results.append({
#                 "Run": i + 1,
#                 "Kp Lower Bound": kp,
#                 "Kp Upper Bound": kp_upper,
#                 "Ki Lower Bound": ki,
#                 "Ki Upper Bound": ki_upper,
#                 "Kp": best_pi_params[0],
#                 "Ki": best_pi_params[1],
#                 "Fixed Kd": fixed_Kd,
#                 "ITAE": best_fitness,
#             })
#         except TypeError as e:
#             print(f"Error encountered during optimization: {e}")
#             break
    
#     return results

# # Generate bound combinations by incrementing range
# random_bound_combinations = []
# for upper_limit in range(range_step, 40, range_step):
#     for _ in range(2000):  # Adjust number of random samples per range as needed
#         kp = np.random.uniform(1e-6, upper_limit)
#         kp_upper = np.random.uniform(kp, upper_limit)
#         ki = np.random.uniform(1e-6, upper_limit)
#         ki_upper = np.random.uniform(ki, upper_limit)
#         random_bound_combinations.append((kp, kp_upper, ki, ki_upper))

# # Run optimizations in parallel using joblib
# num_cores = -1  # Use all available CPU cores
# optimization_results = Parallel(n_jobs=num_cores)(
#     delayed(edbo_optimization_run)(kp, kp_upper, ki, ki_upper, num_iterations=10)
#     for kp, kp_upper, ki, ki_upper in random_bound_combinations
# )

# # Flatten the results (since joblib returns a list of lists)
# optimization_results = [item for sublist in optimization_results for item in sublist]
# df = pd.DataFrame(optimization_results)

# opt_csv_path = "edbo_itae_results.csv"
# df.to_csv(opt_csv_path, index=False)

# # Ensure file is written before opening
# time.sleep(2)

# # Open the file automatically (Windows)
# os.startfile(opt_csv_path)
