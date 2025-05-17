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
import heapq
from multiprocessing import Process, Queue
from multiprocessing import Manager


warnings.filterwarnings("ignore", category=UserWarning, module="scipy.signal")
unique_entries = set()
 

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
        self.historyTop5 = []  # Maintains top 5 lowest ITAE values

    @staticmethod
    @staticmethod
    def insert_top_5(history_top5, new_entry):
        """Maintain only the top 5 lowest ITAE values using a min-heap."""
        
        if "ITAE" not in new_entry:
            print("Warning: New entry does not contain ITAE key!", new_entry)
            return  # Skip invalid entries
        
        entry_tuple = (new_entry["ITAE"], new_entry)  # Ensure first element is numeric
        
        if len(history_top5) < 5:
            heapq.heappush(history_top5, entry_tuple)  # Push normally
        else:
            heapq.heappushpop(history_top5, entry_tuple)  # Push & pop to maintain top 5


    def evaluate_fitness(self, index):
        """Evaluate fitness function in parallel."""
        fitness = self.obj_func(self.population[index])
        return index, fitness

    @staticmethod
    def update_beetle(i, population, history_top5, bounds, best_solution, best_fitness, obj_func, t, max_iter):
        """Parallelized update for a single beetle in the population."""
        
        # Evaluate fitness
        fitness = obj_func(population[i])
        
        # Local best tracking (to avoid race conditions)
        local_best_solution = best_solution
        local_best_fitness = best_fitness

        # Update best solution locally
        if fitness < local_best_fitness:
            local_best_fitness = fitness
            local_best_solution = population[i]

        # Merit-Oriented Search Mechanism
        h = np.random.uniform(0, 1)
        I = np.random.uniform(1, 2)

        if len(history_top5) >= 5:
            top_kp_ki = [entry[1] for entry in history_top5[:5]]  # Extract top 5 solutions
            x_s = np.array([
                top_kp_ki[np.random.randint(0, len(top_kp_ki))]["Kp"], 
                top_kp_ki[np.random.randint(0, len(top_kp_ki))]["Ki"]
            ])
        else:
            x_s = population[i]

        merit_position = population[i] + h * (x_s - I * population[i]) 
        new_position = np.clip(merit_position, bounds[:, 0], bounds[:, 1])

        # Sine Learning Factor
        r_min, r_max = 0.1, 1.0
        t_theta = np.tan(np.random.uniform(0, np.pi))
        r = r_min + (r_max - r_min) * np.sin(np.pi * t / max_iter)

        prev_position = population[i-1] if i > 0 else population[i]
        new_position = r * new_position + (1 - r) * t_theta * np.abs(new_position - prev_position)
        new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])

        # Dynamic Spiral Search
        q = np.exp(np.cos(np.pi * t / max_iter))
        l = np.random.uniform(-1, 1)
        spiral_move = q * np.cos(2 * np.pi * l) * np.random.uniform(-1, 1, population.shape[1])
        new_position += spiral_move
        new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])

        # Adaptive t-Distribution Disturbance
        df = np.exp((t / max_iter) ** 2)  
        t_disturbance = np.random.standard_t(df=df, size=population.shape[1])
        if np.random.rand() < 0.5:
            new_position += t_disturbance
        else:
            new_position = local_best_solution + local_best_solution * t_disturbance
        new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])

        # Compute Performance Metrics
        Kp, Ki = new_position
        ITAE, overshoot, settling_time, steady_state_error = ca.get_performance_metrics((Kp, Ki))

        new_entry = None

        bounds_array = np.array(bounds)
        if bounds_array.shape == (2, 2):
            Kp_lower, Kp_upper = bounds_array[0]
            Ki_lower, Ki_upper = bounds_array[1]
        else:
            bounds_array = bounds_array.reshape(2, 2)  # Force reshape
            Kp_lower, Kp_upper = bounds_array[0]
            Ki_lower, Ki_upper = bounds_array[1]
        if ITAE < 1 and settling_time < 2 and steady_state_error < 1:
            new_entry_tuple = (round(Kp, 8), round(Ki, 8), round(ITAE, 8))  # Use tuple for set

            if new_entry_tuple not in unique_entries:
                unique_entries.add(new_entry_tuple)  # Add to set
           
                new_entry = {
           
                    "Kp Lower": [Kp_lower],
                    "Kp Upper": [Kp_upper],
                    "Ki Lower": [Ki_lower],
                    "Ki Upper": [Ki_upper],
                    "Kp": float(f"{Kp:.8f}"),
                    "Ki": float(f"{Ki:.8f}"),
                    "ITAE": float(f"{ITAE:.8f}"),
                    "Settling Time": float(f"{settling_time:.8f}"),
                    "Steady State Error": float(f"{steady_state_error:.8f}"),
                    

                    
                }
        
        return i, new_position, fitness, local_best_solution, local_best_fitness, new_entry

    def optimize(self):
        for t in range(self.max_iter):
            # Run parallelized updates for the entire population
            results = Parallel(n_jobs=-1)(
                delayed(EDBO_PI.update_beetle)(
                    i, self.population, self.historyTop5, self.bounds, 
                    self.best_solution, self.best_fitness, 
                    self.obj_func, t, self.max_iter
                ) for i in range(self.pop_size)
            )

            # Process the results (update population & best solution)
            for i, new_position, fitness, best_sol, best_fit, new_entry in results:
                self.population[i] = new_position

                # Update best solution and best fitness
                if best_fit < self.best_fitness:
                    self.best_fitness = best_fit
                    self.best_solution = best_sol
                
                if new_entry:
                    self.history.append(new_entry)

                # Maintain the top 5 solutions
                if new_entry:
                    self.insert_top_5(self.historyTop5, new_entry)

            # Logging progress
            if (t+1) % 10 == 0:
                print(f"Iteration {t+1}")

        print(f"Final Best ITAE = {self.best_fitness}, Best Solution = {self.best_solution}")
        return self.best_solution, self.best_fitness, self.history



max = 20+1e-6
min = 1e-6

step_size = 10
bounds = [(min, max), (min, max)] 


bound_generator = ca.BoundGenerator(bounds, step_size)  
b = bound_generator.bound_combinations_ranges  

num_jobs = -1  # Use all CPU cores
 

 
def edbo_optimization_run(kp, kp_upper, ki, ki_upper, num_iterations=1):
    """
    Runs the EDBO_PI optimization for a given Kp and Ki bound combination.
    """
    results = []
    bounds = [(kp, kp_upper), (ki, ki_upper)]

    for i in range(num_iterations):
        try:
            
            optimizer = EDBO_PI(lambda params: ca.ITAE(params), dim=2, bounds=bounds, pop_size=10, max_iter=400)
            best_pi_params, best_fitness, history = optimizer.optimize()
            results.append(pd.DataFrame(history))
            results_df = pd.concat(results, ignore_index=True)


            
        except TypeError as e:
            print(f"Error encountered during optimization: {e}")
            break
        except ValueError as e:
            print(f"ValueError encountered: {e}")
            break

    return results



num_jobs = -1  # Use all available CPU cores
csv_path = "edbo_itae_results.csv"

 
def save_results_worker(queue, csv_path):
    """Worker function that continuously saves results from the queue to a CSV file."""
    while True:
        results_df = queue.get()  # Get DataFrame from the queue
        if results_df is None:
            break  # Exit when None is received (graceful shutdown)

        file_exists = os.path.exists(csv_path)

        # Drop NaN rows to avoid corruption
        results_df = results_df.dropna()

        expected_columns = ["Kp Lower", "Kp Upper", "Ki Lower", "Ki Upper", "Kp", "Ki", "ITAE", "Settling Time", "Steady State Error"]
        results_df = results_df[expected_columns]  # Ensure consistent columns

        results_df.to_csv(
            csv_path,
            mode='a',  # Append mode
            index=False,
            header=not file_exists,  # Write header only if the file is new
            float_format="%.8f",
            encoding="utf-8"
        )

        print(f"Saved {len(results_df)} new results to {csv_path}")
        
def process_bounds(kp, kp_upper, ki, ki_upper, total_bounds, counter, queue):
    """Runs optimization for a given set of bounds and sends results to queue."""
    counter[0] += 1
    print(f"Running optimization {counter[0]}/{total_bounds} for bounds: Kp=({kp}, {kp_upper}), Ki=({ki}, {ki_upper})")

    optimization_results = edbo_optimization_run(kp, kp_upper, ki, ki_upper, num_iterations=1)

    filtered_results = [df for df in optimization_results if not df.empty]
    if filtered_results:
        df_final = pd.concat(filtered_results, ignore_index=True)
        queue.put(df_final)  # Send to queue for writing
    else:
        print(f"No valid results for bounds Kp=({kp}, {kp_upper}), Ki=({ki}, {ki_upper})")

    print(f"Completed {counter[0]}/{total_bounds}: Sent results to queue.")


def run_opt():
    """Runs optimization in parallel while saving results asynchronously."""
    total_bounds = len(b)
    counter = [0]  # Mutable counter for tracking progress

    manager = Manager()
    queue = manager.Queue()  # Use Manager().Queue() for compatibility
    writer_process = Process(target=save_results_worker, args=(queue, csv_path))
    writer_process.start()  # Start the CSV writer process

    # Run parallel jobs, explicitly passing `queue`
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_bounds)(kp, kp_upper, ki, ki_upper, total_bounds, counter, queue)
        for kp, kp_upper, ki, ki_upper in b
    )

    # Signal the writer process to stop by sending `None`
    queue.put(None)
    writer_process.join()  # Wait for writer to finish

    print(f"Optimization complete! Final results stored in {csv_path}")
    return os.startfile(csv_path)

if __name__ == '__main__':
    df_final = run_opt()



 
# import time
# import pandas as pd

# # Define different test configurations for (pop_size, max_iter)
# test_configs = [
#     (30, 800),   # Baseline case
#     (45, 800),   # Lower population
#     (30, 400),    # Lower iterations
#     (45, 400),    # Both reduced
#     (30, 800),  # Higher population & iterations (worst case)
# ]

# # Function to benchmark EDBO_PI execution time
# def benchmark_edbo(pop_size, max_iter):
#     optimizer = EDBO_PI(lambda params: np.random.random(), dim=2, bounds=[(1e-6, 40), (1e-6, 40)],
#                         pop_size=pop_size, max_iter=max_iter)
#     start_time = time.time()
#     optimizer.optimize()
#     end_time = time.time()
#     return end_time - start_time

# # Run benchmarks for different configurations
# benchmark_results = []
# for pop_size, max_iter in test_configs:
#     exec_time = benchmark_edbo(pop_size, max_iter)
#     benchmark_results.append((pop_size, max_iter, exec_time))

# # Convert results to DataFrame and save
# benchmark_df = pd.DataFrame(benchmark_results, columns=["Population Size", "Max Iterations", "Execution Time (s)"])
# print(benchmark_df)

# # Save results to CSV
# benchmark_df.to_csv("edbo_benchmark_results.csv", index=False)
# print("Benchmark results saved to edbo_benchmark_results.csv")
