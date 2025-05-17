import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.signal import lti, step
import pandas as pd
from functools import singledispatch
from itertools import product
 
def time_constant():
    return  np.linspace(0, 6, 1600)  

def calculate_rise_time(t, y, steady_state_value):
    """Calculates rise time (time taken from 10% to 90% of steady-state value)."""
    y_10 = 0.1 * steady_state_value
    y_90 = 0.9 * steady_state_value
    idx_10 = np.where(y >= y_10)[0][0]  # First index where response exceeds 10%
    idx_90 = np.where(y >= y_90)[0][0]  # First index where response exceeds 90%
    return t[idx_90] - t[idx_10]
def system_SEDM():
    G1 = lti([1], [0.005, 0.5])  # 1 / (0.005s + 0.5)
    G2 = lti([1], [1.3, 0.01])  # 1 / (1.3s + 0.01)




    # Open-loop transfer function G_OL = G1 * G2
    G_OL_num = np.polymul(G1.num, G2.num)
    G_OL_den = np.polymul(G1.den, G2.den)
    # Closed-loop SEDM system (negative feedback)
    G_SEDM_num = G_OL_num
    G_SEDM_den = np.polyadd(G_OL_den, G_OL_num)
    return lti(G_SEDM_num, G_SEDM_den)
def system_SEDM_CL(Kp,Ki):
    C = lti([Kp, Ki], [1, 0])
    G_CL_num = np.polymul(C.num, system_SEDM().num)
    G_CL_den = np.polymul(C.den, system_SEDM().den)
  
    # Closed-loop PI-controlled system with feedback
    G_SEDM_CL_num = G_CL_num
    G_SEDM_CL_den = np.polyadd(G_CL_den, G_CL_num)
    return lti(G_SEDM_CL_num, G_SEDM_CL_den)


def simulate_response(Kp, Ki):
    """Computes the step response for the PI-controlled system."""
    system = system_SEDM_CL(Kp, Ki)
    
    t_ref = time_constant()  # Ensure time matches the step response
    t, y = step(system, T=t_ref)  # Ensure t matches t_ref

    # Check if step() returned fewer time points than expected
    if len(t) != len(t_ref):
        y = np.interp(t_ref, t, y)  # Interpolate to match time constant
        t = t_ref  # Now they are the same length

    e = 1 - y   # Compute error

    return t, e, y
def ITAE(params):
    """Computes Integral of Time-weighted Absolute Error (ITAE)."""
    Kp, Ki = params
    t,e,y = simulate_response(Kp, Ki)

    return simpson(t * np.abs(e), t)  # Integration using Simpson's rule
def peakTime(params):
    Kp, Ki = params
    t,e,y = simulate_response(Kp, Ki)
    peak_index = np.argmax(y)  
    return t[peak_index]   
@singledispatch
def get_performance_metrics(arg):
    raise TypeError(f"Unsupported type: {type(arg)}")

@get_performance_metrics.register
def _(lti_system: lti):
    """Compute performance metrics from an LTI system."""
    t_ref = time_constant()
    t, y = step(lti_system, T=t_ref)
    e = 1 - y 
  

    plt.figure(figsize=(8, 4))
    plt.plot(t, y*220, label="System Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Output Voltage (V)")
    plt.title("")
    plt.grid(True)
    plt.legend()
    plt.show()

    ITAE = simpson(t * np.abs(e), t)
    y = y * 220
    steady_state_value = 220

    settling_indices = np.where(np.abs(y - steady_state_value) > 0.02 * steady_state_value)[0]
    settling_time = t[settling_indices[-1]] if len(settling_indices) > 0 else 0

    steady_state_error = np.abs(y[-1] - steady_state_value) / steady_state_value * 100
    overshoot = ((np.max(y) - steady_state_value) / steady_state_value) * 100 if np.max(y) > steady_state_value else 0

    results = pd.DataFrame({
        "Metric": ["ITAE", "Settling Time (s)", "Steady-State Error (%)", "Percent Overshoot (%)"],
        "Value": [ITAE, settling_time, steady_state_error, overshoot]
    }) 
    print(results.to_string(index=False))
def _(params: tuple):
    """Handles (Kp, Ki) tuple by generating an LTI system and computing metrics."""
    if len(params) != 2:
        raise ValueError("Expected a tuple of (Kp, Ki)")
    
    Kp, Ki = params
    return get_performance_metrics(system_SEDM_CL(Kp, Ki))  # Ensure `system_SEDM_CL` is correctly defined
@get_performance_metrics.register
def _(Kp_Ki: tuple):
    """Compute performance metrics from Kp and Ki values."""
    Kp, Ki = Kp_Ki
    t, e, y = simulate_response(Kp, Ki)
    ITAE = simpson(t * np.abs(e), t)
    steady_state_value = 220
    y = y * 220
    settling_indices = np.where(np.abs(y - steady_state_value) > 0.02 * steady_state_value)[0]
    settling_time = t[settling_indices[-1]] if len(settling_indices) > 0 else 0

    steady_state_error = np.abs(y[-1] - steady_state_value) / steady_state_value * 100
    overshoot = ((np.max(y) - steady_state_value) / steady_state_value) * 100 if np.max(y) > steady_state_value else 0

    return ITAE, overshoot, settling_time, steady_state_error  


import numpy as np

class BoundGenerator:
    def __init__(self, bounds, step_size):
        self.bounds = bounds
        self.step_size = step_size

        # Define range attributes
        self.kp_range = (self.bounds[0][0], self.bounds[0][1])  
        self.ki_range = (self.bounds[1][0], self.bounds[1][1])  

        # Generate values
        self.kp_values = self._generate_kp_values()
        self.ki_values = self._generate_ki_values()
        self.bound_combinations = self._generate_combinations()
        self.bound_combinations_ranges = self._generate_combinations_ranges()


    def _generate_kp_values(self):
        """Generate Kp values from min to max using arange instead of linspace to control step limits."""
        return np.arange(self.kp_range[0], self.kp_range[1] + self.step_size / 2, self.step_size)

    def _generate_ki_values(self):
        """Generate Ki values from min to max using arange instead of linspace to control step limits."""
        return np.arange(self.ki_range[0], self.ki_range[1] + self.step_size / 2, self.step_size)

    def _generate_combinations(self):
        """Generate all (Kp, Ki) combinations from normal ranges using itertools.product."""
        return list(product(self.kp_values, self.ki_values))



    def _generate_combinations_ranges(self):
        step_size = self.step_size
        Kp = self.bounds[1][0]  +step_size# Initial Kp
        Ki = self.bounds[0][0]  +step_size# Initial Ki
        Kp_upper = self.bounds[1][1]  # Max Kp
        Ki_upper = self.bounds[0][1]  # Max Ki
        kp_ki_pairs = []  # List to store (Kp, Ki) pairs

        while Ki <= Ki_upper:  # Loop until Ki exceeds upper bound
            while Kp <= Kp_upper:  # Loop until Kp exceeds upper bound
                if not (Kp == self.bounds[1][0] and Ki == self.bounds[0][0]):  
                    if not (Ki == Kp and Ki ==self.bounds[0][1]  and  Kp == self.bounds[1][1]):  #  
                        kp_ki_pairs.append((self.bounds[1][0], Kp, self.bounds[0][0], Ki))
                Kp += step_size  # Increment Kp
            
            # Reset Kp and increment Ki
            Kp = self.bounds[1][0] + step_size
            Ki += step_size
        
        # Print outside the loop to see results properly
        print("Generated pairs:", kp_ki_pairs)
        
        return kp_ki_pairs

