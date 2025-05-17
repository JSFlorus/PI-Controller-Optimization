import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.signal import lti, step , TransferFunction

import pandas as pd
import control_analysis as ca
def run():
    Kp = 1
    Ki = 10
    
    # Define system components
    G1 = lti([1], [0.005, 0.5])  # 1 / (0.005s + 0.5)
    G2 = lti([1], [1.3, 0.01])  # 1 / (1.3s + 0.01)
    # PI Controller Transfer Function: (Kp * s + Ki) / s
    C = lti([Kp, Ki], [1, 0])
    # Open-loop transfer function G_OL = G1 * G2
    G_OL_num = np.polymul(G1.num, G2.num)
    G_OL_den = np.polymul(G1.den, G2.den)
    G_OL = lti(G_OL_num, G_OL_den)
    # Closed-loop SEDM system (negative feedback)
    G_SEDM_num = G_OL_num
    G_SEDM_den = np.polyadd(G_OL_den, G_OL_num)
    #SEDMN ITAE Characteristics
    print("-"*5+"Closed-loop SEDM system (negative feedback)"+"-"*5)
    G_SEDM = lti(G_SEDM_num, G_SEDM_den)
    ca.get_performance_metrics(G_SEDM)



    # PI-controlled system: G_CL = C * G_SEDM
    G_CL_num = np.polymul(C.num, G_SEDM.num)
    G_CL_den = np.polymul(C.den, G_SEDM.den)
    G_CL = lti(G_CL_num, G_CL_den)

    # Closed-loop PI-controlled system with feedback
    G_SEDM_CL_num = G_CL_num
    G_SEDM_CL_den = np.polyadd(G_CL_den, G_CL_num)
    print("-"*5+"Closed-loop PI-controlled system with feedback"+"-"*5)
    G_SEDM_CL = lti(G_SEDM_CL_num, G_SEDM_CL_den)
    ca.get_performance_metrics(G_SEDM_CL)


    
    prefilter_num = [79]
    prefilter_den = [1.0, 79, 79]

    G_prefilter = TransferFunction(prefilter_num, prefilter_den)
    new_num = np.polymul(G_SEDM.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM.den, G_prefilter.den)
    G_SEDM_Prefilter = TransferFunction(new_num, new_den)
    new_num = np.polymul(G_SEDM_CL.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM_CL.den, G_prefilter.den)
    G_SEDM_CL_Prefilter = TransferFunction(new_num, new_den)
    print("-"*5+"Pre-filtered Closed-loop SEDM system (negative feedback)"+"-"*5)
    ca.get_performance_metrics(G_SEDM_Prefilter)
    print("-"*5+"Pre-filtered Closed-loop PI-controlled system with feedback"+"-"*5)
    ca.get_performance_metrics(G_SEDM_CL_Prefilter)

    # Enter ITAE value: 0.002
    # Enter Settling Time: 0.05
    # Enter Steady State Error: 0.0002
    # Predicted Kp: 21.6706
    # Predicted Ki: 36.5979
    # Actual ITAE Ki: 0.001923570865736093

    Kp = 21.6706
    Ki = 36.5979
    C = lti([Kp, Ki], [1, 0])
    # PI-controlled system: G_CL = C * G_SEDM
    G_CL_num = np.polymul(C.num, G_SEDM.num)
    G_CL_den = np.polymul(C.den, G_SEDM.den)
    G_CL = lti(G_CL_num, G_CL_den)

    # Closed-loop PI-controlled system with feedback
    G_SEDM_CL_num = G_CL_num
    G_SEDM_CL_den = np.polyadd(G_CL_den, G_CL_num)
    print("-"*5+"EDBO Closed-loop PI-controlled system with feedback"+"-"*5)
    G_SEDM_CL = lti(G_SEDM_CL_num, G_SEDM_CL_den)
    ca.get_performance_metrics(G_SEDM_CL)
 

    G_prefilter = TransferFunction(prefilter_num, prefilter_den)
    new_num = np.polymul(G_SEDM.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM.den, G_prefilter.den)
    G_SEDM_Prefilter = TransferFunction(new_num, new_den)
    new_num = np.polymul(G_SEDM_CL.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM_CL.den, G_prefilter.den)
    G_SEDM_CL_Prefilter = TransferFunction(new_num, new_den)
 
    print("-"*5+"EDBO Pre-filtered Closed-loop PI-controlled system with feedback"+"-"*5)
    ca.get_performance_metrics(G_SEDM_CL_Prefilter)

    # Enter ITAE value: 0.002
    # Enter Settling Time: 0.05
    # Enter Steady State Error: 0.0002
    # Predicted Kp: 27.3293
    # Predicted Ki: 47.8029
    # Actual ITAE0: 0.001887112295657378
    
    Kp = 27.3293
    Ki = 47.8029
    C = lti([Kp, Ki], [1, 0])
    # PI-controlled system: G_CL = C * G_SEDM
    G_CL_num = np.polymul(C.num, G_SEDM.num)
    G_CL_den = np.polymul(C.den, G_SEDM.den)
    G_CL = lti(G_CL_num, G_CL_den)

    # Closed-loop PI-controlled system with feedback
    G_SEDM_CL_num = G_CL_num
    G_SEDM_CL_den = np.polyadd(G_CL_den, G_CL_num)
    print("-"*5+"Grid Search Closed-loop PI-controlled system with feedback"+"-"*5)
    G_SEDM_CL = lti(G_SEDM_CL_num, G_SEDM_CL_den)
    ca.get_performance_metrics(G_SEDM_CL)
 

    G_prefilter = TransferFunction(prefilter_num, prefilter_den)
    new_num = np.polymul(G_SEDM.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM.den, G_prefilter.den)
    G_SEDM_Prefilter = TransferFunction(new_num, new_den)
    new_num = np.polymul(G_SEDM_CL.num, G_prefilter.num)
    new_den = np.polymul(G_SEDM_CL.den, G_prefilter.den)
    G_SEDM_CL_Prefilter = TransferFunction(new_num, new_den)
 
    print("-"*5+"Grid Search Pre-filtered Closed-loop PI-controlled system with feedback"+"-"*5)
    ca.get_performance_metrics(G_SEDM_CL_Prefilter)




 
run()

 
 