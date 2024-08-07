import control as ctrl
import numpy as np
import sympy as sp

# Define the continuous-time system matrices
A = np.array([[0, 1],
              [-1, -2]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Create the continuous-time state-space model
sys_continuous = ctrl.StateSpace(A, B, C, D)

# Sampling time
Ts = 0.1  # e.g., 0.1 seconds

# Discretize the system using Zero-Order Hold
sys_discrete = ctrl.sample_system(sys_continuous, Ts, method='zoh')

# Print the discrete-time system
print("Discrete-time state-space model:")
print(sys_discrete)

