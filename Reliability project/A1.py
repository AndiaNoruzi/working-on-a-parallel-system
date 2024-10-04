import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# Constants for model A1
alpha = [1e-5, 2.3e-5, 3e-5, 4e-5]
beta = [1.5, 1.5, 1.5, 1.5]
v = [1, 2, 3, 4]
w = [6, 6, 8, 8]
t = 10  # Assuming a fixed value for t

# Reliability function
def R_i(t, alpha_i, beta_i):
    return np.exp(-alpha_i * t * (np.log(t))**beta_i)

# System reliability function
def R_sys(n, t, alpha, beta):
    R = [R_i(t, alpha[i], beta[i]) for i in range(4)]
    return np.prod([1 - (1 - R[i])**n[i] for i in range(4)])

# Objective function to maximize
def objective(n, t, alpha, beta):
    return -R_sys(n, t, alpha, beta)

# Constraints
def constraint1(n):
    return 400 - np.sum([alpha[i] * t * (n[i] + np.exp(0.25 * n[i])) * (np.log(t))**beta[i] for i in range(4)])

def constraint2(n):
    return 250 - np.sum([v[i] * n[i]**2 for i in range(4)])

def constraint3(n):
    return 500 - np.sum([w[i] * n[i] * np.exp(-0.25 * n[i]) for i in range(4)])

# Bounds for n_i
bounds = [(1, 10) for _ in range(4)]

# Initial guess
n0 = [1, 1, 1, 1]

# Constraints in dictionary form
cons = [{'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}]

# Solving the optimization problem
solution = minimize(objective, n0, args=(t, alpha, beta), method='SLSQP', bounds=bounds, constraints=cons)

# Extract results and round to nearest integers
n_opt = np.round(solution.x).astype(int)

# Check and adjust if the rounded solution violates any constraints
def is_feasible(n):
    return (constraint1(n) >= 0 and constraint2(n) >= 0 and constraint3(n) >= 0)

if not is_feasible(n_opt):
    n_opt = solution.x.astype(int)
    n_opt = np.clip(n_opt, 1, 10)
    while not is_feasible(n_opt):
        for i in range(4):
            if constraint1(n_opt) < 0:
                n_opt[i] = max(n_opt[i] - 1, 1)
            elif constraint2(n_opt) < 0:
                n_opt[i] = max(n_opt[i] - 1, 1)
            elif constraint3(n_opt) < 0:
                n_opt[i] = max(n_opt[i] - 1, 1)

R_opt = R_sys(n_opt, t, alpha, beta)

# Prepare the results table
data = {
    'ni': n_opt,
    'R(t)': [R_i(t, alpha[i], beta[i]) for i in range(4)]
}

df = pd.DataFrame(data)
df['Rsys(t)'] = R_opt

# Display results
print("Optimization Results for Model A1 with Integer Constraints:")
print(df)

# Plotting the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
ax.axis('off')

plt.show()
