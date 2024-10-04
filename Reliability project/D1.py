import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
import pandas as pd
import matplotlib.pyplot as plt

alpha = [1e-5, 2.3e-5, 0.3e-5, 2.3e-5]
beta = [1.5, 1.5, 1.5, 1.5]
v = [1, 2, 3, 2]
w = [6, 6, 8, 7]
t = 10

def R_i(t, alpha_i, beta_i):
    return np.exp(-(t / alpha_i)**beta_i)

def R_sys(n, t, alpha, beta):
    R = [R_i(t, alpha[i], beta[i]) for i in range(4)]
    return np.prod([1 - (1 - R[i])**n[i] for i in range(4)])

def safe_log(x):
    return np.log(np.maximum(x, 1e-10))

def cost_constraint_safe(n):
    cost = sum(alpha[i] * ((-safe_log(R_i(t, alpha[i], beta[i])))**beta[i]) * (n[i] + np.exp(0.25 * n[i])) for i in range(4))
    return 400 - cost

def volume_constraint(n):
    volume = sum(v[i] * n[i]**2 for i in range(4))
    return 250 - volume

def weight_constraint(n):
    weight = sum(w[i] * n[i] * np.exp(0.25 * n[i]) for i in range(4))
    return 500 - weight

bounds = [(1, 10) for _ in range(4)]

n0 = [1, 1, 1, 1]

cons_safe = [{'type': 'ineq', 'fun': cost_constraint_safe},
             {'type': 'ineq', 'fun': volume_constraint},
             {'type': 'ineq', 'fun': weight_constraint}]

solution_safe = minimize(lambda n: -R_sys(n, t, alpha, beta), n0, method='SLSQP', bounds=bounds, constraints=cons_safe)

n_opt_safe = np.round(solution_safe.x).astype(int)

def VTTF_sys(n, alpha, beta):
    variances = [(alpha[i]**2 * gamma(1 + 2 / beta[i]) - (alpha[i] * gamma(1 + 1 / beta[i]))**2) * n[i] for i in range(4)]
    return sum(variances)

VTTF_sys_value = VTTF_sys(n_opt_safe, alpha, beta)

data_safe = {
    'ni': n_opt_safe,
    'MTTF': [alpha[i] * gamma(1 + 1 / beta[i]) for i in range(4)],
    'VTTF': [alpha[i]**2 * gamma(1 + 2 / beta[i]) - (alpha[i] * gamma(1 + 1 / beta[i]))**2 for i in range(4)],
    'VTTF_sys': [VTTF_sys_value] * 4
}

df_safe = pd.DataFrame(data_safe)

print("Optimization Results for Model D1:")
print(df_safe)

fig, ax = plt.subplots(figsize=(10, 6))
ax.table(cellText=df_safe.values, colLabels=df_safe.columns, loc='center')
ax.axis('off')

plt.show()
