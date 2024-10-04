import numpy as np
import matplotlib.pyplot as plt

alpha_now = [1e-5, 2.3e-5, 3e-5, 4e-5]
beta_now = [1.5, 1.5, 1.5, 1.5]
alpha_new = [1.1e-5, 2.5e-5, 3.2e-5, 4.2e-5]  
beta_new = [1.6, 1.6, 1.6, 1.6] 


t = np.linspace(1, 10000, 500)

def R_i(t, alpha_i, beta_i):
    return np.exp(-alpha_i * t * (np.log(t))**beta_i)

def R_sys(t, alpha, beta):
    Ri = [R_i(t, alpha[i], beta[i]) for i in range(4)]
    return np.prod(Ri, axis=0)

def f_sys(t, alpha, beta):
    R = R_sys(t, alpha, beta)
    dRdt = -np.gradient(R, t)
    return dRdt / R

def h_sys(t, alpha, beta):
    R = R_sys(t, alpha, beta)
    f = f_sys(t, alpha, beta)
    return f / R

R_sys_now = R_sys(t, alpha_now, beta_now)
R_sys_new = R_sys(t, alpha_new, beta_new)
h_sys_now = h_sys(t, alpha_now, beta_now)
h_sys_new = h_sys(t, alpha_new, beta_new)
f_sys_now = f_sys(t, alpha_now, beta_now)
f_sys_new = f_sys(t, alpha_new, beta_new)

plt.figure(figsize=(18, 12))

plt.subplot(3, 1, 1)
plt.plot(t, R_sys_now, label='R_sys_now(t)', linestyle='-', color='blue')
plt.plot(t, R_sys_new, label='R_sys_new(t)', linestyle='--', color='red')
plt.title('System Reliability: Now vs. New')
plt.xlabel('Time (t)')
plt.ylabel('Reliability')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, h_sys_now, label='h_sys_now(t)', linestyle='-', color='blue')
plt.plot(t, h_sys_new, label='h_sys_new(t)', linestyle='--', color='red')
plt.title('Hazard Function: Now vs. New')
plt.xlabel('Time (t)')
plt.ylabel('Hazard Rate')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, f_sys_now, label='f_sys_now(t)', linestyle='-', color='blue')
plt.plot(t, f_sys_new, label='f_sys_new(t)', linestyle='--', color='red')
plt.title('Failure Rate Function: Now vs. New')
plt.xlabel('Time (t)')
plt.ylabel('Failure Rate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
