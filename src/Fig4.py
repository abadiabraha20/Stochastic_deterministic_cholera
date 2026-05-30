"""
Fig4.py - Vaccination intervention strategies
Generates Figure 4: Impact of vaccination coverage on outbreak dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

np.random.seed(42)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

# Model parameters
params = {
    'Lambda': 1000, 'mu': 4e-5, 'beta0': 0.25, 'eta': 0.001,
    'theta': 0.4, 'K': 1e6, 'rho': 0.3, 'nu_V': 0.003,
    'epsilon': 0.75, 'omega': 0.0005, 'p': 0.65, 'alpha': 0.1,
    'gamma_A': 0.14, 'gamma_T': 0.3, 'gamma': 0.2, 'd': 0.005,
    'tau': 0.001, 'delta': 0.0003, 'xi_A': 1e6, 'xi_I': 1e8, 'mu_B': 0.33,
}

def cholera_model(y, t, params):
    S, V, A, I, T, R, B = y
    Lambda = params['Lambda']; mu = params['mu']; beta0 = params['beta0']
    eta = params['eta']; theta = params['theta']; K = params['K']
    rho = params['rho']; nu_V = params['nu_V']; epsilon = params['epsilon']
    omega = params['omega']; p = params['p']; alpha = params['alpha']
    gamma_A = params['gamma_A']; gamma_T = params['gamma_T']; gamma = params['gamma']
    d = params['d']; tau = params['tau']; delta = params['delta']
    xi_A = params['xi_A']; xi_I = params['xi_I']; mu_B = params['mu_B']
    
    N = max(S + V + A + I + T + R, 1)
    beta_t = beta0 * (1 + rho * np.cos(2 * np.pi * t / 365))
    lambd = beta_t * (I + theta * A) / N + eta * B / (K + B)
    
    dS = Lambda + omega * V + delta * R - lambd * S - (mu + nu_V) * S
    dV = nu_V * S - (1 - epsilon) * lambd * V - (mu + omega) * V
    dA = p * lambd * (S + (1 - epsilon) * V) - (mu + alpha + gamma_A) * A
    dI = (1 - p) * lambd * (S + (1 - epsilon) * V) + alpha * A - (mu + d + gamma_T) * I
    dT = gamma_T * I - (mu + tau + gamma) * T
    dR = gamma * T + gamma_A * A - (mu + delta) * R
    dB = xi_A * A + xi_I * I - mu_B * B
    return [dS, dV, dA, dI, dT, dR, dB]

def main():
    print("\n" + "=" * 60)
    print("FIGURE 4: VACCINATION INTERVENTION")
    print("=" * 60 + "\n")
    
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    
    # Corrected nu_V values for 20%, 40%, 60%, 70% coverage
    nu_V_values = [0.000135, 0.000360, 0.000810, 0.001260]
    labels = ['20% coverage', '40% coverage', '60% coverage', '70% coverage']
    colors = ['green', 'blue', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    peaks, finals, durations = [], [], []
    
    for nu_V, label, color in zip(nu_V_values, labels, colors):
        params_temp = params.copy()
        params_temp['nu_V'] = nu_V
        
        S0 = N0 * (params_temp['mu'] + params_temp['omega']) / (params_temp['mu'] + params_temp['omega'] + nu_V)
        V0 = N0 * nu_V / (params_temp['mu'] + params_temp['omega'] + nu_V)
        y0 = [S0, V0, 0, 10, 0, 0, 1000]
        
        sol = odeint(cholera_model, y0, t, args=(params_temp,))
        I = sol[:, 3]
        axes[0, 0].plot(t, I, color=color, lw=2, label=label)
        peaks.append(np.max(I))
        finals.append(np.trapezoid(I, t))
        
        threshold = 0.05 * np.max(I)
        above = np.where(I > threshold)[0]
        if len(above) > 1:
            durations.append((above[-1] - above[0]) * 200 / len(t))
        else:
            durations.append(0)
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious individuals')
    axes[0, 0].set_title('(a) Outbreak trajectories', fontweight='bold')
    axes[0, 0].legend()
    
    cov = [20, 40, 60, 70]
    axes[0, 1].plot(cov, peaks, 'bo-', lw=2.5, markersize=8)
    axes[0, 1].set_xlabel('Vaccination coverage (%)')
    axes[0, 1].set_ylabel('Peak infection size')
    axes[0, 1].set_title('(b) Peak infection vs coverage', fontweight='bold')
    
    axes[1, 0].plot(cov, durations, 'go-', lw=2.5, markersize=8)
    axes[1, 0].set_xlabel('Vaccination coverage (%)')
    axes[1, 0].set_ylabel('Outbreak duration (days)')
    axes[1, 0].set_title('(c) Outbreak duration vs coverage', fontweight='bold')
    
    axes[1, 1].bar(cov, [f/1e6 for f in finals], color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Vaccination coverage (%)')
    axes[1, 1].set_ylabel('Total cases (millions)')
    axes[1, 1].set_title('(d) Final epidemic size vs coverage', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/Fig4.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 4 saved: figures/Fig4.png")
    print(f"   Peak reduction: {(1 - peaks[-1]/peaks[0])*100:.1f}%")

if __name__ == "__main__":
    main()
