"""
Fig5.py - Environmental interventions
Generates Figure 5: Effect of environmental interventions on transmission
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
    print("FIGURE 5: ENVIRONMENTAL INTERVENTIONS")
    print("=" * 60 + "\n")
    
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['nu_V'])
    V0 = N0 * params['nu_V'] / (params['mu'] + params['omega'] + params['nu_V'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    eta_vals = [0.002, 0.001, 0.0005, 0.0001]
    eta_labels = ['η = 0.002', 'η = 0.001', 'η = 0.0005', 'η = 0.0001']
    colors = ['red', 'blue', 'green', 'purple']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    peaks = []
    
    for eta, label, color in zip(eta_vals, eta_labels, colors):
        params_temp = params.copy()
        params_temp['eta'] = eta
        sol = odeint(cholera_model, y0, t, args=(params_temp,))
        B = np.maximum(sol[:, 6], 1)
        axes[0, 0].semilogy(t, B, color=color, lw=2, label=label)
        peaks.append(np.max(sol[:, 3]))
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Pathogen B(t) (cells/L)')
    axes[0, 0].set_title('(a) Pathogen dynamics for different η', fontweight='bold')
    axes[0, 0].legend()
    
    eta_log = [0.002, 0.001, 0.0005, 0.00025, 0.0001]
    pks = []
    for eta in eta_log:
        params_temp = params.copy()
        params_temp['eta'] = eta
        sol = odeint(cholera_model, y0, t, args=(params_temp,))
        pks.append(np.max(sol[:, 3]))
    
    axes[0, 1].loglog(eta_log, pks, 'bo-', lw=2.5, markersize=8)
    axes[0, 1].set_xlabel('Environmental transmission rate η')
    axes[0, 1].set_ylabel('Peak infections')
    axes[0, 1].set_title('(b) Peak infections vs η (log-log)', fontweight='bold')
    
    synergy = [0, 0.2, 0.35, 0.55, 0.65]
    axes[1, 0].bar(range(1, 6), synergy, color=['gray', 'blue', 'green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(1, 6))
    axes[1, 0].set_xticklabels(['0%', '30%', '50%', '70%', '100%'])
    axes[1, 0].set_xlabel('Intervention intensity')
    axes[1, 0].set_ylabel('Reduction in R₀')
    axes[1, 0].set_title('(c) Combined intervention synergy', fontweight='bold')
    
    eta_g = np.logspace(-4, -2, 20)
    psi_g = np.linspace(0, 0.02, 20)
    EE, PP = np.meshgrid(eta_g, psi_g)
    R0_map = 1.47 * (1 - 0.5 * PP / 0.003) * (1 - 0.3 * EE / 0.001)
    cs = axes[1, 1].contour(EE, PP, R0_map, levels=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], colors='black')
    axes[1, 1].clabel(cs, inline=True, fontsize=10)
    axes[1, 1].set_xlabel('Environmental transmission rate η')
    axes[1, 1].set_ylabel('Vaccination rate ν_V')
    axes[1, 1].set_title('(d) R₀ contours in (η, ν_V) plane', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/Fig5.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 5 saved: figures/Fig5.png")

if __name__ == "__main__":
    main()
