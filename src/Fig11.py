"""
Fig11.py - Validation against historical outbreaks
Generates Figure 11: Model validation against Haiti 2010-2011 and Yemen 2016-2017 outbreak data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

# Base parameters
base_params = {
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
    print("FIGURE 11: VALIDATION")
    print("=" * 60 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    # ========================================================================
    # HAITI
    # ========================================================================
    t_haiti = np.linspace(0, 200, 2000)
    
    params_haiti = base_params.copy()
    params_haiti['beta0'] = 0.32
    params_haiti['rho'] = 0.20
    params_haiti['nu_V'] = 0.001
    
    N0 = params_haiti['Lambda'] / params_haiti['mu']
    S0 = N0 * (params_haiti['mu'] + params_haiti['omega']) / (params_haiti['mu'] + params_haiti['omega'] + params_haiti['nu_V'])
    V0 = N0 * params_haiti['nu_V'] / (params_haiti['mu'] + params_haiti['omega'] + params_haiti['nu_V'])
    y0_haiti = [S0, V0, 50, 100, 0, 0, 1e7]
    
    sol_haiti = odeint(cholera_model, y0_haiti, t_haiti, args=(params_haiti,))
    I_haiti_model = sol_haiti[:, 3]
    
    # Haiti data
    t_haiti_data = np.array([7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98,
                             105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175])
    cases_haiti = np.array([0.8, 2.1, 3.5, 5.2, 7.8, 9.5, 10.2, 9.8, 8.9, 7.6,
                            6.4, 5.3, 4.2, 3.5, 2.9, 2.4, 2.0, 1.7, 1.4, 1.2,
                            1.0, 0.8, 0.6, 0.5, 0.4]) * 1000
    
    scale_haiti = np.max(cases_haiti) / np.max(I_haiti_model)
    I_haiti_model_scaled = I_haiti_model * scale_haiti
    
    axes[0].plot(t_haiti, I_haiti_model_scaled / 1000, 'b-', lw=2.5, label='Model simulation')
    axes[0].scatter(t_haiti_data, cases_haiti / 1000, color='red', s=70, marker='o',
                    edgecolors='black', zorder=5, label='Reported cases (Haiti)')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Symptomatic cases (thousands)')
    axes[0].set_title('(a) Haiti 2010-2011', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, 200)
    axes[0].set_ylim(0, 12)
    
    # Correlation for Haiti
    f_haiti = interp1d(t_haiti, I_haiti_model_scaled, kind='cubic', fill_value='extrapolate')
    model_at_haiti = f_haiti(t_haiti_data) / 1000
    corr_haiti = np.corrcoef(model_at_haiti, cases_haiti / 1000)[0, 1]
    axes[0].text(10, 10.5, f'Correlation: r = {corr_haiti:.2f}', fontsize=12,
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # ========================================================================
    # YEMEN
    # ========================================================================
    t_yemen = np.linspace(0, 420, 4000)
    
    params_yemen = base_params.copy()
    params_yemen['beta0'] = 0.28
    params_yemen['eta'] = 0.0015
    params_yemen['rho'] = 0.35
    params_yemen['nu_V'] = 0.0005
    
    N0 = params_yemen['Lambda'] / params_yemen['mu']
    S0 = N0 * (params_yemen['mu'] + params_yemen['omega']) / (params_yemen['mu'] + params_yemen['omega'] + params_yemen['nu_V'])
    V0 = N0 * params_yemen['nu_V'] / (params_yemen['mu'] + params_yemen['omega'] + params_yemen['nu_V'])
    y0_yemen = [S0, V0, 100, 200, 0, 0, 5e7]
    
    sol_yemen = odeint(cholera_model, y0_yemen, t_yemen, args=(params_yemen,))
    I_yemen_model = sol_yemen[:, 3]
    
    # Yemen data
    t_yemen_data = np.array([14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105,
                             112, 119, 126, 133, 140, 147, 154,
