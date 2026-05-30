"""
Fig7.py - Seasonal forcing dynamics
Generates Figure 7: Influence of seasonal forcing on outbreak patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

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
    print("FIGURE 7: SEASONAL FORCING")
    print("=" * 60 + "\n")
    
    t = np.linspace(0, 730, 3000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['nu_V'])
    V0 = N0 * params['nu_V'] / (params['mu'] + params['omega'] + params['nu_V'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # Panel (a)
    rho_vals = [0, 0.15, 0.3, 0.5]
    colors = ['black', 'blue', 'red', 'green']
    for rho, col in zip(rho_vals, colors):
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        axes[0, 0].plot(t, sol[:, 3], color=col, lw=2, label=f'ρ = {rho}')
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious individuals')
    axes[0, 0].set_title('(a) Infectious dynamics for different seasonality', fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    
    # Panel (b)
    p = params.copy()
    p['rho'] = 0.3
    sol = odeint(cholera_model, y0, t, args=(p,))
    f, Pxx = signal.periodogram(sol[:, 3], fs=1 / 0.243, window='hann', nfft=4096)
    axes[0, 1].semilogy(f, Pxx, 'b-', lw=2)
    axes[0, 1].axvline(1 / 365, color='red', ls='--', lw=2, label='1 year⁻¹')
    axes[0, 1].set_xlabel('Frequency (day⁻¹)')
    axes[0, 1].set_ylabel('Power spectral density')
    axes[0, 1].set_title('(b) Power spectral density for ρ = 0.3', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 0.015)
    
    # Panel (c)
    rho_scan = np.linspace(0, 0.6, 20)
    periods = []
    for rho in rho_scan:
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        I = sol[1000:, 3]
        peaks_idx = signal.find_peaks(I, height=np.max(I) * 0.3)[0]
        if len(peaks_idx) > 1:
            period = np.mean(np.diff(peaks_idx)) * 730 / len(t)
            periods.append(period)
        else:
            periods.append(365)
    
    axes[1, 0].plot(rho_scan, periods, 'ro-', lw=2.5, markersize=6)
    axes[1, 0].axhline(365, color='black', ls='--', alpha=0.5)
    axes[1, 0].set_xlabel('Seasonality amplitude ρ')
    axes[1, 0].set_ylabel('Outbreak period (days)')
    axes[1, 0].set_title('(c) Outbreak period vs seasonality', fontweight='bold')
    
    # Panel (d)
    peaks_incidence = []
    for rho in rho_scan:
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        peaks_incidence.append(np.max(sol[:, 3]))
    
    axes[1, 1].plot(rho_scan, peaks_incidence, 'go-', lw=2.5, markersize=6)
    axes[1, 1].set_xlabel('Seasonality amplitude ρ')
    axes[1, 1].set_ylabel('Peak incidence')
    axes[1, 1].set_title('(d) Peak incidence vs seasonality', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/Fig7.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 7 saved: figures/Fig7.png")

if __name__ == "__main__":
    main()
