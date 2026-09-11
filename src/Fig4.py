"""
Fig4.py - Vaccination intervention strategies (with R0 calibration)
Recalibrates eta for each coverage level so that R0 = 1.34 throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import brentq
import os

if not os.path.exists('Fig'):
    os.makedirs('Fig')

np.random.seed(42)
plt.rcParams['font.size'] = 11

# Base parameters
params_base = {
    'Lambda': 1000.0, 'mu': 4e-5, 'd': 0.005, 'tau': 0.001,
    'beta0': 0.38, 'eta': 2.664e-4,
    'theta': 0.4, 'K': 1e6, 'rho': 0.3,
    'nu_V': 0.003, 'epsilon': 0.75, 'omega': 0.0005,
    'p': 0.65, 'alpha': 0.1, 'gamma_A': 0.14,
    'gamma_T': 0.3, 'gamma': 0.2, 'delta': 0.0003,
    'xi_A': 1e6, 'xi_I': 1e8, 'mu_B': 0.33, 'Vw': 1e6,
    'I0': 1,
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
    Vw = params['Vw']

    N = max(S + V + A + I + T + R, 1)
    beta_t = beta0 * (1 + rho * np.cos(2 * np.pi * t / 365))
    lambd = beta_t * (I + theta * A) / N + eta * B / (K + B)

    dS = Lambda + omega * V + delta * R - lambd * S - (mu + nu_V) * S
    dV = nu_V * S - (1 - epsilon) * lambd * V - (mu + omega) * V
    dA = p * lambd * (S + (1 - epsilon) * V) - (mu + alpha + gamma_A) * A
    dI = (1 - p) * lambd * (S + (1 - epsilon) * V) + alpha * A - (mu + d + gamma_T) * I
    dT = gamma_T * I - (mu + tau + gamma) * T
    dR = gamma * T + gamma_A * A - (mu + delta) * R
    dB = (xi_A * A + xi_I * I) / Vw - mu_B * B
    return [dS, dV, dA, dI, dT, dR, dB]

def compute_R0(params):
    mu = params['mu']; omega = params['omega']; nu_V = params['nu_V']
    Lambda = params['Lambda']; epsilon = params['epsilon']
    beta0 = params['beta0']; eta = params['eta']; K = params['K']
    xi_A = params['xi_A']; xi_I = params['xi_I']; mu_B = params['mu_B']
    Vw = params['Vw']; p = params['p']; alpha = params['alpha']
    gamma_A = params['gamma_A']; gamma_T = params['gamma_T']; d = params['d']
    theta = params['theta']

    N0 = Lambda / mu
    S0 = Lambda * (mu + omega) / (mu * (mu + omega + nu_V))
    V0 = Lambda * nu_V / (mu * (mu + omega + nu_V))
    Seff = S0 + (1 - epsilon) * V0
    exit_A = mu + alpha + gamma_A
    exit_I = mu + d + gamma_T

    R_H = (beta0 * Seff / (exit_I * N0)) * \
          (alpha * theta * p + (1 - p) * exit_A) / exit_A
    R_E = (eta * Seff / (K * exit_I * mu_B * Vw)) * \
          (xi_A * p * exit_I + xi_I * (alpha * p + (1 - p) * exit_A)) / exit_A
    return 0.5 * (R_H + np.sqrt(R_H**2 + 4 * R_E))

def calibrate_eta(params, target_R0=1.34):
    def obj(eta):
        p = params.copy()
        p['eta'] = eta
        return compute_R0(p) - target_R0
    return brentq(obj, 1e-8, 1e-2, xtol=1e-12)

def main():
    print("=" * 60)
    print("FIGURE 4: VACCINATION WITH R0 CALIBRATION")
    print("=" * 60)

    t = np.linspace(0, 200, 2001)
    N0 = params_base['Lambda'] / params_base['mu']

    coverage_levels = [0.20, 0.40, 0.60, 0.70]
    labels = [f'{int(c*100)}% coverage' for c in coverage_levels]
    colors = ['green', 'blue', 'orange', 'red']

    mu_plus_omega = params_base['mu'] + params_base['omega']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    peaks, finals, durations = [], [], []

    print("\nCoverage | nu_V      | calibrated eta | R0  | Peak")
    print("-" * 60)

    for cov, label, color in zip(coverage_levels, labels, colors):
        nu_V = cov * mu_plus_omega / (1 - cov)
        params_temp = params_base.copy()
        params_temp['nu_V'] = nu_V
        eta_cal = calibrate_eta(params_temp, target_R0=1.34)
        params_temp['eta'] = eta_cal

        R0_check = compute_R0(params_temp)

        S0 = N0 * (params_temp['mu'] + params_temp['omega']) / \
             (params_temp['mu'] + params_temp['omega'] + nu_V)
        V0 = N0 * nu_V / \
             (params_temp['mu'] + params_temp['omega'] + nu_V)
        y0 = [S0, V0, 0, params_base['I0'], 0, 0, 1000]

        sol = odeint(cholera_model, y0, t, args=(params_temp,))
        I = sol[:, 3]
        peak = np.max(I)
        peaks.append(peak)

        axes[0, 0].plot(t, I, color=color, lw=2, label=label)

        finals.append(np.trapezoid(I, t))
        threshold = 0.05 * peak
        above = np.where(I > threshold)[0]
        durations.append((above[-1] - above[0]) * (t[1] - t[0]) if len(above) > 1 else 0)

        print(f"  {int(cov*100):3d}%   | {nu_V:.6f} | {eta_cal:.4e}  | {R0_check:.3f} | {peak:.0f}")

    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious individuals')
    axes[0, 0].set_title('(a) Outbreak trajectories', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 200)

    cov_arr = [int(c*100) for c in coverage_levels]
    axes[0, 1].plot(cov_arr, peaks, 'bo-', lw=2.5, markersize=8, markeredgecolor='black')
    axes[0, 1].set_xlabel('Vaccination coverage (%)')
    axes[0, 1].set_ylabel('Peak infection size')
    axes[0, 1].set_title('(b) Peak infection vs coverage', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(cov_arr, durations, 'go-', lw=2.5, markersize=8, markeredgecolor='black')
    axes[1, 0].set_xlabel('Vaccination coverage (%)')
    axes[1, 0].set_ylabel('Outbreak duration (days)')
    axes[1, 0].set_title('(c) Outbreak duration vs coverage', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].bar(cov_arr, [f/1e6 for f in finals], color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Vaccination coverage (%)')
    axes[1, 1].set_ylabel('Total cases (millions)')
    axes[1, 1].set_title('(d) Total epidemic size vs coverage', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig/Fig4.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure 4 saved: Fig/Fig4.png")

    if len(peaks) >= 2:
        red = 100 * (peaks[-2] - peaks[-1]) / peaks[-2]
        print(f"Reduction from 60% to 70%: {red:.1f}%")

if __name__ == "__main__":
    main()
