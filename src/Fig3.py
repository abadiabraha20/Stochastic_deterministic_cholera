"""
Fig3.py - Amplification vs basic reproduction number
Generates Figure 3: Median amplification of peak infections vs R0
for varying noise intensities.

Sweeps environmental transmission rate eta to vary R0 across a wide range.
Shows that amplification is largest near the epidemic threshold, and
decreases as R0 increases due to saturation of the environmental feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

if not os.path.exists('Fig'):
    os.makedirs('Fig')
    print("Created directory: Fig/")

np.random.seed(42)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

# ============================================================================
# MODEL PARAMETERS (calibrated)
# ============================================================================

params = {
    'Lambda': 1000.0, 'mu': 4e-5, 'd': 0.005, 'tau': 0.001,
    'beta0': 0.38, 'eta': 2.664e-4,
    'theta': 0.4, 'K': 1e6, 'rho': 0.3,
    'nu_V': 0.003, 'epsilon': 0.75, 'omega': 0.0005,
    'p': 0.65, 'alpha': 0.1, 'gamma_A': 0.14,
    'gamma_T': 0.3, 'gamma': 0.2, 'delta': 0.0003,
    'xi_A': 1e6, 'xi_I': 1e8, 'mu_B': 0.33, 'Vw': 1e6,
    'I0': 1,
}

# ============================================================================
# DETERMINISTIC MODEL
# ============================================================================

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

# ============================================================================
# BASIC REPRODUCTION NUMBER
# ============================================================================

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

# ============================================================================
# STOCHASTIC SIMULATION (CONSERVATIVE NOISE ON ENVIRONMENTAL PATHWAY)
# ============================================================================

def run_stochastic(params, y0, t, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = t[1] - t[0]
    n_steps = len(t)
    y = np.array(y0, dtype=float)
    I_traj = np.zeros(n_steps)
    I_traj[0] = y[3]
    dW = np.random.normal(0, np.sqrt(dt), n_steps - 1)

    for i in range(n_steps - 1):
        S, V, A, I, T, R, B = y
        N = max(S + V + A + I + T + R, 1)
        beta_t = params['beta0'] * (1 + params['rho'] * np.cos(2 * np.pi * t[i] / 365))
        Seff = S + (1 - params['epsilon']) * V
        direct_det = beta_t * (I + params['theta'] * A) / N
        env_det = params['eta'] * B / (params['K'] + B)
        env_noise = sigma * env_det * dW[i]

        lambda_direct_dt = direct_det * dt
        lambda_env_dt = env_det * dt + env_noise

        dS = (params['Lambda'] + params['omega'] * V + params['delta'] * R) * dt \
             - lambda_direct_dt * S - lambda_env_dt * S \
             - (params['mu'] + params['nu_V']) * S * dt
        dV = params['nu_V'] * S * dt \
             - (1 - params['epsilon']) * (lambda_direct_dt + lambda_env_dt) * V \
             - (params['mu'] + params['omega']) * V * dt
        dA = params['p'] * (lambda_direct_dt + lambda_env_dt) * Seff \
             - (params['mu'] + params['alpha'] + params['gamma_A']) * A * dt
        dI = (1 - params['p']) * (lambda_direct_dt + lambda_env_dt) * Seff \
             + params['alpha'] * A * dt \
             - (params['mu'] + params['d'] + params['gamma_T']) * I * dt
        dT = params['gamma_T'] * I * dt \
             - (params['mu'] + params['tau'] + params['gamma']) * T * dt
        dR = params['gamma'] * T * dt + params['gamma_A'] * A * dt \
             - (params['mu'] + params['delta']) * R * dt
        dB = (params['xi_A'] * A + params['xi_I'] * I) / params['Vw'] * dt \
             - params['mu_B'] * B * dt

        y = np.maximum(y + np.array([dS, dV, dA, dI, dT, dR, dB]), 1e-10)
        I_traj[i + 1] = y[3]

    extinct = I_traj[-1] < 1.0
    return extinct, I_traj

# ============================================================================
# MAIN
# ============================================================================

def main():
    t = np.linspace(0, 200, 4001)

    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / \
         (params['mu'] + params['omega'] + params['nu_V'])
    V0 = N0 * params['nu_V'] / \
         (params['mu'] + params['omega'] + params['nu_V'])
    y0 = [S0, V0, 0, params['I0'], 0, 0, 1000]

    # Sweep eta to vary R0 across a wide range
    eta_baseline = params['eta']
    eta_range = eta_baseline * np.array(
        [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7, 2.0, 2.3, 2.7, 3.1]
    )

    sigma_vals = [0.25, 0.50, 0.75]
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5), facecolor='white')

    print("Computing amplification vs R0 (sweeping eta)...")
    print(f"  eta range: {eta_range[0]:.2e} to {eta_range[-1]:.2e}")
    print(f"  sigma values: {sigma_vals}\n")

    results = {}

    for sigma, color, marker in zip(sigma_vals, colors, markers):
        R0_list = []
        amp_list = []

        for eta_val in eta_range:
            params_temp = params.copy()
            params_temp['eta'] = eta_val
            R0_temp = compute_R0(params_temp)

            # Deterministic peak
            sol = odeint(cholera_model, y0, t, args=(params_temp,))
            det_peak = np.max(sol[:, 3])

            # Stochastic ensemble
            peaks = []
            for sim in range(200):
                _, I_traj = run_stochastic(params_temp, y0, t, sigma,
                                            seed=10000 + sim)
                peaks.append(np.max(I_traj))
            peaks = np.array(peaks)

            amp = 100 * (np.median(peaks) - det_peak) / det_peak

            R0_list.append(R0_temp)
            amp_list.append(amp)

        # Sort by R0 for smooth plotting
        order = np.argsort(R0_list)
        R0_arr = np.array(R0_list)[order]
        amp_arr = np.array(amp_list)[order]

        results[sigma] = (R0_arr, amp_arr)

        # ---- Plot with markevery=2 ----
        ax.plot(R0_arr, amp_arr, marker + '-', color=color, lw=2.5,
                markersize=9, markeredgecolor='black', markeredgewidth=0.6,
                markevery=2,
                label=f'$\\sigma$ = {sigma}')

        print(f"  sigma = {sigma}: {len(R0_arr)} points")
        print(f"    R0 range: {R0_arr[0]:.2f} to {R0_arr[-1]:.2f}")
        print(f"    amplification range: {amp_arr.min():.1f}% to {amp_arr.max():.1f}%")

    # ------------------------------------------------------------------
    # Reference lines and markers
    # ------------------------------------------------------------------
    ax.axhline(0, color='gray', ls='--', lw=1.5,
               label='Deterministic (no change)')
    ax.axvline(1.0, color='gray', ls=':', lw=2,
               label=r'$\mathcal{R}_0 = 1$')

    # Baseline marker
    R0_baseline = 1.34
    ax.axvline(R0_baseline, color='purple', ls='--', lw=2.5,
               label=f'Baseline: $\\mathcal{{R}}_0$ = {R0_baseline}')

    # Diamond at baseline on sigma=0.50 curve
    R0_arr_50, amp_arr_50 = results[0.50]
    idx = np.argmin(np.abs(R0_arr_50 - R0_baseline))
    ax.scatter([R0_arr_50[idx]], [amp_arr_50[idx]],
               color='purple', s=250, marker='D',
               edgecolors='black', linewidths=1.2, zorder=10)

    # ------------------------------------------------------------------
    # Axes, labels, legend
    # ------------------------------------------------------------------
    ax.set_xlabel(r'Basic reproduction number $\mathcal{R}_0$', fontsize=14)
    ax.set_ylabel('Median amplification (%)', fontsize=14)
    ax.set_title('Stochastic amplification of cholera outbreaks',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(1.0, 1.9)
    ax.set_ylim(-5, 45)

    plt.tight_layout()
    plt.savefig('Fig/Fig3.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure 3 saved: Fig/Fig3.png")

if __name__ == "__main__":
    main()
