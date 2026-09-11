"""
Fig2.py - Deterministic vs stochastic comparison
Generates Figure 2: Comparison of deterministic and stochastic model predictions
"""

PANELS:
(a) Infectious individuals over time (deterministic + stochastic trajectories)
(b) Environmental pathogen B(t) on semi-log scale
(c) Distribution of peak infections
(d) Amplification vs noise intensity (supports the paper's main finding)
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
# MODEL PARAMETERS
# ============================================================================

params = {
    'Lambda': 1000.0, 'mu': 4e-5, 'd': 0.005, 'tau': 0.001,
    'beta0': 0.38,
    'eta': 2.664e-4,
    'theta': 0.4, 'K': 1e6, 'rho': 0.3,
    'nu_V': 0.003, 'epsilon': 0.75, 'omega': 0.0005,
    'p': 0.65, 'alpha': 0.1, 'gamma_A': 0.14,
    'gamma_T': 0.3, 'gamma': 0.2, 'delta': 0.0003,
    'xi_A': 1e6, 'xi_I': 1e8, 'mu_B': 0.33, 'Vw': 1e6,
    'sigma': 0.50,
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
# STOCHASTIC SIMULATION: CONSERVATIVE NOISE ON ENVIRONMENTAL PATHWAY
# ============================================================================

def run_stochastic(params, y0, t, sigma, seed=None):
    """Euler-Maruyama with conservative noise on environmental transmission."""
    if seed is not None:
        np.random.seed(seed)

    dt = t[1] - t[0]
    n_steps = len(t)
    y = np.array(y0, dtype=float)
    I_traj = np.zeros(n_steps)
    B_traj = np.zeros(n_steps)
    I_traj[0] = y[3]
    B_traj[0] = y[6]

    dW = np.random.normal(0, np.sqrt(dt), n_steps - 1)

    for i in range(n_steps - 1):
        S, V, A, I, T, R, B = y
        N = max(S + V + A + I + T + R, 1)
        t_i = t[i]

        beta_t = params['beta0'] * (1 + params['rho'] * np.cos(2 * np.pi * t_i / 365))
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
        B_traj[i + 1] = y[6]

    extinct = I_traj[-1] < 1.0
    return extinct, I_traj, B_traj

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

    # Deterministic
    sol = odeint(cholera_model, y0, t, args=(params,))
    I_det = sol[:, 3]
    B_det = sol[:, 6]
    DET_PEAK = np.max(I_det)
    DET_PEAK_TIME = t[np.argmax(I_det)]

    print(f"Deterministic peak: {DET_PEAK:.0f} cases at day {DET_PEAK_TIME:.1f}")

    # Stochastic ensemble
    n_sims = 5000
    n_traj_to_plot = 50
    sigma = params['sigma']

    peaks = np.zeros(n_sims)
    trajectories_I = []
    trajectories_B = []
    extinction_count = 0

    print(f"\nRunning {n_sims} stochastic simulations (sigma = {sigma})...")
    for sim in range(n_sims):
        extinct, I_traj, B_traj = run_stochastic(params, y0, t, sigma, seed=42 + sim)
        peaks[sim] = np.max(I_traj)
        if extinct:
            extinction_count += 1
        if sim < n_traj_to_plot:
            trajectories_I.append(I_traj)
            trajectories_B.append(B_traj)
        if (sim + 1) % 1000 == 0:
            print(f"  Completed {sim + 1}/{n_sims} simulations")

    STOCH_MEDIAN = np.median(peaks)
    STOCH_MEAN = np.mean(peaks)
    STOCH_STD = np.std(peaks)
    Q05 = np.percentile(peaks, 5)
    Q25 = np.percentile(peaks, 25)
    Q75 = np.percentile(peaks, 75)
    Q95 = np.percentile(peaks, 95)
    EXTINCTION_RATE = 100.0 * extinction_count / n_sims

    print(f"\nStochastic statistics (sigma = {sigma}):")
    print(f"  Median peak:    {STOCH_MEDIAN:.0f}")
    print(f"  Mean peak:      {STOCH_MEAN:.0f}")
    print(f"  Std dev:        {STOCH_STD:.0f}")
    print(f"  25-75% range:   [{Q25:.0f}, {Q75:.0f}]")
    print(f"  5-95% range:    [{Q05:.0f}, {Q95:.0f}]")
    print(f"  Median change:  {100.0 * (STOCH_MEDIAN - DET_PEAK) / DET_PEAK:+.1f}%")
    print(f"  Extinction:     {EXTINCTION_RATE:.1f}%")

    # ========================================================================
    # Compute amplification vs sigma for panel (d)
    # ========================================================================
    sigma_sweep = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
    amplification = []
    amplification_std = []

    print(f"\nComputing amplification vs sigma (panel d)...")
    for s in sigma_sweep:
        peaks_sweep = []
        n_sweep = 200
        for sim in range(n_sweep):
            _, I_traj_sweep, _ = run_stochastic(params, y0, t, s, seed=5000 + sim)
            peaks_sweep.append(np.max(I_traj_sweep))
        peaks_sweep = np.array(peaks_sweep)
        amp = 100 * (np.median(peaks_sweep) - DET_PEAK) / DET_PEAK
        amplification.append(amp)
        amplification_std.append(np.std(peaks_sweep))
        print(f"  sigma = {s:.2f}: amplification = {amp:+.1f}%")

    # ========================================================================
    # FIGURE
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')

    # ------------------------------------------------------------------
    # Panel (a)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for I_traj in trajectories_I:
        ax.plot(t, I_traj, 'r-', lw=0.5, alpha=0.25)
    ax.plot(t, I_det, 'b-', lw=2.5, label='Deterministic', zorder=10)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Infectious $I(t)$')
    ax.set_title('(a) Infectious Individuals Over Time', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 200)

    # ------------------------------------------------------------------
    # Panel (b)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for B_traj in trajectories_B[:20]:
        ax.semilogy(t, np.maximum(B_traj, 1), 'r-', lw=0.5, alpha=0.25)
    ax.semilogy(t, np.maximum(B_det, 1), 'b-', lw=2.5, label='Deterministic', zorder=10)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Pathogen $B(t)$ (cells/L)')
    ax.set_title('(b) Environmental Pathogen (semi-log)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 200)

    # ------------------------------------------------------------------
    # Panel (c)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.hist(peaks, bins=60, color='red', alpha=0.6, density=True, edgecolor='black')
    ax.axvline(DET_PEAK, color='blue', ls='--', lw=2.5,
               label=f'Deterministic: {DET_PEAK:.0f}')
    ax.axvline(STOCH_MEDIAN, color='red', ls='--', lw=2.5,
               label=f'Median: {STOCH_MEDIAN:.0f}')
    ax.axvline(STOCH_MEAN, color='darkred', ls=':', lw=2.0,
               label=f'Mean: {STOCH_MEAN:.0f}')
    ax.axvspan(Q25, Q75, alpha=0.15, color='orange', label='25-75%')
    ax.set_xlabel('Peak infections')
    ax.set_ylabel('Density')
    ax.set_title('(c) Distribution of Peak Infections', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # ------------------------------------------------------------------
    # Panel (d): Amplification vs noise intensity (from actual simulations)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    sigma_arr = np.array(sigma_sweep)
    amp_arr = np.array(amplification)

    ax.plot(sigma_arr, amp_arr, 'o-', color='red', lw=2.5, markersize=8,
            markeredgecolor='black', markeredgewidth=0.8,
            label='Median amplification')

    # Zero reference line
    ax.axhline(0, color='gray', ls='--', lw=1.5, label='Deterministic (no change)')

    # Baseline sigma
    ax.axvline(params['sigma'], color='purple', ls=':', lw=2,
               label=f'Baseline $\\sigma$ = {params["sigma"]}')

    # Marker at baseline
    baseline_idx = sigma_sweep.index(params['sigma'])
    ax.scatter([params['sigma']], [amp_arr[baseline_idx]],
               color='purple', s=250, marker='D',
               edgecolors='black', linewidths=1.0, zorder=10)

    ax.set_xlabel(r'Noise intensity $\sigma$', fontsize=12)
    ax.set_ylabel('Median amplification (%)', fontsize=12)
    ax.set_title('(d) Amplification vs Noise Intensity', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig('Fig/Fig2.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nFigure 2 saved: Fig/Fig2.png")
    print(f"  Deterministic peak:  {DET_PEAK:.0f}")
    print(f"  Stochastic median:   {STOCH_MEDIAN:.0f}  "
          f"({100.0 * (STOCH_MEDIAN - DET_PEAK) / DET_PEAK:+.1f}%)")
    print(f"  Stochastic mean:     {STOCH_MEAN:.0f}")
    print(f"  Extinction rate:     {EXTINCTION_RATE:.1f}%")


if __name__ == "__main__":
    main()
