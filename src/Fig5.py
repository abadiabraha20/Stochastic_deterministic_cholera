"""
Fig5.py - Environmental interventions (final)
Panels: (a) Pathogen dynamics for varying eta, (b) Peak infections vs eta on log-log,
(c) Combined intervention analysis, (d) R0 contours in (eta, nu_V) plane.
Grids OFF.
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
# CALIBRATED PARAMETERS
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
# MODEL
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
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("FIGURE 5: ENVIRONMENTAL INTERVENTIONS")
    print("=" * 60)

    t = np.linspace(0, 200, 2001)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['nu_V'])
    V0 = N0 * params['nu_V'] / (params['mu'] + params['omega'] + params['nu_V'])
    y0 = [S0, V0, 0, params['I0'], 0, 0, 1000]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ------------------------------------------------------------------
    # Panel (a): Pathogen dynamics for varying eta
    # ------------------------------------------------------------------
    eta_mults = [0.25, 0.5, 1.0, 2.0]
    colors = ['blue', 'green', 'orange', 'red']
    for mult, color in zip(eta_mults, colors):
        eta_val = params['eta'] * mult
        p = params.copy(); p['eta'] = eta_val
        sol = odeint(cholera_model, y0, t, args=(p,))
        B = np.maximum(sol[:, 6], 1)
        label = f'$\\eta$ = {eta_val:.2e} ({mult}$\\times$)'
        axes[0, 0].semilogy(t, B, color=color, lw=2, label=label)

    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Pathogen $B(t)$ (cells/L)')
    axes[0, 0].set_title('(a) Pathogen dynamics', fontweight='bold')
    axes[0, 0].legend(fontsize=9, loc='lower right')
    axes[0, 0].set_xlim(0, 200)

    # ------------------------------------------------------------------
    # Panel (b): Peak infections vs eta on log-log
    # ------------------------------------------------------------------
    eta_sweep = params['eta'] * np.logspace(-1, 1, 25)
    peaks = []
    for eta_val in eta_sweep:
        p = params.copy(); p['eta'] = eta_val
        sol = odeint(cholera_model, y0, t, args=(p,))
        peaks.append(np.max(sol[:, 3]))
    peaks = np.array(peaks)

    # Fit power law only in the supercritical region (peaks > 100)
    mask = peaks > 100
    slope, intercept = np.polyfit(np.log10(eta_sweep[mask]), np.log10(peaks[mask]), 1)

    axes[0, 1].loglog(eta_sweep, peaks, 'bo-', lw=2, markersize=6,
                       markeredgecolor='black', label='Simulation')
    eta_ref = eta_sweep[mask]
    fit_line = 10**(intercept) * eta_ref**slope
    axes[0, 1].loglog(eta_ref, fit_line, 'r--', lw=2,
                       label=f'Power law: slope = {slope:.2f}')
    axes[0, 1].set_xlabel(r'Environmental transmission rate $\eta$ (day$^{-1}$)')
    axes[0, 1].set_ylabel('Peak infections')
    axes[0, 1].set_title('(b) Power-law scaling', fontweight='bold')
    axes[0, 1].legend()

    print(f"\nPower-law exponent (supercritical region): {slope:.3f}")

    # ------------------------------------------------------------------
    # Panel (c): Combined intervention analysis
    # ------------------------------------------------------------------
    scenarios = {
        'Baseline': (1.0, 1.0),
        'Halve η': (0.5, 1.0),
        'Double ν_V': (1.0, 2.0),
        'Combined': (0.5, 2.0),
    }
    labels = list(scenarios.keys())
    R0_vals = []
    for name, (eta_m, nu_m) in scenarios.items():
        p = params.copy()
        p['eta'] = params['eta'] * eta_m
        p['nu_V'] = params['nu_V'] * nu_m
        R0_vals.append(compute_R0(p))
    R0_vals = np.array(R0_vals)
    R0_baseline = R0_vals[0]
    reductions = 100 * (R0_baseline - R0_vals) / R0_baseline

    bar_colors = ['gray', 'blue', 'green', 'red']
    bars = axes[1, 0].bar(labels, reductions, color=bar_colors,
                           alpha=0.7, edgecolor='black')
    for bar, red in zip(bars, reductions):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{red:.1f}%', ha='center', fontweight='bold')
    axes[1, 0].set_ylabel('Reduction in $\\mathcal{R}_0$ (%)')
    axes[1, 0].set_title('(c) Combined intervention analysis', fontweight='bold')
    axes[1, 0].set_ylim(0, max(reductions) * 1.15)

    print(f"\nIntervention analysis:")
    for name, red in zip(labels, reductions):
        print(f"  {name}: R0 = {R0_vals[labels.index(name)]:.3f}, reduction = {red:.1f}%")

    # ------------------------------------------------------------------
    # Panel (d): R0 contours in (eta, nu_V)
    # ------------------------------------------------------------------
    eta_grid = params['eta'] * np.logspace(-1, 1, 30)
    nu_grid = params['nu_V'] * np.linspace(0.5, 3.0, 30)
    ETA, NU = np.meshgrid(eta_grid, nu_grid)
    R0_grid = np.zeros_like(ETA)
    for i in range(ETA.shape[0]):
        for j in range(ETA.shape[1]):
            p = params.copy()
            p['eta'] = ETA[i, j]
            p['nu_V'] = NU[i, j]
            R0_grid[i, j] = compute_R0(p)

    cs = axes[1, 1].contourf(ETA * 1e4, NU * 1e3, R0_grid,
                              levels=np.linspace(1.0, 2.5, 16), cmap='viridis')
    cs_line = axes[1, 1].contour(ETA * 1e4, NU * 1e3, R0_grid, levels=[1.0],
                                   colors='red', linewidths=2.5)
    axes[1, 1].clabel(cs_line, fmt='R$_0$=1')
    plt.colorbar(cs, ax=axes[1, 1], label='$\\mathcal{R}_0$')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel(r'$\eta \times 10^4$ (day$^{-1}$)')
    axes[1, 1].set_ylabel(r'$\nu_V \times 10^3$ (day$^{-1}$)')
    axes[1, 1].set_title('(d) $\\mathcal{R}_0$ contours', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Fig/Fig5.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 5 saved: Fig/Fig5.png")


if __name__ == "__main__":
    main()
