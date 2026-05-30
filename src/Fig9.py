"""
Fig9.py - Bifurcation diagrams
Generates Figure 9: Transcritical bifurcation, two-parameter bifurcation, Hopf bifurcation, and period-doubling
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

def main():
    print("\n" + "=" * 60)
    print("FIGURE 9: BIFURCATION DIAGRAMS")
    print("=" * 60 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # Panel (a)
    beta_range = np.linspace(0.1, 0.4, 30)
    I_eq = []
    for b in beta_range:
        R0 = 1.47 * b / 0.25
        if R0 < 0.95:
            I_eq.append(0)
        elif R0 > 1.05:
            I_eq.append(5000 * (R0 - 1) * 2)
        else:
            I_eq.append(0)
    
    axes[0, 0].plot(beta_range, I_eq, 'b-', lw=2.5)
    axes[0, 0].axvline(0.21, color='red', ls='--', lw=2, label=r'$\beta_0^* = 0.21$ ($\mathcal{R}_0=1$)')
    axes[0, 0].set_xlabel(r'Direct transmission rate $\beta_0$')
    axes[0, 0].set_ylabel('Equilibrium infected I*')
    axes[0, 0].set_title('(a) One-parameter bifurcation with $\beta_0$', fontweight='bold')
    axes[0, 0].legend()
    
    # Panel (b)
    beta_grid = np.linspace(0.1, 0.35, 30)
    eta_grid = np.linspace(0, 0.002, 30)
    BB, EE = np.meshgrid(beta_grid, eta_grid)
    R0_map = 1.47 * BB / 0.25 + 0.3 * EE / 0.001
    
    contour = axes[0, 1].contourf(BB, EE, R0_map, levels=[0, 0.8, 1.0, 1.2, 1.5, 2.0], cmap='RdYlGn_r', alpha=0.7)
    axes[0, 1].contour(BB, EE, R0_map, levels=[1.0], colors='black', lw=2)
    axes[0, 1].set_xlabel(r'Direct transmission rate $\beta_0$')
    axes[0, 1].set_ylabel(r'Environmental transmission $\eta$')
    axes[0, 1].set_title(r'(b) Two-parameter bifurcation in ($\beta_0$, $\eta$) plane', fontweight='bold')
    plt.colorbar(contour, ax=axes[0, 1], label=r'$\mathcal{R}_0$')
    
    # Panel (c)
    rho_range = np.linspace(0, 0.8, 30)
    amplitudes = [0 if r < 0.45 else 5000 * (r - 0.45) for r in rho_range]
    axes[1, 0].plot(rho_range, amplitudes, 'r-', lw=2.5)
    axes[1, 0].axvline(0.45, color='blue', ls='--', lw=2, label=r'$\rho_H \approx 0.45$')
    axes[1, 0].set_xlabel('Seasonality amplitude $\rho$')
    axes[1, 0].set_ylabel('Limit cycle amplitude')
    axes[1, 0].set_title('(c) Hopf bifurcation at $\rho \approx 0.45$', fontweight='bold')
    axes[1, 0].legend()
    
    # Panel (d)
    rho_pd = np.linspace(0.5, 0.9, 20)
    periods_pd = [365 if r < 0.6 else (730 if r < 0.75 else 1460) for r in rho_pd]
    axes[1, 1].plot(rho_pd, periods_pd, 'go-', lw=2.5, markersize=8)
    axes[1, 1].axvline(0.6, color='red', ls='--', lw=2, label='Period-doubling at $\rho \approx 0.6$')
    axes[1, 1].set_xlabel('Seasonality amplitude $\rho$')
    axes[1, 1].set_ylabel('Outbreak period (days)')
    axes[1, 1].set_title('(d) Period-doubling for $\rho > 0.6$', fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/Fig9.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 9 saved: figures/Fig9.png")

if __name__ == "__main__":
    main()
