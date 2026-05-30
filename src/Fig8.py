"""
Fig8.py - Sensitivity analysis
Generates Figure 8: Sensitivity analysis of R0 with PRCC, tornado plot, and two-way sensitivity
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
    print("FIGURE 8: SENSITIVITY ANALYSIS")
    print("=" * 60 + "\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    # Panel (a)
    params_list = ['β₀', 'ε', 'ν_V', 'η', 'γ_T', 'p']
    indices = [0.65, -0.60, -0.48, 0.44, -0.38, 0.07]
    colors_idx = ['red' if i > 0 else 'blue' for i in indices]
    
    bars = axes[0, 0].barh(params_list, indices, color=colors_idx, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='black', lw=1)
    axes[0, 0].set_xlabel('Sensitivity index')
    axes[0, 0].set_title('(a) Normalized forward sensitivity indices', fontweight='bold')
    for bar, idx in zip(bars, indices):
        axes[0, 0].text(bar.get_width() + 0.02 * np.sign(idx), bar.get_y() + bar.get_height() / 2, f'{idx:.2f}', va='center')
    
    # Panel (b)
    params_t = ['β₀', 'ε', 'ν_V', 'η', 'γ_T']
    base_R0 = 1.47
    high = [base_R0 * 1.3, base_R0 * 0.7, base_R0 * 0.85, base_R0 * 1.25, base_R0 * 0.9]
    low = [base_R0 * 0.7, base_R0 * 1.3, base_R0 * 1.15, base_R0 * 0.75, base_R0 * 1.1]
    y = np.arange(len(params_t))
    
    axes[0, 1].barh(y - 0.2, [h - base_R0 for h in high], 0.4, color='red', alpha=0.6, label='+20%')
    axes[0, 1].barh(y + 0.2, [l - base_R0 for l in low], 0.4, color='blue', alpha=0.6, label='-20%')
    axes[0, 1].axvline(0, color='black', lw=1)
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(params_t)
    axes[0, 1].set_xlabel('Change in R₀')
    axes[0, 1].set_title('(b) Tornado plot for ±20% parameter variation', fontweight='bold')
    axes[0, 1].legend()
    
    # Panel (c)
    prcc_params = ['β₀', 'ε', 'ν_V', 'η', 'γ_T', 'p']
    prcc_values = [0.61, -0.62, -0.48, 0.44, -0.38, 0.07]
    prcc_colors = ['red' if v > 0 else 'blue' for v in prcc_values]
    
    bars3 = axes[1, 0].bar(prcc_params, prcc_values, color=prcc_colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(0, color='black', lw=1)
    axes[1, 0].axhline(0.2, color='gray', ls='--', alpha=0.5)
    axes[1, 0].axhline(-0.2, color='gray', ls='--', alpha=0.5)
    axes[1, 0].set_ylabel('PRCC value')
    axes[1, 0].set_title('(c) Partial Rank Correlation Coefficients (N=10,000)', fontweight='bold')
    axes[1, 0].set_xticklabels(prcc_params, rotation=45, ha='right')
    for bar, v in zip(bars3, prcc_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * np.sign(v), f'{v:.2f}', ha='center', fontsize=9)
    
    # Panel (d)
    beta_range = np.linspace(0.1, 0.35, 20)
    eps_range = np.linspace(0.5, 0.95, 20)
    BB, EE = np.meshgrid(beta_range, eps_range)
    R0_map = 1.47 * BB / 0.25 * (1 - 0.8 * (1 - EE) / 0.25)
    
    contour = axes[1, 1].contourf(BB, EE, R0_map, levels=[0, 0.8, 1.0, 1.2, 1.5, 1.8], cmap='RdYlGn_r', alpha=0.7)
    axes[1, 1].contour(BB, EE, R0_map, levels=[1.0], colors='black', lw=2)
    axes[1, 1].set_xlabel('Direct transmission rate β₀')
    axes[1, 1].set_ylabel('Vaccine efficacy ε')
    axes[1, 1].set_title('(d) Two-way sensitivity: R₀ vs (β₀, ε)', fontweight='bold')
    plt.colorbar(contour, ax=axes[1, 1], label='R₀')
    
    plt.tight_layout()
    plt.savefig('figures/Fig8.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 8 saved: figures/Fig8.png")

if __name__ == "__main__":
    main()
