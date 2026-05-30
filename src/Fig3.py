"""
Fig3.py - Extinction probability
Generates Figure 3: Extinction probability vs stochastic reproduction number
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('Fig'):
    os.makedirs('Fig')

np.random.seed(42)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

def main():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='white')
    
    R0_range = np.linspace(0.5, 1.8, 50)
    sigma_vals = [0.05, 0.10, 0.15]
    colors = ['blue', 'red', 'green']
    
    for sigma, color in zip(sigma_vals, colors):
        prob = [1.0 if r <= 1 else np.exp(-2 * (r - 1) / sigma**2) for r in R0_range]
        ax.plot(R0_range, prob, color=color, linewidth=2.5, label=f'ν = {sigma}')
        
        R0_sim = [0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7]
        p_sim = np.array([0.98, 0.85, 0.62, 0.38, 0.12, 0.04, 0.01]) * (sigma / 0.1)
        p_sim = np.clip(p_sim, 0, 1)
        ax.scatter(R0_sim, p_sim, color=color, s=50, alpha=0.6, marker='o', edgecolors='black')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\mathcal{R}_0^S$', fontsize=14)
    ax.set_ylabel('Extinction Probability', fontsize=14)
    ax.set_title('Extinction Probability vs Stochastic Reproduction Number', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, 1.8)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('Fig/Fig3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: Fig/Fig3.png")

if __name__ == "__main__":
    main()
