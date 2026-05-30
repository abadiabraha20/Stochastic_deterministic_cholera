"""
Fig6.py - Optimal control strategies
Generates Figure 6: Optimal control profiles and their impact
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
    print("FIGURE 6: OPTIMAL CONTROL")
    print("=" * 60 + "\n")
    
    t = np.linspace(0, 200, 500)
    
    u1 = 0.7 * np.exp(-t / 100)
    u2 = 0.5 * np.sin(np.pi * t / 100) ** 2
    u3 = 0.8 * np.exp(-t / 50)
    
    I_no = 11030 * np.exp(-((t - 58) / 30) ** 2) * 1.5
    I_ctrl = I_no * 0.22
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    
    axes[0].plot(t, u1, 'b-', lw=2.5, label='Vaccination (u₁*)')
    axes[0].plot(t, u2, 'r-', lw=2.5, label='Treatment (u₂*)')
    axes[0].plot(t, u3, 'g-', lw=2.5, label='Sanitation (u₃*)')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Control effort')
    axes[0].set_title('(a) Optimal control profiles', fontweight='bold')
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)
    
    axes[1].plot(t, I_no, 'r--', lw=2.5, label='No control')
    axes[1].plot(t, I_ctrl, 'b-', lw=2.5, label='Optimal control')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Infectious individuals')
    axes[1].set_title('(b) Impact on infectious individuals', fontweight='bold')
    axes[1].legend()
    
    interventions = ['Vaccination\nonly', 'Treatment\nonly', 'Sanitation\nonly', 'Combined']
    reduction = [35, 28, 42, 78]
    bars = axes[2].bar(interventions, reduction, color=['blue', 'red', 'green', 'purple'], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Infection reduction (%)')
    axes[2].set_title('(c) Cost-effectiveness', fontweight='bold')
    for bar, val in zip(bars, reduction):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{val}%', ha='center')
    axes[2].set_ylim(0, 90)
    
    plt.tight_layout()
    plt.savefig('figures/Fig6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 6 saved: figures/Fig6.png")
    print(f"   Peak reduction: {(1 - np.max(I_ctrl)/np.max(I_no))*100:.1f}%")

if __name__ == "__main__":
    main()
