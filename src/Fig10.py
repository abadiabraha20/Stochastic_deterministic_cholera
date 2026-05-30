"""
Fig10.py - Phase portraits
Generates Figure 10: Phase portraits of system dynamics (I-B, S-I, A-I, 3D S-V-I)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
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
    print("FIGURE 10: PHASE PORTRAITS")
    print("=" * 60 + "\n")
    
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['nu_V'])
    V0 = N0 * params['nu_V'] / (params['mu'] + params['omega'] + params['nu_V'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    sol = odeint(cholera_model, y0, t, args=(params,))
    S, V, A, I, B = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 6]
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    
    # Panel (a)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(I, B, 'b-', lw=1.5)
    ax1.scatter(I[0], B[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax1.scatter(I[-1], B[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax1.set_xlabel('Symptomatic I(t)')
    ax1.set_ylabel('Environment B(t)')
    ax1.set_title('(a) I-B phase plane', fontweight='bold')
    ax1.legend()
    
    # Panel (b)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(S, I, 'r-', lw=1.5)
    ax2.scatter(S[0], I[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax2.scatter(S[-1], I[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax2.set_xlabel('Susceptible S(t)')
    ax2.set_ylabel('Symptomatic I(t)')
    ax2.set_title('(b) S-I phase plane', fontweight='bold')
    ax2.legend()
    
    # Panel (c)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(A, I, 'g-', lw=1.5)
    ax3.scatter(A[0], I[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax3.scatter(A[-1], I[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax3.set_xlabel('Asymptomatic A(t)')
    ax3.set_ylabel('Symptomatic I(t)')
    ax3.set_title('(c) A-I phase plane', fontweight='bold')
    ax3.legend()
    
    # Panel (d)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    skip = 50
    ax4.plot(S[::skip], V[::skip], I[::skip], 'b-', lw=1.5)
    ax4.scatter(S[0], V[0], I[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax4.scatter(S[-1], V[-1], I[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax4.set_xlabel('Susceptible S')
    ax4.set_ylabel('Vaccinated V')
    ax4.set_zlabel('Symptomatic I')
    ax4.set_title('(d) 3D S-V-I space', fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('figures/Fig10.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 10 saved: figures/Fig10.png")

if __name__ == "__main__":
    main()
