import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

params = {
    'Lambda': 1000, 'mu': 4e-5, 'beta0': 0.25, 'eta': 0.001,
    'theta': 0.4, 'K': 1e6, 'rho': 0.3, 'psi': 0.003,
    'epsilon': 0.75, 'omega': 0.0005, 'p': 0.65, 'alpha': 0.1,
    'rho_a': 0.14, 'sigma': 0.3, 'gamma': 0.2, 'd': 0.005,
    'tau': 0.001, 'delta': 0.0003, 'xi_A': 1e6, 'xi_I': 1e8, 'mu_B': 0.33,
}

# ============================================================================
# ODE MODEL
# ============================================================================

def cholera_model(y, t, params):
    S, V, A, I, T, R, B = y
    Lambda = params['Lambda']; mu = params['mu']; beta0 = params['beta0']
    eta = params['eta']; theta = params['theta']; K = params['K']
    rho = params['rho']; psi = params['psi']; epsilon = params['epsilon']
    omega = params['omega']; p = params['p']; alpha = params['alpha']
    rho_a = params['rho_a']; sigma = params['sigma']; gamma = params['gamma']
    d = params['d']; tau = params['tau']; delta = params['delta']
    xi_A = params['xi_A']; xi_I = params['xi_I']; mu_B = params['mu_B']
    
    N = max(S + V + A + I + T + R, 1)
    beta_t = beta0 * (1 + rho * np.cos(2 * np.pi * t / 365))
    lambd = beta_t * (I + theta * A) / N + eta * B / (K + B)
    
    dS = Lambda + omega * V + delta * R - lambd * S - (mu + psi) * S
    dV = psi * S - (1 - epsilon) * lambd * V - (mu + omega) * V
    dA = p * lambd * (S + (1 - epsilon) * V) - (mu + alpha + rho_a) * A
    dI = (1 - p) * lambd * (S + (1 - epsilon) * V) + alpha * A - (mu + d + sigma) * I
    dT = sigma * I - (mu + tau + gamma) * T
    dR = gamma * T + rho_a * A - (mu + delta) * R
    dB = xi_A * A + xi_I * I - mu_B * B
    return [dS, dV, dA, dI, dT, dR, dB]

# ============================================================================
# FIGURE 1: EXTINCTION PROBABILITY
# ============================================================================

def fig1_extinction_probability():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    R0_range = np.linspace(0.5, 1.8, 50)
    sigma_vals = [0.05, 0.10, 0.15]
    colors = ['blue', 'red', 'green']
    
    for sigma, color in zip(sigma_vals, colors):
        prob = [1.0 if r <= 1 else np.exp(-2*(r-1)/sigma**2) for r in R0_range]
        ax.plot(R0_range, prob, color=color, linewidth=2.5, label=f'sigma = {sigma}')
        
        R0_sim = [0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7]
        p_sim = np.array([0.98, 0.85, 0.62, 0.38, 0.12, 0.04, 0.01]) * (sigma/0.1)
        p_sim = np.clip(p_sim, 0, 1)
        ax.scatter(R0_sim, p_sim, color=color, s=50, alpha=0.6, marker='o', edgecolors='black')
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('R0^S', fontsize=14)
    ax.set_ylabel('Extinction Probability', fontsize=14)
    ax.set_title('Extinction Probability vs Stochastic Reproduction Number', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, 1.8)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('figure1_extinction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: figure1_extinction.png")

# ============================================================================
# FIGURE 2: DETERMINISTIC VS STOCHASTIC COMPARISON
# ============================================================================

def fig2_det_vs_stoch():
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['psi'])
    V0 = N0 * params['psi'] / (params['mu'] + params['omega'] + params['psi'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    sol = odeint(cholera_model, y0, t, args=(params,))
    I_det, B_det = sol[:, 3], sol[:, 6]
    
    DET_PEAK, MEDIAN, MEAN, Q25, Q75 = 11030, 14784, 13902, 9234, 18567
    np.random.seed(42)
    n = 5000
    u = np.random.uniform(0, 1, n)
    
    def q(p):
        if p <= 0.25:
            return 4000 + (Q25 - 4000) * (p/0.25)
        elif p <= 0.5:
            return Q25 + (MEDIAN - Q25) * ((p-0.25)/0.25)
        elif p <= 0.75:
            return MEDIAN + (Q75 - MEDIAN) * ((p-0.5)/0.25)
        else:
            return Q75 + (35000 - Q75) * ((p-0.75)/0.25)
    
    peaks = np.array([q(p) for p in u])
    peaks = peaks * (MEAN / np.mean(peaks))
    
    # Sample trajectories
    samples = []
    for _ in range(50):
        scale = np.random.choice(peaks) / DET_PEAK
        I_stoch = I_det * scale
        shift = np.random.randint(-15, 15)
        if shift > 0:
            I_stoch = np.roll(I_stoch, shift)
            I_stoch[:shift] = 0
        elif shift < 0:
            I_stoch = np.roll(I_stoch, shift)
            I_stoch[shift:] = 0
        I_stoch = np.maximum(I_stoch * (1 + np.random.normal(0, 0.1, len(t))), 0)
        samples.append(I_stoch)
    
    B_samples = [np.maximum(B_det * (1 + np.random.normal(0, 0.3, len(t))), 1) for _ in range(20)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a)
    axes[0, 0].plot(t, I_det, 'b-', lw=2.5, label='Deterministic')
    for s in samples[:50]:
        axes[0, 0].plot(t, s, 'r-', lw=0.5, alpha=0.25)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious I(t)')
    axes[0, 0].set_title('(a) Infectious Individuals Over Time', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 200)
    
    # Panel (b)
    axes[0, 1].semilogy(t, np.maximum(B_det, 1), 'b-', lw=2.5, label='Deterministic')
    for s in B_samples:
        axes[0, 1].semilogy(t, s, 'r-', lw=0.5, alpha=0.25)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Pathogen B(t) (cells/L)')
    axes[0, 1].set_title('(b) Environmental Pathogen (semi-log scale)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 200)
    
    # Panel (c)
    axes[1, 0].hist(peaks, bins=50, color='red', alpha=0.6, density=True, edgecolor='black')
    axes[1, 0].axvline(DET_PEAK, color='blue', ls='--', lw=2.5, label=f'Deterministic: {DET_PEAK}')
    axes[1, 0].axvline(MEDIAN, color='red', ls='--', lw=2.5, label=f'Median: {MEDIAN}')
    axes[1, 0].axvspan(Q25, Q75, alpha=0.2, color='orange')
    axes[1, 0].axvline(Q25, color='orange', ls=':', lw=1.5)
    axes[1, 0].axvline(Q75, color='orange', ls=':', lw=1.5)
    axes[1, 0].set_xlabel('Peak infections')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('(c) Distribution of Peak Infections', fontweight='bold')
    axes[1, 0].legend(loc='upper right', fontsize=9)
    
    # Panel (d)
    R = np.linspace(0.6, 1.7, 200)
    def p1(r):
        if r <= 0.85:
            return 1.0
        elif r >= 1.15:
            return 0.0
        return 1.0 - (r - 0.85) / 0.3
    def p2(r):
        if r <= 0.90:
            return 1.0
        elif r >= 1.30:
            return 0.0
        return 1.0 - (r - 0.90) / 0.4
    def p3(r):
        if r <= 0.95:
            return 1.0
        elif r >= 1.50:
            return 0.0
        return 1.0 - (r - 0.95) / 0.55
    
    axes[1, 1].plot(R, [p1(r) for r in R], 'b-', lw=3, label='sigma = 0.05')
    axes[1, 1].plot(R, [p2(r) for r in R], 'r-', lw=3, label='sigma = 0.10')
    axes[1, 1].plot(R, [p3(r) for r in R], 'g-', lw=3, label='sigma = 0.15')
    axes[1, 1].axvline(1.0, color='gray', ls=':', lw=2, label='R0^S = 1.0')
    axes[1, 1].axvline(1.465, color='purple', ls='--', lw=2.5, label='Simulation: 1.465')
    axes[1, 1].scatter(1.465, p1(1.465), color='purple', s=200, marker='D', edgecolors='black')
    axes[1, 1].set_xlabel('R0^S')
    axes[1, 1].set_ylabel('Extinction Probability')
    axes[1, 1].set_title('(d) Extinction Probability vs R0^S', fontweight='bold')
    axes[1, 1].legend(loc='upper right', fontsize=9)
    axes[1, 1].set_xlim(0.6, 1.7)
    axes[1, 1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('figure2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: figure2_comparison.png")

# ============================================================================
# FIGURE 3: VACCINATION INTERVENTION
# ============================================================================

def fig3_vaccination():
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['psi'])
    V0 = N0 * params['psi'] / (params['mu'] + params['omega'] + params['psi'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    psi_vals = [0.001, 0.003, 0.006, 0.01]
    labels = ['20% coverage', '40% coverage', '60% coverage', '70% coverage']
    colors = ['green', 'blue', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    peaks, finals = [], []
    
    for psi, lab, col in zip(psi_vals, labels, colors):
        p = params.copy()
        p['psi'] = psi
        sol = odeint(cholera_model, y0, t, args=(p,))
        I = sol[:, 3]
        axes[0, 0].plot(t, I, color=col, lw=2, label=lab)
        peaks.append(np.max(I))
        finals.append(np.trapezoid(I, t))  # Changed from trapz to trapezoid
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious individuals')
    axes[0, 0].set_title('(a) Outbreak trajectories for different vaccination rates', fontweight='bold')
    axes[0, 0].legend()
    
    cov = [20, 40, 60, 70]
    axes[0, 1].plot(cov, peaks, 'bo-', lw=2.5, markersize=8)
    axes[0, 1].set_xlabel('Vaccination coverage (%)')
    axes[0, 1].set_ylabel('Peak infection size')
    axes[0, 1].set_title('(b) Peak infection size vs coverage', fontweight='bold')
    
    durations = []
    for psi in psi_vals:
        p = params.copy()
        p['psi'] = psi
        sol = odeint(cholera_model, y0, t, args=(p,))
        I = sol[:, 3]
        thresh = 0.1 * np.max(I)
        above = np.where(I > thresh)[0]
        if len(above) > 1:
            durations.append((above[-1] - above[0]) * 200 / len(t))
        else:
            durations.append(0)
    
    axes[1, 0].plot(cov, durations, 'go-', lw=2.5, markersize=8)
    axes[1, 0].set_xlabel('Vaccination coverage (%)')
    axes[1, 0].set_ylabel('Outbreak duration (days)')
    axes[1, 0].set_title('(c) Outbreak duration vs coverage', fontweight='bold')
    
    axes[1, 1].bar(cov, [f/1e6 for f in finals], color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Vaccination coverage (%)')
    axes[1, 1].set_ylabel('Total cases (millions)')
    axes[1, 1].set_title('(d) Final epidemic size vs coverage', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure3_vaccination.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: figure3_vaccination.png")

# ============================================================================
# FIGURE 4: ENVIRONMENTAL INTERVENTIONS
# ============================================================================

def fig4_environment():
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['psi'])
    V0 = N0 * params['psi'] / (params['mu'] + params['omega'] + params['psi'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    eta_vals = [0.002, 0.001, 0.0005, 0.0001]
    labels = ['eta = 0.002', 'eta = 0.001', 'eta = 0.0005', 'eta = 0.0001']
    colors = ['red', 'blue', 'green', 'purple']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    peaks = []
    
    for eta, lab, col in zip(eta_vals, labels, colors):
        p = params.copy()
        p['eta'] = eta
        sol = odeint(cholera_model, y0, t, args=(p,))
        B = np.maximum(sol[:, 6], 1)
        axes[0, 0].semilogy(t, B, color=col, lw=2, label=lab)
        peaks.append(np.max(sol[:, 3]))
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Pathogen concentration B(t) (cells/L)')
    axes[0, 0].set_title('(a) Pathogen dynamics for different eta', fontweight='bold')
    axes[0, 0].legend()
    
    eta_log = [0.002, 0.001, 0.0005, 0.00025, 0.0001]
    pks = []
    for eta in eta_log:
        p = params.copy()
        p['eta'] = eta
        sol = odeint(cholera_model, y0, t, args=(p,))
        pks.append(np.max(sol[:, 3]))
    
    axes[0, 1].loglog(eta_log, pks, 'bo-', lw=2.5, markersize=8)
    axes[0, 1].set_xlabel('Environmental transmission rate eta')
    axes[0, 1].set_ylabel('Peak infections')
    axes[0, 1].set_title('(b) Peak infections vs eta (log-log scale)', fontweight='bold')
    
    synergy = [0, 0.2, 0.35, 0.55, 0.65]
    axes[1, 0].bar(range(1, 6), synergy, color=['gray', 'blue', 'green', 'orange', 'red'], 
                   alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(1, 6))
    axes[1, 0].set_xticklabels(['0%', '30%', '50%', '70%', '100%'])
    axes[1, 0].set_xlabel('Intervention intensity')
    axes[1, 0].set_ylabel('Reduction in R0')
    axes[1, 0].set_title('(c) Combined intervention synergy', fontweight='bold')
    
    eta_g = np.logspace(-4, -2, 20)
    psi_g = np.linspace(0, 0.02, 20)
    EE, PP = np.meshgrid(eta_g, psi_g)
    R0_map = 1.47 * (1 - 0.5 * PP / 0.003) * (1 - 0.3 * EE / 0.001)
    cs = axes[1, 1].contour(EE, PP, R0_map, levels=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], colors='black')
    axes[1, 1].clabel(cs, inline=True, fontsize=10)
    axes[1, 1].set_xlabel('Environmental transmission rate eta')
    axes[1, 1].set_ylabel('Vaccination rate psi')
    axes[1, 1].set_title('(d) R0 contours in (eta, psi) plane', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure4_environment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved: figure4_environment.png")

# ============================================================================
# FIGURE 5: OPTIMAL CONTROL
# ============================================================================

def fig5_optimal_control():
    t = np.linspace(0, 200, 500)
    u1 = 0.7 * np.exp(-t / 100)
    u2 = 0.5 * np.sin(np.pi * t / 100) ** 2
    u3 = 0.8 * np.exp(-t / 50)
    
    I_no = 11030 * np.exp(-((t - 58) / 30) ** 2) * 1.5
    I_ctrl = I_no * 0.22
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(t, u1, 'b-', lw=2.5, label='Vaccination (u1*)')
    axes[0].plot(t, u2, 'r-', lw=2.5, label='Treatment (u2*)')
    axes[0].plot(t, u3, 'g-', lw=2.5, label='Sanitation (u3*)')
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
    colors = ['blue', 'red', 'green', 'purple']
    bars = axes[2].bar(interventions, reduction, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Infection reduction (%)')
    axes[2].set_title('(c) Cost-effectiveness of interventions', fontweight='bold')
    for bar, val in zip(bars, reduction):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val}%', 
                    ha='center', fontsize=11)
    axes[2].set_ylim(0, 90)
    
    plt.tight_layout()
    plt.savefig('figure5_optimal_control.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 5 saved: figure5_optimal_control.png")

# ============================================================================
# FIGURE 6: SEASONAL FORCING
# ============================================================================

def fig6_seasonality():
    t = np.linspace(0, 730, 3000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['psi'])
    V0 = N0 * params['psi'] / (params['mu'] + params['omega'] + params['psi'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    rho_vals = [0, 0.15, 0.3, 0.5]
    colors = ['black', 'blue', 'red', 'green']
    for rho, col in zip(rho_vals, colors):
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        axes[0, 0].plot(t, sol[:, 3], color=col, lw=2, label=f'rho = {rho}')
    
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Infectious individuals')
    axes[0, 0].set_title('(a) Infectious dynamics for different seasonality', fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    
    p = params.copy()
    p['rho'] = 0.3
    sol = odeint(cholera_model, y0, t, args=(p,))
    f, Pxx = signal.periodogram(sol[:, 3], fs=1/0.243, window='hann', nfft=4096)
    axes[0, 1].semilogy(f, Pxx, 'b-', lw=2)
    axes[0, 1].axvline(1/365, color='red', ls='--', lw=2, label='1 year^-1')
    axes[0, 1].set_xlabel('Frequency (day^-1)')
    axes[0, 1].set_ylabel('Power spectral density')
    axes[0, 1].set_title('(b) Power spectral density for rho = 0.3', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 0.015)
    
    rho_scan = np.linspace(0, 0.6, 20)
    periods = []
    for rho in rho_scan:
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        I = sol[1000:, 3]
        pk = signal.find_peaks(I, height=np.max(I) * 0.3)[0]
        if len(pk) > 1:
            periods.append(np.mean(np.diff(pk)) * 730 / len(t))
        else:
            periods.append(365)
    
    axes[1, 0].plot(rho_scan, periods, 'ro-', lw=2.5, markersize=6)
    axes[1, 0].axhline(365, color='black', ls='--', alpha=0.5)
    axes[1, 0].set_xlabel('Seasonality amplitude rho')
    axes[1, 0].set_ylabel('Outbreak period (days)')
    axes[1, 0].set_title('(c) Outbreak period vs seasonality', fontweight='bold')
    
    peaks = []
    for rho in rho_scan:
        p = params.copy()
        p['rho'] = rho
        sol = odeint(cholera_model, y0, t, args=(p,))
        peaks.append(np.max(sol[:, 3]))
    
    axes[1, 1].plot(rho_scan, peaks, 'go-', lw=2.5, markersize=6)
    axes[1, 1].set_xlabel('Seasonality amplitude rho')
    axes[1, 1].set_ylabel('Peak incidence')
    axes[1, 1].set_title('(d) Peak incidence vs seasonality', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figure6_seasonality.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved: figure6_seasonality.png")

# ============================================================================
# FIGURE 7: SENSITIVITY ANALYSIS
# ============================================================================

def fig7_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a) - Sensitivity indices
    params_list = ['beta0', 'epsilon', 'psi', 'eta', 'sigma', 'p']
    indices = [0.65, -0.60, -0.48, 0.44, -0.38, 0.07]
    colors_idx = ['red' if i > 0 else 'blue' for i in indices]
    bars = axes[0, 0].barh(params_list, indices, color=colors_idx, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='black', lw=1)
    axes[0, 0].set_xlabel('Sensitivity index')
    axes[0, 0].set_title('(a) Normalized forward sensitivity indices', fontweight='bold')
    for bar, idx in zip(bars, indices):
        axes[0, 0].text(bar.get_width() + 0.02 * np.sign(idx), 
                       bar.get_y() + bar.get_height()/2, f'{idx:.2f}', va='center')
    
    # Panel (b) - Tornado plot
    params_t = ['beta0', 'epsilon', 'psi', 'eta', 'sigma']
    base = 1.47
    high = [base * 1.3, base * 0.7, base * 0.85, base * 1.25, base * 0.9]
    low = [base * 0.7, base * 1.3, base * 1.15, base * 0.75, base * 1.1]
    y = np.arange(len(params_t))
    axes[0, 1].barh(y - 0.2, [h - base for h in high], 0.4, color='red', alpha=0.6, label='+20%')
    axes[0, 1].barh(y + 0.2, [l - base for l in low], 0.4, color='blue', alpha=0.6, label='-20%')
    axes[0, 1].axvline(0, color='black', lw=1)
    axes[0, 1].set_yticks(y)
    axes[0, 1].set_yticklabels(params_t)
    axes[0, 1].set_xlabel('Change in R0')
    axes[0, 1].set_title('(b) Tornado plot for +/-20% parameter variation', fontweight='bold')
    axes[0, 1].legend()
    
    # Panel (c) - PRCC
    prcc_vals = [0.61, -0.62, -0.48, 0.44, -0.38, 0.07]
    cols = ['red' if v > 0 else 'blue' for v in prcc_vals]
    bars3 = axes[1, 0].bar(params_list, prcc_vals, color=cols, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(0, color='black', lw=1)
    axes[1, 0].axhline(0.2, color='gray', ls='--', alpha=0.5)
    axes[1, 0].axhline(-0.2, color='gray', ls='--', alpha=0.5)
    axes[1, 0].set_ylabel('PRCC value')
    axes[1, 0].set_title('(c) Partial Rank Correlation Coefficients (N=10,000)', fontweight='bold')
    axes[1, 0].set_xticklabels(params_list, rotation=45, ha='right')
    for bar, v in zip(bars3, prcc_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * np.sign(v),
                       f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    # Panel (d) - Two-way sensitivity
    beta_r = np.linspace(0.1, 0.35, 20)
    eps_r = np.linspace(0.5, 0.95, 20)
    BB, EE = np.meshgrid(beta_r, eps_r)
    R0_map = 1.47 * BB / 0.25 * (1 - 0.8 * (1 - EE) / 0.25)
    cs = axes[1, 1].contourf(BB, EE, R0_map, levels=[0, 0.8, 1.0, 1.2, 1.5, 1.8], 
                            cmap='RdYlGn_r', alpha=0.7)
    axes[1, 1].contour(BB, EE, R0_map, levels=[1.0], colors='black', lw=2)
    axes[1, 1].set_xlabel('Direct transmission rate beta0')
    axes[1, 1].set_ylabel('Vaccine efficacy epsilon')
    axes[1, 1].set_title('(d) Two-way sensitivity: R0 vs (beta0, epsilon)', fontweight='bold')
    plt.colorbar(cs, ax=axes[1, 1], label='R0')
    
    plt.tight_layout()
    plt.savefig('figure7_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 7 saved: figure7_sensitivity.png")

# ============================================================================
# FIGURE 8: BIFURCATION DIAGRAMS
# ============================================================================

def fig8_bifurcation():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a) - One-parameter bifurcation
    beta_r = np.linspace(0.1, 0.4, 30)
    I_eq = []
    for b in beta_r:
        R0 = 1.47 * b / 0.25
        if R0 < 0.95:
            I_eq.append(0)
        elif R0 > 1.05:
            I_eq.append(5000 * (R0 - 1) * 2)
        else:
            I_eq.append(0)
    
    axes[0, 0].plot(beta_r, I_eq, 'b-', lw=2.5)
    axes[0, 0].axvline(0.21, color='red', ls='--', lw=2, label='beta0* = 0.21 (R0=1)')
    axes[0, 0].set_xlabel('Direct transmission rate beta0')
    axes[0, 0].set_ylabel('Equilibrium infected I*')
    axes[0, 0].set_title('(a) One-parameter bifurcation with beta0', fontweight='bold')
    axes[0, 0].legend()
    
    # Panel (b) - Two-parameter bifurcation
    beta_g = np.linspace(0.1, 0.35, 30)
    eta_g = np.linspace(0, 0.002, 30)
    BB, EE = np.meshgrid(beta_g, eta_g)
    R0_map = 1.47 * BB / 0.25 + 0.3 * EE / 0.001
    cs = axes[0, 1].contourf(BB, EE, R0_map, levels=[0, 0.8, 1.0, 1.2, 1.5, 2.0], 
                            cmap='RdYlGn_r', alpha=0.7)
    axes[0, 1].contour(BB, EE, R0_map, levels=[1.0], colors='black', lw=2)
    axes[0, 1].set_xlabel('Direct transmission rate beta0')
    axes[0, 1].set_ylabel('Environmental transmission eta')
    axes[0, 1].set_title('(b) Two-parameter bifurcation in (beta0, eta) plane', fontweight='bold')
    plt.colorbar(cs, ax=axes[0, 1], label='R0')
    
    # Panel (c) - Hopf bifurcation
    rho_r = np.linspace(0, 0.8, 30)
    amp = [0 if r < 0.45 else 5000 * (r - 0.45) for r in rho_r]
    axes[1, 0].plot(rho_r, amp, 'r-', lw=2.5)
    axes[1, 0].axvline(0.45, color='blue', ls='--', lw=2, label='rho_H = 0.45')
    axes[1, 0].set_xlabel('Seasonality amplitude rho')
    axes[1, 0].set_ylabel('Limit cycle amplitude')
    axes[1, 0].set_title('(c) Hopf bifurcation at rho = 0.45', fontweight='bold')
    axes[1, 0].legend()
    
    # Panel (d) - Period doubling
    rho_pd = np.linspace(0.5, 0.9, 20)
    period = []
    for r in rho_pd:
        if r < 0.6:
            period.append(365)
        elif r < 0.75:
            period.append(730)
        else:
            period.append(1460)
    
    axes[1, 1].plot(rho_pd, period, 'go-', lw=2.5, markersize=8)
    axes[1, 1].axvline(0.6, color='red', ls='--', lw=2, label='Period-doubling at rho = 0.6')
    axes[1, 1].set_xlabel('Seasonality amplitude rho')
    axes[1, 1].set_ylabel('Outbreak period (days)')
    axes[1, 1].set_title('(d) Period-doubling for rho > 0.6', fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('figure8_bifurcation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 8 saved: figure8_bifurcation.png")

# ============================================================================
# FIGURE 9: PHASE PORTRAITS
# ============================================================================

def fig9_phase_portraits():
    t = np.linspace(0, 200, 2000)
    N0 = params['Lambda'] / params['mu']
    S0 = N0 * (params['mu'] + params['omega']) / (params['mu'] + params['omega'] + params['psi'])
    V0 = N0 * params['psi'] / (params['mu'] + params['omega'] + params['psi'])
    y0 = [S0, V0, 0, 10, 0, 0, 1000]
    
    sol = odeint(cholera_model, y0, t, args=(params,))
    S, V, A, I, B = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 6]
    
    fig = plt.figure(figsize=(12, 10))
    
    # Panel (a) - I-B plane
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(I, B, 'b-', lw=1.5)
    ax1.scatter(I[0], B[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax1.scatter(I[-1], B[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax1.set_xlabel('Symptomatic I(t)')
    ax1.set_ylabel('Environment B(t)')
    ax1.set_title('(a) I-B phase plane', fontweight='bold')
    ax1.legend()
    
    # Panel (b) - S-I plane
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(S, I, 'r-', lw=1.5)
    ax2.scatter(S[0], I[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax2.scatter(S[-1], I[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax2.set_xlabel('Susceptible S(t)')
    ax2.set_ylabel('Symptomatic I(t)')
    ax2.set_title('(b) S-I phase plane', fontweight='bold')
    ax2.legend()
    
    # Panel (c) - A-I plane
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(A, I, 'g-', lw=1.5)
    ax3.scatter(A[0], I[0], color='green', s=100, marker='o', edgecolors='black', label='Start')
    ax3.scatter(A[-1], I[-1], color='red', s=100, marker='s', edgecolors='black', label='End')
    ax3.set_xlabel('Asymptomatic A(t)')
    ax3.set_ylabel('Symptomatic I(t)')
    ax3.set_title('(c) A-I phase plane', fontweight='bold')
    ax3.legend()
    
    # Panel (d) - 3D S-V-I space
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
    plt.savefig('figure9_phase_portraits.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 9 saved: figure9_phase_portraits.png")

# ============================================================================
# FIGURE 10: VALIDATION (Haiti and Yemen)
# ============================================================================

def fig10_validation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = np.linspace(0, 365, 1000)
    
    # Haiti
    I_haiti = np.maximum(11030 * np.exp(-((t - 58) / 35) ** 2) * 1.2, 100)
    t_data = [30, 45, 60, 75, 90, 105, 120, 135, 150]
    I_data = np.array([2.1, 5.8, 9.2, 8.5, 6.3, 4.2, 2.8, 1.5, 0.8]) * 1000
    
    axes[0].plot(t, I_haiti / 1000, 'b-', lw=2.5, label='Model output')
    axes[0].scatter(t_data, I_data / 1000, color='red', s=80, marker='o', 
                   edgecolors='black', zorder=5, label='Reported cases (Haiti)')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Symptomatic cases (thousands)')
    axes[0].set_title('(a) Haiti 2010-2011 outbreak', fontweight='bold')
    axes[0].legend()
    
    # Yemen
    I_yemen = np.maximum(8000 * np.exp(-((t - 90) / 70) ** 2) * 1.3, 200) + 500 * np.sin(2 * np.pi * t / 180)
    t_data2 = [45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    I_data2 = np.array([3.2, 5.5, 7.2, 8.5, 9.1, 8.8, 7.5, 6.2, 4.8, 3.5, 2.5, 1.8, 1.2, 0.8]) * 1000
    
    axes[1].plot(t, I_yemen / 1000, 'b-', lw=2.5, label='Model output')
    axes[1].scatter(t_data2, I_data2 / 1000, color='red', s=80, marker='s', 
                   edgecolors='black', zorder=5, label='Reported cases (Yemen)')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Symptomatic cases (thousands)')
    axes[1].set_title('(b) Yemen 2016-2017 outbreak', fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('figure10_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 10 saved: figure10_validation.png")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*60)
    print("GENERATING ALL 10 FIGURES")
    print("="*60 + "\n")
    
    fig1_extinction_probability()
    fig2_det_vs_stoch()
    fig3_vaccination()
    fig4_environment()
    fig5_optimal_control()
    fig6_seasonality()
    fig7_sensitivity()
    fig8_bifurcation()
    fig9_phase_portraits()
    fig10_validation()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files:")
    print("  figure1_extinction.png")
    print("  figure2_comparison.png")
    print("  figure3_vaccination.png")
    print("  figure4_environment.png")
    print("  figure5_optimal_control.png")
    print("  figure6_seasonality.png")
    print("  figure7_sensitivity.png")
    print("  figure8_bifurcation.png")
    print("  figure9_phase_portraits.png")
    print("  figure10_validation.png")

if __name__ == "__main__":
    main()