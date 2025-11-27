import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# ACTUAL EXPERIMENT: Dynamic Optical Homeostat with PSON & Wormhole Tunneling
# =============================================================================
# Implements the "Complexity from Constraints" theory:
# 1. Physics: Double-slit interference with irregular prime gaps (Aliasing).
# 2. Control: Iterative relaxation loop (Homeostat) finding optimal phase.
# 3. Wormhole: Non-local gradient update based on global visibility benefit.
# 4. PSON: Precision-Scaled Orthogonal Noise for safe exploration.
# =============================================================================

np.random.seed(42)

# --- Physical Constants (He-Ne laser) ---
lambda_nm = 633.0
k = 2 * np.pi / (lambda_nm * 1e-9)
L = 1.0
x_screen = np.linspace(-0.005, 0.005, 500)

# --- Slit Params ---
slit_sep_base = 100e-6
slit_width = 10e-6
amp_per_slit = 0.5

# --- Prime Gaps Setup ---
first_25_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
prime_sets = {
    'uniform': [100] * 25,
    'primes': [p * 10 for p in first_25_primes]
}
n_layers = 10

# =============================================================================
# OPTICAL PHYSICS ENGINE
# =============================================================================

def simulate_interference(gaps, phase_shift=0.0):
    """
    Calculates interference intensity I(x) given a specific phase shift.
    Here, 'phase_shift' is the order parameter (eta) being evolved.
    """
    d_eff = np.mean(gaps) * 1e-6
    theta = x_screen / L
    phi_base = k * d_eff * np.sin(theta)
    
    # Cumulative aliasing over layers
    I_layers = []
    for _ in range(n_layers):
        I_layer_stack = []
        for g in gaps:
            d_layer = g * 1e-6
            phi_layer = k * d_layer * np.sin(theta)
            
            # Field 1: Reference
            field1 = amp_per_slit * np.exp(1j * 0)
            
            # Field 2: Modulated by path AND the Homeostat's phase shift
            # This is the "Gate" being controlled non-locally
            field2 = amp_per_slit * np.exp(1j * (phi_layer + phase_shift))
            
            I_layer = np.abs(field1 + field2)**2
            I_layer_stack.append(I_layer)
        I_aliased_layer = np.mean(I_layer_stack, axis=0)
        I_layers.append(I_aliased_layer)
        
    return np.mean(I_layers, axis=0)

def calculate_visibility(I):
    I_max = np.max(I)
    I_min = np.min(I)
    return (I_max - I_min) / (I_max + I_min + 1e-8)

# =============================================================================
# HOMEOSTAT CONTROLLER (The "Actual" Experiment)
# =============================================================================

def run_homeostat_relaxation(gaps, n_steps=100, w=0.5, noise_scale=0.01):
    """
    Evolves the phase shift (eta) using the Wormhole update rule + PSON.
    """
    # 1. Initialization
    current_phi = 0.0
    
    # Precision (Lambda): Inverse of structural variance (stiffness)
    # High gap variance = Low Precision (floppy) -> More Noise allowed
    gap_var = np.var(gaps)
    precision = 1.0 / (gap_var + 1e-6)
    
    trajectory_V = []
    trajectory_Phi = []
    
    print(f"Starting relaxation: Gap Var={gap_var:.1f}, Precision={precision:.4f}")
    
    for t in range(n_steps):
        # 2. Measurement (The "Forward Pass")
        I_current = simulate_interference(gaps, phase_shift=current_phi)
        V_current = calculate_visibility(I_current)
        
        # 3. Energy Definition
        # Energy = Defect (1.0 - Visibility). We want to minimize this.
        energy = 1.0 - V_current
        
        # 4. Wormhole Gradient (Eq. 3: dF/d_eta = -w * Delta_Benefit)
        # Here, "Delta Benefit" is the potential gain. The gradient points DOWNHILL
        # on the energy landscape. The "benefit" of the gate is roughly proportional
        # to the current error (simple proportional control/Hebb-like term).
        # We push 'phi' in a direction that opposes the energy (simple gradient descent view)
        # BUT strictly: The "Wormhole" concept says we credit the gate based on downstream outcome.
        grad_wormhole = -w * energy 
        
        # 5. PSON Injection (Eq. 1)
        # Noise scaled by INVERSE precision.
        # Uniform gaps (High Precision) -> Low Noise.
        # Prime gaps (Low Precision) -> High Noise (Exploration).
        pson_noise = np.random.normal(0, 1) * (1.0 / precision) * noise_scale
        
        # 6. Update State
        # Simple relaxation step
        current_phi -= (grad_wormhole + pson_noise)
        
        # Logging
        trajectory_V.append(V_current)
        trajectory_Phi.append(current_phi)
        
    return current_phi, trajectory_V, trajectory_Phi

# =============================================================================
# EXECUTION
# =============================================================================

print("--- Running Uniform Control ---")
phi_uni, traj_uni, _ = run_homeostat_relaxation(prime_sets['uniform'], w=0.2)
print(f"Final V (Uniform): {traj_uni[-1]:.4f}")

print("\n--- Running Prime Gap Homeostat ---")
phi_prime, traj_prime, phi_hist_prime = run_homeostat_relaxation(prime_sets['primes'], w=0.2)
print(f"Final V (Primes): {traj_prime[-1]:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(traj_uni, label='Uniform Gaps (Control)', linestyle='--')
plt.plot(traj_prime, label='Prime Gaps (Homeostat Active)', linewidth=2)
plt.xlabel('Relaxation Steps')
plt.ylabel('Visibility (Coherence)')
plt.title('Emergent Causality: Homeostat Relaxation Trajectory')
plt.legend()
plt.grid(True)
plt.savefig('actual_experiment_trajectory.png')
print("\nTrajectory plot saved to actual_experiment_trajectory.png")

# T-Test verification (Statistically distinct final states?)
# We take the last 20 steps as "converged" samples
v_steady_prime = traj_prime[-20:]
v_steady_uniform = traj_uni[-20:]
t_stat, p_val = stats.ttest_ind(v_steady_prime, v_steady_uniform, equal_var=False)
print(f"\nStatistical Check (Steady State): t={t_stat:.3f}, p={p_val:.5f}")
