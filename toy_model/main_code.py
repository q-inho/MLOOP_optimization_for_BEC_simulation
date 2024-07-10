import numpy as np
from scipy.constants import h, hbar, k, atomic_mass, epsilon_0, c, mu_0
from scipy.interpolate import CubicSpline
from numba import jit

# Rb87 constants
Rb87_mass = 86.909180527 * atomic_mass
Rb87_wavelength_D1 = 794.979e-9  # meters
Rb87_wavelength_D2 = 780.241e-9  # meters
Rb87_gamma_D1 = 2 * np.pi * 5.746e6  # rad/s
Rb87_gamma_D2 = 2 * np.pi * 6.065e6  # rad/s
Rb87_a_s = 98 * 5.29e-11  # s-wave scattering length in meters
Rb87_g_F = 1/2  # Landé g-factor for F=2 state
Rb87_mu_B = 9.274e-24  # Bohr magneton in J/T
Rb87_Er_D1 = (h / Rb87_wavelength_D1)**2 / (2 * Rb87_mass)  # Recoil energy for D1 transition
Rb87_I_sat_D1 = 1.49  # mW/cm^2
Rb87_I_sat_D2 = 1.67  # mW/cm^2
Rb87_T_r_eff = 2.8e-6  # Effective recoil temperature

@jit(nopython=True)
def calculate_trap_frequency(P, w0, wavelength):
    polarizability = 5.3e-39  # m^3, approximate value for 1064 nm
    U0 = 2 * polarizability * P / (np.pi * c * epsilon_0 * w0**2)
    return np.sqrt(4 * U0 / (Rb87_mass * w0**2))

@jit(nopython=True)
def calculate_trap_depth(P_y, P_z, w0_y, w0_z, is_crossed):
    U_y = 2 * 5.3e-39 * P_y / (np.pi * c * epsilon_0 * w0_y**2)
    if is_crossed:
        U_z = 2 * 5.3e-39 * P_z / (np.pi * c * epsilon_0 * w0_z**2)
        return U_y + U_z
    else:
        return U_y

@jit(nopython=True)
def calculate_scattering_rate(P, w0, detuning):
    I = 2 * P / (np.pi * w0**2)
    return Rb87_gamma_D1 / 2 * (I / Rb87_I_sat_D1) / (1 + I / Rb87_I_sat_D1 + 4 * (detuning / Rb87_gamma_D1)**2)

@jit(nopython=True)
def raman_cooling_rate(T, delta_R, Omega_R, Gamma_sc):
    cooling_efficiency = 1 - np.exp(-delta_R / (k * T))
    if T > Rb87_T_r_eff:
        return Gamma_sc * Omega_R**2 * cooling_efficiency
    else:
        return Gamma_sc * Omega_R**2 * (T / Rb87_T_r_eff) * cooling_efficiency

@jit(nopython=True)
def evaporation_rate(T, trap_depth, omega_mean):
    eta = trap_depth / (k * T)
    return omega_mean * eta * np.exp(-eta)

@jit(nopython=True)
def three_body_loss_rate(n, T):
    K3_0 = 4.3e-29  # cm^6/s for Rb-87 in |2, -2⟩ state
    T_scale = 100e-6  # 100 μK
    K3 = K3_0 * (1 + (T / T_scale)**2) * 1e-12  # Convert to m^6/s
    return K3 * n**2

@jit(nopython=True)
def light_induced_loss_rate(n, Gamma_sc, detuning):
    return 1e-12 * n * Gamma_sc / (detuning / Rb87_gamma_D1)**2

@jit(nopython=True)
def reabsorption_heating_rate(n, T, Gamma_sc, omega_mean):
    return 1e-30 * n * Gamma_sc * (Rb87_Er_D1 / k) * (Gamma_sc / omega_mean)

@jit(nopython=True)
def optimize_raman_detuning(T, omega_mean):
    return max(8 * Rb87_Er_D1 / hbar, k * T / hbar)

def expand_parameters(params):
    total_time = 0.475  # 575ms - 100ms (MOT time)
    time_points = np.linspace(0, total_time, 11)
    
    P_y = params[0:11]
    P_z = params[11:22]
    P_R = params[22:33]
    P_p = params[33:44]
    B_z = params[44:55]
    
    return (
        CubicSpline(time_points, P_y),
        CubicSpline(time_points, P_z),
        CubicSpline(time_points, P_R),
        CubicSpline(time_points, P_p),
        CubicSpline(time_points, B_z)
    )

@jit(nopython=True)
def simulation_step(t, N, T, P_y, P_z, P_R, P_p, B_z, is_crossed, is_raman, dt):
    omega_x = calculate_trap_frequency(P_y, 10e-6, 1064e-9)
    omega_y = calculate_trap_frequency(P_y, 10e-6, 1064e-9)
    omega_z = calculate_trap_frequency(P_z, 18e-6, 1064e-9) if is_crossed else omega_y
    omega_mean = (omega_x * omega_y * omega_z)**(1/3)
    
    trap_depth = calculate_trap_depth(P_y, P_z, 10e-6, 18e-6, is_crossed)
    
    n = N * (Rb87_mass * omega_mean**2 / (4 * np.pi * k * T))**(3/2)
    
    Gamma_sc = calculate_scattering_rate(P_p, 30e-6, -2*np.pi*4.33e9)
    Omega_R = np.sqrt(P_R * Gamma_sc / (P_p + 1e-10))
    
    delta_R = optimize_raman_detuning(T, omega_mean) + Rb87_g_F * Rb87_mu_B * B_z / hbar
    
    if is_raman:
        cooling_rate = raman_cooling_rate(T, delta_R, Omega_R, Gamma_sc)
    else:
        cooling_rate = 0
    
    evap_rate = evaporation_rate(T, trap_depth, omega_mean)
    three_body_rate = three_body_loss_rate(n, T)
    light_induced_loss = light_induced_loss_rate(n, Gamma_sc, -2*np.pi*4.33e9)
    reabsorption_heating = reabsorption_heating_rate(n, T, Gamma_sc, omega_mean)
    
    # Reduce loss rates
    evap_rate *= 0.1
    three_body_rate *= 0.01
    light_induced_loss *= 0.1
    
    dT = (-cooling_rate * T + (evap_rate * trap_depth - 3 * k * T) / (N + 1e-10) + reabsorption_heating) * dt
    dN = (-evap_rate * N - three_body_rate * N - light_induced_loss * N) * dt
    
    T_new = max(T + dT, 1e-6)  # Increase minimum temperature to 1 μK
    N_new = max(N + dN, 1)  # Prevent atom number from going below 1
    
    PSD = N_new * (h * omega_mean / (k * T_new))**3
    
    # Calculate condensate fraction
    if PSD > 2.612:
        condensate_fraction = 1 - (T_new / (0.94 * h * omega_mean / k * N_new**(1/3)))**3
    else:
        condensate_fraction = 0
    
    return N_new, T_new, PSD, n, condensate_fraction


def run_simulation(P_y, P_z, P_R, P_p, B_z):
    time = np.arange(0, total_time, dt)
    N = np.zeros_like(time)
    T = np.zeros_like(time)
    PSD = np.zeros_like(time)
    n = np.zeros_like(time)
    condensate_fraction = np.zeros_like(time)
    cooling_efficiency = np.zeros_like(time)
    
    N[0] = N_atoms_initial
    T[0] = T_initial
    
    stages = [
        {'name': 'S1', 'start': 0, 'end': 0.063, 'is_raman': True, 'is_crossed': False},
        {'name': 'S2', 'start': 0.063, 'end': 0.125, 'is_raman': True, 'is_crossed': False},
        {'name': 'X1', 'start': 0.125, 'end': 0.188, 'is_raman': True, 'is_crossed': True},
        {'name': 'X2', 'start': 0.188, 'end': 0.251, 'is_raman': True, 'is_crossed': True},
        {'name': 'X3', 'start': 0.251, 'end': 0.314, 'is_raman': True, 'is_crossed': True},
        {'name': 'E1', 'start': 0.314, 'end': 0.341, 'is_raman': False, 'is_crossed': True},
        {'name': 'E2', 'start': 0.341, 'end': 0.368, 'is_raman': False, 'is_crossed': True},
        {'name': 'E3', 'start': 0.368, 'end': 0.395, 'is_raman': False, 'is_crossed': True},
        {'name': 'E4', 'start': 0.395, 'end': 0.422, 'is_raman': False, 'is_crossed': True},
        {'name': 'E5', 'start': 0.422, 'end': 0.449, 'is_raman': False, 'is_crossed': True},
        {'name': 'E6', 'start': 0.449, 'end': 0.475, 'is_raman': False, 'is_crossed': True},
    ]
    
    for i in range(1, len(time)):
        t = time[i]
        stage = next(stage for stage in stages if stage['start'] <= t < stage['end'])
        
        N[i], T[i], PSD[i], n[i], condensate_fraction[i] = simulation_step(
            t, N[i-1], T[i-1],
            P_y(t), P_z(t), P_R(t), P_p(t), B_z(t),
            stage['is_crossed'], stage['is_raman'],
            dt
        )
        
        if i > 1:
            dN = N[i] - N[i-1]
            dPSD = PSD[i] - PSD[i-1]
            if abs(dN) > 1e-10 and abs(dPSD) > 1e-10:
                cooling_efficiency[i] = -np.log(PSD[i] / PSD[i-1]) / np.log(N[i] / N[i-1])
            else:
                cooling_efficiency[i] = cooling_efficiency[i-1]
        
        # Add debugging: print values every 1000 steps
        if i % 1000 == 0:
            print(f"Step {i}: t={t:.3f}, N={N[i]:.2e}, T={T[i]:.2e}, PSD={PSD[i]:.2e}")
    
    return {
        'time': time,
        'N': N,
        'T': T,
        'PSD': PSD,
        'n': n,
        'condensate_fraction': condensate_fraction,
        'cooling_efficiency': cooling_efficiency
    }

# Simulation parameters
N_atoms_initial = int(1e5)
T_initial = 30e-6  # 30 μK
dt = 1e-5  # Time step
total_time = 0.475  # Total simulation time (excluding MOT)

# Initial parameter values (adjusted)
P_y_init = [1.0, 0.6, 0.4, 0.15, 0.02, 0.01, 0.008, 0.02, 0.01, 0.0075, 0.005]
P_z_init = [0.01, 0.012, 0.01, 0.025, 0.02, 0.01, 0.008, 0.06, 0.5, 0.015, 0.003]
P_R_init = [1, 4, 3, 0, 1, 0.1, 0, 0, 0, 0, 0]
P_p_init = [0.0008, 0.0009, 0.001, 0.001, 0.0001, 0.0005, 0, 0, 0, 0, 0]
B_z_init = [3.25e-4, 3.15e-4, 3.25e-4, 3.2e-4, 2.8e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4]

params_init = P_y_init + P_z_init + P_R_init + P_p_init + B_z_init

# Create interpolation functions
P_y, P_z, P_R, P_p, B_z = expand_parameters(params_init)

# Run simulation
results = run_simulation(P_y, P_z, P_R, P_p, B_z)

# Print final results
print(f"Final atom number: {results['N'][-1]:.2e}")
print(f"Final temperature: {results['T'][-1]*1e6:.2f} μK")
print(f"Final PSD: {results['PSD'][-1]:.2e}")
print(f"Final condensate fraction: {results['condensate_fraction'][-1]:.2%}")
print(f"Average cooling efficiency (γ): {np.mean(results['cooling_efficiency'][1:]):.2f}")
print(f"Initial cooling efficiency (γ): {np.mean(results['cooling_efficiency'][1:100]):.2f}")
print(f"Final cooling efficiency (γ): {np.mean(results['cooling_efficiency'][-100:]):.2f}")

# Plot results
import matplotlib.pyplot as plt

def plot_results(results):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    axs[0, 0].plot(results['time'], results['N'])
    axs[0, 0].set_ylabel('Atom Number')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_yscale('log')
    
    axs[0, 1].plot(results['time'], results['T'])
    axs[0, 1].set_ylabel('Temperature (K)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_yscale('log')
    
    axs[1, 0].plot(results['time'], results['PSD'])
    axs[1, 0].set_ylabel('Phase Space Density')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_yscale('log')
    
    axs[1, 1].plot(results['time'], results['condensate_fraction'])
    axs[1, 1].set_ylabel('Condensate Fraction')
    axs[1, 1].set_xlabel('Time (s)')
    
    axs[2, 0].plot(results['time'], results['n'])
    axs[2, 0].set_ylabel('Peak Density (m^-3)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_yscale('log')
    
    axs[2, 1].plot(results['time'][1:], results['cooling_efficiency'][1:])
    axs[2, 1].set_ylabel('Cooling Efficiency (γ)')
    axs[2, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

plot_results(results)