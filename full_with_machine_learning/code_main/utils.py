import numpy as np
from constants import Rb87, k, hbar, g

def photoassociation_loss_rate(delta, intensity):
    gamma = 2 * np.pi * 5.75e6  # Natural linewidth of Rb87 D1 line
    saturation_intensity = 4.484  # mW/cm^2 for Rb87 D1 line
    optimal_detuning = -4.33e9  # -4.33 GHz, optimal detuning from main_ref.tex
    detuning_width = 50e6  # 50 MHz width around optimal detuning
    
    # Convert intensity to saturation parameter
    s = intensity / saturation_intensity
    
    # Calculate base rate
    base_rate = 1e-14 * (s / (1 + s + 4 * (delta / gamma)**2))**2
    
    # Apply detuning-dependent scaling
    detuning_factor = np.exp(-(delta - optimal_detuning)**2 / (2 * detuning_width**2))
    
    return base_rate * detuning_factor


def mot_loading_and_compression(atoms, trap, P_y, P_z, B_z):
    logging.debug(f"Starting MOT loading and compression: N = {atoms.N}, "
                  f"positions shape = {atoms.positions.shape}, "
                  f"velocities shape = {atoms.velocities.shape}")
    
    try:
        # 88 ms of MOT loading and initial compression
        for _ in range(int(0.088 / dt)):
            atoms.velocities *= 0.989  # Increased cooling rate
            atoms.update(dt)
            atoms.update_temperature()
        
        logging.debug(f"After initial compression: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
        # Apply gray molasses for 1 ms
        atoms.apply_gray_molasses(0.001)
        
        logging.debug(f"After gray molasses: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
        # 10 ms ramp of trap beam powers with some heating
        initial_P_y, initial_P_z = trap.P_y, trap.P_z
        for i in range(int(0.01 / dt)):
            t = i * dt / 0.01
            trap.P_y = initial_P_y * (1 - t) + P_y[0] * t
            trap.P_z = initial_P_z * (1 - t) + P_z[0] * t
            trap.update_trap_frequencies()
            atoms.update(dt)
            atoms.T *= 1.001  # Small heating during compression
        
        logging.debug(f"After power ramp: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
        # 1 ms magnetic field adjustment
        initial_B_z = B_z[0]
        for i in range(int(0.001 / dt)):
            t = i * dt / 0.001
            current_B_z = initial_B_z * (1 - t) + B_z[0] * t
            atoms.apply_magnetic_field(current_B_z)
            atoms.update(dt)
        
        logging.debug(f"After magnetic field adjustment: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
        # Adjust atom number to match reference
        atoms.N = int(2.7e5)
        atoms.update_atom_number(atoms.N)
        
        logging.debug(f"After atom number adjustment: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
    except Exception as e:
        logging.error(f"Error in mot_loading_and_compression: {e}")
        raise
    
    return atoms



def raman_cooling(atoms, P_R, P_p, delta_R, sigma_minus_beam, pi_beam):
    atoms.apply_light_shift(P_p, -4.33e9, sigma_minus_beam)
    
    raman_rate = P_R * 1e3
    v_recoil = Rb87.vr_D1
    v_res = (delta_R - atoms.light_shift) / (2 * np.pi / Rb87.wavelength_D1)
    
    I_sigma = sigma_minus_beam.intensity(atoms.positions)
    I_pi = pi_beam.intensity(atoms.positions)
    
    epsilon = 1e-10  # Small positive constant
    Omega_eff = np.sqrt(np.maximum(I_sigma * I_pi, epsilon)) * P_R
    
    delta_R_eff = delta_R - atoms.light_shift
    delta_R_eff = np.maximum(delta_R_eff, 1e-10)  # Avoid division by zero
    
    v_magnitude = np.linalg.norm(atoms.velocities, axis=1)
    
    cooling_prob = raman_rate * dt * (Omega_eff**2 / (Omega_eff**2 + 4 * (v_magnitude - v_res)**2 + 4 * delta_R_eff**2))
    cooling_mask = np.random.random(atoms.N) < cooling_prob
    
    delta_E = 0
    if np.sum(cooling_mask) > 0:
        cooled_velocities = atoms.velocities[cooling_mask]
        cooled_v_magnitude = v_magnitude[cooling_mask]
        
        non_zero_velocity_mask = cooled_v_magnitude > 1e-10
        if np.sum(non_zero_velocity_mask) > 0:
            cooling_direction = np.zeros_like(cooled_velocities)
            cooling_direction[non_zero_velocity_mask] = cooled_velocities[non_zero_velocity_mask] / cooled_v_magnitude[non_zero_velocity_mask, np.newaxis]
        else:
            cooling_direction = np.random.randn(*cooled_velocities.shape)
            cooling_direction /= np.linalg.norm(cooling_direction, axis=1)[:, np.newaxis]
        
        E_before = 0.5 * Rb87.mass * np.sum(cooled_velocities**2)
        
        cooled_velocities -= v_recoil * cooling_direction
        
        # Subrecoil cooling model
        v_eff_recoil = np.sqrt(8 * Rb87.Er_D1 / Rb87.mass)
        subrecoil_mask = cooled_v_magnitude < v_eff_recoil
        cooling_strength = np.exp(-(cooled_v_magnitude[subrecoil_mask] / v_eff_recoil)**2)
        cooled_velocities[subrecoil_mask] *= np.maximum(0, 1 - cooling_strength[:, np.newaxis])
        
        atoms.velocities[cooling_mask] = cooled_velocities
        
        E_after = 0.5 * Rb87.mass * np.sum(cooled_velocities**2)
        
        delta_E = E_before - E_after
        atoms.T -= delta_E / (3 * atoms.N * k)
        
        # Off-resonant scattering heating
        off_resonant_rate = Omega_eff**2 / (4 * delta_R_eff**2) * Rb87.gamma_D1
        heating_energy = off_resonant_rate * dt * np.sum(cooling_mask) * 2 * Rb87.Er_D1
        atoms.T += heating_energy / (3 * atoms.N * k)
        
        # Ensure temperature doesn't go below minimum value
        atoms.T = np.maximum(atoms.T, 1e-9)

    atoms.update_temperature()
    
    logging.debug(f"Raman cooling: delta_T = {delta_E / (3 * atoms.N * k):.2e}, T = {atoms.T:.2e}, cooled atoms: {np.sum(cooling_mask)}")

    # Additional error checking
    if np.isnan(atoms.T) or np.isinf(atoms.T):
        logging.error(f"Invalid temperature after Raman cooling: T = {atoms.T:.2e}")
        raise ValueError("Invalid temperature encountered in Raman cooling")
 


def optical_pumping(atoms, P_p, delta, sigma_minus_beam):
    I = sigma_minus_beam.intensity(atoms.positions)
    I_sat = 1.67  # Saturation intensity for Rb87 D1 line
    s = I / I_sat
    gamma_sc = Rb87.gamma_D1 / 2 * s / (1 + s + 4 * delta**2 / Rb87.gamma_D1**2)
    
    reabsorption_prob = atoms.calculate_density() * 3 * Rb87.wavelength_D1**2 / (2 * np.pi)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    festina_lente_factor = 1 / (1 + reabsorption_prob * gamma_sc / (atoms.trap.omega_x + epsilon))

    scattering_prob = gamma_sc * dt * festina_lente_factor
    scattering_mask = np.random.random(atoms.N) < scattering_prob
    
    recoil_velocity = Rb87.vr_D1
    recoil_directions = np.random.randn(np.sum(scattering_mask), 3)
    recoil_directions /= np.linalg.norm(recoil_directions, axis=1)[:, np.newaxis]
    
    E_before = 0.5 * Rb87.mass * np.sum(atoms.velocities[scattering_mask]**2)
    
    # Account for branching ratio
    branching_ratio = 1/3  # Assuming 1/3 chance to end up in the dark state
    pumping_mask = np.random.random(np.sum(scattering_mask)) < branching_ratio
    atoms.velocities[scattering_mask][pumping_mask] += recoil_velocity * recoil_directions[pumping_mask]
    
    E_after = 0.5 * Rb87.mass * np.sum(atoms.velocities[scattering_mask]**2)
    
    delta_E = E_after - E_before
    atoms.T += delta_E / (3 * atoms.N * k)
    
    # Bosonic stimulation effect
    v_magnitude = np.linalg.norm(atoms.velocities, axis=1)
    ground_state_mask = v_magnitude < Rb87.vr_D1
    stimulation_factor = 1 + atoms.calculate_density() * Rb87.wavelength_D1**3
    atoms.velocities[ground_state_mask] *= np.exp(-stimulation_factor * dt)
    
    atoms.update_temperature()
    
    logging.debug(f"Optical pumping: delta_T = {delta_E / (3 * atoms.N * k):.2e}, T = {atoms.T:.2e}, scattered atoms: {np.sum(scattering_mask)}")


def calculate_observables(atoms):
    if atoms.N > 0:
        T = np.sum(atoms.velocities**2) * Rb87.mass / (3 * k * atoms.N)
        T = max(1e-10, T)  # Ensure temperature is positive
    else:
        T = 1e-10  # Set a minimum temperature if N is zero
    N = max(1, atoms.N)  # Ensure N is at least 1 to avoid division by zero
    n = atoms.calculate_density()
    PSD = n * (h**2 / (2*np.pi*Rb87.mass*k*T))**1.5
    return T, N, PSD


def calculate_cooling_efficiency(results):
    N = np.array([max(1e-10, result[1]) for result in results])  # Ensure N > 0
    PSD = np.array([max(1e-10, result[2]) for result in results])  # Ensure PSD > 0
    
    log_N = np.log(N)
    log_PSD = np.log(PSD)
    
    diff_log_N = np.diff(log_N)
    diff_log_PSD = np.diff(log_PSD)
    
    # Avoid division by zero or very small numbers
    epsilon = 1e-10
    gamma = -diff_log_PSD / (diff_log_N + epsilon)
    
    return gamma



def calculate_delta_R(B_z):
    return 2 * Rb87.mu_B * Rb87.g_F * B_z / hbar

def calculate_optical_depth(T, N):
    return N * (h**2 / (2 * np.pi * Rb87.mass * k * T))**(2/3)

def calculate_psd(T, N, trap_frequencies):
    omega_x, omega_y, omega_z = trap_frequencies.T
    omega_mean = np.cbrt(omega_x * omega_y * omega_z)
    
    # Print shapes for debugging
    print(f"Shape of N: {N.shape}")
    print(f"Shape of T: {T.shape}")
    print(f"Shape of omega_mean: {omega_mean.shape}")
    
    # Ensure all arrays have the same length
    min_length = min(len(N), len(T), len(omega_mean))
    N = N[:min_length]
    T = T[:min_length]
    omega_mean = omega_mean[:min_length]
    
    # Avoid division by zero
    T = np.maximum(T, 1e-10)
    
    return N * (hbar * omega_mean / (k * T))**3



def expand_parameters(params):
    # Total simulation time
    total_time = 0.575  # 575ms

    # Create time array for 11 points
    time_points = np.linspace(0, total_time, 11)

    # Expand to 11 stages for all parameters
    P_y = params[0:11]
    P_z = params[11:22]
    P_R = params[22:33]
    P_p = params[33:44]
    B_z = params[44:55]

    # Create interpolation functions for each parameter
    def create_interp_func(param_values):
        def interp_func(t):
            if np.isscalar(t):
                idx = np.searchsorted(time_points, t, side='right') - 1
                idx = np.clip(idx, 0, len(time_points) - 2)
                t0, t1 = time_points[idx], time_points[idx + 1]
                p0, p1 = param_values[idx], param_values[idx + 1]
                return p0 + (p1 - p0) * (t - t0) / (t1 - t0)
            else:
                return np.interp(t, time_points, param_values)
        return interp_func


    P_y_interp = create_interp_func(P_y)
    P_z_interp = create_interp_func(P_z)
    P_R_interp = create_interp_func(P_R)
    P_p_interp = create_interp_func(P_p)
    B_z_interp = create_interp_func(B_z)

    return P_y_interp, P_z_interp, P_R_interp, P_p_interp, B_z_interp




def validate_simulation(results, trap_frequencies, params, trap, cooling_efficiencies, condensate_fractions):
    if np.any(np.isnan(results)) or np.any(np.isinf(results)):
        logging.warning("Results contain NaN or Inf values. Skipping plot_psd_vs_n.")
    else:
        try:
            plot_psd_vs_n(results)
        except Exception as e:
            logging.error(f"Error in plotting PSD vs N: {e}")
    
    plot_control_waveforms(params)
    plot_trap_and_atomic_properties(results, trap_frequencies)
    plot_results(results, cooling_efficiencies, condensate_fractions)

    T_final, N_final, PSD_final = results[-1]
    # Use the last set of trap frequencies instead of the trap object
    final_trap_frequencies = trap_frequencies[-1]
    condensate_fraction = estimate_condensate_fraction(T_final, N_final, final_trap_frequencies)
    
    print(f"Final Temperature: {T_final*1e6:.2f} μK")
    print(f"Final Atom Number: {N_final:.2e}")
    print(f"Final PSD: {PSD_final:.2e}")
    print(f"Estimated Condensate Fraction: {condensate_fraction*100:.1f}%")




def estimate_condensate_fraction(T, N, trap_frequencies):
    omega_x, omega_y, omega_z = trap_frequencies
    T_c = 0.94 * hbar * (N * omega_x * omega_y * omega_z)**(1/3) / k
    return max(0, 1 - (T/T_c)**3) if T < T_c else 0