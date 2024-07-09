import numpy as np
from constants import dt, N_atoms_initial, T_initial
from atom_cloud import AtomCloud
from dipole_trap import TiltedDipleTrap
from laser_beam import LaserBeam
from utils import mot_loading_and_compression, raman_cooling, optical_pumping, calculate_observables, expand_parameters
import logging


def run_full_sequence(params):
    P_y_interp, P_z_interp, P_R_interp, P_p_interp, B_z_interp = expand_parameters(params)
    
    try:
        total_time = 0.575  # 575ms
        tilt_angle = 17 * np.pi / 180  # 17 degrees tilt
        
        # Start with initial powers based on the paper
        initial_P_y, initial_P_z = P_y_interp(0), P_z_interp(0)  # Initial powers in Watts
        trap = TiltedDipleTrap(initial_P_y, initial_P_z, 18e-6, 14e-6, 1064e-9, tilt_angle)
        
        atoms = AtomCloud(N_atoms_initial, T_initial, trap)
        
        sigma_minus_beam = LaserBeam(P_p_interp(0), 30e-6, 1e-3, Rb87.wavelength_D1)
        pi_beam = LaserBeam(P_R_interp(0), 0.5e-3, 0.5e-3, Rb87.wavelength_D1)
        
        results = []
        trap_frequencies = []
        cooling_efficiencies = []
        condensate_fractions = []
        
        # Create time array
        time_array = np.linspace(0, total_time, int(total_time / dt))

    
        logging.debug("Starting MOT loading and compression")
        atoms = mot_loading_and_compression(atoms, trap, P_y_interp(time_array[:int(0.099/dt)]), 
                                            P_z_interp(time_array[:int(0.099/dt)]), B_z_interp(time_array[:int(0.099/dt)]))
        logging.debug("Finished MOT loading and compression")

        atoms.force_sync()
        results.append(calculate_observables(atoms))
        logging.info(f"After MOT: N = {atoms.N:.2e}, T = {atoms.T*1e6:.2f} μK")
        
        logging.debug(f"Before main cooling loop: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        

        
        logging.debug(f"After forced synchronization: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        
        logging.debug(f"Before main cooling loop: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")

        
        
        # Pre-calculate some values
        raman_cooling_end = 0.414  # End of Raman cooling phase
        xodt_start = 0.3  # Start of crossed ODT
        record_interval = 1000
        min_trap_depth = 1e-6  # Minimum trap depth in Kelvin
        
        # Variables for detecting sudden changes
        prev_N = atoms.N
        prev_T = atoms.T
        
        # Main cooling and evaporation loop
        for i, t in enumerate(time_array[int(0.099/dt):], start=int(0.099/dt)):
            if i == int(0.099/dt):
                logging.debug(f"First iteration of main loop: t = {t:.3f}s, N = {atoms.N}, "
                              f"positions shape = {atoms.positions.shape}, "
                              f"velocities shape = {atoms.velocities.shape}")
            
            atoms.force_sync()
            # Update trap parameters
            P_y_target, P_z_target = P_y_interp(t), P_z_interp(t)
            trap.ramp_powers(t - 0.099, 0.01, P_y_target, P_z_target)  # 10 ms ramp duration
            
            # Transition to crossed ODT
            if t >= xodt_start:
                trap.update_waists(18e-6, 14e-6)  # Adjust waists for crossed ODT
            
            # Calculate trap depth
            trap_depth = trap.calculate_trap_depth()
            if trap_depth < min_trap_depth:
                #logging.warning(f"Trap depth too low at t = {t:.3f}s: {trap_depth:.2e} K")
                trap_depth = min_trap_depth
            
            sigma_minus_beam.power = P_p_interp(t)
            pi_beam.power = P_R_interp(t)
            
            atoms.update(dt)
            
            if t < raman_cooling_end:  # Raman cooling phase (until 414ms)
                raman_cooling(atoms, P_R_interp(t), P_p_interp(t), calculate_delta_R(B_z_interp(t)), sigma_minus_beam, pi_beam)
                optical_pumping(atoms, P_p_interp(t), -4.33e9, sigma_minus_beam)
                atoms.apply_light_assisted_collisions(P_p_interp(t), -4.33e9)
            
            # Apply temperature-dependent three-body recombination
            atoms.apply_three_body_recombination()
            
            atoms.apply_photon_reabsorption_heating(P_p_interp(t), -4.33e9)
            atoms.apply_evaporative_cooling(trap_depth)
            
            # Detect sudden changes
            if abs(atoms.N - prev_N) / prev_N > 0.1 or abs(atoms.T - prev_T) / prev_T > 0.1:
                logging.warning(f"Sudden change detected at t = {t:.3f}s: "
                                f"ΔN/N = {(atoms.N - prev_N)/prev_N:.2f}, "
                                f"ΔT/T = {(atoms.T - prev_T)/prev_T:.2f}")
            
            prev_N, prev_T = atoms.N, atoms.T
            
            if i % record_interval == 0:  # Record results every 1000 steps
                
                logging.info(f"At t = {t:.3f}s: N = {atoms.N:.2e}, T = {atoms.T*1e6:.2f} μK")
        
            results.append(calculate_observables(atoms))
            trap_frequencies.append((trap.omega_x, trap.omega_y, trap.omega_z))
            # Add logging for array sizes
            logging.debug(f"At t = {t:.3f}s: atoms.positions shape = {atoms.positions.shape}, "
                          f"atoms.velocities shape = {atoms.velocities.shape}")
        

        # Final results
        results.append(calculate_observables(atoms))
        trap_frequencies.append((trap.omega_x, trap.omega_y, trap.omega_z))

        # Convert results and trap_frequencies to numpy arrays
        results = np.array(results)
        trap_frequencies = np.array(trap_frequencies)
        

        logging.info(f"Final simulation results: shape = {results.shape}")
        logging.info(f"Final trap frequencies: shape = {trap_frequencies.shape}")
        

        for i in range(len(results) - 1):
            cooling_efficiencies.append(calculate_cooling_efficiency(results[i:i+2]))
            condensate_fractions.append(detect_bec(atoms))
        
        return (np.array(results), np.array(trap_frequencies), trap,
                np.array(cooling_efficiencies), np.array(condensate_fractions))
    
    except Exception as e:
        logging.error(f"Error in run_full_sequence: {e}")
        logging.error(f"Error occurred at t = {t:.3f}s")
        logging.error(f"Current state: N = {atoms.N}, "
                      f"positions shape = {atoms.positions.shape}, "
                      f"velocities shape = {atoms.velocities.shape}")
        raise



def initial_simulation_run():
    P_y_init = [1.0, 0.6, 0.4, 0.15, 0.02, 0.01, 0.008, 0.02, 0.01, 0.0075, 0.005]
    P_z_init = [0.01, 0.012, 0.01, 0.025, 0.02, 0.01, 0.008, 0.06, 0.5, 0.015, 0.003]
    P_R_init = [10, 40, 30, 0, 10, 1, 0, 0, 0, 0, 0]
    P_p_init = [0.008, 0.009, 0.01, 0.01, 0.001, 0.005, 0, 0, 0, 0, 0]
    B_z_init = [3.25e-4, 3.15e-4, 3.25e-4, 3.2e-4, 2.8e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4, 3.05e-4]

    params_init = P_y_init + P_z_init + P_R_init + P_p_init + B_z_init

    return run_full_sequence(params_init)