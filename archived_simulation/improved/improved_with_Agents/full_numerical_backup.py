import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp, nquad
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.constants import h, hbar, k, atomic_mass, mu_0, epsilon_0, g
from scipy.special import zeta

class BECSimulation:
    def __init__(self):
        self.m = 87 * atomic_mass  # Mass of Rb-87 atom in kg
        self.a = 5.77e-9  # Scattering length for Rb-87
        self.wavelength = 795e-9  # Wavelength of cooling light
        self.trap_wavelength = 1064e-9  # Wavelength of trapping light
        self.k_L = 2 * np.pi / self.wavelength
        self.mu_B = 9.274e-24  # Bohr magneton

        # Trap properties
        self.w_x = 18e-6  # Beam waist for x direction
        self.w_y = 18e-6  # Beam waist for y direction
        self.w_z = 14e-6  # Beam waist for z direction

        self.params = {
            'initial_N': 2.7e5,  # Initial atom number from paper
            'initial_T': 30e-6,  # Initial temperature of 30 μK from paper
            'gamma_bg': 0.05,  # Background loss rate
            'wx': 18e-6,  # 18 μm horizontal beam waist
            'wy': 14e-6,  # 14 μm vertical beam waist
            'w_R': 500e-6,  # Raman beam waist
            'theta_R': np.pi/4,  # Angle between Raman beams
            'raman_cooling_efficiency': 0.9,  # Base Raman cooling efficiency
            'evap_efficiency': 0.98,  # Base evaporative cooling efficiency
            'min_temperature': 50e-9,  # 50 nK minimum temperature
            'Omega_R_0': 2 * np.pi * 50e3,  # Base Raman Rabi frequency
            'Gamma_OP_0': 2 * np.pi * 2e3,  # Base optical pumping rate
            'Gamma': 2 * np.pi * 5.75e6,  # Natural linewidth of Rb87 D1 line
            'E_r': (h / self.wavelength)**2 / (2 * self.m),  # Recoil energy'
            'interaction_shift': -1.33,
            'tilt_factor': 2.0,  # Tilt factor for evaporation from paper
            'mot_cooling_rate': 1e3,
            'mot_final_temperature': 20e-6,
            'compression_cooling_rate': 1e3,
            'K_3': 4e-29,  # Three-body loss coefficient for Rb-87
            'imaging_resolution': 5e-6,  # 5 μm imaging resolution
            'imaging_noise': 0.05,  # 5% imaging noise
            'mot_loading_time': 89e-3,  # 89 ms MOT loading time
            'mot_compression_time': 10e-3,  # 10 ms compression time
        }

        self.stages = []
        total_time = 0
        for stage_info in [
            {'name': 'MOTLoading', 'duration': 89e-3},
            {'name': 'MOTCompression', 'duration': 10e-3},
            {'name': 'MagneticFieldAdjustment', 'duration': 1e-3},
            {'name': 'Raman1', 'duration': 63e-3},
            {'name': 'Raman2', 'duration': 63e-3},
            {'name': 'Raman3', 'duration': 63e-3},
            {'name': 'Raman4', 'duration': 63e-3},
            {'name': 'Raman5', 'duration': 63e-3},
            {'name': 'Evap1', 'duration': 27e-3},
            {'name': 'Evap2', 'duration': 27e-3},
            {'name': 'Evap3', 'duration': 27e-3},
            {'name': 'Evap4', 'duration': 27e-3},
            {'name': 'Evap5', 'duration': 27e-3},
            {'name': 'Evap6', 'duration': 25e-3},
        ]:
            stage = stage_info.copy()
            stage['start_time'] = total_time
            total_time += stage['duration']
            stage['end_time'] = total_time
            self.stages.append(stage)

        self.stage_boundaries = [(s['start_time'], s['end_time'], s['name']) for s in self.stages]

        self.stage_params = {
            'MOTLoading': {'P_p': 0, 'P_R': 0, 'P_y': 1.0, 'P_z': 0, 'B_z': 0},
            'MOTCompression': {'P_p': 0, 'P_R': 0, 'P_y': 1.0, 'P_z': 0, 'B_z': 0},
            'MagneticFieldAdjustment': {'P_p': 0, 'P_R': 0, 'P_y': 1.0, 'P_z': 0.01, 'B_z': 0},
            'Raman1': {'P_p': 0.008, 'P_R': 0.01, 'P_y': 1.0, 'P_z': 0.01, 'B_z': 3.25},
            'Raman2': {'P_p': 0.009, 'P_R': 0.04, 'P_y': 0.6, 'P_z': 0.012, 'B_z': 3.15},
            'Raman3': {'P_p': 0.01, 'P_R': 0.03, 'P_y': 0.4, 'P_z': 0.01, 'B_z': 3.25},
            'Raman4': {'P_p': 0.01, 'P_R': 0, 'P_y': 0.15, 'P_z': 0.025, 'B_z': 3.2},
            'Raman5': {'P_p': 0.001, 'P_R': 0.01, 'P_y': 0.02, 'P_z': 0.02, 'B_z': 2.8},
            'Evap1': {'P_p': 0.005, 'P_R': 0.0001, 'P_y': 0.01, 'P_z': 0.01, 'B_z': 3.05},
            'Evap2': {'P_p': 0, 'P_R': 0, 'P_y': 0.008, 'P_z': 0.008, 'B_z': 3.05},
            'Evap3': {'P_p': 0, 'P_R': 0, 'P_y': 0.02, 'P_z': 0.06, 'B_z': 3.05},
            'Evap4': {'P_p': 0, 'P_R': 0, 'P_y': 0.01, 'P_z': 0.5, 'B_z': 3.05},
            'Evap5': {'P_p': 0, 'P_R': 0, 'P_y': 0.0075, 'P_z': 0.015, 'B_z': 3.05},
            'Evap6': {'P_p': 0, 'P_R': 0, 'P_y': 0.005, 'P_z': 0.003, 'B_z': 3.05},
        }



    def initialize_atoms(self, N, T):
        # Ensure N is an integer
        N = int(np.ceil(N))  # Round up to the nearest integer
        # Initialize positions and velocities based on temperature
        self.positions = np.random.normal(0, np.sqrt(k * T / (self.m * self.omega_x**2)), (N, 3))
        self.velocities = np.random.normal(0, np.sqrt(k * T / self.m), (N, 3))
    
    def update_trap_frequencies(self, P_x, P_y, P_z):
        # Calculate trap frequencies based on beam powers and waists
        self.omega_x = np.sqrt(4 * P_x / (np.pi * self.w_x**2 * self.w_y * self.m * 299792458))
        self.omega_y = np.sqrt(4 * P_y / (np.pi * self.w_x * self.w_y**2 * self.m * 299792458))
        self.omega_z = np.sqrt(4 * P_z / (np.pi * self.w_x * self.w_z**2 * self.m * 299792458))
    



    def interpolate_params(self, t):
        for i, (start, end, name) in enumerate(self.stage_boundaries):
            if start <= t < end:
                progress = (t - start) / (end - start)
                current_params = self.stage_params[name]
                
                if i == len(self.stage_boundaries) - 1:
                    return current_params
                
                next_params = self.stage_params[self.stage_boundaries[i+1][2]]
                
                return {key: current_params[key] + (next_params[key] - current_params[key]) * progress 
                        for key in current_params}
        
        return self.stage_params[self.stages[-1]['name']]

    def trap_potential(self, x, y, z, P_y, P_z):
        U_y = self.gaussian_beam_potential(y, z, P_y, self.params['wx'])
        U_z = self.gaussian_beam_potential(x, y, P_z, self.params['wy'])
        return U_y + U_z + self.m * g * z  # Include gravitational potential

    def gaussian_beam_potential(self, r, z, P, w0):
        z_R = np.pi * w0**2 / self.trap_wavelength
        w = w0 * np.sqrt(1 + (z / z_R)**2)
        U_0 = 4 * P / (np.pi * w0**2)
        return U_0 * (w0 / w)**2 * np.exp(-2 * r**2 / w**2)
    

    def trap_frequencies(self, P_y, P_z):
        omega_x = np.sqrt(4 * P_y / (self.m * np.pi * self.params['wx']**2) + 4 * P_z / (self.m * np.pi * self.params['wy']**2))
        omega_y = np.sqrt(4 * P_y / (self.m * np.pi * self.params['wx']**2))
        omega_z = np.sqrt(max(4 * P_z / (self.m * np.pi * self.params['wy']**2) - g / self.params['wy'], 1e-10))
        return omega_x, omega_y, omega_z

    def geometric_mean_freq(self, P_y, P_z):
        omega_x, omega_y, omega_z = self.trap_frequencies(P_y, P_z)
        return (omega_x * omega_y * omega_z)**(1/3)

    def peak_density(self, N, T, P_y, P_z):
        omega_mean = self.geometric_mean_freq(P_y, P_z)
        T = max(T, 1e-15)  # Avoid division by zero
        return N * (self.m * omega_mean**2 / (2 * np.pi * k * T))**(3/2)

    def psd(self, N, T, P_y, P_z):
        n = N / (4/3 * np.pi * np.prod(np.sqrt(2*k*T / (self.m * np.array([self.omega_x, self.omega_y, self.omega_z])**2))))
        lambda_dB = h / np.sqrt(2 * np.pi * self.m * k * max(T, 1e-15))
        return n * lambda_dB**3

    def collision_rate(self, N, T, P_y, P_z):
        n_peak = self.peak_density(N, T, P_y, P_z)
        T = max(T, 1e-15)  # Avoid division by zero
        v_thermal = np.sqrt(8 * k * T / (np.pi * self.m))
        return np.sqrt(2) * n_peak * self.a**2 * v_thermal

    def bec_fraction(self, N, T, P_y, P_z):
        if N <= 0 or T <= 0:
            return 0  # Return 0 for non-positive atom numbers or temperatures
        Tc = self.critical_temperature(N, P_y, P_z)
        if T >= Tc:
            return 0
        else:
            return 1 - (T / Tc)**3

    def critical_temperature(self, N, P_y, P_z):
        if N <= 0:
            return 0  # Return 0 for non-positive atom numbers
        omega_mean = self.geometric_mean_freq(P_y, P_z)
        return (hbar * omega_mean / k) * (N / zeta(3))**(1/3)

    def mot_cooling_rate(self, T):
        return (T - self.params['mot_final_temperature']) / self.params['mot_loading_time']

    def compression_cooling_rate(self, T):
        return (T - self.params['mot_final_temperature']) / self.params['mot_compression_time']


    def raman_cooling_rate(self, N, T, P_R, P_y, P_z, P_p, B_z, delta, dt):
        # Ensure N and T are not too small
        N = max(int(np.ceil(N)), 1)  # Ensure N is at least 1 and an integer
        T = max(T, 1e-9)  # Set a minimum temperature of 1 nK

        # Initialize atom positions and velocities
        self.initialize_atoms(N, T)

        # Raman transition probability
        k_eff = 2 * np.pi / self.wavelength
        Omega_R = np.sqrt(P_R / (h * self.params['Gamma']))  # Rabi frequency

        # Apply Raman cooling
        for i in range(N):
            v_z = self.velocities[i, 2]
            delta_eff = delta - k_eff * v_z  # Effective detuning including Doppler shift
            P_raman = (Omega_R**2 / 2) / (delta_eff**2 + Omega_R**2 / 2 + self.params['Gamma']**2 / 4) * dt
            
            if np.random.random() < P_raman:
                # Velocity change due to Raman transition
                delta_v = hbar * k_eff / self.m
                self.velocities[i, 2] -= delta_v
                
                # Optical pumping (three photon scattering events on average)
                for _ in range(3):
                    recoil_v = hbar * k_eff / self.m
                    self.velocities[i] += np.random.normal(0, recoil_v, 3)
    
        # Update positions and velocities
        self.positions += self.velocities * dt
        self.velocities += self.calculate_acceleration(B_z) * dt
    
        # Calculate cooling rate
        cooling_rate = np.sum(self.m * np.sum(self.velocities**2, axis=1)) / (3 * N * k)
    
        return cooling_rate

    def calculate_acceleration(self, B_z):
        # Calculate acceleration due to trap potential and magnetic field
        ax = -self.omega_x**2 * self.positions[:, 0]
        ay = -self.omega_y**2 * self.positions[:, 1]
        az = -self.omega_z**2 * self.positions[:, 2] + 9.8  # Include gravity
        
        # Add magnetic field gradient effect (simplified)
        mu_B = 9.274e-24  # Bohr magneton
        dB_dz = 0.1  # Tesla/m, adjust as needed
        a_B = mu_B * dB_dz / self.m
        az += a_B * B_z
        
        return np.column_stack((ax, ay, az))

    def raman_cooling_efficiency(self, T, P_R, P_p):
        base_efficiency = self.params['raman_cooling_efficiency']
        T_scale = 1e-6  # Temperature scale (1 µK)
        P_scale = 0.01  # Power scale (10 mW)
        
        T_factor = 1 / (1 + np.exp(-(T - T_scale) / (T_scale / 10)))
        P_factor = 1 / (1 + np.exp((P_R + P_p - P_scale) / (P_scale / 10)))
        
        return base_efficiency * T_factor * P_factor

    def heating_rate(self, N, T, P_y, P_z, P_R, P_p, delta):
        nu_c = self.collision_rate(N, T, P_y, P_z)
        recoil_energy = h * 3.6e3  # Recoil energy in Hz
        photon_reabsorption_heating = nu_c * recoil_energy * self.wavelength**2 / (2 * np.pi)
        
        # Three-body recombination heating
        n_peak = self.peak_density(N, T, P_y, P_z)
        K3 = self.params['K_3']
        E_bind = h * 300e9  # Approximate binding energy of Rb2
        three_body_heating = K3 * n_peak**2 * E_bind / 6
        
        # Off-resonant light scattering heating
        detuning = 2 * np.pi * 4.33e9
        Gamma_sc = P_p * self.params['Gamma_OP_0']
        off_resonant_heating = (P_R + P_p) * (self.wavelength / (2 * np.pi))**3 * Gamma_sc**2 / (8 * delta**2) * n_peak * recoil_energy
        
        # Technical heating (e.g., trap depth fluctuations)
        trap_depth = self.trap_depth(P_y, P_z)
        technical_heating_rate = 1e-3 * trap_depth * k  # Assuming 0.1% fluctuations in trap depth
        
        total_heating_rate = (photon_reabsorption_heating + three_body_heating + 
                              off_resonant_heating + technical_heating_rate) / (n_peak * k)
        
        return total_heating_rate

    def light_assisted_loss_rate(self, N, T, P_y, P_z, P_p):
        n_peak = self.peak_density(N, T, P_y, P_z)
        beta = 5e-14 * P_p  # Adjusted coefficient based on optical pumping power
        return beta * n_peak

    def three_body_loss_rate(self, N, T, P_y, P_z):
        n_peak = self.peak_density(N, T, P_y, P_z)
        return min(self.params['K_3'] * n_peak**2, 1e3)  # Further reduce maximum loss rate

    def evaporation_rate(self, N, T, P_y, P_z, t):
        eta = self.trap_depth(P_y, P_z) / (k * T)
        collision_rate = self.collision_rate(N, T, P_y, P_z)
        
        evap_rate = collision_rate * eta * np.exp(-eta) * (eta - 4) / (eta - 5)
        
        efficiency = self.evaporation_efficiency(t)
        
        return evap_rate * efficiency

    def evaporation_efficiency(self, t):
        base_efficiency = self.params['evap_efficiency']
        t_raman_end = self.stages[7]['end_time']  # End of Raman5 stage
        t_scale = 30e-3  # Time scale (30 ms)
        
        t_factor = 1 / (1 + np.exp(-(t - t_raman_end) / t_scale))
        
        return base_efficiency * (1 + 0.5 * t_factor)

    def trap_depth(self, P_y, P_z):
        U_y = 4 * P_y / (np.pi * self.params['wx']**2)
        U_z = 4 * P_z / (np.pi * self.params['wy']**2)
        return min(U_y, U_z)

    def system_evolution(self, t, state):
        
        N, T = state
        N = max(N, 1e-10)
        T = max(T, self.params['min_temperature'])

        dt = 1e-4  # Time step for rate calculations

        stage = next((name for start, end, name in self.stage_boundaries if start <= t < end), self.stages[-1]['name'])
        params = self.interpolate_params(t)

        P_y, P_z, B_z, P_p, P_R = [params.get(key, 0) for key in ['P_y', 'P_z', 'B_z', 'P_p', 'P_R']]
        self.update_trap_frequencies(P_y, P_y, P_z)

        if int(t * 1e5) % 100 == 0:
            print(f"t: {t*1e3:.2f} ms, Stage: {stage}, N: {N:.2e}, T: {T*1e6:.2f} µK")  # Debug print

        if stage == 'MOTLoading':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = -self.mot_cooling_rate(T)
        elif stage == 'MOTCompression':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = -self.compression_cooling_rate(T)
        elif stage == 'MagneticFieldAdjustment':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = 0
        elif stage.startswith('Raman'):
            delta = 2 * np.pi * 4.33e9  # Detuning from paper
            gamma_cool = self.raman_cooling_rate(N, T, P_R, P_y, P_z, P_p, B_z, delta, dt)
            gamma_heat = self.heating_rate(N, T, P_y, P_z, P_R, P_p, delta)
            gamma_light_loss = self.light_assisted_loss_rate(N, T, P_y, P_z, P_p)
            gamma_3body = self.three_body_loss_rate(N, T, P_y, P_z)
            dNdt = -self.params['gamma_bg'] * N - gamma_light_loss * N - gamma_3body * N**2
            dNdt = max(dNdt, -N / (1e-6))  # Limit atom loss rate

            dTdt = -gamma_cool * T + gamma_heat
            dTdt = max(min(dTdt, T / (1e-6)), -T / (1e-6))  # Limit cooling and heating rates
            print("gamma_cool: ", gamma_cool)
            print("-gamma_cool * T: ", -gamma_cool * T)
            print("gamma_heat: ", gamma_heat)
        elif stage.startswith('Evap'):
            gamma_evap = self.evaporation_rate(N, T, P_y, P_z, t)
            gamma_3body = self.three_body_loss_rate(N, T, P_y, P_z)
            dNdt = -self.params['gamma_bg'] * N - gamma_evap * N - gamma_3body * N**2
            dNdt = max(dNdt, -N / (1e-6))  # Limit atom loss rate
            dTdt = max(min(-(self.params['tilt_factor'] - 2) * gamma_evap * T, T / (1e-6)), -T / (1e-6))  # Limit cooling rate
        else:
            dNdt = -self.params['gamma_bg'] * N
            dTdt = 0

        return [dNdt, dTdt]

    def calculate_cooling_efficiency(self, results, start_idx, end_idx):
        start_psd = results[start_idx]['PSD']
        end_psd = results[end_idx]['PSD']
        start_N = results[start_idx]['N']
        end_N = results[end_idx]['N']
        
        # Check for valid PSD and N values
        if start_psd <= 0 or end_psd <= 0 or start_N <= 0 or end_N <= 0:
            return float('inf')  # Return infinity for invalid cases
        
        log_psd_ratio = np.log(end_psd / start_psd)
        log_N_ratio = np.log(end_N / start_N)
        
        if log_N_ratio == 0:
            return float('inf')  # Avoid division by zero
        
        return -log_psd_ratio / log_N_ratio

    def analyze_cooling_stages(self, results):
        stage_boundaries = []
        
        try:
            raman_index = next(i for i, r in enumerate(results) if r['stage'].startswith('Raman'))
            stage_boundaries.append(('Raman', raman_index))
        except StopIteration:
            print("Warning: Raman stage not reached in the simulation.")
        
        try:
            evap_index = next(i for i, r in enumerate(results) if r['stage'].startswith('Evap'))
            stage_boundaries.append(('Evaporation', evap_index))
        except StopIteration:
            print("Warning: Evaporation stage not reached in the simulation.")
        
        stage_boundaries.append(('End', len(results) - 1))
        
        analysis = []
        for i in range(len(stage_boundaries) - 1):
            start_stage, start_idx = stage_boundaries[i]
            end_stage, end_idx = stage_boundaries[i+1]
            
            efficiency = self.calculate_cooling_efficiency(results, start_idx, end_idx)
            start_T = results[start_idx]['T']
            end_T = results[end_idx]['T']
            start_N = results[start_idx]['N']
            end_N = results[end_idx]['N']
            start_PSD = results[start_idx]['PSD']
            end_PSD = results[end_idx]['PSD']
            
            print(f"Stage: {start_stage} to {end_stage}")
            print(f"  Start N: {start_N:.2e}, End N: {end_N:.2e}")
            print(f"  Start PSD: {start_PSD:.2e}, End PSD: {end_PSD:.2e}")
            print(f"  Cooling efficiency: {efficiency:.2f}")
            
            analysis.append({
                'stage': f"{start_stage} to {end_stage}",
                'cooling_efficiency': efficiency,
                'temperature_change': end_T - start_T,
                'atom_number_change': end_N - start_N,
                'psd_start': results[start_idx]['PSD'],
                'psd_end': results[end_idx]['PSD']
            })
        
        return analysis

    def plot_results(self, results):
        times = [r['time'] * 1e3 for r in results]  # Convert to ms
        N_values = [r['N'] for r in results]
        T_values = [r['T'] * 1e6 for r in results]  # Convert to µK
        PSD_values = [r['PSD'] for r in results]
        BEC_fraction_values = [r['BEC_fraction'] for r in results]

        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        axs[0].semilogy(times, N_values)
        axs[0].set_ylabel('Atom Number')

        axs[1].semilogy(times, T_values)
        axs[1].set_ylabel('Temperature (µK)')

        axs[2].semilogy(times, PSD_values)
        axs[2].set_ylabel('Phase Space Density')

        axs[3].plot(times, BEC_fraction_values)
        axs[3].set_ylabel('BEC Fraction')
        axs[3].set_xlabel('Time (ms)')

        for ax in axs:
            ax.grid(True, which="both", ls="-", alpha=0.2)

        for stage in self.stages:
            for ax in axs:
                ax.axvline(x=stage['start_time']*1e3, color='gray', linestyle='--', alpha=0.5)

        stages = [stage['name'] for stage in self.stages]
        unique_stages = []
        for stage in stages:
            if stage not in unique_stages:
                unique_stages.append(stage)

        plt.xticks([stage['start_time']*1e3 for stage in self.stages], unique_stages, rotation=45, ha='right')

        fig.suptitle('BEC Formation Simulation Results')
        plt.tight_layout()
        plt.savefig('bec_formation_results.png')
        plt.close()

    def plot_parameters(self, results):
        times = [r['time'] * 1e3 for r in results]  # Convert to ms
        P_p = [r['P_p'] for r in results]
        P_R = [r['P_R'] * 1e3 for r in results]  # Convert to mW
        P_y = [r['P_y'] for r in results]
        P_z = [r['P_z'] for r in results]
        B_z = [r['B_z'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.semilogy(times, P_p, label='P_p')
        ax1.semilogy(times, P_R, label='P_R (x10³)')
        ax1.semilogy(times, P_y, label='P_y')
        ax1.semilogy(times, P_z, label='P_z')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        ax2.plot(times, B_z)
        ax2.set_ylabel('B_z (G)')
        ax2.set_xlabel('Time (ms)')
        ax2.grid(True, which="both", ls="-", alpha=0.2)

        for stage in self.stages:
            ax1.axvline(x=stage['start_time']*1e3, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=stage['start_time']*1e3, color='gray', linestyle='--', alpha=0.5)

        stages = [stage['name'] for stage in self.stages]
        unique_stages = []
        for stage in stages:
            if stage not in unique_stages:
                unique_stages.append(stage)

        ax1.set_title('Parameter Evolution During BEC Formation')
        plt.xticks([stage['start_time']*1e3 for stage in self.stages], unique_stages, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('parameter_evolution.png')
        plt.close()

    def plot_collision_rate(self, results):
        times = [r['time'] * 1e3 for r in results]  # Convert to ms
        collision_rates = [r['nu_c'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.semilogy(times, collision_rates)
        plt.xlabel('Time (ms)')
        plt.ylabel('Collision Rate (Hz)')
        plt.title('Evolution of Collision Rate')
        plt.grid(True)
        plt.savefig('collision_rate_evolution.png')
        plt.close()

    def plot_psd_vs_atom_number(self, results):
        N_values = [r['N'] for r in results]
        PSD_values = [r['PSD'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.loglog(N_values, PSD_values)
        plt.xlabel('Atom Number')
        plt.ylabel('Phase Space Density')
        plt.title('Phase Space Density vs. Atom Number')
        plt.grid(True)
        plt.savefig('psd_vs_atom_number.png')
        plt.close()

    def plot_condensate_fraction(self, results):
        T_values = [r['T'] * 1e6 for r in results]  # Convert to µK
        BEC_fraction_values = [r['BEC_fraction'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(T_values, BEC_fraction_values)
        plt.xlabel('Temperature (µK)')
        plt.ylabel('Condensate Fraction')
        plt.title('Condensate Fraction vs. Temperature')
        plt.grid(True)
        plt.savefig('condensate_fraction_vs_temperature.png')
        plt.close()

    def print_stage_info(self, results):
        stages = set(r['stage'] for r in results)
        print("Stages reached in the simulation:")
        for stage in stages:
            stage_results = [r for r in results if r['stage'] == stage]
            start_time = stage_results[0]['time'] * 1e3  # Convert to ms
            end_time = stage_results[-1]['time'] * 1e3  # Convert to ms
            print(f"  {stage}: {start_time:.2f} ms to {end_time:.2f} ms")

    def run_simulation(self):
        t_span = (0, self.stages[-1]['end_time'])
        t_eval = np.linspace(0, t_span[1], 1000)

        print("Stage boundaries:")
        for start, end, name in self.stage_boundaries:
            print(f"{name}: {start*1e3:.2f} ms to {end*1e3:.2f} ms")

        sol = solve_ivp(
            self.system_evolution,
            t_span,
            [self.params['initial_N'], self.params['initial_T']],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            max_step=1e-4,
            first_step=1e-6  # Add this line to set a small initial step size
        )

        results = []
        for t, N, T in zip(sol.t, sol.y[0], sol.y[1]):
            stage = next((name for start, end, name in self.stage_boundaries if start <= t <= end), self.stages[-1]['name'])
            params = self.interpolate_params(t)
            
            psd = self.psd(N, T, params['P_y'], params['P_z'])
            bec_frac = self.bec_fraction(N, T, params['P_y'], params['P_z'])
            collision_rate = self.collision_rate(N, T, params['P_y'], params['P_z'])

            results.append({
                'time': t,
                'stage': stage,
                'N': N,
                'T': T,
                'P_p': params['P_p'],
                'P_R': params['P_R'],
                'P_y': params['P_y'],
                'P_z': params['P_z'],
                'B_z': params['B_z'],
                'PSD': psd,
                'BEC_fraction': bec_frac,
                'nu_c': collision_rate
            })

        self.plot_results(results)
        self.plot_parameters(results)
        self.plot_collision_rate(results)
        self.plot_psd_vs_atom_number(results)
        self.plot_condensate_fraction(results)

        self.print_stage_info(results)  # Add this line to print stage information

        analysis = self.analyze_cooling_stages(results)
        for stage_analysis in analysis:
            print(f"Stage: {stage_analysis['stage']}")
            print(f"  Cooling efficiency (γ): {stage_analysis['cooling_efficiency']:.2f}")
            print(f"  Temperature change: {stage_analysis['temperature_change']*1e6:.2f} µK")
            print(f"  Atom number change: {stage_analysis['atom_number_change']:.2e}")
            print(f"  PSD change: {stage_analysis['psd_start']:.2e} -> {stage_analysis['psd_end']:.2e}")
            print()

        return results

# Main execution
if __name__ == "__main__":
    sim = BECSimulation()
    results = sim.run_simulation()

    final_result = results[-1]
    print(f"Final atom number: {final_result['N']:.2e}")
    print(f"Final temperature: {final_result['T']*1e6:.2f} µK")
    print(f"Final BEC fraction: {final_result['BEC_fraction']:.2%}")
    print(f"Final phase space density: {final_result['PSD']:.2e}")

    # Compare with paper results
    paper_final_N = 2.8e3
    paper_final_BEC_fraction = 0.76

    N_difference = (final_result['N'] - paper_final_N) / paper_final_N * 100
    BEC_fraction_difference = (final_result['BEC_fraction'] - paper_final_BEC_fraction) / paper_final_BEC_fraction * 100

    print(f"\nComparison with paper results:")
    print(f"Final atom number: {final_result['N']:.2e} (Difference: {N_difference:.2f}%)")
    print(f"Final BEC fraction: {final_result['BEC_fraction']:.2%} (Difference: {BEC_fraction_difference:.2f}%)")
    print(f"Paper reported cooling slope γ: ~16")
    print(f"Our overall cooling slope γ: {sim.calculate_cooling_efficiency(results, 0, -1):.2f}")