import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
from scipy.integrate import solve_ivp, nquad
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn.linear_model import RANSACRegressor
import warnings
from scipy.constants import h, hbar, k, atomic_mass, mu_0, epsilon_0, g
from scipy.special import zeta

class BECSimulation:
    def __init__(self):
        self.m = 87 * atomic_mass  # Mass of Rb-87 atom in kg
        self.a = 5.77e-9  # Scattering length for Rb-87
        self.wavelength = 780e-9  # Wavelength of cooling light
        self.trap_wavelength = 1064e-9  # Wavelength of trapping light
        self.k_L = 2 * np.pi / self.wavelength
        self.mu_B = 9.274e-24  # Bohr magneton

        self.params = {
            'initial_N': 2.7e5,  # Initial atom number from paper
            'initial_T': 50e-6,  # Initial temperature before MOT loading
            'gamma_bg': 0.05,  # Background loss rate
            'wx': 18e-6,  # 18 μm horizontal beam waist
            'wy': 14e-6,  # 14 μm vertical beam waist
            'w_R': 500e-6,  # Raman beam waist
            'theta_R': np.pi/4,  # Angle between Raman beams
            'raman_cooling_efficiency': 0.5,
            'evap_efficiency': 0.98,
            'min_temperature': 50e-9,  # 50 nK minimum temperature
            'Omega_R_0': 2 * np.pi * 50e3,  # Base Raman Rabi frequency
            'Gamma_OP_0': 2 * np.pi * 2e3,  # Base optical pumping rate
            'interaction_shift': -1.33,
            'tilt_factor': 2.0,  # Tilt factor for evaporation from paper
            'mot_cooling_rate': 1e3,
            'mot_final_temperature': 20e-6,
            'compression_cooling_rate': 1e3,
            'mot_loading_time': 89e-3,  # 89 ms MOT loading time
            'mot_compression_time': 10e-3,  # 10 ms compression time
        }

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

        self.stages = []
        total_time = 0
        for stage_info in [
            {'name': 'MOTLoading', 'duration': self.params['mot_loading_time']},
            {'name': 'MOTCompression', 'duration': self.params['mot_compression_time']},
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

    def calculate_trap_potential(self, x, y, z, P_y, P_z):
        U_y = self.gaussian_beam_potential(y, z, P_y, self.params['wx'])
        U_z = self.gaussian_beam_potential(x, y, P_z, self.params['wy'])
        return U_y + U_z

    def gaussian_beam_potential(self, r, z, P, w0):
        z_R = np.pi * w0**2 / self.trap_wavelength
        w = w0 * np.sqrt(1 + (z / z_R)**2)
        U_0 = 4 * P / (np.pi * w0**2)
        return U_0 * (w0 / w)**2 * np.exp(-2 * r**2 / w**2)

    def boltzmann_factor(self, x, y, z, T, P_y, P_z):
        U = self.calculate_trap_potential(x, y, z, P_y, P_z)
        return np.exp(-U / (k * T))

    def partition_function(self, T, P_y, P_z):
        def integrand(x, y, z):
            return self.boltzmann_factor(x, y, z, T, P_y, P_z)
        
        limit = 1e-3  # 1 mm
        ranges = [[-limit, limit], [-limit, limit], [-limit, limit]]
        
        result, _ = nquad(integrand, ranges)
        return result

    def peak_density(self, N, T, P_y, P_z):
        Z = self.partition_function(T, P_y, P_z)
        x0, y0, z0 = 0, 0, 0  # Assume trap bottom at origin
        n_cp = N * self.boltzmann_factor(x0, y0, z0, T, P_y, P_z) / Z
        return n_cp

    def psd(self, N, T, P_y, P_z):
        n_cp = self.peak_density(N, T, P_y, P_z)
        lambda_dB = h / np.sqrt(2 * np.pi * self.m * k * T)
        return n_cp * lambda_dB**3

    def collision_rate(self, N, T, P_y, P_z):
        sigma = 8 * np.pi * self.a**2
        v_rms = np.sqrt(6 * k * T / self.m)
        
        def integrand(x, y, z):
            f_B = self.boltzmann_factor(x, y, z, T, P_y, P_z)
            Z = self.partition_function(T, P_y, P_z)
            return (f_B / Z)**2
        
        limit = 1e-3  # 1 mm
        ranges = [[-limit, limit], [-limit, limit], [-limit, limit]]
        
        integral, _ = nquad(integrand, ranges)
        
        return N * sigma * v_rms * integral

    def bec_fraction(self, N, T, P_y, P_z):
        Tc = self.critical_temperature(N, P_y, P_z)
        if T >= Tc:
            return 0
        else:
            fraction = 1 - (T / Tc)**3
            return max(0, min(fraction, 1))

    def mot_cooling_rate(self, T):
        return (T - self.params['mot_final_temperature']) / self.params['mot_loading_time']

    def mot_psd(self, N, T):
        mot_volume = (1e-3)**3  # Assume 1 mm^3 MOT volume
        n = N / mot_volume
        T = max(T, 1e-10)  # Prevent division by zero
        return n * (h**2 / (2 * np.pi * self.m * k * T))**(3/2)

    def compression_cooling_rate(self, T):
        return (T - self.params['mot_final_temperature']) / self.params['mot_compression_time']
    
    def raman_cooling_rate(self, T, P_R, P_p, B_z, delta):
        Omega_R = np.sqrt(P_R) * self.params['Omega_R_0']
        Gamma_OP = P_p * self.params['Gamma_OP_0']
        recoil_energy = (h ** 2) / (2 * self.m * self.trap_wavelength ** 2)
        eta = recoil_energy / (k * T)

        detuning = 2 * np.pi * 4.33e9 + (self.mu_B * 1e-4 * B_z) / hbar
    
        Omega_eff = Omega_R**2 / (2 * detuning)
    
        Gamma_eff = Gamma_OP * Omega_eff**2 / (Gamma_OP**2 / 4 + delta**2 + 2 * Omega_eff**2)
    
        cooling_rate = Gamma_eff * eta * recoil_energy / hbar
    
        return cooling_rate * self.params['raman_cooling_efficiency']

    def heating_rate(self, N, T, P_y, P_z):
        nu_c = self.collision_rate(N, T, P_y, P_z)
        return h * nu_c / (3 * k)  # Reduced heating

    def light_assisted_loss_rate(self, N, T, P_y, P_z, P_p):
        n_cp = self.peak_density(N, T, P_y, P_z)
        beta = 5e-15 * P_p  # Adjusted coefficient based on optical pumping power
        return beta * n_cp * (T / (1e-6))**0.5

    def evaporation_rate(self, N, T, P_y, P_z, t):
        tilt_factor = self.params['tilt_factor'] * (1 + 2 * t / self.stages[-1]['end_time'])
        collision_rate = self.collision_rate(N, T, P_y, P_z)
        eta = 10 - 4 * t / self.stages[-1]['end_time']
        evap_rate = tilt_factor * collision_rate * np.exp(-eta) * (eta - 4) / (eta - 5)
        return evap_rate * self.params['evap_efficiency'] * (P_y / P_z)**0.5

    def critical_temperature(self, N, P_y, P_z):
        def geometric_mean_freq(P_y, P_z):
            U_y = 4 * P_y / (np.pi * self.params['wx']**2)
            U_z = 4 * P_z / (np.pi * self.params['wy']**2)
            nu_y = np.sqrt(4 * U_y / (self.m * self.params['wx']**2)) / (2 * np.pi)
            nu_z = np.sqrt(4 * U_z / (self.m * self.params['wy']**2)) / (2 * np.pi)
            nu_x = np.sqrt(2 * g / (self.params['wx'] + self.params['wy'])) / (2 * np.pi)
            return (nu_x * nu_y * nu_z)**(1/3)

        omega_mean = 2 * np.pi * geometric_mean_freq(P_y, P_z)
        N = max(N, 1e-10)  # Ensure N is not too small
        Tc_ideal = hbar * omega_mean * (N / zeta(3))**(1/3) / k
        
        # Finite size correction
        delta_Tc_fs = -0.73 * omega_mean * Tc_ideal / (2*np.pi*N**(1/3))
        
        # Interaction shift
        epsilon = 1e-10  # Small value to prevent division by zero
        a_ho = np.sqrt(hbar / (self.m * max(omega_mean, epsilon)))
        delta_Tc_int = self.params['interaction_shift'] * Tc_ideal * (N**(1/6) * self.a / a_ho)
        
        Tc = max(Tc_ideal + delta_Tc_fs + delta_Tc_int, 1e-9)
        
        return Tc

    def system_evolution(self, t, state):
        N, T = state
        N = max(N, 1e-10)
        T = max(T, self.params['min_temperature'])

        stage = next((name for start, end, name in self.stage_boundaries if start <= t < end), self.stages[-1]['name'])
        params = self.interpolate_params(t)

        P_y, P_z, B_z, P_p, P_R = [params.get(key, 0) for key in ['P_y', 'P_z', 'B_z', 'P_p', 'P_R']]

        if stage == 'MOTLoading':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = -self.mot_cooling_rate(T)
        elif stage == 'MOTCompression':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = -self.compression_cooling_rate(T)
        elif stage.startswith('Raman') or stage == 'Evap1':
            delta = 2 * np.pi * 4.33e9  # Detuning from paper
            gamma_cool = self.raman_cooling_rate(T, P_R, P_p, B_z, delta)
            gamma_heat = self.heating_rate(N, T, P_y, P_z)
            dNdt = -self.params['gamma_bg'] * N - self.light_assisted_loss_rate(N, T, P_y, P_z, P_p)
            dTdt = -gamma_cool * T + gamma_heat
        elif stage.startswith('Evap'):
            gamma_evap = self.evaporation_rate(N, T, P_y, P_z, t)
            dNdt = -self.params['gamma_bg'] * N - gamma_evap * N
            dTdt = -(self.params['tilt_factor'] - 2) * gamma_evap * T
        else:
            dNdt = -self.params['gamma_bg'] * N
            dTdt = 0

        try:
            psd = self.psd(N, T, P_y, P_z)
            bec_frac = self.bec_fraction(N, T, P_y, P_z)
        except Exception as e:
            print(f"Error calculating PSD or BEC fraction: {e}")
            psd = 0
            bec_frac = 0

        print(f"Time: {t*1e3:.1f} ms, Stage: {stage}, T: {T*1e6:.2f} µK, N: {N:.2e}, PSD: {psd:.2e}, BEC fraction: {bec_frac:.2%}")

        return [dNdt, dTdt]

    def run_simulation(self):
        t_span = (0, self.stages[-1]['end_time'])
        t_eval = np.linspace(0, t_span[1], 1000)

        sol = solve_ivp(
            self.system_evolution,
            t_span,
            [self.params['initial_N'], self.params['initial_T']],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-6
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

        return results

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

    def calculate_cooling_slope(self, results):
        if len(results) < 3:
            return float('nan')
        log_psd = np.log10([max(r['PSD'], 1e-10) for r in results])
        log_N = np.log10([max(r['N'], 1e-10) for r in results])
        weights = np.array([r['N'] for r in results])  # Use atom number as weight

        mask = np.isfinite(log_psd) & np.isfinite(log_N) & np.isfinite(weights)
        log_psd = log_psd[mask]
        log_N = log_N[mask]
        weights = weights[mask]

        if len(log_psd) < 3:
            return float('nan')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                slope, _ = np.polyfit(log_N, log_psd, 1, w=weights)
            return -slope
        except (ValueError, np.linalg.LinAlgError):
            return float('nan')

class BECVisualizer:
    def __init__(self, simulation):
        self.sim = simulation
        self.imaging_resolution = 5e-6  # 5 micrometers
        self.tof = 20e-3  # 20 ms time-of-flight
        self.vmin = None
        self.vmax = None
        self.xlim = (-1000e-6, 1000e-6)  # ±1000 µm x-axis range
        self.ylim = (-1000e-6, 1000e-6)  # ±1000 µm y-axis range

    def calculate_density_range(self, results):
        densities = []
        for result in results:
            if result['stage'] != 'MOTLoading' and result['stage'] != 'MOTCompression':  # Only consider frames after MOT stages
                frame_data = self.generate_frame(result)
                densities.extend(frame_data.flatten())
        self.vmin = np.min(densities)
        self.vmax = np.max(densities)

    def generate_frame(self, result, grid_size=200):
        x = np.linspace(self.xlim[0], self.xlim[1], grid_size)
        y = np.linspace(self.ylim[0], self.ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)

        N, T = result['N'], result['T']
        P_y, P_z = result['P_y'], result['P_z']
        bec_fraction = result['BEC_fraction']

        epsilon = 1e-10
        sigma_x = np.sqrt(max(k * T / (self.sim.m * (2*np.pi*self.sim.collision_rate(N, T, P_y, P_z))**2), epsilon) + (h * self.sim.collision_rate(N, T, P_y, P_z) / (2*np.pi*self.sim.m))**2 * self.tof**2)
        sigma_y = sigma_x  # Assuming isotropic expansion

        n_thermal = N * (1 - bec_fraction) * np.exp(-(X**2 / (2*sigma_x**2) + Y**2 / (2*sigma_y**2))) / (2*np.pi*sigma_x*sigma_y)
        
        if bec_fraction > 0:
            R_x = np.sqrt(2 * h * self.sim.collision_rate(N, T, P_y, P_z) * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*self.sim.collision_rate(N, T, P_y, P_z))**2))
            R_y = R_x  # Assuming isotropic expansion
            n_bec = N * bec_fraction * np.maximum(0, 1 - X**2/R_x**2 - Y**2/R_y**2)**(3/2) / (4*np.pi*R_x*R_y/3)
        else:
            n_bec = np.zeros_like(X)

        n_total = n_thermal + n_bec

        # Calculate column density (integrate along z-axis)
        column_density = n_total * np.sqrt(2 * np.pi) * sigma_y

        # Calculate optical depth
        cross_section = 3 * (780e-9)**2 / (2 * np.pi)  # Absorption cross-section for Rb87
        od = column_density * cross_section

        # Add realistic noise
        od_with_noise = od + np.random.normal(0, 0.05 * np.max(od), od.shape)

        # Simulate finite imaging resolution
        od_blurred = gaussian_filter(od_with_noise, sigma=self.imaging_resolution / (self.xlim[1] - self.xlim[0]) * grid_size)

        od_blurred = np.clip(od_blurred, 1e-10, np.inf)

        return od_blurred

    def create_animation(self, results, output_filename):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate global vmin and vmax for frames after MOT stages
        self.calculate_density_range(results)
        
        def update(frame):
            ax.clear()
            result = results[frame]
            frame_data = self.generate_frame(result)
            
            if result['stage'] == 'MOTLoading' or result['stage'] == 'MOTCompression':
                im = ax.imshow(frame_data, cmap='jet', norm=LogNorm(), extent=[self.xlim[0]*1e6, self.xlim[1]*1e6, self.ylim[0]*1e6, self.ylim[1]*1e6])
            else:
                im = ax.imshow(frame_data, cmap='jet', norm=LogNorm(vmin=self.vmin, vmax=self.vmax), extent=[self.xlim[0]*1e6, self.xlim[1]*1e6, self.ylim[0]*1e6, self.ylim[1]*1e6])
            
            ax.set_title(f"Time: {result['time']*1e3:.1f} ms, Stage: {result['stage']}\n"
                         f"T: {result['T']*1e6:.2f} µK, N: {result['N']:.2e}, "
                         f"BEC fraction: {result['BEC_fraction']:.2%}")
            ax.set_xlabel('x (µm)')
            ax.set_ylabel('y (µm)')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(results), interval=200, blit=True)
        
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_filename, writer=writer)
        
        plt.close(fig)

# Main execution
if __name__ == "__main__":
    sim = BECSimulation()
    results = sim.run_simulation()
    
    visualizer = BECVisualizer(sim)
    visualizer.create_animation(results, 'bec_evolution.mp4')

    sim.plot_parameters(results)
    sim.plot_results(results)

    print("Animation creation complete. Check 'bec_evolution.mp4'.")
    print("Parameter evolution plot saved as 'parameter_evolution.png'.")
    print("BEC formation results plot saved as 'bec_formation_results.png'.")

    # Print final results
    final_result = results[-1]
    print(f"Final atom number: {final_result['N']:.2e}")
    print(f"Final temperature: {final_result['T']*1e6:.2f} µK")
    print(f"Final BEC fraction: {final_result['BEC_fraction']:.2%}")
    print(f"Final phase space density: {final_result['PSD']:.2e}")

    # Calculate and print the cooling slope for different stages
    raman_start = next(r for r in results if r['stage'] == 'Raman1')
    raman_end = next(r for r in results if r['stage'] == 'Evap1')
    evap_start = raman_end
    evap_end = results[-1]

    raman_slope = sim.calculate_cooling_slope([r for r in results if raman_start['time'] <= r['time'] < raman_end['time']])
    evap_slope = sim.calculate_cooling_slope([r for r in results if evap_start['time'] <= r['time'] <= evap_end['time']])
    overall_slope = sim.calculate_cooling_slope(results)

    print(f"Raman cooling slope γ: {raman_slope:.2f}")
    print(f"Evaporative cooling slope γ: {evap_slope:.2f}")
    print(f"Overall cooling slope γ: {overall_slope:.2f}")

    # Compare with paper results
    paper_final_N = 2.8e3
    paper_final_BEC_fraction = 0.76

    N_difference = (final_result['N'] - paper_final_N) / paper_final_N * 100
    BEC_fraction_difference = (final_result['BEC_fraction'] - paper_final_BEC_fraction) / paper_final_BEC_fraction * 100

    print(f"Comparison with paper results:")
    print(f"Final atom number: {final_result['N']:.2e} (Difference: {N_difference:.2f}%)")
    print(f"Final BEC fraction: {final_result['BEC_fraction']:.2%} (Difference: {BEC_fraction_difference:.2f}%)")
    print(f"Paper reported cooling slope γ: ~16")
    print(f"Our overall cooling slope γ: {overall_slope:.2f}")

    # Define times for plotting
    times = [r['time'] * 1e3 for r in results]  # Convert to ms

    # Additional analysis: Collision rates
    plt.figure(figsize=(12, 8))
    plt.plot(times, [r['nu_c'] for r in results], label='Collision Rate')
    plt.ylabel('Collision Rate (Hz)')
    plt.xlabel('Time (ms)')
    plt.legend()
    plt.title('Collision Rate Evolution')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('collision_rate_evolution.png')
    plt.close()

    print("Collision rate evolution plot saved as 'collision_rate_evolution.png'.")

    print("Simulation and analysis complete.")