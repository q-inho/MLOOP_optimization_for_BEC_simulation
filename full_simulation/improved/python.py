import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
from scipy.integrate import solve_ivp
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
        self.k_L = 2 * np.pi / self.wavelength

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
            {'name': 'Evap6', 'duration': 25e-3},  # Adjusted to reach 575 ms total
        ]:
            stage = stage_info.copy()
            stage['start_time'] = total_time
            total_time += stage['duration']
            stage['end_time'] = total_time
            self.stages.append(stage)

        self.params = {
            'initial_N': 2.7e5,
            'initial_T': 30e-6,
            'gamma_bg': 0.001,  # Further reduced background loss rate
            'wx': 18e-6,  # 18 µm horizontal beam waist
            'wy': 14e-6,  # 14 µm vertical beam waist
            'w_R': 500e-6,  # Raman beam waist
            'theta_R': np.pi/4,  # Angle between Raman beams
            'B_bias': 20.8e-4,  # Bias field (20.8 G) from paper
            'K2': 1e-20,  # Further reduced two-body loss coefficient
            'K3': 1e-42,  # Adjusted three-body loss coefficient
            'eta_load': 0.5,  # Trap loading efficiency
            'tilt_factor': 3.0,  # Increased tilt factor for more efficient evaporation
            'gray_molasses_duration': 1e-3,  # 1 ms gray molasses at the end of compression
            'gray_molasses_T_final': 35e-6,
            'intensity_noise': 1e-4,
            'anharmonicity': 1e-15,
            'phase_noise_amplitude': 0.1,
            'freq_noise_amplitude': 1e3,
            'Omega_R_0': 2 * np.pi * 50e3,
            'interaction_shift': -1.33,
            'atom_number_calibration': 1.0,
            'atom_number_uncertainty': 0.1,
            'B_field_fluctuation': 1e-7,
            'imaging_resolution': 5e-6,
            'raman_cooling_efficiency': 0.99,  # Increased Raman cooling efficiency
            'evap_efficiency': 0.95,  # Increased evaporation efficiency
            'min_temperature': 1e-9,  # 1 nK minimum temperature
        }

        for i, stage in enumerate(self.stages):
            if stage['name'].startswith('Raman'):
                self.params[f'{stage["name"]}_P_x'] = 1.0 - 0.02 * i
                self.params[f'{stage["name"]}_P_y'] = 1.0 - 0.02 * i
                self.params[f'{stage["name"]}_Omega_R'] = 2 * np.pi * (40e3 - 1e3 * i)  # Increased Rabi frequency
                self.params[f'{stage["name"]}_Gamma_OP'] = 2 * np.pi * (4e3 - 100 * i)  # Increased optical pumping rate
                self.params[f'{stage["name"]}_delta'] = 2 * np.pi * 4.33e9
            elif stage['name'].startswith('Evap'):
                self.params[f'{stage["name"]}_eta'] = 12 - 0.5 * i  # Adjusted eta values for more efficient evaporation
                self.params[f'{stage["name"]}_P_x'] = 1.0 - 0.15 * i
                self.params[f'{stage["name"]}_P_y'] = 1.0 - 0.18 * i

    def trap_frequencies(self, P_x, P_y, t):
        U_x = max(0, 1e-6 * P_x * (1 - 0.2 * t / self.stages[-1]['end_time']))
        U_y = max(0, 1e-6 * P_y * (1 - 0.2 * t / self.stages[-1]['end_time']))

        nu_x = max(1e-10, np.sqrt(4 * U_x / (self.m * self.params['wx']**2)) / (2 * np.pi))
        nu_y = max(1e-10, np.sqrt(4 * U_y / (self.m * self.params['wy']**2)) / (2 * np.pi))
        nu_z = max(1e-10, np.sqrt(2 * g / (self.params['wx'] + self.params['wy'])) / (2 * np.pi))

        nu_mean = np.mean([nu_x, nu_y, nu_z])
        target_freq = 185 + 15 * (1 - t / self.stages[-1]['end_time'])  # Dynamic adjustment of trap frequencies
        adjustment = target_freq / nu_mean
        return nu_x * adjustment, nu_y * adjustment, nu_z * adjustment

    def collision_rate(self, N, T, nu_x, nu_y, nu_z):
        nu_mean = max(np.mean([nu_x, nu_y, nu_z]), 1e-10)
        n0 = max(N * (self.m * nu_mean**2 / (2 * np.pi * k * max(T, 1e-10)))**(3/2), 1e-10)
        v_mean = np.sqrt(8 * k * max(T, 1e-10) / (np.pi * self.m))
        return 8 * np.pi * self.a**2 * n0 * v_mean

    def psd(self, N, T, nu_x, nu_y, nu_z):
        return max(N * (h * np.mean([nu_x, nu_y, nu_z]) / (k * max(T, 1e-10)))**3, 1e-10)

    def bec_fraction(self, N, T, nu_x, nu_y, nu_z):
        Tc = self.critical_temperature(N, nu_x, nu_y, nu_z)
        if T >= Tc:
            return 0
        else:
            fraction = 1 - (T / Tc)**3
            return max(0, min(fraction, 1)) * (1 - np.exp(-(Tc - T) / (0.1 * Tc)))  # Smoother transition

    def raman_cooling_rate(self, T, Omega_R, Gamma_OP, delta):
        recoil_energy = (h ** 2) / (2 * self.m * self.wavelength ** 2)
        eta = recoil_energy / (k * T)
        base_rate = 1e5 * eta * Omega_R**2 * Gamma_OP / (4 * delta**2) * np.sqrt(self.m / (2 * k * T)) * (1 + (T / (1e-6))**2)
        return base_rate * self.params['raman_cooling_efficiency']

    def heating_rate(self, N, T, nu_x, nu_y, nu_z):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        return 1.5 * h * nu_mean * self.collision_rate(N, T, nu_x, nu_y, nu_z) / (3 * k)

    def light_assisted_loss_rate(self, N, T, nu_x, nu_y, nu_z, Gamma_OP):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        n0 = N * (self.m * nu_mean**2 / (2 * np.pi * k * T))**(3/2)
        beta = 5e-15  # Further adjusted coefficient
        return beta * Gamma_OP * n0 * (T / (1e-6))**0.5

    def evaporation_rate(self, N, T, nu_x, nu_y, nu_z, eta, t):
        tilt_factor = self.params['tilt_factor'] * (1 + t / self.stages[-1]['end_time'])  # Increase tilt factor over time
        collision_rate = self.collision_rate(N, T, nu_x, nu_y, nu_z)
        trap_depth = eta * k * T
        evap_rate = tilt_factor * collision_rate * np.exp(-eta) * (eta - 4) / (eta - 5)
        return evap_rate * self.params['evap_efficiency'] * (1 + T / (1e-6)) * (1 + t / self.stages[-1]['end_time'])  # Time-dependent efficiency


    def mot_loading(self, N, T, t, stage_duration):
        N_final = 6e5  # Adjusted final atom number after MOT loading
        T_final = 100e-6  # Final temperature after MOT loading (estimate)
        N_new = N_final + (N - N_final) * np.exp(-3 * t / stage_duration)
        T_new = T_final + (T - T_final) * np.exp(-3 * t / stage_duration)
        return N_new, T_new

    def mot_compression(self, N, T, t, stage_duration):
        N_final = 2.7e5  # Final atom number after compression (from paper)
        T_final = 35e-6  # Temperature at the end of compression (from paper)
        N_new = N_final + (N - N_final) * np.exp(-5 * t / stage_duration)
        T_new = T_final + (T - T_final) * np.exp(-5 * t / stage_duration)
        return N_new, T_new

    def gray_molasses_cooling(self, T, t):
        T_final = self.params['gray_molasses_T_final']
        return T_final + (T - T_final) * np.exp(-10 * t / self.params['gray_molasses_duration'])

    def critical_temperature(self, N, nu_x, nu_y, nu_z):
        omega_mean = 2 * np.pi * np.mean([nu_x, nu_y, nu_z])
        N = max(N, 1e-10)  # Ensure N is not too small
        Tc_ideal = hbar * omega_mean * (N / zeta(3))**(1/3) / k
        
        # Finite size correction
        delta_Tc_fs = -0.73 * (omega_mean / (2*np.pi*max(nu_x, 1e-10)) + omega_mean / (2*np.pi*max(nu_y, 1e-10)) + omega_mean / (2*np.pi*max(nu_z, 1e-10))) * Tc_ideal / N**(1/3)
        
        # Interaction shift
        a_ho = np.sqrt(hbar / (self.m * omega_mean))
        delta_Tc_int = self.params['interaction_shift'] * Tc_ideal * (N**(1/6) * self.a / a_ho)
        
        Tc = max(Tc_ideal + delta_Tc_fs + delta_Tc_int, 1e-9)
        
        return Tc

    def system_evolution(self, t, state):
        N, T = state
        N = max(N, 1e-10)
        T = max(T, self.params['min_temperature'])

        stage = next((s for s in self.stages if s['start_time'] <= t < s['end_time']), self.stages[-1])

        # Initialize nu_x, nu_y, nu_z with default values
        nu_x = nu_y = nu_z = 1.0  # Default to 1 Hz if not set

        if stage['name'] == 'MOTLoading':
            N, T = self.mot_loading(N, T, t - stage['start_time'], stage['duration'])
            dNdt = (N - state[0]) / stage['duration']
            dTdt = (T - state[1]) / stage['duration']
        elif stage['name'] == 'MOTCompression':
            N, T = self.mot_compression(N, T, t - stage['start_time'], stage['duration'])
            dNdt = (N - state[0]) / stage['duration']
            dTdt = (T - state[1]) / stage['duration']
            
            # Apply gray molasses cooling at the end of compression
            if t > (stage['end_time'] - self.params['gray_molasses_duration']):
                T = self.gray_molasses_cooling(T, t - (stage['end_time'] - self.params['gray_molasses_duration']))
                dTdt = (T - state[1]) / self.params['gray_molasses_duration']
        elif stage['name'] == 'MagneticFieldAdjustment':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = 0  # Assume temperature doesn't change during this brief period
        else:
            P_x = self.params[f'{stage["name"]}_P_x'] if f'{stage["name"]}_P_x' in self.params else 1.0
            P_y = self.params[f'{stage["name"]}_P_y'] if f'{stage["name"]}_P_y' in self.params else 1.0
            nu_x, nu_y, nu_z = self.trap_frequencies(P_x, P_y, t)

            if stage['name'].startswith('Raman'):
                Omega_R = self.params[f'{stage["name"]}_Omega_R']
                Gamma_OP = self.params[f'{stage["name"]}_Gamma_OP']
                delta = self.params[f'{stage["name"]}_delta']
                gamma_cool = self.raman_cooling_rate(T, Omega_R, Gamma_OP, delta)
                gamma_heat = self.heating_rate(N, T, nu_x, nu_y, nu_z)
                dNdt = -self.params['gamma_bg'] * N - self.light_assisted_loss_rate(N, T, nu_x, nu_y, nu_z, Gamma_OP)
                dTdt = -gamma_cool * T + gamma_heat
            elif stage['name'].startswith('Evap'):
                eta = self.params[f'{stage["name"]}_eta']
                gamma_evap = self.evaporation_rate(N, T, nu_x, nu_y, nu_z, eta, t)
                dNdt = -self.params['gamma_bg'] * N - gamma_evap * N
                dTdt = -(eta - 3) / 3 * gamma_evap * T

        # Add three-body loss
        n0 = N * (self.m * np.mean([nu_x, nu_y, nu_z])**2 / (2 * np.pi * k * T))**(3/2)
        dNdt -= self.params['K3'] * n0**2 * N

        # Enhance cooling and evaporation effects
        dTdt *= 1.5
        dNdt *= 1.1

        # Prevent temperature from going below minimum
        if T + dTdt * 1e-3 < self.params['min_temperature']:
            dTdt = (self.params['min_temperature'] - T) / 1e-3

        psd = self.psd(N, T, nu_x, nu_y, nu_z)
        bec_frac = self.bec_fraction(N, T, nu_x, nu_y, nu_z)

        print(f"Time: {t*1e3:.1f} ms, Stage: {stage['name']}, T: {T*1e6:.2f} µK, N: {N:.2e}, PSD: {psd:.2e}, BEC fraction: {bec_frac:.2%}")

        return [dNdt, dTdt]

    def calculate_cooling_slope(self, results):
        if len(results) < 3:
            return float('nan')
        log_psd = np.log10([max(r['PSD'], 1e-10) for r in results])
        log_N = np.log10([max(r['N'], 1e-10) for r in results])
        weights = np.array([r['N'] for r in results])  # Use atom number as weight

        # Remove any non-finite values
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

    def run_simulation(self):
        t_span = (0, self.stages[-1]['end_time'])
        t_eval = np.linspace(0, t_span[1], 1000)

        def event_stage_change(t, y):
            for stage in self.stages:
                if stage['end_time'] > t:
                    return t - stage['end_time']
            return 0

        event_stage_change.terminal = True
        event_stage_change.direction = 1

        sol = solve_ivp(
            self.system_evolution,
            t_span,
            [self.params['initial_N'], self.params['initial_T']],
            t_eval=t_eval,
            method='RK45',
            events=event_stage_change,
            rtol=1e-8,
            atol=1e-8
        )

        results = []
        for t, N, T in zip(sol.t, sol.y[0], sol.y[1]):
            stage = next((s for s in self.stages if s['start_time'] <= t <= s['end_time']), self.stages[-1])
            P_x = self.params[f'{stage["name"]}_P_x'] if f'{stage["name"]}_P_x' in self.params else 1.0
            P_y = self.params[f'{stage["name"]}_P_y'] if f'{stage["name"]}_P_y' in self.params else 1.0
            nu_x, nu_y, nu_z = self.trap_frequencies(P_x, P_y, t)
            nu_c = self.collision_rate(N, T, nu_x, nu_y, nu_z)
            psd = self.psd(N, T, nu_x, nu_y, nu_z)
            bec_frac = self.bec_fraction(N, T, nu_x, nu_y, nu_z)

            results.append({
                'time': t,
                'stage': stage['name'],
                'N': N,
                'T': T,
                'PSD': psd,
                'nu_x': nu_x,
                'nu_y': nu_y,
                'nu_z': nu_z,
                'nu_c': nu_c,
                'BEC_fraction': bec_frac
            })

        return results

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
        nu_x, nu_y, nu_z = result['nu_x'], result['nu_y'], result['nu_z']
        bec_fraction = result['BEC_fraction']

        nu_x = max(nu_x, 1e-10)
        nu_y = max(nu_y, 1e-10)

        epsilon = 1e-10
        sigma_x = np.sqrt(max(k * T / (self.sim.m * (2*np.pi*nu_x)**2), epsilon) + (h * nu_x / (2*np.pi*self.sim.m))**2 * self.tof**2)
        sigma_y = np.sqrt(max(k * T / (self.sim.m * (2*np.pi*nu_y)**2), epsilon) + (h * nu_y / (2*np.pi*self.sim.m))**2 * self.tof**2)

        n_thermal = N * (1 - bec_fraction) * np.exp(-(X**2 / (2*sigma_x**2) + Y**2 / (2*sigma_y**2))) / (2*np.pi*sigma_x*sigma_y)
        
        if bec_fraction > 0:
            R_x = np.sqrt(2 * h * nu_x * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_x)**2))
            R_y = np.sqrt(2 * h * nu_y * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_y)**2))
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

# Main script
if __name__ == "__main__":
    # Run simulation and create visualization
    sim = BECSimulation()
    results = sim.run_simulation()

    visualizer = BECVisualizer(sim)
    visualizer.create_animation(results, 'bec_evolution.mp4')

    print("Animation creation complete. Check 'bec_evolution.mp4'.")

    # Plot additional results
    times = [result['time'] * 1e3 for result in results]  # Convert to ms
    N_values = [result['N'] for result in results]
    T_values = [result['T'] for result in results]
    PSD_values = [result['PSD'] for result in results]
    BEC_fraction_values = [result['BEC_fraction'] for result in results]

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.semilogy(times, N_values)
    plt.ylabel('Atom Number')
    plt.title('BEC Formation Simulation Results')

    plt.subplot(4, 1, 2)
    plt.semilogy(times, T_values)
    plt.ylabel('Temperature (K)')

    plt.subplot(4, 1, 3)
    plt.semilogy(times, PSD_values)
    plt.ylabel('Phase Space Density')

    plt.subplot(4, 1, 4)
    plt.plot(times, BEC_fraction_values)
    plt.ylabel('BEC Fraction')
    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig('bec_formation_results.png')
    plt.close()

    print("Additional results plot saved as 'bec_formation_results.png'.")

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

    # Plot cooling process
    plt.figure(figsize=(12, 8))
    for stage in sim.stages:
        stage_results = [r for r in results if r['stage'] == stage['name']]
        if stage_results:
            log_psd = np.log10([max(r['PSD'], 1e-10) for r in stage_results])
            log_N = np.log10([max(r['N'], 1e-10) for r in stage_results])
            plt.plot(log_N, log_psd, 'o-', label=stage['name'])

    plt.xlabel('log(N)')
    plt.ylabel('log(PSD)')
    plt.legend()
    plt.title('Cooling Process: PSD vs N')
    plt.savefig('cooling_process.png')
    plt.close()

    print("Cooling process plot saved as 'cooling_process.png'.")

    # Additional analysis: Trap frequencies and collision rates
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(times, [r['nu_x'] for r in results], label='νx')
    plt.plot(times, [r['nu_y'] for r in results], label='νy')
    plt.plot(times, [r['nu_z'] for r in results], label='νz')
    plt.ylabel('Trap Frequency (Hz)')
    plt.legend()
    plt.title('Trap Frequencies and Collision Rates')

    plt.subplot(2, 1, 2)
    plt.plot(times, [r['nu_c'] for r in results], label='Collision Rate')
    plt.plot(times, [np.mean([r['nu_x'], r['nu_y'], r['nu_z']]) for r in results], label='Mean Trap Frequency')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('trap_and_collision_rates.png')
    plt.close()

    print("Trap frequencies and collision rates plot saved as 'trap_and_collision_rates.png'.")

    # Analyze collision rate vs trap frequency
    collision_rates = [r['nu_c'] for r in results]
    mean_trap_freqs = [np.mean([r['nu_x'], r['nu_y'], r['nu_z']]) for r in results]
    ratio = [c / f if f != 0 else 0 for c, f in zip(collision_rates, mean_trap_freqs)]

    plt.figure(figsize=(10, 6))
    plt.plot(times, ratio)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.ylabel('Collision Rate / Mean Trap Frequency')
    plt.xlabel('Time (ms)')
    plt.title('Ratio of Collision Rate to Mean Trap Frequency')
    plt.savefig('collision_rate_ratio.png')
    plt.close()

    print("Collision rate ratio plot saved as 'collision_rate_ratio.png'.")

    # Print some statistics about the ratio
    print(f"Mean ratio of collision rate to trap frequency: {np.mean(ratio):.2f}")
    print(f"Maximum ratio: {np.max(ratio):.2f}")
    print(f"Minimum ratio: {np.min(ratio):.2f}")

    # Analyze cooling efficiency
    cooling_efficiency = []
    for i in range(1, len(results)):
        dN = results[i]['N'] - results[i-1]['N']
        dPSD = results[i]['PSD'] - results[i-1]['PSD']
        if dN != 0 and results[i-1]['N'] != 0 and results[i-1]['PSD'] > 0 and results[i]['PSD'] > 0:
            efficiency = (np.log10(results[i]['PSD']) - np.log10(results[i-1]['PSD'])) / (np.log10(results[i-1]['N']) - np.log10(results[i]['N']))
            cooling_efficiency.append(efficiency)

    plt.figure(figsize=(10, 6))
    plt.plot(times[1:len(cooling_efficiency)+1], cooling_efficiency)
    plt.ylabel('Cooling Efficiency')
    plt.xlabel('Time (ms)')
    plt.title('Cooling Efficiency over Time')
    plt.savefig('cooling_efficiency.png')
    plt.close()

    print("Cooling efficiency plot saved as 'cooling_efficiency.png'.")

    # Analyze PSD increase rate
    psd_increase_rate = []
    for i in range(1, len(results)):
        dt = results[i]['time'] - results[i-1]['time']
        if dt > 0:
            rate = (np.log10(results[i]['PSD']) - np.log10(results[i-1]['PSD'])) / dt
            psd_increase_rate.append(rate)

    plt.figure(figsize=(10, 6))
    plt.plot(times[1:len(psd_increase_rate)+1], psd_increase_rate)
    plt.ylabel('PSD Increase Rate (log10(PSD)/s)')
    plt.xlabel('Time (ms)')
    plt.title('PSD Increase Rate over Time')
    plt.savefig('psd_increase_rate.png')
    plt.close()

    print("PSD increase rate plot saved as 'psd_increase_rate.png'.")

    print("Simulation and analysis complete.")