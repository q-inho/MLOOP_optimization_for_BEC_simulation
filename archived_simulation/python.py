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

class BECSimulation:
    def __init__(self):
        self.m = 1.44e-25  # Mass of Rb-87 atom in kg
        self.h = 6.626e-34  # Planck constant
        self.kb = 1.38e-23  # Boltzmann constant
        self.a = 5.77e-9  # Scattering length for Rb-87
        self.g = 9.81  # Gravitational acceleration

        self.stages = []
        total_time = 0
        for stage_info in [
            {'name': 'MOT', 'duration': 95e-3},
            {'name': 'GrayMolasses', 'duration': 4e-3},
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
            {'name': 'Evap6', 'duration': 27e-3},
        ]:
            stage = stage_info.copy()
            stage['start_time'] = total_time
            total_time += stage['duration']
            stage['end_time'] = total_time
            self.stages.append(stage)

        self.params = {
            'initial_N': 2.7e5,  # Initial atom number
            'initial_T': 200e-6,  # Initial temperature (200 µK)
            'gamma_bg': 0.05,  # Reduced background loss rate
            'wx': 18e-6,  # 18 µm horizontal beam waist
            'wy': 14e-6,  # 14 µm vertical beam waist
        }

        for i, stage in enumerate(self.stages):
            if stage['name'].startswith('Raman'):
                self.params[f'{stage["name"]}_P_x'] = 1.0 - 0.15 * i  # More gradual power decrease
                self.params[f'{stage["name"]}_P_y'] = 1.0 - 0.15 * i
                self.params[f'{stage["name"]}_Omega_R'] = 2 * np.pi * (15e3 - 1e3 * i)  # Decreasing Raman coupling rate
                self.params[f'{stage["name"]}_Gamma_OP'] = 2 * np.pi * (2e3 - 200 * i)  # Decreasing optical pumping rate
                self.params[f'{stage["name"]}_delta'] = 2 * np.pi * (150e3 + 10e3 * i)  # Increasing detuning
            elif stage['name'].startswith('Evap'):
                self.params[f'{stage["name"]}_eta'] = 7 - 0.3 * i  # More gradual eta decrease
                self.params[f'{stage["name"]}_P_x'] = 1.0 - 0.15 * i  # Horizontal beam power ramp
                self.params[f'{stage["name"]}_P_y'] = 1.0 - 0.20 * i  # Vertical beam power ramp

    def trap_frequencies(self, P_x, P_y, t):
        U_x = max(0, 1e-6 * P_x * (1 - 0.2 * t / self.stages[-1]['end_time']))
        U_y = max(0, 1e-6 * P_y * (1 - 0.2 * t / self.stages[-1]['end_time']))

        nu_x = np.sqrt(4 * U_x * self.kb / (self.m * self.params['wx']**2)) / (2 * np.pi)
        nu_y = np.sqrt(4 * U_y * self.kb / (self.m * self.params['wy']**2)) / (2 * np.pi)
        nu_z = np.sqrt(2 * self.g / (self.params['wx'] + self.params['wy'])) / (2 * np.pi)

        # Implement adaptive adjustment
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        adjustment = 1 + 0.1 * np.sin(2 * np.pi * t / self.stages[-1]['end_time'])  # Example adaptive adjustment
        return nu_x * adjustment, nu_y * adjustment, nu_z * adjustment

    def collision_rate(self, N, T, nu_x, nu_y, nu_z):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        n0 = N * (self.m * nu_mean**2 / (2 * np.pi * self.kb * T))**(3/2)
        v_mean = np.sqrt(8 * self.kb * T / (np.pi * self.m))
        return 8 * np.pi * self.a**2 * n0 * v_mean

    def psd(self, N, T, nu_x, nu_y, nu_z):
        return N * (self.h * np.mean([nu_x, nu_y, nu_z]) / (self.kb * T))**3

    def bec_fraction(self, N, T, nu_x, nu_y, nu_z):
        Tc = 0.94 * self.h * np.mean([nu_x, nu_y, nu_z]) * N**(1/3) / self.kb
        return max(0, 1 - (T / Tc)**3) if T < Tc else 0

    def raman_cooling_rate(self, T, Omega_R, Gamma_OP, delta):
        # Implement a more sophisticated Raman cooling rate
        recoil_energy = (self.h ** 2) / (2 * self.m * (780e-9) ** 2)  # Recoil energy for Rb87
        eta = recoil_energy / (self.kb * T)  # Lamb-Dicke parameter
        return 10 * eta * Omega_R**2 * Gamma_OP / (4 * delta**2) * np.sqrt(self.m / (2 * self.kb * T))

    def heating_rate(self, N, T, nu_x, nu_y, nu_z):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        return 2 * self.h * nu_mean * self.collision_rate(N, T, nu_x, nu_y, nu_z) / (3 * self.kb)

    def light_assisted_loss_rate(self, N, T, nu_x, nu_y, nu_z, Gamma_OP):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        n0 = N * (self.m * nu_mean**2 / (2 * np.pi * self.kb * T))**(3/2)
        beta = 5e-13  # Reduced light-assisted loss coefficient
        return beta * Gamma_OP * n0

    def evaporation_rate(self, N, T, nu_x, nu_y, nu_z, eta):
        return self.collision_rate(N, T, nu_x, nu_y, nu_z) * eta * np.exp(-eta) * (eta - 4) / (eta - 5)

    def gray_molasses_cooling(self, T):
        return 0.5 * T  # Estimate: cool to half the initial temperature

    def system_evolution(self, t, state):
        N, T = state
        N = max(N, 1e-10)  # Prevent N from becoming zero
        T = max(T, 1e-10)  # Prevent T from becoming zero

        stage = next((s for s in self.stages if s['start_time'] <= t < s['end_time']), self.stages[-1])

        P_x = self.params[f'{stage["name"]}_P_x'] if f'{stage["name"]}_P_x' in self.params else 1.0
        P_y = self.params[f'{stage["name"]}_P_y'] if f'{stage["name"]}_P_y' in self.params else 1.0
        nu_x, nu_y, nu_z = self.trap_frequencies(P_x, P_y, t)

        if stage['name'] == 'GrayMolasses':
            dNdt = -self.params['gamma_bg'] * N
            dTdt = (self.gray_molasses_cooling(T) - T) / stage['duration']
        elif stage['name'].startswith('Raman'):
            Omega_R = self.params[f'{stage["name"]}_Omega_R']
            Gamma_OP = self.params[f'{stage["name"]}_Gamma_OP']
            delta = self.params[f'{stage["name"]}_delta']
            gamma_cool = self.raman_cooling_rate(T, Omega_R, Gamma_OP, delta)
            gamma_heat = self.heating_rate(N, T, nu_x, nu_y, nu_z)
            dNdt = -self.params['gamma_bg'] * N - self.light_assisted_loss_rate(N, T, nu_x, nu_y, nu_z, Gamma_OP)
            dTdt = -gamma_cool * T + gamma_heat
        elif stage['name'].startswith('Evap'):
            eta = self.params[f'{stage["name"]}_eta']
            gamma_evap = self.evaporation_rate(N, T, nu_x, nu_y, nu_z, eta)
            dNdt = -self.params['gamma_bg'] * N - gamma_evap * N
            dTdt = -(eta - 3) / 3 * gamma_evap * T
        else:  # MOT
            dNdt = 0
            dTdt = 0

        dNdt = np.clip(dNdt, -N/10, N/10)  # Limit the rate of atom loss
        dTdt = np.clip(dTdt, -T/10, T/10)  # Limit the rate of temperature change

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
        self.xlim = (-1000e-6, 1000e-6)  # Increased x-axis range to ±500 µm
        self.ylim = (-1000e-6, 1000e-6)  # Increased y-axis range to ±500 µm

    def calculate_density_range(self, results):
        densities = []
        for result in results:
            if result['stage'] != 'MOT':  # Only consider frames after MOT
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

        sigma_x = np.sqrt(self.sim.kb * T / (self.sim.m * (2*np.pi*nu_x)**2) + (self.sim.h * nu_x / (2*np.pi*self.sim.m))**2 * self.tof**2)
        sigma_y = np.sqrt(self.sim.kb * T / (self.sim.m * (2*np.pi*nu_y)**2) + (self.sim.h * nu_y / (2*np.pi*self.sim.m))**2 * self.tof**2)

        n_thermal = N * (1 - bec_fraction) * np.exp(-(X**2 / (2*sigma_x**2) + Y**2 / (2*sigma_y**2))) / (2*np.pi*sigma_x*sigma_y)
        
        if bec_fraction > 0:
            R_x = np.sqrt(2 * self.sim.h * nu_x * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_x)**2))
            R_y = np.sqrt(2 * self.sim.h * nu_y * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_y)**2))
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

        # Calculate global vmin and vmax for frames after MOT
        self.calculate_density_range(results)
        
        def update(frame):
            ax.clear()
            result = results[frame]
            frame_data = self.generate_frame(result)
            
            if result['stage'] == 'MOT':
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