import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

class BECSimulation:
    def __init__(self):
        self.m = 1.44e-25  # Mass of Rb-87 atom in kg
        self.h = 6.626e-34  # Planck constant
        self.kb = 1.38e-23  # Boltzmann constant
        self.a = 5.77e-9  # Scattering length for Rb-87
        self.g = 9.81  # Gravitational acceleration

        self.stages = [
            {'name': 'MOT', 'duration': 99e-3},
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
        ]

        self.params = {
            'initial_N': 1e6,  # 1 million atoms
            'initial_T': 100e-6,  # 100 µK
            'gamma_bg': 0.1,  # Background loss rate
            'wx': 100e-6,  # 100 µm horizontal beam waist
            'wy': 100e-6,  # 100 µm vertical beam waist
        }

        for stage in self.stages:
            if stage['name'].startswith('Raman'):
                self.params[f'{stage["name"]}_P'] = 1.0  # Full power for Raman cooling
            elif stage['name'].startswith('Evap'):
                self.params[f'{stage["name"]}_eta'] = 6  # Reduced truncation parameter for evaporation

    def trap_frequencies(self, P):
        U = 1e-6 * P  # Trap depth in microkelvin, adjust as needed
        nu_x = np.sqrt(4 * U * self.kb / (self.m * self.params['wx']**2)) / (2 * np.pi)
        nu_y = np.sqrt(4 * U * self.kb / (self.m * self.params['wy']**2)) / (2 * np.pi)
        nu_z = np.sqrt(2 * self.g / (self.params['wx'] + self.params['wy'])) / (2 * np.pi)
        return min(nu_x, 1e5), min(nu_y, 1e5), nu_z  # Limit frequencies to realistic values

    def collision_rate(self, N, T, nu_x, nu_y, nu_z):
        nu_mean = np.mean([nu_x, nu_y, nu_z])
        n0 = N * (self.m * nu_mean**2 / (2 * np.pi * self.kb * T))**(3/2)
        v_mean = np.sqrt(8 * self.kb * T / (np.pi * self.m))
        return min(8 * np.pi * self.a**2 * n0 * v_mean, 1e4)  # Limit collision rate to realistic values

    def psd(self, N, T, nu_x, nu_y, nu_z):
        # Add small epsilon to avoid division by zero
        epsilon = 1e-15
        return N * (self.h * np.mean([nu_x, nu_y, nu_z]) / (self.kb * max(T, epsilon)))**3

    def bec_fraction(self, N, T, nu_x, nu_y, nu_z):
        Tc = 0.94 * self.h * np.mean([nu_x, nu_y, nu_z]) * N**(1/3) / self.kb
        return max(0, min(1, 1 - (T / Tc)**3)) if T < Tc else 0

    def system_evolution(self, state, t, stage):
        N, T = state
        N = max(N, 1)  # Ensure N is at least 1
        T = max(T, 1e-9)  # Ensure T is positive and not too close to zero

        if stage['name'].startswith('Raman'):
            P = self.params[f'{stage["name"]}_P']
            nu_x, nu_y, nu_z = self.trap_frequencies(P)
            gamma_cool = 0.1  # Adjust cooling rate as needed
            dNdt = -self.params['gamma_bg'] * N
            dTdt = -gamma_cool * T
        elif stage['name'].startswith('Evap'):
            P = 1.0 - 0.8 * (int(stage['name'][-1]) - 1) / 5  # Power ramp down
            nu_x, nu_y, nu_z = self.trap_frequencies(P)
            eta = self.params[f'{stage["name"]}_eta']
            gamma_evap = self.collision_rate(N, T, nu_x, nu_y, nu_z) * np.exp(-eta)
            dNdt = -self.params['gamma_bg'] * N - gamma_evap * N
            dTdt = -(eta - 3) / 3 * gamma_evap * T
        else:  # MOT
            dNdt = 0
            dTdt = 0
        
        # Ensure dNdt and dTdt are finite
        dNdt = np.clip(dNdt, -N/10, N/10)  # Limit the rate of atom loss
        dTdt = np.clip(dTdt, -T/10, T/10)  # Limit the rate of temperature change
        
        return [dNdt, dTdt]

    def run_simulation(self):
        results = []
        N, T = self.params['initial_N'], self.params['initial_T']

        for stage in self.stages:
            sol = solve_ivp(
                lambda t, y: self.system_evolution(y, t, stage),
                [0, stage['duration']],
                [N, T],
                method='RK45',
                rtol=1e-8,
                atol=1e-8
            )
            N, T = sol.y[:, -1]

            if stage['name'].startswith('Raman') or stage['name'].startswith('Evap'):
                P = self.params[f'{stage["name"]}_P'] if stage['name'].startswith('Raman') else 1.0 - 0.8 * (int(stage['name'][-1]) - 1) / 5
            else:
                P = 1.0
            nu_x, nu_y, nu_z = self.trap_frequencies(P)
            nu_c = self.collision_rate(N, T, nu_x, nu_y, nu_z)
            psd = self.psd(N, T, nu_x, nu_y, nu_z)
            bec_frac = self.bec_fraction(N, T, nu_x, nu_y, nu_z)

            print(f"Stage: {stage['name']}")
            print(f"N: {N}, T: {T}")
            print(f"nu_x: {nu_x}, nu_y: {nu_y}, nu_z: {nu_z}")
            print(f"nu_c: {nu_c}")
            print(f"PSD: {psd}")
            print(f"BEC fraction: {bec_frac}")
            print("---")

            results.append({
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

    def generate_frame(self, result, grid_size=100, xlim=(-100e-6, 100e-6), ylim=(-100e-6, 100e-6)):
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)

        N, T = result['N'], result['T']
        nu_x, nu_y, nu_z = result['nu_x'], result['nu_y'], result['nu_z']
        bec_fraction = result['BEC_fraction']

        sigma_x = np.sqrt(self.sim.kb * T / (self.sim.m * (2*np.pi*nu_x)**2) + (self.sim.h * nu_x / (2*np.pi*self.sim.m))**2 * self.tof**2)
        sigma_y = np.sqrt(self.sim.kb * T / (self.sim.m * (2*np.pi*nu_y)**2) + (self.sim.h * nu_y / (2*np.pi*self.sim.m))**2 * self.tof**2)

        n_thermal = N * (1 - bec_fraction) * np.exp(-(X**2 / (2*sigma_x**2) + Y**2 / (2*sigma_y**2))) / (2*np.pi*sigma_x*sigma_y)
        
        if bec_fraction > 0:
            # Ensure all values are non-negative before sqrt
            R_x = np.sqrt(np.maximum(0, 2 * self.sim.h * nu_x * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_x)**2)))
            R_y = np.sqrt(np.maximum(0, 2 * self.sim.h * nu_y * N * bec_fraction * self.sim.a / (self.sim.m * (2*np.pi*nu_y)**2)))
            n_bec = N * bec_fraction * np.maximum(0, 1 - X**2/R_x**2 - Y**2/R_y**2)**(3/2) / (4*np.pi*R_x*R_y/3)
        else:
            n_bec = np.zeros_like(X)

        n_total = n_thermal + n_bec

        # Add noise and apply imaging resolution limit
        n_with_noise = n_total + np.random.normal(0, 0.1 * np.max(n_total), n_total.shape)
        n_blurred = gaussian_filter(n_with_noise, sigma=self.imaging_resolution / (xlim[1] - xlim[0]) * grid_size)

        # Handle NaN and inf values
        if np.all(np.isnan(n_blurred)):
            n_blurred = np.zeros_like(n_blurred)
        else:
            n_blurred = np.nan_to_num(n_blurred, nan=0, posinf=np.nanmax(n_blurred), neginf=0)

        # Ensure non-negative values
        n_blurred = np.maximum(n_blurred, 0)

        return n_blurred

    def create_animation(self, results, output_filename):
        frames = []
        for result in results:
            frame = self.generate_frame(result)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Handle the case where all values are zero or negative
            positive_frame = frame[frame > 0]
            if positive_frame.size > 0:
                vmin = np.min(positive_frame)
                vmax = np.max(frame)
            else:
                vmin = 1e-10  # Set a small positive value
                vmax = 1.0    # Set a default maximum value
            
            im = ax.imshow(frame, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), extent=[-100, 100, -100, 100])
            plt.colorbar(im, label='Density (a.u.)')
            ax.set_title(f"Stage: {result['stage']}\nT: {result['T']*1e6:.2f} µK, N: {result['N']:.2e}, BEC fraction: {result['BEC_fraction']:.2%}")
            ax.set_xlabel('x (µm)')
            ax.set_ylabel('y (µm)')
            
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            image = np.array(fig.canvas.buffer_rgba())
            frames.append(image)
            
            plt.close(fig)

        # Save frames as GIF
        imageio.mimsave(output_filename, frames, fps=2)

# Run simulation and create visualization
sim = BECSimulation()
results = sim.run_simulation()

visualizer = BECVisualizer(sim)
visualizer.create_animation(results, 'bec_evolution.gif')

print("Animation creation complete. Check 'bec_evolution.gif'.")

# Plot additional results
stages = [result['stage'] for result in results]
N_values = [result['N'] for result in results]
T_values = [result['T'] for result in results]
PSD_values = [result['PSD'] for result in results]

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.semilogy(stages, N_values)
plt.ylabel('Atom Number')
plt.title('BEC Formation Simulation Results')

plt.subplot(3, 1, 2)
plt.semilogy(stages, T_values)
plt.ylabel('Temperature (K)')

plt.subplot(3, 1, 3)
plt.semilogy(stages, PSD_values)
plt.ylabel('Phase Space Density')
plt.xlabel('Stages')

plt.tight_layout()
plt.savefig('bec_formation_results.png')
plt.close()

print("Additional results plot saved as 'bec_formation_results.png'.")