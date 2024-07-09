import numpy as np
from constants import Rb87, k, h, hbar
from utils import photoassociation_loss_rate

class AtomCloud:
    def __init__(self, N, T, trap):
        self.N = int(max(N, 1))  # Ensure at least one atom
        self.T = max(T, 1e-9)  # Ensure non-zero temperature (1 nK minimum)
        self.trap = trap
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities()
        self.light_shift = 0

    def initialize_positions(self):
        sigma = np.sqrt(k * self.T / (Rb87.mass * (2*np.pi*100)**2))
        return np.random.normal(0, sigma, (self.N, 3))

    def initialize_velocities(self):
        sigma_v = np.sqrt(k * self.T / Rb87.mass)
        return np.random.normal(0, sigma_v, (self.N, 3)) 
    
    def force_sync(self):
        self.N = int(self.N)
        self.N = min(self.N, len(self.positions), len(self.velocities))
        self.positions = self.positions[:self.N]
        self.velocities = self.velocities[:self.N]
        logging.debug(f"After force_sync: N = {self.N}, "
                      f"positions shape = {self.positions.shape}, "
                      f"velocities shape = {self.velocities.shape}")

    def update(self, dt):
        forces = self.trap.force(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
        self.N = int(self.N)
        self.velocities += forces * dt / Rb87.mass
        self.positions += self.velocities * dt

    def update_temperature(self):
        kinetic_energy = 0.5 * Rb87.mass * np.sum(self.velocities**2)
        self.T = max(2 * kinetic_energy / (3 * k * self.N), 1e-9)

    def update_velocities(self):
        sigma_v = np.sqrt(k * self.T / Rb87.mass)
        self.velocities = np.random.normal(0, sigma_v, (self.N, 3))

    def apply_evaporation(self, trap_depth):
        tilt = self.trap.calculate_tilt()
        kinetic_energy = 0.5 * Rb87.mass * np.sum(self.velocities**2, axis=1)
        potential_energy = self.trap.calculate_potential(self.positions)
        total_energy = kinetic_energy + potential_energy
        
        escape_prob = np.exp(-(trap_depth - total_energy) / (k * self.T)) * (1 + np.sin(tilt))
        mask = np.random.random(self.N) > escape_prob
        
        self.positions = self.positions[mask]
        self.velocities = self.velocities[mask]
        self.update_atom_number(np.sum(mask))

    def apply_light_shift(self, P_p, delta, sigma_minus_beam):
        I = sigma_minus_beam.intensity(self.positions)
        I_sat = 1.67
        self.light_shift = hbar * Rb87.gamma_D1**2 * I / (8 * delta * I_sat)

    def apply_gray_molasses(self, duration):
        cooling_rate = 1e-1
        self.T *= np.exp(-cooling_rate * duration)

    def apply_light_assisted_collisions(self, P_p, delta):
        n = self.calculate_density()
        n = min(n, 1e19)  # Cap density at a realistic maximum value
        
        # Adjust intensity calculation
        beam_area = np.pi * self.trap.w_y * self.trap.w_z
        intensity = min(P_p / beam_area, 1e3)  # Cap intensity at 1000 W/m^2
        intensity_mW_cm2 = intensity * 1e-1  # Convert to mW/cm^2
    
        K_2 = photoassociation_loss_rate(delta, intensity_mW_cm2)
        
        loss_rate = K_2 * n * dt
        survival_prob = np.exp(-loss_rate)
        new_N = int(self.N * survival_prob)
        self.N = max(new_N, 1)  # Ensure at least 1 atom remains

        # Only update positions and velocities if atoms are actually lost
        if new_N < self.N:
            if len(self.positions) > new_N:
                indices = np.random.choice(len(self.positions), new_N, replace=False)
                self.positions = self.positions[indices]
                self.velocities = self.velocities[indices]

    def apply_three_body_recombination(self):
        n = self.calculate_density()
        K_3 = Rb87.calculate_K3(self.T)
        loss_rate = K_3 * n**2 * dt
        survival_prob = np.exp(-loss_rate)
        initial_N = self.N
        self.N = max(int(self.N * survival_prob), 1)  # Ensure at least one atom remains
        atoms_lost = initial_N - self.N
        
        if self.N < len(self.positions):
            indices = np.random.choice(len(self.positions), self.N, replace=False)
            self.positions = self.positions[indices]
            self.velocities = self.velocities[indices]

        heating_rate = K_3 * n**2 * (Rb87.a_s * hbar)**2 / (2 * Rb87.mass) * 1e3
        self.T += heating_rate * dt

        # Add small probability of hot atoms remaining trapped
        hot_atom_prob = 0.01
        hot_atom_heating = atoms_lost * hot_atom_prob * 10 * k * self.T / (self.N * 3 * k)
        self.T += hot_atom_heating

        self.update_temperature()

    def apply_evaporative_cooling(self, trap_depth):
        # Simple evaporative cooling model
        eta = 10  # truncation parameter
        evap_prob = np.exp(-eta * self.T / trap_depth)
        atoms_to_remove = int(self.N * evap_prob)
        self.N -= atoms_to_remove
        self.T *= 1 - evap_prob / 3  # Cooling effect
        

    def apply_photon_reabsorption_heating(self, P_p, delta):
        n = self.calculate_density()
        reabsorption_prob = n * 3 * Rb87.wavelength_D1**2 / (2 * np.pi)
        scattering_rate = P_p / (1 + 4 * delta**2 / Rb87.gamma_D1**2)
        heating_rate = reabsorption_prob * scattering_rate * 2 * Rb87.Er_D1 / (3 * k)
        self.T += heating_rate * dt

    def apply_magnetic_field(self, B_z):
        # Calculate the Zeeman shift
        zeeman_shift = Rb87.mu_B * Rb87.g_F * B_z
        
        # Apply the shift to the atoms' energies
        # This is a simplified approach; you may need to adjust it based on your specific requirements
        self.light_shift += zeeman_shift
        
    # In the AtomCloud class, modify the calculate_density method:
    def calculate_density(self):
        omega = (self.trap.omega_x * self.trap.omega_y * self.trap.omega_z)**(1/3)
        return self.N * (Rb87.mass * omega / (2 * np.pi * k * self.T))**(3/2)
    
    def update_atom_number(self, new_N):
        new_N = int(max(new_N, 1))  # Ensure at least one atom
        if new_N < self.N:
            # Randomly select atoms to keep
            indices = np.random.choice(self.N, new_N, replace=False)
            self.positions = np.array(self.positions[indices])
            self.velocities = np.array(self.velocities[indices])
        elif new_N > self.N:
            # Add new atoms
            additional_positions = self.initialize_positions()[:new_N - self.N]
            additional_velocities = self.initialize_velocities()[:new_N - self.N]
            self.positions = np.vstack((self.positions, additional_positions))
            self.velocities = np.vstack((self.velocities, additional_velocities))
        self.N = new_N
        
        logging.debug(f"After update_atom_number: N = {self.N}, "
                      f"positions shape = {self.positions.shape}, "
                      f"velocities shape = {self.velocities.shape}")

    def __init__(self, N, T, trap):
        self.N = int(max(N, 1))
        self.T = max(T, 1e-9)
        self.trap = trap
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities()
        self.light_shift = 0

    def initialize_positions(self):
        sigma = np.sqrt(k * self.T / (Rb87.mass * (2*np.pi*100)**2))
        return np.random.normal(0, sigma, (self.N, 3))

    def initialize_velocities(self):
        sigma_v = np.sqrt(k * self.T / Rb87.mass)
        return np.random.normal(0, sigma_v, (self.N, 3))

    # ... (include all other methods from the AtomCloud class)

    def calculate_density(self):
        omega = (self.trap.omega_x * self.trap.omega_y * self.trap.omega_z)**(1/3)
        return self.N * (Rb87.mass * omega / (2 * np.pi * k * self.T))**(3/2)

    def update_atom_number(self, new_N):
        new_N = int(max(new_N, 1))
        if new_N < self.N:
            indices = np.random.choice(self.N, new_N, replace=False)
            self.positions = np.array(self.positions[indices])
            self.velocities = np.array(self.velocities[indices])
        elif new_N > self.N:
            additional_positions = self.initialize_positions()[:new_N - self.N]
            additional_velocities = self.initialize_velocities()[:new_N - self.N]
            self.positions = np.vstack((self.positions, additional_positions))
            self.velocities = np.vstack((self.velocities, additional_velocities))
        self.N = new_N