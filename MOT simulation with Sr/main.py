import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, mu_0, c, elementary_charge as e, k as kB

# Constants
m = 87.62 * 1.66e-27  # mass of Sr-88 in kg
g = 9.81  # acceleration due to gravity in m/s^2
gamma = 2 * np.pi * 7.5e3  # natural linewidth
lambda_light = 689e-9  # wavelength of light in m
k = 2 * np.pi / lambda_light  # wavevector
gamma_B = 0.08  # magnetic field gradient in T/m
mu_B = 9.274e-24  # Bohr magneton
g_factor = 1.5  # Landé g-factor for 3P1 state

# Simulation parameters
N_atoms = 5000  # number of atoms
dt = 0.1 / gamma  # time step
t_total = 15e-3  # total simulation time
steps = int(t_total / dt)

# Define laser beam directions and polarizations
beam_directions = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
beam_polarizations = np.array([
    [0, -1j, 1], [0, 1j, 1],  # σ- for +x, σ+ for -x
    [1, 0, -1j], [1, 0, 1j],  # σ- for +y, σ+ for -y
    [-1, -1j, 0], [-1, 1j, 0]  # σ- for +z, σ+ for -z
]) / np.sqrt(2)

def initialize_atoms(N, initial_temp):
    positions = np.random.uniform(-1e-4, 1e-4, (N, 3))
    velocities = np.random.normal(0, np.sqrt(kB * initial_temp / m), (N, 3))
    # Initialize all atoms in ground state (J=0, mj=0)
    states = np.zeros(N, dtype=int)
    return positions, velocities, states

def quadrupole_field(pos):
    x, y, z = pos
    return gamma_B * np.array([x, y, -2*z])

def calculate_coupling_strengths(B_field, laser_polarization):
    B_unit = B_field / np.linalg.norm(B_field)
    # Rotation matrix to transform from lab frame to local B-field frame
    R = np.array([
        [np.cos(np.arccos(B_unit[2])), 0, -B_unit[0]],
        [0, 1, 0],
        [B_unit[0], 0, np.cos(np.arccos(B_unit[2]))]
    ])
    local_polarization = R @ laser_polarization
    # Coupling strengths for mj = -1, 0, 1 transitions
    W = np.abs(local_polarization)**2
    return W

def detuning(pos, vel, laser_k, Delta):
    B = quadrupole_field(pos)
    B_mag = np.linalg.norm(B)
    delta_z = mu_B * g_factor * B_mag / hbar
    return Delta - np.dot(laser_k, vel) - delta_z

def scattering_rate(pos, vel, laser_k, laser_pol, Delta, S, state):
    delta = detuning(pos, vel, laser_k, Delta)
    B = quadrupole_field(pos)
    W = calculate_coupling_strengths(B, laser_pol)
    # Use coupling strength corresponding to the current state
    return (gamma / 2) * (W[state] * S / (1 + W[state] * S + 4 * (delta / gamma)**2))

def run_simulation(positions, velocities, states, Delta, S):
    for step in range(steps):
        for i in range(N_atoms):
            # Calculate total scattering rate
            total_rate = sum(scattering_rate(positions[i], velocities[i], k * beam_directions[j],
                                             beam_polarizations[j], Delta, S, states[i])
                             for j in range(6))
            
            # Probabilistic photon absorption
            if np.random.random() < total_rate * dt:
                # Choose which beam caused the absorption
                beam_probs = [scattering_rate(positions[i], velocities[i], k * beam_directions[j],
                                              beam_polarizations[j], Delta, S, states[i])
                              for j in range(6)]
                beam_index = np.random.choice(6, p=beam_probs/np.sum(beam_probs))
                
                # Momentum kick from absorbed photon
                velocities[i] += hbar * k * beam_directions[beam_index] / m
                
                # Update atomic state (simplified - always goes to mj = -1 due to σ- light)
                states[i] = 0  # mj = -1 state
                
                # Momentum kick from emitted photon (random direction)
                phi = np.random.uniform(0, 2*np.pi)
                theta = np.arccos(1 - 2*np.random.random())
                kick_dir = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
                velocities[i] += hbar * k * kick_dir / m
                
                # Reset to ground state after emission
                states[i] = 0
            
            # Update positions and velocities
            positions[i] += velocities[i] * dt
            velocities[i] += np.array([0, 0, -g]) * dt  # gravity
        
        if step % 100 == 0:
            print(f"Step {step}/{steps} completed")
    
    return positions, velocities, states

# Set initial conditions and run simulation
Delta = -2 * np.pi * 200e3  # detuning
S = 10  # saturation parameter
initial_temp = 1e-6  # initial temperature in K

positions, velocities, states = initialize_atoms(N_atoms, initial_temp)
final_positions, final_velocities, final_states = run_simulation(positions, velocities, states, Delta, S)

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(final_positions[:, 0], final_positions[:, 2], alpha=0.5)
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Final atom positions')

plt.subplot(132)
plt.hist(np.linalg.norm(final_velocities, axis=1), bins=30)
plt.xlabel('Velocity (m/s)')
plt.ylabel('Count')
plt.title('Velocity distribution')

plt.subplot(133)
plt.hist(final_states, bins=[-0.5, 0.5, 1.5, 2.5])
plt.xlabel('mj state')
plt.ylabel('Count')
plt.title('Final state distribution')

plt.tight_layout()
plt.show()