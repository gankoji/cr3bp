import numpy as np
import math

mu_e = 3.986e14 # m^3/s^2, Earth's gravitational parameter
mu_s = 1.327e20 # m^3/s^2, Sol's gravitational parameter
R_e = 6378137 # m, Earth's equatorial radius
m_e = 5.97237e24# kg, Earth's mass
m_sc = 1000 # kg, Spacecraft's mass

# Unit scaling here as well.
mu_e = mu_e/(R_e**3)
mu_s = mu_s/(R_e**3)

def dynamics(t, y):
    # Initialize outputs
    y_dot = np.zeros((12,))

    # Velocities
    y_dot[0:3] = y[3:6]
    y_dot[6:9] = y[9:12]

    # Accelerations
    r_e = y[0:3]
    r_sc = y[6:9]
    r_sc_e = r_sc - r_e # Relative position of Earth
    sc_mag = np.linalg.norm(r_sc)**3
    sc_e_mag = np.linalg.norm(r_sc_e)**3
    e_mag = np.linalg.norm(r_e)**3

    y_dot[3:6] = -(mu_s/e_mag)*r_e
    y_dot[9:12] = -(mu_s/sc_mag)*r_sc - (mu_e/sc_e_mag)*r_sc_e

    return y_dot

def hamiltonian(y):
    # This function calculates the value of the Hamiltonian of the
    # system at a given state.

    # Preliminaries
    r_e = y[0:3]
    r_sc = y[6:9]
    v_e = y[3:6]
    v_sc = y[9:12]
    r_sc_e = r_sc - r_e
    
    r_sc_mag = np.linalg.norm(r_sc)
    r_sc_e_mag = np.linalg.norm(r_sc_e)
    r_e_mag = np.linalg.norm(r_e)
    v_e_mag = np.linalg.norm(v_e)
    v_sc_mag = np.linalg.norm(v_sc)
    #v_sc_e_mag = np.linalg.norm(v_sc_e)

    T_e = (m_e/2.0)*(v_e_mag**2)
    T_sc = (m_sc/2.0)*(v_sc_mag**2)

    U_e = -mu_s*m_e/r_e_mag
    U_sc = -mu_s*m_sc/r_sc_mag
    U_sc_e = -mu_e*m_sc/r_sc_e_mag

    hamiltonian = T_e + T_sc + U_e + U_sc + U_sc_e

    return hamiltonian

    
