import numpy as np
import math


# Physical Constants
mu_e = 3.986e14 # m^3/s^2, Earth's gravitational parameter
mu_s = 1.327e20 # m^3/s^2, Sol's gravitational parameter
R_e = 6378137 # m, Earth's equatorial radius
m_e = 5.97237e24# kg, Earth's mass
m_sc = 1000 # kg, Spacecraft's mass
smaEarth = 1.49598023e11 # m, Earth's semimajor axis

# Unit scaling here as well.
mu_e = mu_e/(R_e**3)
mu_s = mu_s/(R_e**3)
smaEarth = smaEarth/R_e
wEarth = math.sqrt((mu_e + mu_s)/(smaEarth**3))
R_SoI = (mu_e/mu_s)**(2.0/5.0)*smaEarth

def dynamics(t, y):
    # Initialize outputs
    y_dot = np.zeros((6,))

    # Velocities
    y_dot[0:3] = y[3:6]

    # Accelerations
    lonEarth = wEarth*t
    r_e = smaEarth*np.array([math.cos(lonEarth), math.sin(lonEarth), 0])
    r_sc = y[0:3]
    r_sc_e = r_sc - r_e # Relative position of Earth
    sc_mag = np.linalg.norm(r_sc)**3
    sc_e_mag = np.linalg.norm(r_sc_e)**3
    e_mag = np.linalg.norm(r_e)**3

    y_dot[3:6] = -(mu_s/sc_mag)*r_sc - (mu_e/sc_e_mag)*r_sc_e

    return y_dot

def hamiltonian(t, y):
    # This function calculates the value of the Hamiltonian of the
    # system at a given state.
    lonEarth = wEarth*t
    r_e = smaEarth*np.array([math.cos(lonEarth), math.sin(lonEarth), 0])
    
    # Preliminaries
    r_sc = y[0:3]
    v_e = smaEarth*np.array([-math.sin(lonEarth), math.cos(lonEarth), 0])
    v_sc = y[3:6]
    r_sc_e = r_sc - r_e
    
    r_sc_mag = np.linalg.norm(r_sc)
    r_sc_e_mag = np.linalg.norm(r_sc_e)
    r_e_mag = np.linalg.norm(r_e)
    v_e_mag = np.linalg.norm(v_e)
    v_sc_mag = np.linalg.norm(v_sc)

    T_sc = (m_sc/2.0)*(v_sc_mag**2)

    U_sc = -mu_s*m_sc/r_sc_mag
    U_sc_e = -mu_e*m_sc/r_sc_e_mag

    hamiltonian = T_sc + U_sc + U_sc_e

    return hamiltonian

    
