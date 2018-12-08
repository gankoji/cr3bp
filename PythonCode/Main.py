import numpy as np
import math
import matplotlib.pyplot as plt
# import orbital.elements
import time

from rkf45 import r8_rkf45
from Dynamics import dynamics, hamiltonian

print("Started: " + time.strftime("%H:%M:%S"))

# Physical Constants
mu_e = 3.986e14  # m^3/s^2, Earth's gravitational parameter
R_e = 6378137  # m, Earth's equatorial radius

# Simulation Conditions
debug = False

if debug:
    simLength = 100*24*3600
    dt = 60*1.50
else:
    simLength = 980*24*3600
    dt = 60*1.50

N = int(math.floor(simLength/dt))
simTime = np.linspace(0, simLength, N)
flag = 1
relerr = 1e-8
abserr = 1e-3

# Sim output storage
y_out = np.zeros((N, 12))
H_out = np.zeros((N,))

# Initial Conditions
# m, Earth's average orbital
# distance, along the vernal
# equinox
r_e0 = np.array([1.4960e11, 0, 0])

# m/s, Earth's average orbital velocity
# in the y inertial direction
v_e0 = np.array([0, 3e4, 0])

# m, Spacecraft's initial
# position added to Earth's
r_sc0 = r_e0 + np.array([6.7e6, 0, 0])

# m/s, Spacecraft's initial
# velocity added to Earth's
v_sc0 = v_e0 + np.array([0, 1.3e4, 0])

# Unit Conversion
# Meters was a very clearly poor choice for distance units
# Transform instead to units of Earth Equatorial Radius
# This is a compromise between S/C and Earth dynamics
r_e0 = r_e0/R_e
v_e0 = v_e0/R_e
r_sc0 = r_sc0/R_e
v_sc0 = v_sc0/R_e

# Thus we have a 12x1 initial state vector
y0 = np.append(r_e0, v_e0)
y0 = np.append(y0, r_sc0)
y0 = np.append(y0, v_sc0)

y_out[0, :] = y0
yp = np.zeros((12,))

for i, x in enumerate(simTime):
    if i == 0:
        print("Initializing Done.")
        H_out[i] = hamiltonian(y_out[i, :])
    else:
        y = np.reshape(y_out[i-1, :], (12,))
        t0 = simTime[i-1]

        y, yp, t, flag = r8_rkf45(dynamics, 12, y, yp, t0, x,
                                  relerr, abserr, flag)
        y_out[i, :] = y
        H_out[i] = hamiltonian(y_out[i, :])

        if flag == 7 or flag == 6:
            flag = 2

print("Finished: " + time.strftime("%H:%M:%S"))

fig, ax = plt.subplots()
ax.plot(y_out[:, 0], y_out[:, 1])
ax.plot(y_out[:, 6], y_out[:, 7])
ax.plot([0.0], [0.0], 'r+')
ax.set(xlabel='X - Earth Radii', ylabel='Y - Earth Radii',
       title='Planar Motion of Two Bodies')
ax.legend(['Earth', 'Spacecraft', 'Sol'])
fig.savefig('PlanarPath.png')

fig1, ax1 = plt.subplots()
ax1.plot(simTime, H_out)
ax1.set(xlabel='Time', ylabel='Energy',
        title='Hamiltonian (Total Energy)')
fig1.savefig('Hamiltonian.png')

plt.show()
