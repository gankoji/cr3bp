import numpy as np
import math
import matplotlib.pyplot as plt
## import orbital.elements
#import time
import datetime

from rkf45 import *
from Dynamics import dynamics, hamiltonian
from Dynamics_KS import dynamics_ks, ks_init, ks2cart

print("Started: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

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

# Simulation Conditions
debug = False
secondsPerDay = 3600*24

if debug:
    ## This is an approximation to the fictitious time (rather than
    ## add the expense of an event function
    simLength = 400*secondsPerDay
    simLength = simLength/smaEarth
    dt = 10.5
else:
    simLength = 546*secondsPerDay
    simLength = simLength/smaEarth
    dt = 1
    
N= int(math.floor(simLength/dt))
simTime = np.linspace(0,simLength, N)
flag = 1
relerr = 1e-8
abserr = 1e-3

# Sim output storage
y_out = np.zeros((N,10))
x_out = np.zeros((N,3))
e_out = np.zeros((N,3))
H_out = np.zeros((N,))

print(smaEarth*wEarth*R_e)
# Initial Conditions
r_e0 = np.array([1.4960e11, 0, 0]) # m, Earth's average orbital
                                   # distance, along the vernal
                                   # equinox
v_e0 = np.array([0, 3e4, 0]) # m/s, Earth's average orbital velocity
                             # in the y inertial direction

r_sc0 = r_e0 + np.array([6.7e6, 0, 0]) # m, Spacecraft's initial
                                       # position added to Earth's
v_sc0 = v_e0 + np.array([-1e3, 7.0e3, 0]) # m/s, Spacecraft's initial
                                       # velocity added to Earth's

# Unit Conversion
# Meters was a very clearly poor choice for distance units
# Transform instead to units of Earth Equatorial Radius
# This is a compromise between S/C and Earth dynamics
r_e0 = r_e0/R_e
v_e0 = v_e0/R_e
r_sc0 = r_sc0/R_e
v_sc0 = v_sc0/R_e
print(v_sc0)

x_out[0,:] = r_sc0
e_out[0,:] = smaEarth*np.array([1,0,0])

u0 = ks_init(r_sc0, v_sc0, 0, mu_s, 0)

# Thus we have a 10x1 initial state vector
y0 = np.append(r_e0, v_e0)
y0 = np.append(y0, r_sc0)
y0 = np.append(y0, v_sc0)

#u0 = np.append(u0, e_out[0,:])
y_out[0,:] = u0
yp = np.zeros((10,))
Hrel = np.zeros((N,))

for i,x in enumerate(simTime):
    if i==0:
        print("Initializing Done.")
        H_out[i] = u0[8]
        Hrel[i] = 0
    else:
        y = np.reshape(y_out[i-1,:],(10,))
        t0 = simTime[i-1]

        y, yp, t, flag = r8_rkf45( dynamics_ks, 10, y, yp, t0, x, relerr, abserr, flag)
        y_out[i,:] = y
        H_out[i] = y[8]
        Hrel[i] = (H_out[i] - H_out[i-1])/H_out[i]
        
        if flag == 7 or flag == 6:
            flag = 2

        p, pdot = ks2cart(y)
        x_out[i,:] = p
        t = y[9]*np.linalg.norm(y[0:4])
        lonEarth = wEarth*t
        e_out[i,:] = smaEarth*np.array([math.cos(lonEarth), math.sin(lonEarth), 0])

print("Finished: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

fig, ax = plt.subplots()
ax.plot(e_out[:,0], e_out[:,1])
ax.plot(x_out[:,0], x_out[:,1])
ax.plot([0.0],[0.0],'r+')
ax.set(xlabel='X - Earth Radii',ylabel='Y - Earth Radii',
       title='Planar Motion of Two Bodies')
ax.legend(['Earth','Spacecraft','Sol'])
fig.savefig('PlanarPath.png')

fig1, ax1 = plt.subplots()
ax1.plot(simTime/secondsPerDay*smaEarth, Hrel)
ax1.set(xlabel='Time (Days)',ylabel='Energy',
        title='Hamiltonian (Total Energy)')
fig1.savefig('Hamiltonian.png')

plt.show()
