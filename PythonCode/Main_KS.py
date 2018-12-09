import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

from Constants import *
from rkf45 import *
from Dynamics import dynamics, hamiltonian
from Dynamics_KS import dynamics_ks, ks_init, ks2cart, setUnits, inSoI

print("Started: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

# Initial Conditions
r_e0 = np.array([smaEarth, 0, 0]) # km, Earth's average orbital
# distance, along the vernal
                                   # equinox
v_e0 = np.array([0, smaEarth*wEarth, 0]) # km/s, Earth's average orbital velocity
                             # in the y inertial direction

r_sc0 = r_e0 + np.array([6.5e3, 0, 0]) # km, Spacecraft's initial
                                       # position added to Earth's
v_sc0 = v_e0 + np.array([1, 4.0, 0]) # km/s, Spacecraft's initial
                                       # velocity added to Earth's

# Unit Conversion
DU = np.linalg.norm(r_sc0)
TU = math.sqrt(mu_e/(DU**3))
setUnits(DU, TU)

# Simulation Conditions
debug = False

if debug:
    simLength = 400*secondsPerDay
    dt = 900
    simLength = simLength/DU
    dt = dt/DU
else:
    simLength = 350*secondsPerDay
    dt = 300
    simLength = simLength/DU
    dt = dt/DU
    
N= int(math.floor(simLength/dt))
simTime = np.linspace(0,simLength, N)
flag = 1
relerr = 1e-8
abserr = 1e-6

# Sim output storage
y_out = np.zeros((N,10))
x_out = np.zeros((N,3))
e_out = np.zeros((N,3))
H_out = np.zeros((N,))

r_sc0 = r_sc0/DU
v_sc0 = v_sc0/(DU*TU)

x_out[0,:] = r_sc0
e_out[0,:] = (smaEarth/DU)*np.array([1,0,0])

u0 = ks_init(r_sc0, v_sc0, 0, mu_s, 0)

y_out[0,:] = u0
yp = np.zeros((10,))
Hrel = np.zeros((N,))

for i,x in enumerate(simTime):
    if i==0:
        H_out[i] = u0[8]
        Hrel[i] = 0
    else:
        y = np.reshape(y_out[i-1,:],(10,))
        t0 = simTime[i-1]
        y, yp, t, flag = r8_rkf45( dynamics_ks, 10,
                                   y, yp, t0, x, relerr, abserr, flag)
        y_out[i,:] = y
        H_out[i] = y[8]
        Hrel[i] = (H_out[i] - H_out[i-1])/H_out[i]
        
        if flag == 7 or flag == 6:
            flag = 2

        p, pdot = ks2cart(y)
        x_out[i,:] = p
        t = y[9]*np.linalg.norm(y[0:4])
        lonEarth = wEarth*t/TU
        e_out[i,:] = (smaEarth/DU)*np.array([math.cos(lonEarth),
                                             math.sin(lonEarth), 0])
        ve = (smaEarth/(DU*TU))*wEarth*np.array([-math.sin(lonEarth),
                                            math.cos(lonEarth), 0])

        #print(np.linalg.norm(ve))
        
        if not inSoI:
            r_sc_e = p - e_out[i,:]
        else:
            r_sc_e = p
        
        if np.linalg.norm(r_sc_e) <= R_SoI/DU:
            # We're in Earth's SoI
            if not inSoI:
                print("Capturing")
                inSoI = True
                r_sc = r_sc_e
                print(np.linalg.norm(r_sc))
                v_sc = pdot - ve
                y = ks_init(r_sc, v_sc, y[9], mu_e, 0)
                y_out[i, :] = y
                p, pdot = ks2cart(y)
                yp = dynamics_ks(x, y)

            x_out[i, :] = p + e_out[i, :]
        else:
            # We're outside of it, back to heliocentric
            if inSoI:
                print("Releasing")
                inSoI = False
                r_sc = p + e_out[i, :]
                print(np.linalg.norm(r_sc))
                v_sc = pdot + ve
                y = ks_init(r_sc, v_sc, y[9], mu_s, 0)
                yp = dynamics_ks(x, y)
                y_out[i, :] = y

print("Finished: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

fig, ax = plt.subplots()
ax.plot(e_out[:,0], e_out[:,1])
ax.plot(x_out[:,0], x_out[:,1])
ax.plot([0.0],[0.0],'r+')
ax.set(xlabel='X - AU',ylabel='Y - AU',
       title='Planar Motion of Two Bodies')
ax.legend(['Earth','Spacecraft','Sol'])
fig.savefig('PlanarPath.png')

fig1, ax1 = plt.subplots()
ax1.plot(simTime/secondsPerDay*smaEarth, Hrel)
ax1.set(xlabel='Time (Days)',ylabel='Energy',
        title='Hamiltonian (Total Energy)')
fig1.savefig('Hamiltonian.png')

plt.show()
