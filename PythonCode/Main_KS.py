import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

from Constants import *
from rkf45 import *
from Dynamics import dynamics, hamiltonian
from Dynamics_KS import dynamics_ks, ks_init, ks2cart, setUnits, inSoI

print("Started: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

## Escape velocity calculation
r0 = 7.5e3
V0 = 1.0

V_circ = math.sqrt(mu_e/r0)
V_e = math.sqrt(2)*V_circ

print("Circular Speed: " + str(V_circ) + " km/s. Escape Speed: " + str(V_e) + " km/s.")

# Initial Conditions

# km, Earth's average orbital distance, along the vernal equinox
r_e0 = np.array([smaEarth, 0, 0])

# km/s, Earth's average orbital velocity in the y inertial direction
v_e0 = np.array([0, smaEarth*wEarth, 0]) 
                                         
# km, Spacecraft's initial position added to Earth's
r_sc0 = r_e0 + np.array([r0, 0, 0])

# km/s, Spacecraft's initial velocity added to Earth's
v_sc0 = v_e0 + np.array([0, V0, 0]) 

## Check our starting position before normalizing
r_sc_e = np.linalg.norm(r_sc0 - r_e0)

simLength = 3650*secondsPerDay
endDate = simLength

if 0: #r_sc_e < R_SoI:
    # Start in Geocentric
    r_sc0 = r_sc0 - r_e0
    v_sc0 = v_sc0 - v_e0
    inSoI = True
    DU = np.linalg.norm(r_sc0)
    TU = math.sqrt((mu_e/1e7)/(DU**3))
    dt = 3600
    simLength = simLength*TU
    dt = dt*TU
else:
    # Start in Heliocentric
    inSoI = False
    DU = np.linalg.norm(r_sc0)
    TU = math.sqrt(mu_e/(DU**3))
    dt = 180
    simLength = simLength*TU
    dt = dt*TU
    
# Unit Conversion
setUnits(DU, TU)

print("Sim Length: " + str(simLength))
J = int(math.floor(simLength/dt))
N = int(2e3)
simTime = np.linspace(0,simLength, N)
flag = 1
relerr = 1e-8
abserr = 1e-6

# Sim output storage
y_out = np.zeros((N,10))
x_out = np.zeros((N,3))
x_gc_out = np.zeros((N,3))
e_out = np.zeros((N,3))
H_out = np.zeros((N,))

r_sc0 = r_sc0/DU
v_sc0 = v_sc0/(DU*TU)

x_out[0,:] = r_sc0
e_out[0,:] = (smaEarth/DU)*np.array([1,0,0])
x_gc_out[0, :] = x_out[0, :] - e_out[0, :]

u0 = ks_init(r_sc0, v_sc0, 0, mu_s, 0)

y_out[0,:] = u0
yp = np.zeros((10,))
Hrel = np.zeros((N,))

exitFlag = True
i = 0
while exitFlag:
    if i==0:
        H_out[i] = u0[8]
        Hrel[i] = 0
    else:
        x = simTime[i]
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
        t = y[9]
        lonEarth = wEarth*t/TU
        e_out[i,:] = (smaEarth/DU)*np.array([math.cos(lonEarth),
                                             math.sin(lonEarth), 0])
        ve = (smaEarth/(DU*TU))*wEarth*np.array([-math.sin(lonEarth),
                                                 math.cos(lonEarth), 0])

        if fabs(y[9] - endDate*TU) < 1e-3:
            exitFlag = False
            print("End Time: " + str(y[9]))
        elif i >= N-1:
            print("Exceeded number of steps at: " + str(y[9]))
            exitFlag = False
            
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
                v_sc = pdot + ve
                y = ks_init(r_sc, v_sc, y[9], mu_s, 0)
                yp = dynamics_ks(x, y)
                y_out[i, :] = y
                x_out[i, :] = p + e_out[i, :]
        # print(y[9])
    x_gc_out[i, :] = x_out[i, :] - e_out[i, :]
    i = i + 1
    # print(i)


new_e_out = np.trim_zeros(e_out[:,0])
a = new_e_out.size

new_e_out = np.zeros((a, 2))
new_e_out[:, 0] = np.trim_zeros(e_out[:,0])
new_e_out[:, 1] = np.resize(np.trim_zeros(e_out[:,1]), (a,))
new_e_out = new_e_out[:-1, :]

new_x_out = np.zeros((a, 2))
new_x_out[:, 0] = np.trim_zeros(x_out[:,0])
new_x_out[:, 1] = np.resize(np.trim_zeros(x_out[:,1]), (a,))
new_x_out = new_x_out[:-1, :]

new_x_gc_out = np.zeros((a, 2))
new_x_gc_out[:, 0] = np.trim_zeros(x_gc_out[:,0])
new_x_gc_out[:, 1] = np.resize(np.trim_zeros(x_gc_out[:,1]), (a,))
new_x_gc_out = new_x_gc_out[:-1, :]

print("Finished: " + datetime.datetime.now().strftime("%H:%M:%S.%f"))

fig2, ax2 = plt.subplots()
ax2.plot(new_x_gc_out[:, 0], new_x_gc_out[:, 1])
ax2.set(xlabel='X - AU',ylabel='Y - AU',
       title='Geocentric Satellite Motion')
ax2.legend(['Spacecraft'])
fig2.savefig('GeoPlanarPath.png')

fig, ax = plt.subplots()
ax.plot(new_e_out[:,0], new_e_out[:,1])
ax.plot(new_x_out[:,0], new_x_out[:,1])
ax.plot([0.0],[0.0],'r+')
ax.set(xlabel='X - AU',ylabel='Y - AU',
       title='Planar Motion of Two Bodies')
ax.legend(['Earth','Spacecraft','Sol'])
fig.savefig('PlanarPath.png')

fig1, ax1 = plt.subplots()
ax1.plot(simTime/secondsPerDay*smaEarth, H_out)
ax1.set(xlabel='Time (Days)',ylabel='Energy',
        title='Hamiltonian (Total Energy)')
fig1.savefig('Hamiltonian.png')

plt.show()
