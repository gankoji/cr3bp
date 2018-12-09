import numpy as np
import math

from Constants import *

DU = 1.0
TU = 1.0
inSoI = False
def setUnits(d, t):
    global DU, TU

    DU = d
    TU = t
    
def dynamics_ks(t, y):
    global inSoI
    # State Vector
    # u[0:4] -> u1-u4, KS-position
    # u[4:8] -> u1'-u4', KS-velocity
    
    ## u[8] -> h (-total energy) = (-Keplerian energy)
    ## + (-potential)

    # u[9]   -> t, physical time

    # Initialize the output
    y_dot = np.zeros((10,))
    
    # Radius of position and time
    x, xdot = ks2cart(y)
    r = np.linalg.norm(y[0:4])
    t = y[9]

    # Perturbing potential
    rVpot = 0.0
    drVdu = 0.0
    lonEarth = wEarth*t/TU
    
    # Perturbing Accelerations
    P = np.zeros((4,))
    rEarth = (smaEarth/DU)*np.array([math.cos(lonEarth),
                                     math.sin(lonEarth), 0])
    r_sc_e = np.linalg.norm(rEarth - (x))

    muFactor = 1/((TU**2)*(DU**3))
    if inSoI:
        # Third body is the sun
        ## print("Third body is the sun")
        r2 = -(smaEarth/DU)*np.array([math.cos(lonEarth),
                                      math.sin(lonEarth), 0])
        P[0:3] = accel3B(x, r2, mu_e*muFactor, mu_s*muFactor)
    else:
        ## print("Third body is the Earth")
        r2 = (smaEarth/DU)*np.array([math.cos(lonEarth),
                                     math.sin(lonEarth), 0])
        P[0:3] = accel3B(x, r2, mu_s*muFactor, mu_e*muFactor)

    L = ksmat(y)

    # Total acceleration
    Q = -drVdu/4.0 + 0.5*r*np.dot(np.transpose(L), P)

    # Velocities
    y_dot[0:4] = y[4:8]

    # Accelerations
    y_dot[4:8] = -0.5*y[8]*y[0:4] + Q
    y_dot[8] = -2*np.dot(y[4:8], np.dot(np.transpose(L),P))
    y_dot[9] = r

    return y_dot

def accel3B(r3, r2, mu1, mu2):
    r23 = (r3 - r2)
    r2N = np.linalg.norm(r2)
    r23N = np.linalg.norm(r23)
    accel = -(mu2/mu1)*((r23/(r23N**3)) + (r2/(r2N**3)))

    return accel

def ksmat(u):
    mat = np.array([[ u[0],  u[1], u[2],  u[3]],
                    [-u[1],  u[0], u[3], -u[2]],
                    [-u[2], -u[3], u[0],  u[1]],
                    [ u[3], -u[2], u[1], -u[0]]])

    return mat

def ks2cart(u):
    x = np.zeros((3,))
    xdot = np.zeros((3,))
    
    x[0] = u[0]**2 - u[1]**2 - u[2]**2 + u[3]**2
    x[1] = 2*(u[0]*u[1] - u[2]*u[3])
    x[2] = 2*(u[0]*u[2] + u[1]*u[3])
    r = np.linalg.norm(u[0:4])**2

    xdot[0] = 2*(u[0]*u[4] - u[1]*u[5] - u[2]*u[6] + u[3]*u[7])/r
    xdot[1] = 2*(u[1]*u[4] + u[0]*u[5] - u[3]*u[6] - u[2]*u[7])/r
    xdot[2] = 2*(u[2]*u[4] + u[3]*u[5] + u[0]*u[6] + u[1]*u[7])/r

    return x, xdot

def ks_init(r0, v0, t0, mu, V):
    u0 = np.zeros((10,))

    r = np.linalg.norm(r0)
    Ksq = mu/((DU**3)*(TU**2))

    if r0[0] >= 0.0:
        u0[0] = 0.0
        u0[3] = math.sqrt(.5*(r + r0[0]) - u0[0]**2)
        u0[1] = (r0[1]*u0[0] + r0[2]*u0[3])/(r + r0[0])
        u0[2] = (r0[2]*u0[0] - r0[1]*u0[3])/(r + r0[0])
    else:
        u0[1] = 0.0
        u0[2] = math.sqrt(.5*(r - r0[0]) - u0[1]**2)
        u0[0] = (r0[1]*u0[1] + r0[2]*u0[2])/(r + r0[0])
        u0[3] = (r0[2]*u0[1] - r0[1]*u0[2])/(r + r0[0])

    u0[4] = 0.5*np.dot(u0[0:3], v0)
    u0[5] = 0.5*np.dot(np.array([-u0[1],u0[0],u0[3]]), v0)
    u0[6] = 0.5*np.dot(np.array([-u0[2],-u0[3],u0[0]]), v0)
    u0[7] = 0.5*np.dot(np.array([u0[3],-u0[2],u0[1]]), v0)

    Kin = 0.5*np.dot(v0, v0)
    h = Ksq/r - Kin - V
    u0[8] = h

    u0[9] = t0

    return u0
