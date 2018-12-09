import math

# Physical Constants
mu_e = 3.98601e5 # km^3/s^2, Earth's gravitational parameter
mu_s = 1.32712428e11 # km^3/s^2, Sol's gravitational parameter
R_e = 6.37122E3 # km, Earth's equatorial radius
smaEarth = 1.49598023e8 # m, Earth's semimajor axis
wEarth = math.sqrt((mu_e + mu_s)/(smaEarth**3))
R_SoI = (mu_e/mu_s)**(2.0/5.0)*smaEarth
secondsPerDay = 86400
