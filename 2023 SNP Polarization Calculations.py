'''
This is the neutron going across the center line
'''
import numpy as np
import matplotlib.pyplot as plt
import math

def dP_dt(P, B):
    gamma = -1*1.832 * (10**8)
    # Calculate the derivative of P with respect to time
    return np.cross(P, B) * gamma

def runge_kutta_step(P, B, dt):
    # Perform a single Runge-Kutta step
    k1 = dt * dP_dt(P, B)
    k2 = dt * dP_dt(P + 0.5 * k1, B)
    k3 = dt * dP_dt(P + 0.5 * k2, B)
    k4 = dt * dP_dt(P + k3, B)
    return P + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def solve_runge_kutta(P0, B, v, dt, num_steps, x_range):
    # Solve the equation using the Runge-Kutta method
    P = np.zeros((num_steps, 3))
    P[0] = P0
    x = np.linspace(0.0, x_range, num_steps)
    for i in range(1, num_steps):
        P[i] = runge_kutta_step(P[i-1], B[i], dt)
        if x[i] >= x_range:
            break
            
    return P[:i+1]

# Parameters
dt = 0.01  # Time step
num_steps = 496  # Number of steps
xyz_step = 0.2
v = 395600/2  # Neutron velocity cm/s
# Adjust time step according to velocity
dt = xyz_step / v

# Initial conditions
P0 = np.array([0.0, 0.0, 1.0])  # Initial polarization vector

# Read magnetic field distribution (B) from file
data = np.loadtxt(r"C:\Users\3qi\OneDrive - Oak Ridge National Laboratory\Instrument_Table sol to sol 3A 2 mu [-43, .2, 496].txt")
zcoordinates = data[:, 1]  # x, y, z coordinates

# Find min and max z values
min_z = np.min(zcoordinates)
max_z = np.max(zcoordinates)
# Adjust z range
z_range = max_z - min_z
# find number of data pts along each direction
#znum = int(z_range / xyz_step + 1)

B = data[:, 2:5]  # Bx, By, Bz values

#print(B[:,1])
# Interpolate magnetic field values along the neutron trajectory
z = np.linspace(min_z, max_z, num_steps)
B_interpolated = np.zeros((num_steps, 3))
timepts = np.linspace(0, z_range / v, num_steps)

print("Initial Polarization:", P0)

def Pmag(x, y, z):
    return math.sqrt(abs(x**2) + abs(y**2) + abs(z**2))

def P0mag(P):
    return math.sqrt(P[0]**2 + P[1]**2 + P[2]**2)

for i in range(3):
    #B_interpolated[:, i] = np.interp(x, coordinates[:, 0], B[:, i])
    B_interpolated[:, i] = np.interp(z, zcoordinates[:], B[:, i])
    #TEMPORARY DOUBLE FIELD VALUES
    for ir, row in enumerate(B_interpolated):
        for ic, col in enumerate(row):
            B_interpolated[ir][ic]=B_interpolated[ir][ic]*1
    
# Solve the equation
P = solve_runge_kutta(P0, B_interpolated, v, dt, num_steps, z_range)

# Final polarization when exit
final_polarization = P[-1]
print("polarization update:", final_polarization)
    
print("Change in polarization:", (final_polarization - P0))
print("done")

init_Pmag = P0mag(P0)
fin_Pmag = P0mag(final_polarization)
print("average final magnitude:", fin_Pmag)
print("change in Pmag:", init_Pmag - fin_Pmag)

plt.plot(timepts, P, label = ['Px', 'Py', 'Pz'])
plt.legend()
plt.title("Px, Py, Pz, vs Time through the center")
plt.xlabel("Time (s)")
plt.ylabel("Px, Py, Pz")