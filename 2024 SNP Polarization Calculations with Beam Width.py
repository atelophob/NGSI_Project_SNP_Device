# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:09:12 2024

@author: klx
"""


import numpy as np

import matplotlib.pyplot as plt

import math

import pandas as pd


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




def Pmag(x, y, z):

    return math.sqrt(abs(x**2) + abs(y**2) + abs(z**2))



def P0mag(P):

    return math.sqrt(P[0]**2 + P[1]**2 + P[2]**2)


# Parameters

dt = 0.01  # Time step

num_steps = 391  # Number of steps

xyz_step = 0.2 #dist b/w steps

beam_radius = 1 #in cm (xyz step size also applies to this)
print("beam radius:", beam_radius)

v = 395600/2.46 #speed of neutrons, divided by the angstrom speed value

# Adjust time step according to velocity
dt = xyz_step / v

# Initial conditions

P0 = np.array([1.0, 0.0, 0.0])  # Initial polarization vector
print("Initial Polarization:", P0)


# Read magnetic field distribution (B) from file
#data = np.loadtxt(r"C:\Users\klx\OneDrive - Oak Ridge National Laboratory\SNP Device Nutator Work SC_Table 1cm beam [78, .2, 391] [260An] [+X].txt")

xVals = []
zVals = []
initialP = []
finalP = []
deltaP = []
finalPmag = []
deltaPMag = []

pddata = pd.read_csv(r"C:\Users\klx\OneDrive - Oak Ridge National Laboratory\SNP Device Nutator Work SC_Table 1cm beam [78, .2, 391] [260An] [+X].txt", header = 0, sep='\s+')
df = pd.DataFrame(pddata)

for k in range(int(beam_radius * -100), int(beam_radius * 100 + 1), int(xyz_step * 100)):
    for j in range(int(beam_radius * -100), int(beam_radius * 100 + 1), int(xyz_step * 100)):
        mask = (df['X'] == k/100) & (df['Z'] == j/100)
        #mask = (df['X'] == 0) & (df['Z'] == 0)
        #print(k/100, j/100)
        result = df[mask]
        data = result[['Y', 'Bx', 'By', 'Bz']].to_numpy()

        #print(data)
        
        
        #zcoordinates = data[:, 1]  # x, y, z coordinates
        zcoordinates = data[:, 0]  # x, y, z coordinates
        
        
        # Find min and max z values
        
        min_z = np.min(zcoordinates)
        
        max_z = np.max(zcoordinates)
        
        # Adjust z range
        
        z_range = max_z - min_z
        
        # find number of data pts along each direction
        
        #znum = int(z_range / xyz_step + 1)
        
        
        
        #B = data[:, 2:5]  # Bx, By, Bz values
        B = data[:, 1:4]  # Bx, By, Bz values
        
        
        
        #print(B[:,1])
        
        # Interpolate magnetic field values along the neutron trajectory
        
        z = np.linspace(min_z, max_z, num_steps)
        
        B_interpolated = np.zeros((num_steps, 3))
        
        timepts = np.linspace(0, z_range / v, num_steps)
        
        
        for i in range(3):
        
            #B_interpolated[:, i] = np.interp(x, coordinates[:, 0], B[:, i])
        
            B_interpolated[:, i] = np.interp(z, zcoordinates[:], B[:, i])
        
            for ir, row in enumerate(B_interpolated):
        
                for ic, col in enumerate(row):
        
                    B_interpolated[ir][ic]=B_interpolated[ir][ic]*1 ##change this number to change field values
        
            
        # Solve the equation
        
        P = solve_runge_kutta(P0, B_interpolated, v, dt, num_steps, z_range)
        
        # Final polarization when exit

        final_polarization = P[-1]
        delta_polarization = final_polarization - P0
        init_Pmag = P0mag(P0)
        fin_Pmag = P0mag(final_polarization)
        delta_Pmag = init_Pmag - fin_Pmag
        
        #inserting calculated values into lists
        xVals.append(k/100)
        zVals.append(j/100)
        initialP.append(P0)
        finalP.append(final_polarization)
        deltaP.append(delta_polarization)
        finalPmag.append(fin_Pmag)
        deltaPMag.append(delta_Pmag)

df2 = pd.DataFrame({'X': xVals, 'Z': zVals, 'initial polarization': initialP, 'final polarization': finalP, 'change in polarization': deltaP, 'final magnitude': finalPmag, 'change in Pmag': deltaPMag})

avg_finalp = sum(finalP)/len(finalP)
print("average final polarization", avg_finalp)

avg_deltap = sum(deltaP)/len(deltaP)
print("average change in polarization", avg_deltap)

avg_finalpmag = sum(finalPmag)/len(finalPmag)
print("average final polarization", avg_finalpmag)

avg_deltapmag = sum(deltaPMag)/len(deltaPMag)
print("average change in polarization", avg_deltapmag)


print(df2)

