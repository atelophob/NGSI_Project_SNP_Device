#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 07:47:56 2024

@author: kean

This is the full translated python version of a MatLab program developed my Sichao
This code cannot really be used as solving through all path points takes many days


"""
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LogLocator

import matplotlib.pyplot as plt
import pandas as pd
import os


#this bloch equation solver has not been tested
def EulerFixed_BlochEq(z_data, B_data, h, gaman, vn, P0, tol):
    z = np.arange(min(z_data), max(z_data) + h, h) #in cm
    B = np.interp(z, z_data, B_data, left = None, right = None) #in T
    P = np.zeros((len(z), 3))
    P[0, :] = P0
    z = z/100 # in m
    h = h/100 #in m
    gaman = gaman * 10000 # in /s/T
    
    for i in range(len(z) - 1):
        Bp = B[i+1, :]
        Pp = P[i, :]
        Pi0 = Pp
        err = 1
        
        n = 0
        while err>tol and n<100:
            Pp = P[i, :] + h * gaman / vn * np.cross(Pp, Bp)
            Pi1 = Pp
            err = np.rad2deg(np.arccos(np.dot(Pi1, Pi0) / (np.linalg.norm(Pi1) * np.linalg.norm(Pi0))))
            Pi0 = Pi1
            n += 1
        
        P[i+1, :] = Pi1
        P[i+1, :] = P[i+1, :] / np.linalg.norm(P[i+1, :])
        
    P_data = np.interp(z_data / 100, z, P, left = None, right = None)
    return P_data

def fit_function_1(x, a, b):
    return a * x**2 + b * x + 1

def fit_function_2(x, a, b):
    return a * x**2 + b * x

# The code is used to solve the spin transport for every flight path in a 
# horizontally divergent scattering beam, and calculates the good paths to 
# all paths ratio and averaged deviation as well as their evolutions with 
# coverage angle.  


#CONSTANTS
gaman = 1.83247172e4 # in /G/s
hplank = 6.626e-34
mneutron = 1.674e-27
lamdaneutron = [2.33]

lstp = 0.1 # element spacing in cm on line path
hstp = np.concatenate((np.arange(0.1, 1.0, 0.1), np.arange(1, 11), np.array([100])))
 # step size for explicit ode if not use adaptive ode
tol = .001

filenameid = 'SNP Development 60 deg Tri' # name of .mn work file. The data files are named in a pattern as (.mn )-p(problem no.)_3DTable.txt.
loglocation = r'/Users/kean/Downloads' # file location

log_filepath = os.path.join(loglocation, f"{filenameid}_log.txt")  #must be already created on your device

CRa = [0.2] # current ratio (center coil /side coil)
pro = [1] # problem number
Bamp = [2500] # B field amplification


writedata = 0 # write polarization for every path. 1-yes, 0-no.
writeiv = 0 # record the vector plot of spin transport for all paths and save as .avi. 1-yes, 0-no.
pausetime = 0 # pause between vector plot for every path
plotflag = 0 # plot the path, B and spin transport for all paths and save as .jpg. 1-yes, 0-no.
statplotflag = 1 # plot statictis results. 1-yes, 0-no.


count = 0 #used to see how far you are in the solve -- count is printed in a for loop later


with open(log_filepath, 'a') as logid:  # you must create a log file so the program can write results into it
    
    figurefolder = 'Figures-nofit\ ' # folder name to save figures (if choose not to fit calculated polarization)
    
    
    # Film200x90 configuration geometry
    x1 = -20.5 
    x2 = 0
    y1 = 0.58
    y2 = 9.58
    z1 = -0.06
    z2 = 17.84
    nx = 205
    ny = 91
    nz = 180
    
    bd1 = np.array([-2.847, 1.791]) # path boundary
    bd2 = np.array([-9.853, 16.611])
    
    l1x = np.array([-0.5, -20.5])
    l1z = np.array([-0.01, -0.01])
    l2x = np.array([-0.241, -10.241])
    l2z = np.array([0.438, 17.759])
    #coil 1 - bottom left
    c1x = np.array([-2.974, -2.974])
    c1z = np.array([0.228, 1.571])
    # coil 2 - up left
    c2x = np.array([-2.847, -1.684])
    c2z = np.array([1.791, 2.462])
    # coil3 - bottom right
    c3x = np.array([-19.313, -19.313])
    c3z = np.array([0.228, 1.571])
    # coil 4 - center right
    c4x = np.array([-19.009, -11.215])
    c4z = np.array([1.975, 15.475])
    # coil 5 - up right
    c5x = np.array([-11.016, -9.853])
    c5z = np.array([15.940, 16.611])
    
    ps1 = np.array([-3.857, -0.01]) # path's starting point postion range
    ps2 = np.array([-19.478, -0.01])
    pe1 = np.array([-1.922, 3.348]) # path's ending point postion range
    pe2 = np.array([-9.730, 16.873])
    
    cs = (ps1+ps2)/2 # reference center path starting point
    ce = (pe1+pe2)/2 # reference center path ending point
    vc = ce-cs # reference center path vector
    angc = np.rad2deg(np.arccos(np.dot(vc,np.array([1,0]))/np.linalg.norm(vc)))
    
    axisrange = np.array([-21, 0, -2, 18])
    
    #numerical operations below
    nyp = 0 # plane of path. 0-central plane, >0-offset from central plane
    nyf = (ny+1)/2
    nyfadd = nyp/((y2-y1)/(ny-1))
    nyf = nyf+nyfadd
    
    dis0 = 5 # sample-to-assembly (first film) distance (cm?)
    divang_Array = np.arange(16, 51, 2) # scattering angle range to cover (horizontal), in deg
    bsz = 0.2 # beam size
    angnum = 3 # define path number to calculate or step size in deg
    angstp = 0.4 # set as 0 if define path number, else 0.2 or 0.4
    
    positionx_Array = np.arange(-2, 2.1, .4) # offset of center path starting point from the reference center path
    positionang_Array = np.arange(-10, 5, 2) # offset of center path orientation from the reference center path, in deg
    
    
    angPZ_A1_tolerance0 = 175 # define max deviation of good path
    angPZ_A1_tolerance1 = 170 # max deviation of not bad path
    angPZ_A1_tolerance2 = 160 # min deviation of bad path
    
    ode23flag = 1 # ode method, 1-implicit solution, 0-explict solution
    
    fitflag = 1
    
    figurefolder = 'Figures-fit\ ' # folder name to save figures (if choose to fit calculated polarization)
    
    print("Variable Initialization Complete")
    
        # every current ratio
    for icr in range(0, len(CRa)):
        
        icrstr = str(icr)
        # read data file
        # filename = str(icr+1)+ '_Table.txt' #still need to change filename--this is not what the current files are named as
        filenameid = f"{filenameid}{pro[icr]}_Table.txt"
        # read data file
        # fileid = open([filenameid,'-p',num2str(pro(icr+1)),'_3DTable.txt']) # filename of field data of problem no. x
        
        data = np.loadtxt(filenameid, skiprows=1)
        print("data = np.loadtxt complete")
        
        with open(filenameid, 'r') as fileid:
            # Skip the first line (header)
            for _ in range(1):
                next(fileid)
            
            # Read the remaining data
            data0 = pd.read_csv(fileid, delim_whitespace=True, header=None)
            
            # Extract relevant columns (assuming columns are zero-indexed in Python)
            data2 = data0.iloc[:, :7].to_numpy()  # Adjust column range if necessary
            print("Data2 Extracted")
            
        CR = CRa[icr]
    
        # permute data matrix
        for i in range(1,7):     
            tmp1 = data2[:,i].reshape((nz,ny,nx),order = 'F')
            tmp = np.zeros((ny,nx,nz))
            tmp2 = np.zeros((nz,nx,ny))
            for j in range(0, nz):
                for k in range(0, nx):
                    tmp[:,k,j] = tmp1[j,:,k]
            
            for j in range(0, ny):
                for k in range(0, nx):
                    tmp2[:,k,j] = tmp1[:,j,k]
            
            if i == 1:
                # case 1
                X = tmp # xy z
                X1 = tmp1 #yz x
                X2 = tmp2 #xz y
            elif i == 2:
                # case 2
                Y = tmp
                Y1 = tmp1
                Y2 = tmp2
            elif i == 3:
                # case 3
                Z = tmp
                Z1 = tmp1
                Z2 = tmp2
            elif i == 4:
                # case 4
                Bx = tmp
                Bx1 = tmp1
                Bx2 = tmp2
            elif i == 5:
                # case 5
                By = tmp
                By1 = tmp1
                By2 = tmp2
            elif i == 6:
                # case 6
                Bz = tmp
                Bz1 = tmp1
                Bz2 = tmp2
    
        # plane field to analyze
        Xf = X2[:,:,int(nyf-1)]
        Zf = Z2[:,:,int(nyf-1)]
        Bxf = Bx2[:,:,int(nyf-1)]
        Byf = By2[:,:,int(nyf-1)]
        Bzf = Bz2[:,:,int(nyf-1)]
        
        print("Xf,Zf,Bxf,Byf,Bzf 1st Round Set")
    
        for iBamp in range(0, len(Bamp)):
    
            for ilamda in range(0, len(lamdaneutron)):
    
                vn = hplank / mneutron / (lamdaneutron[ilamda] * 1e-10)  # in m/s
            
                logid.write('\n')
                logid.write(f'{nyp} cm plane, {lamdaneutron[ilamda]} Angstroms\n')
            
            
                #if block below does not work/isn't fully fixed
                if writeiv == 1:
                    video_filename = f"{figurefolder}{filenameid}-CR{CR}_x{Bamp[iBamp]}B_PL{nyp}y_lamda{lamdaneutron[ilamda]}.avi"
                    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    # videof1 = cv2.VideoWriter(video_filename, fourcc, 3.0, (640, 480))  # Assuming frame size (640, 480)
                    #can't get cv2 imported atm
        
                stat1_Array = np.array([]) 
                print("stat1_Array initialized")
                
                for iposx in range (0, len(positionx_Array)): # calculate for every center path starting point 
                    posx = positionx_Array[iposx]
                    
                    for iposang in range (0, len(positionang_Array)): # calculate for every center path orientation
                        posang = positionang_Array[iposang]
                        Rpos = np.array([[np.cos(np.deg2rad(posang)), -1*np.sin(np.deg2rad(posang))], [np.sin(np.deg2rad(posang)), np.cos(np.deg2rad(posang))]]) # rotation matrix, from x//film to path
    
                        uvc = vc/np.linalg.norm(vc)
                        uvc = np.dot(Rpos, uvc) # rotate reference center path vector to center path direction
    
    
                        if vc[0] == 0:
                            uvb = np.array([1,0])
                        elif vc[1] == 0:
                            uvb = np.array([0,1])
                        else:
                            uvb = np.array([1, -1*uvc[0]/uvc[1]])
                            uvb = uvb/np.linalg.norm(uvb) # vector on beam cross section, normal to center path
    
    
                        alpha = 90-np.rad2deg(np.arccos(np.dot(uvc,(ps1-ps2))/np.linalg.norm(ps1-ps2))) # angle between center path vector and film normal
                        dis = dis0/np.cos(np.deg2rad(alpha)) # sample to film distance on path
                        pbc = cs + np.array([posx,0]) - uvc * dis # beam center at sample position (vector + vector) - vector * constant
                        print("currently right before: for idivang in range(0, len(divang_Array)):")

                        for idivang in range(0, len(divang_Array)): # calculate for every divergent path
                            divang = divang_Array[idivang]
                            if angstp == 0:
                                angarray = np.linspace(-1*divang/2, divang/2, angnum)
                            else:
                                
                                angarray = np.arange(-1*divang/2, divang/2 + angstp, angstp)
    
                                angnum = len(angarray)
                            
    
                            bszarray = np.linspace(-1*bsz/2, bsz/2, angnum)
    
                            hstpbestarray = np.zeros((angnum, 2))
    
                            angPZ_A1_AllArray = [] #note these are lists
                            angPZ_A1_5Array = []  #note these are lists
                            shadowflag = 0
                            
                            print("currently right before: for ix in range(0, angnum):")
                            
                            # every angle
                            for ix in range(0, angnum):
                                iang = ix
                                pb = pbc - uvb * bszarray[ix] # assume average spaced scattering position, may not be the real case
                                dist = dis * np.cos(np.deg2rad(alpha)) / np.cos(np.deg2rad(alpha - angarray[iang])) # sample-to-film distance on divergent path
                                Rt = np.array([[np.cos(np.deg2rad(angarray[iang])),  -1*np.sin(np.deg2rad(angarray[iang]))], [np.sin(np.deg2rad(angarray[iang])), np.cos(np.deg2rad(angarray[iang]))]])    
                                uvt = np.dot(Rt, uvc) # rotate center path vector to divergent path direction
                                uvt = uvt/np.linalg.norm(uvt)
    
                                # interpolation and extract line field on
                                # path
                                lxstp = lstp * uvt[0]
                                lzstp = lstp * uvt[1]
                                # test = np.meshgrid(np.arange(x1, x2 + abs(lxstp), abs(lxstp)), np.arange(z1, z2 + lzstp, lzstp))
                                # Xq, Zq = np.meshgrid(np.arange(x1, x2 + abs(lxstp), abs(lxstp)), np.arange(z1, z2 + lzstp, lzstp))
                                Xq, Zq = np.meshgrid(np.arange(x1, x2, np.abs(lxstp)), np.arange(z1, z2, lzstp))
                                count += 1
                                print("Xq and Zq initilization done (np.meshgrid), count: " + str(count))
                                
                                #this section to the next print statement is the most time-consuming step in this program
                                Xf_flat = Xf.flatten()
                                Zf_flat = Zf.flatten()
                                
                                Bxq = griddata((Xf_flat, Zf_flat), Bxf.flatten(), (Xq, Zq), method='linear')
                                Byq = griddata((Xf_flat, Zf_flat), Byf.flatten(), (Xq, Zq), method='linear')
                                Bzq = griddata((Xf_flat, Zf_flat), Bzf.flatten(), (Xq, Zq), method='linear')
                                
                                Xq = np.flip(Xq, axis=1)
                                Zq = np.flip(Zq, axis=1)
                                Bxq = np.flip(Bxq, axis=1)
                                Byq = np.flip(Byq, axis=1)
                                Bzq = np.flip(Bzq, axis=1)
    
                                print("Xq,Zq,Bxq,Byq,Bzq griddata and flipping complete")
    
                                xs = pb + uvt * dist
                                xs = xs[0]
    
                                nxs = np.argmin(np.abs(Xq[0,:] - xs)) +1 #NXS WAS SMALLER THAN MATLAB'S NXS SO +1
    
                                nzs = 1
    
                                lang = 90 - alpha + angarray[iang]
                                if lang - 90 == 0:
                                    nxe = nxs
                                    nze = Zq.shape[0]
                                elif lang - 90 < 0:
                                    lnum = min(Zq.shape[0], nxs)
                                    nxe = nxs - lnum + 1
                                    nze = lnum
                                else:
                                    lnum = min(Zq.shape[0], Zq.shape[1] - nxs + 1)
                                    nxe = nxs + lnum - 1
                                    nze = lnum
                                
    
                                data = np.zeros((lnum,7))
                                data[:,0] = np.arange(1, lnum + 1)
                                data[:,2] = np.repeat(Y2[0 ,0, int(nyf-1)], lnum)
    
                                if nxs <= nxe:
                                    tmp2 = np.arange(nxs - 1, nxe)
                                else:
                                    tmp2 = np.arange(nxs - 1, nxe - 2, -1)
                                
                                if tmp2.size == 0:
                                    tmp2 = np.arange(nxs - 1, nxe - 2, -1)
                                
                                tmp1 = np.arange(nzs - 1, nze)
                                
                                for il in range(0, lnum):
                                    data[il,1] = Xq[tmp1[il], tmp2[il]]
                                    data[il,3] = Zq[tmp1[il], tmp2[il]]
                                    data[il,4] = Bxq[tmp1[il], tmp2[il]]
                                    data[il,5] = Byq[tmp1[il], tmp2[il]]
                                    data[il,6] = Bzq[tmp1[il], tmp2[il]]
                                
    
    
                                s = data[:, 1:4]
                                Bbase = data[:, 4:7]
                                B = Bbase * Bamp[iBamp] # B components
                                
                                print("s and B set once")
                                
                                # restrict the range of date to solve
                                ind2 = None
                                ind1 = 0
                                for i in range(0, s.shape[0] - 1):
                                    test = np.isnan(B[i+1, 0])
                                    if np.isnan(B[i+1, 0]) == True:
                                        B1check = 1
                                    else:
                                        B1check = 0
                                    if np.isnan(B[i, 0]) == True:
                                        B2check = 1
                                    else:
                                        B2check = 0
                                    if B1check - B2check == 1:
                                        ind2 = i
                                        break
                                
                                s_ind2 = 1
                                
                                if ind2 is None:
                                    ind2 = s.shape[0] - 1
                                    s_ind2 = 0
                                
    
                                s = s[ind1:ind2+1,:] #+1 bc these ranges are exclusive in python but not in matlab
                                B = B[ind1:ind2+1,:]
    
                                endid = s.shape[0] - 1
                                while np.arccos(np.dot(s[endid, 0:3:2] - np.array([l2x[0], l2z[0]]), np.array([1, 0])) / np.linalg.norm(s[endid, 0:3:2] - np.array([l2x[0], l2z[0]]))) <= np.arccos(np.dot(np.array([l2x[1], l2z[1]]) - np.array([l2x[0], l2z[0]]), np.array([1, 0])) / np.linalg.norm(np.array([l2x[1], l2z[1]]) - np.array([l2x[0], l2z[0]]))):
                                    endid = endid - 1
                                
    
                                s = s[0:endid+1, :]
                                B = B[0:endid+1, :]
    
                                print("s and B finalized")
    
                                # define the conditions for unshadowed path
                                bd1v = (bd1 - s[0, [0,2]])
                                bd2v = (bd2-s[0, [0,2]])
                                
                                bd1ang = np.rad2deg(np.arccos(np.dot(bd1v, np.array([1, 0]))/np.linalg.norm(bd1v)))
                                bd2ang = np.rad2deg(np.arccos(np.dot(bd2v, np.array([1, 0]))/np.linalg.norm(bd2v)))
    
                                if lang >= bd1ang and lang <= bd2ang: # for every unshadowed path
    
                                    filename = f"{filenameid}-CR{CR}_ang{lang}xs{xs}_PL{nyp}y_lamda{lamdaneutron[ilamda]}"
                                    
                                    vl = s[-1, :] - s[0, :]
                                    thetadeg = np.rad2deg(np.arccos(np.dot([1, 0, 0], vl) / np.linalg.norm(vl)))
    
                                    theta1deg = 90 - 60
    
                                    theta = np.deg2rad(thetadeg)
                                    A = np.array([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-1*np.sin(theta), 0, np.cos(theta)]]) # coordinate transformation matrix, reference coordinate (x//upstream film and B, z_|_film) to path coordinate (x//path, z_|_path)
                                    theta1 = np.deg2rad(theta1deg)  
                                    A1 = np.array([[np.cos(theta1), 0, np.sin(theta1)], [0,1,0], [-1*np.sin(theta1), 0, np.cos(theta1)]]) # coordinate transformation matrix, reference coordinate (x//upstream film and B, z_|_film) to downstream film coordinate (x//downstream film and B, z_|_downstream film)
    
                                    B_A = np.zeros_like(B)
                                    s_A = np.zeros_like(s)
                                    s_A1 = np.zeros_like(s)
                                    B_A1 = np.zeros_like(B)
                                    
                                    for i in range(0, len(s)):
                                        s_A[i,:] = np.dot(A, s[i, :])
                                        B_A[i,:] = np.dot(A, B[i, :]) # path field in path coordinate system
                                        s_A1[i,:] = np.dot(A1, s[i, :])
                                        B_A1[i,:] = np.dot(A1, B[i, :]) # path field in downstream film coordinate system
                                    
    
                                    l = s_A[:, 0]
    
                                    l_Adia = l[2:]
                                    B_A_Adia = B_A[2:, :] * 1 * 10**4
                                    fl = np.gradient(B_A_Adia,lstp, axis = 0) #do i care about the axis being 0?????
                                    Adia = 1. / np.sum((B_A_Adia)**2, axis = 1) * np.sqrt(np.sum(fl**2, axis = 1)) * vn * 100 / gaman # adiabaticity on path
                                    print("adiabaticity on path calculated")


                                    # solve Bloch equation on path
                                    if ode23flag == 1:
                                        hnum = 1
                                    else:
                                        hnum = len(hstp)
    
                                    Pararray = np.zeros((hnum, 2*3))
                                    resarray = np.zeros(hnum)
                                    Poriarray = []
    
                                    for ihstp in range(0, hnum):
                                        h = lstp / hstp[ihstp]
                                        P0 = np.array([1, 0, 0])
    
                                        if ode23flag == 0:
                                            
                                            # Pori = RK4_BlochEq(l,B,h,gaman,vn,P0)
                                            # Pori = MidPt_BlochEq(l,B,h,gaman,vn,P0)
                                            Pori = EulerFixed_BlochEq(l,B,h,gaman,vn,P0,tol)
                                            # Pori = Radau2_BlochEq(l,B,h,gaman,vn,P0,tol)
                                            '''
                                            above here: need to go over the other files and look at the 
                                            correct bloch equation solver method to use for this and 
                                            fix the code for that as well
                                            '''
                                        elif ode23flag == 1:
                                            l_data = l / 100
                                            tspan = [l_data[0], l_data[-1]]
                                            
                                            B_interp = interp1d(l_data, B, kind='linear', fill_value='extrapolate', axis=0)
                                            
                                            def odefun(ll, Pori):
                                                B_ll = B_interp(ll)
                                                cross_prod = np.cross(Pori, B_ll)
                                                return gaman * 10000 / vn * cross_prod
                                            
                                            sol = solve_ivp(odefun, tspan, P0, method='RK23', rtol=tol, atol=tol)
                                            Pori = sol.y.T
                                            
                                            Poripart1 = np.interp(l_data, np.linspace(tspan[0], tspan[1], len(Pori)), Pori[:, 0], left=None, right=None, period=None)
                                            Poripart2 = np.interp(l_data, np.linspace(tspan[0], tspan[1], len(Pori)), Pori[:, 1], left=None, right=None, period=None)
                                            Poripart3 = np.interp(l_data, np.linspace(tspan[0], tspan[1], len(Pori)), Pori[:, 2], left=None, right=None, period=None)
                                            Pori = np.stack((Poripart1, Poripart2, Poripart3), axis = 1)
                                            
                                            for i in range(Pori.shape[0]):
                                                Pori[i, :] = Pori[i, :] / np.linalg.norm(Pori[i, :])
                                            
                                            print("Pori Set")
                                            
    
                                        # fit the results if needed
                                        Par0 = [0, 0]
                                        l0 = l - l[0]
                                        Parifit = np.zeros(6)
                                        resifit = 0
                                        
                                        
                                        for ifit in range(0,3):
                                            tmp1 = interp1d(l0, Pori[:,ifit], kind = 'linear', fill_value = 'extrapolate')(1)
                                            tmp2 = interp1d(l0, Pori[:,ifit], kind = 'linear', fill_value = 'extrapolate')(-1)
                                            
                                            if ifit == 0:
                                                fit_func = fit_function_1
                                                Par0[0] = (tmp1 + tmp2 - 2) / 2
                                                Par0[1] = (tmp1 - tmp2) / 2
                                            else:
                                                fit_func = fit_function_2
                                                Par0[0] = (tmp1 + tmp2) / 2
                                                Par0[1] = (tmp1 - tmp2) / 2
                                            
                                            
                                            Par, res = curve_fit(fit_func, l0, Pori[:, ifit], p0=Par0)
    
                                            Parifit[(ifit * 2):(ifit * 2 + 2)] = Par
                                            resifit += np.sum((fit_func(l0, *Par) - Pori[:, ifit])**2)
         
                                        Pararray[ihstp, :] = Parifit
                                        resarray[ihstp] = resifit
                                        Poriarray.append(Pori)
                                        print("pori added to pori array")
    
                                    
                                    # get the best solution with the best
                                    # step size which yields the min residuals if
                                    # not using adaptive ode 
                                    idbest = np.argmin(resarray)
                                    Parbest = Pararray[idbest,:]
                                    hstpbest = hstp[idbest]
                                    hstpbestarray[ix,:] = [ix, lstp / hstpbest] # in cm
                                    Pori = Poriarray[idbest]
                                    print("best soln saved to Pori")
    
                                    P = Pori.copy()
                                    
                                    angBZ_A1 = np.rad2deg(np.arccos(np.dot(B_A1[-1,:],[0, 0, 1]) / np.linalg.norm(B_A1[-1, :])))
    
                                    # fit the best solution for more smooth
                                    # P evolution
                                    if fitflag == 1:
                                        if angBZ_A1 >= angPZ_A1_tolerance1:
                                            for ifit in range (0,3):
                                                if ifit == 0:
                                                    g = lambda Par, x: Par[0] * x ** 2 + Par[1] * x + 1
                                                else:
                                                    g = lambda Par, x: Par[0] * x ** 2 + Par[1] * x
                                                
                                                P[:,ifit] = g(Parbest[(ifit)*2:(ifit)*2+2], l0)
                                        
                                        
                                    print("best soln fitted")
                                    
                                    P_A1 = np.zeros_like(P)
                                    
                                    for i in range(0,len(l)):
                                        P_A1[i,:] = np.dot(A1,P[i,:]) # transform P to downstream film coordinate system
                                    
                                    
                                    print("P transformed to downstream film coordinate system")
                                    
                                    # plot flight path, B vector and P
                                    # vector
                                    fontsizexl = 24
                                    fontsizet = 28
                                    fontsizeax = 20
    
                                    if plotflag == 1:
                                        
                                        skipdata = 1
                                        
                                        f1 = plt.figure(1)
                                        fig = plt.figure(figsize = (18, 6))
                                        plt.suptitle(f'Current Ration = {CR}, B field x {Bamp[iBamp]} on {nyp} cm plane, {lamdaneutron[ilamda]} Angstroms', ha='left')
                                        # Beam Trajectory subplot
                                        ax1 = plt.subplot(1, 3, 1)
                                        ax1.plot(l1x, l1z, linewidth=1)
                                        ax1.plot(l2x, l2z, linewidth=1)
                                        ax1.plot(c1x, c1z, linewidth=1)
                                        ax1.plot(c2x, c2z, linewidth=1)
                                        ax1.plot(c3x, c3z, linewidth=1)
                                        ax1.plot(c4x, c4z, linewidth=1)
                                        ax1.plot(c5x, c5z, linewidth=1)
                                        ax1.plot([data[0, 1], data[-1, 1]], [data[0, 3], data[-1, 3]], 'r', linewidth=1.5)
                                        ax1.invert_xaxis()
                                        ax1.set_xlabel('X, cm', fontsize=fontsizexl)
                                        ax1.set_ylabel('Z, cm', fontsize=fontsizexl)
                                        ax1.set_title('Beam Trajectory', fontsize=fontsizet)
                                        ax1.tick_params(axis='both', which='both', width=1.5, labelsize=fontsizeax)
                                        ax1.grid(True)
                                        ax1.set_aspect('equal')
                                        ax1.set_xlim(axisrange[0:2])
                                        ax1.set_ylim(axisrange[2:4])
                                        print("ax1 done")
                                        
                                        # B Field subplot
                                        ax2 = plt.subplot(1, 3, 2)
                                        ax2.plot(l1x, l1z, linewidth=1)
                                        ax2.plot(l2x, l2z, linewidth=1)
                                        ax2.plot(c1x, c1z, linewidth=1)
                                        ax2.plot(c2x, c2z, linewidth=1)
                                        ax2.plot(c3x, c3z, linewidth=1)
                                        ax2.plot(c4x, c4z, linewidth=1)
                                        ax2.plot(c5x, c5z, linewidth=1)
                                        
                                        ax2.quiver(s[::skipdata, 0], s[::skipdata, 2], B[::skipdata, 0], B[::skipdata, 2], color='r', linewidth=1.5)
                                        ax2.invert_xaxis()
                                        ax2.set_xlabel('X, cm', fontsize=fontsizexl)
                                        ax2.set_ylabel('Z, cm', fontsize=fontsizexl)
                                        ax2.set_title('B Field', fontsize=fontsizet)
                                        ax2.tick_params(axis='both', which='both', width=1.5, labelsize=fontsizeax)
                                        ax2.grid(True)
                                        ax2.set_aspect('equal')
                                        ax2.set_xlim(axisrange[0:2])
                                        ax2.set_ylim(axisrange[2:4])
    
    
                                        # Spin Transport subplot
                                        ax3 = plt.subplot(1, 3, 3)
                                        ax3.plot(l1x, l1z, linewidth=1)
                                        ax3.plot(l2x, l2z, linewidth=1)
                                        ax3.plot(c1x, c1z, linewidth=1)
                                        ax3.plot(c2x, c2z, linewidth=1)
                                        ax3.plot(c3x, c3z, linewidth=1)
                                        ax3.plot(c4x, c4z, linewidth=1)
                                        ax3.plot(c5x, c5z, linewidth=1)

                                        ax3.quiver(s[::skipdata, 0], s[::skipdata, 2], P[::skipdata, 0], P[::skipdata, 2], color='r', linewidth=1.5)
                                        ax3.invert_xaxis()
                                        ax3.set_xlabel('X, cm', fontsize=fontsizexl)
                                        ax3.set_ylabel('Z, cm', fontsize=fontsizexl)
                                        ax3.set_title('Spin Transport', fontsize=fontsizet)
                                        ax3.tick_params(axis='both', which='both', width=1.5, labelsize=fontsizeax)
                                        ax3.grid(True)
                                        ax3.set_aspect('equal')
                                        ax3.set_xlim(axisrange[0:2])
                                        ax3.set_ylim(axisrange[2:4])
                                        
    
                                    #this if block does not work atm
                                    if writeiv == 1:
                                        f1.savefig('temp_frame.png')
                                        frame = plt.imread('temp_frame.png')
                                        # videof1.write(frame) #commented out for now bc not doing video and couldn't get cv2 imported
                                        
                                        
                                        # getf1 = getframe(f1)
                                        # writeVideo(videof1,getf1)

    
    
                                    # calculate the deviation angle of the
                                    # spin at downstream film
                                    angPZ_YZ_A1 = np.rad2deg(np.arccos(np.dot([P_A1[endid, 1], P_A1[endid, 2]], [0, 1]) / np.linalg.norm([P_A1[endid, 1], P_A1[endid, 2]]))) # projection on plane normal to centeral plane
                                    angPZ_XZ_A1 = np.rad2deg(np.arccos(np.dot([P_A1[endid, 0], P_A1[endid, 2]], [0, 1]) / np.linalg.norm([P_A1[endid, 0], P_A1[endid, 2]]))) # projection on centeral plane
    
                                    angPZ_A1 = np.rad2deg(np.arccos(np.dot([P_A1[endid, 0], P_A1[endid, 1], P_A1[endid, 2]], [0, 0, 1]) / np.linalg.norm([P_A1[endid, 0], P_A1[endid,1], P_A1[endid,2]]))) # deviation angle in 3-D space
    
                                    print("deviation angles calculated")
    
                                    # save polarization results for every path except the
                                    # bad paths
                                    if writedata == 1:
                                        if angPZ_A1 > angPZ_A1_tolerance2:
                                            if angPZ_A1 > angPZ_A1_tolerance2:
                                                with open(f'{filename}_P_x{Bamp[iBamp]}B_PL{nyp}y_lamda{lamdaneutron[ilamda]}.txt', 'w') as fid:
                                                    fid.write(' Distance     Px           Py           Pz\n')
                                                    for i in range(len(l)):
                                                        fid.write(f'{l[i]:12.9f} {P[i, 0]:12.9f} {P[i, 1]:12.9f} {P[i, 2]:12.9f}\n')
                                        
                                                if theta1deg != 0:
                                                    with open(f'{filename}_P (Z on ds_ybco)_x{Bamp[iBamp]}B_PL{nyp}y_lamda{lamdaneutron[ilamda]}.txt', 'w') as fid:
                                                        fid.write(' Distance     Px           Py           Pz\n')
                                                        for i in range(len(l)):
                                                            fid.write(f'{l[i]:12.9f} {P[i, 0]:12.9f} {P[i, 1]:12.9f} {P[i, 2]:12.9f}\n')
      
    
                                    logid.write(f'{CR:.2f} {xs:.2f} {lang:.2f} {bd1ang:.2f} {bd2ang:.2f} {angPZ_A1:.4f} {angPZ_XZ_A1:.4f} {angPZ_YZ_A1:.4f} {Bamp[iBamp]:.0f}\n')
    
                                    angPZ_A1_AllArray = np.append(angPZ_A1_AllArray, angPZ_A1)
                                    
                                    if angPZ_A1 > angPZ_A1_tolerance0:
                                        angPZ_A1_5Array = np.append(angPZ_A1_5Array, angPZ_A1)
    
    
                                    # plot data in different color,
                                    # good-green, 170-175-red,
                                    # 160-170-yellow, bad-delete or red
                                    if plotflag == 1:
                                        plt.pause(pausetime)
                                        # plt.pause(10)

                                        if np.isnan(angPZ_A1):
                                            angPZ_A1 = -1
                                        
                                        if angPZ_A1 < angPZ_A1_tolerance2:
                                            # delete(Lplot)
                                            # delete(Bplot)
                                            # delete(Pplot)
                                            pass
                                        elif angPZ_A1 < angPZ_A1_tolerance1:
                                            #Lplot = ax1, Bplot = ax2, Pplot = ax3
                                            ax1.Color = 'y'
                                            ax2.Color = 'y'
                                            ax3.Color = 'y'
                                            
                                            
                                        elif angPZ_A1 > angPZ_A1_tolerance0:
                                            ax1.Color = 'g'
                                            ax2.Color = 'g'
                                            ax3.Color = 'g'
                                        
                                    #this if block does not work atm
                                    if writeiv == 1:
                                        f1.savefig('frame1.png')
    
                                        # getf1 = getframe(f1)
                                        # writeVideo(videof1,getf1)
                            
    
                                    # plot adiabaticity and P compenents 
                                    if plotflag == 1:
                                        f2 = plt.figure(2)
                                        f2.set_size_inches(18.5, 10.5, forward=True)
                                        plt.suptitle(f'Current Ratio = {CR}, B field x{Bamp[iBamp]} on {nyp} cm plane, {lamdaneutron[ilamda]} Angstroms', horizontalalignment='left')
    
                                        
                                        fAdia = plt.subplot(1, 3, 1)
                                        plt.semilogy(l_Adia, Adia)
                                        plt.xlabel('Flight Distance, cm', fontsize=fontsizexl)
                                        plt.ylabel('Adiabaticity', fontsize=fontsizexl)
                                        ax = plt.gca()
                                        ax.tick_params(axis='both', which='minor', direction='in', width=1.5)
                                        ax.tick_params(axis='both', which='major', direction='in', width=1.5)
                                        ax.xaxis.set_minor_locator(AutoMinorLocator())
                                        ax.yaxis.set_minor_locator(LogLocator())
                                        ax.grid(True)
                                        ax.set_box_aspect(1)
               
                                        
                                        fs = plt.subplot(1, 3, 2)
                                        plt.plot(l, P[:, 0], 'k', l, P[:, 1], 'k', l, P[:, 2], 'k')
                                        plt.plot(l, Pori[:, 0], 'r', l, Pori[:, 1], 'r', l, Pori[:, 2], 'r')
                                        plt.xlabel('Flight Distance, cm', fontsize=fontsizexl)
                                        plt.ylabel('Polarization Component', fontsize=fontsizexl)
                                        ax = plt.gca()
                                        ax.tick_params(axis='both', which='minor', direction='in', width=1.5)
                                        ax.tick_params(axis='both', which='major', direction='in', width=1.5)
                                        ax.xaxis.set_minor_locator(AutoMinorLocator())
                                        ax.yaxis.set_minor_locator(AutoMinorLocator())
                                        ax.grid(True)
                                        ax.set_box_aspect(1)
    
                                else:
                                    shadowflag = 1 
    
                            #writeiv if block does not work
                            if writeiv == 1:
                                f2.savefig('video_frame2.png')
                            
                            if plotflag == 1:
                                print("at flightpath plot")
                                fh = plt.subplot(1, 3, 3)
                                plt.plot(hstpbestarray[:, 0], hstpbestarray[:, 1], '-ro', markersize=10)
                                plt.xlabel('Flight Path Number, cm', fontsize=fontsizexl)
                                plt.ylabel('Best Step Size, cm', fontsize=fontsizexl)
                                ax = plt.gca()
                                ax.tick_params(axis='both', which='minor', direction='in', width=1.5)
                                ax.tick_params(axis='both', which='major', direction='in', width=1.5)
                                ax.xaxis.set_minor_locator(AutoMinorLocator())
                                ax.yaxis.set_minor_locator(AutoMinorLocator())
                                ax.grid(True)
                                ax.set_box_aspect(1)
                                

                            # write good path proportion, average diviation angle 
                            # and shadowed or not info 
                            # for every coverage angle with various beam position/direction
                            
                        # Compute the statistical values
                        stat1_row = [divang, posx, posang, len(angPZ_A1_5Array) / len(angPZ_A1_AllArray), abs(180 - np.mean(angPZ_A1_AllArray)), shadowflag]
                        
                        # Append the new row to stat1_Array
                        if stat1_Array.size == 0:
                            stat1_Array = np.array([stat1_row])
                        else:
                            stat1_Array = np.vstack((stat1_Array, stat1_row))

                        
    stat1_Array[:, 1] += cs[0]
    stat1_Array[:, 2] += angc


# plot good path proportion and average deviation angle 
# for every coverage angle with various beam position/direction
# if choose to
    if statplotflag == 1:
        print("at statplotflag if chunk")
        fontsizexl = 16
        # fontsizet = 28
        fontsizeax = 14
        
        fig3 = plt.figure(3, figsize=(22, 8))
        fig3.subplots_adjust(wspace=0.3, hspace=0.3)
        fig3.suptitle(f'Current Ratio = {CR}, B field x{Bamp[iBamp]} on {nyp} cm plane, {lamdaneutron[ilamda]} Angstroms', ha='left')
    
    
        sca1 = fig3.add_subplot(1, 2, 1, projection='3d')
        a = sca1.scatter(stat1_Array[:, 0], stat1_Array[:, 1], stat1_Array[:, 2], c=stat1_Array[:, 4], s=50, cmap='viridis')
        sca1.set_xlabel('Coverage Angle, deg', fontsize=fontsizexl)
        sca1.set_ylabel('Center Position, cm', fontsize=fontsizexl)
        sca1.set_zlabel('Center Angle, deg', fontsize=fontsizexl)
        sca1.set_title('Good Paths/All Paths', fontsize=fontsizexl)
        sca1.grid(True)
        sca1.xaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
        sca1.yaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
        sca1.zaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
    
        
        sca2 = fig3.add_subplot(1, 2, 2, projection='3d')
        b = sca2.scatter(stat1_Array[:, 0], stat1_Array[:, 1], stat1_Array[:, 2], c=stat1_Array[:, 5], s=50, cmap='viridis')
        sca2.set_xlabel('Coverage Angle, deg', fontsize=fontsizexl)
        sca2.set_ylabel('Center Position, cm', fontsize=fontsizexl)
        sca2.set_zlabel('Center Angle, deg', fontsize=fontsizexl)
        sca2.set_title('Average Deviation, deg', fontsize=fontsizexl)
        sca2.grid(True)
        sca2.xaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
        sca2.yaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
        sca2.zaxis._axinfo['grid'].update(color = 'b', linestyle = '-', linewidth = 1.5)
    
        plt.colorbar(a)
        plt.colorbar(b)
        # calculate and plot the max good path proportion and min average deviation angle vs.
        # the coverage angle
     
        angPZ_A1_min = np.zeros_like(divang_Array)
        angPZ_A1_max = np.zeros_like(divang_Array)
        
        for i in range(len(divang_Array)):
            for j in range(len(stat1_Array)):
                templistmin = []
                templistmax = []
                if stat1_Array[j,0] == divang_Array[i]:
                    templistmin.append(stat1_Array[j,4])
                    templistmax.append(stat1_Array[j,3])
            angPZ_A1_min[i] = min(templistmin)
            angPZ_A1_max[i] = max(templistmax)
            
            
        fontsizexl = 24
        fontsizet = 28
        fontsizeax = 20
        
        
        fig4, axs = plt.subplots(1, 2, figsize=(15, 7))
        fig4.suptitle(f'Current Ratio = {CR}, B field x{Bamp[iBamp]} on {nyp} cm plane, {lamdaneutron[ilamda]} Angstroms', ha='left', fontsize=fontsizet)
        
        # Plot Good Paths/All Paths
        axs[0].plot(divang_Array, angPZ_A1_max, '-ro', markersize=8, markeredgecolor='auto', markerfacecolor='r', linewidth=1.5)
        axs[0].set_xlabel('Coverage Angle, deg', fontsize=fontsizexl)
        axs[0].set_ylabel('Good Paths/All Paths', fontsize=fontsizexl)
        axs[0].tick_params(axis='both', which='major', labelsize=fontsizeax)
        axs[0].grid(True)
        axs[0].set_box_aspect(1)
        
        # Plot Average Deviation
        axs[1].plot(divang_Array, angPZ_A1_min, '-ro', markersize=8, markeredgecolor='auto', markerfacecolor='r', linewidth=1.5)
        axs[1].set_xlabel('Coverage Angle, deg', fontsize=fontsizexl)
        axs[1].set_ylabel('Average Deviation, deg', fontsize=fontsizexl)
        axs[1].tick_params(axis='both', which='major', labelsize=fontsizeax)
        axs[1].grid(True)
        axs[1].set_box_aspect(1)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
    plt.show()    
    print(CRa)
    logid.close()
