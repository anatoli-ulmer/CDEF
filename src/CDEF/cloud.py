#!/usr/bin/env python
# coding: utf-8
#
#  (C) Copyright 2022 Physikalisch-Technische Bundesanstalt (PTB)
#  Jerome Deumer
#  
#   This file is part of CDEF.
#
#   CDEF is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   CDEF is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with CDEF.  If not, see <https://www.gnu.org/licenses/>.
#
#


import numpy as np

from scipy.spatial import distance

#Halton
from . import debyer

#Sobol
from . import sobol_seq

#Examples of self-scripted functions to generate point clouds without using any stl-files

#Finds row indices 
def find_index(M,a):
    return np.where(np.all(M==a,axis=1))[0]
    


#Deleting scatterer coordinates(row of matrix) of cloud(matrix)
def delete_point(M, point):
    index_array = find_index(M, point)
    M = np.delete(M, index_array, axis=0)
    return M, len(index_array)



#Bounding box of point cloud
def bounding_box(pt):
    x_coordinate=pt[:,0] 
    y_coordinate=pt[:,1]
    z_coordinate=pt[:,2]
    x_size=np.amax(x_coordinate)-np.amin(x_coordinate)
    y_size=np.amax(y_coordinate)-np.amin(y_coordinate)
    z_size=np.amax(z_coordinate)-np.amin(z_coordinate)
    return abs(np.array([x_size, y_size, z_size])) 



#Max edge length of bounding box
def max_dimensionens(pt):
    box=bounding_box(pt)
    return np.amax(box)


#Simple cubic point cloud with scatterers' coordinates as rows
#N - number of scatterers
#K - edge length
#qr - quasi-random True/False (False means true-random)
#kwhalton - halton True/False (False means Sobol algorithsm in case qr=False)
#dim - dimension of cloud
def cube(N=30000, K=2, qr=True, kwhalton=False, dim=3):
    
    global filling_factor
    filling_factor = 1

    #quasi-random, Sobol
    if qr==True and kwhalton==False:
        #soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=False) #no scrambled sobol
        #return K*soboleng.draw(N, dtype=float).numpy()-(K/2)
        return K*sobol_seq.i4_sobol_generate(3, N)-(K/2)
    
    #quasi-random, Halton
    elif qr==True and kwhalton==True:
        return K*debyer.kwhalton(N,dim)-(K/2)
    
    #true-random
    else:
        return K*np.random.rand(N,dim)-(K/2)
    
    
    
#Cubic point cloud as periodic lattice   
#N - number of scatterers
#K - edge length
def cube_lattice(N=30000, K=20):
    n = N**(1/3)
    M = []
    for i in np.linspace(-K/2, K/2, n+1):
        for j in np.linspace(-K/2, K/2, n+1):
            for k in np.linspace(-K/2, K/2, n+1):
                a = [i,j,k]
                M.append(a)
    
    global filling_factor
    filling_factor = 1
    
    return np.array(M)


#Cube with rounded edges
#L - face-to-face distance
#R - radius of edge curvature relative to L/2: 0 < R < L/2
def cube_round_edges(N=30000, L=2, R=0.2, qr=True, kwhalton=False):   
    
    #ideal cube
    pt = cube(N, L, qr, kwhalton, 3)
    pt2 = pt
    
    for point in pt:
        
        if (0 < point[0] and 0 < point[1]):
            M = [(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[1]-M[1])**2)
            if (M[0] < point[0] and M[1] < point[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
        
        elif (0 < point[0] and point[1] < 0):
            M = [(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[1]-M[1])**2)
            if (M[0] < point[0] and point[1] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        elif (0 < point[1] and point[0] < 0):
            M = [-(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[1]-M[1])**2)
            if (M[1] < point[1] and point[0] < M[0] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        else:
            M = [-(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[1]-M[1])**2)
            if (point[0] < M[0] and point[1] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
                
        if (0 < point[1] and 0 < point[2]):
            M = [(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[1]-M[0])**2 + (point[2]-M[1])**2)
            if (M[0] < point[1] and M[1] < point[2] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
        
        elif (0 < point[1] and point[2] < 0):
            M = [(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[1]-M[0])**2 + (point[2]-M[1])**2)
            if (M[0] < point[1] and point[2] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        elif (0 < point[2] and point[1] < 0):
            M = [-(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[1]-M[0])**2 + (point[2]-M[1])**2)
            if (M[1] < point[2] and point[1] < M[0] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        else:
            M = [-(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[1]-M[0])**2 + (point[2]-M[1])**2)
            if (point[1] < M[0] and point[2] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
        
        
        if (0 < point[0] and 0 < point[2]):
            M = [(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[2]-M[1])**2)
            if (M[0] < point[0] and M[1] < point[2] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
        
        elif (0 < point[0] and point[2] < 0):
            M = [(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[2]-M[1])**2)
            if (M[0] < point[0] and point[2] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        elif (0 < point[2] and point[0] < 0):
            M = [-(1-R)*(L/2), (1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[2]-M[1])**2)
            if (M[1] < point[2] and point[0] < M[0] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
                
        else:
            M = [-(1-R)*(L/2), -(1-R)*(L/2)]
            d = np.sqrt((point[0]-M[0])**2 + (point[2]-M[1])**2)
            if (point[0] < M[0] and point[2] < M[1] and R*(L/2) < d):
                pt2 = delete_point(pt2, point)[0]
                continue
    
    global filling_factor
    filling_factor = len(pt2) / len(pt)
    print('filling factor = %s' % filling_factor)

    return pt2


    

    
#Cube with cut edges 
#K - face-to-face distance
#T - truncation factor
#for further information refer to assiciated paper
def cube_cut_edges(N=30000, K=2, qr=True, kwhalton=False, T = 0.9):
    
    #ideal cube
    pt = cube(N, K, qr, kwhalton, 3)
    
    #Hessian normal form
    #Normal unit vectors of the 12 sectional planes:
    normalization_factor = 1 / np.sqrt(2)
    n1 = normalization_factor * np.array([1,0,1])
    n2 = normalization_factor * np.array([1,0,-1])
    n3 = normalization_factor * np.array([0,1,1])
    n4 = normalization_factor * np.array([0,1,-1])
    n5 = normalization_factor * np.array([-1,0,1])
    n6 = normalization_factor * np.array([-1,0,-1])
    n7 = normalization_factor * np.array([0,-1,1])
    n8 = normalization_factor * np.array([0,-1,-1])
    n9 = normalization_factor * np.array([-1,-1,0])
    n10 = normalization_factor * np.array([1,1,0])
    n11 = normalization_factor * np.array([1,-1,0])
    n12 = normalization_factor * np.array([-1,1,0])
    normal_vectors = np.vstack([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12])

    #edge length
    L = max_dimensionens(pt)
    print('edge length L = ', L)
    #Support vectors of the 12 sectional planes:
    s1 = n1 * L / np.sqrt(2) * T
    s2 = n2 * L / np.sqrt(2) * T
    s3 = n3 * L / np.sqrt(2) * T
    s4 = n4 * L / np.sqrt(2) * T
    s5 = n5 * L / np.sqrt(2) * T
    s6 = n6 * L / np.sqrt(2) * T
    s7 = n7 * L / np.sqrt(2) * T
    s8 = n8 * L / np.sqrt(2) * T
    s9 = n9 * L / np.sqrt(2) * T
    s10 = n10 * L / np.sqrt(2) * T
    s11 = n11 * L / np.sqrt(2) * T
    s12 = n12 * L / np.sqrt(2) * T
    support_vectors = np.vstack([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12])
    
    wuerfel = pt
    for point in pt:
        for i in range(0, len(normal_vectors)):
            #Distance of scattering point to each sectional plane
            d = float(np.sum(normal_vectors[i] * (point - support_vectors[i])))
            if 0 < d:
                wuerfel = delete_point(wuerfel, point)[0]
                break
    
    global filling_factor
    filling_factor = len(wuerfel) / N
    print('filling factor = %s' % filling_factor)
    
    return wuerfel



#Distance of a point to the origin
def Distance(args):
    result = 0
    for arg in args:
        result += arg**2
    return np.sqrt(result)



#Spherical cloud
def sphere(N=30000, diameter=2, qr=True, kwhalton=False, dim=3):
    
    K = diameter
    pt = cube(N, K, qr, kwhalton, dim)
    
    pt_new = pt
    for point in pt:
        d = Distance(point)
        if K/2 < d:
            pt_new = delete_point(pt_new, point)[0]
    
    global filling_factor
    filling_factor = len(pt_new) / N
    print('filling factor = %s' % filling_factor)
    
    return pt_new



#Spherical cloud with structure of a periodic lattice
#K - diameter
def spherical_lattice(N=30000, diameter=2):
    
    K = diameter
    pt = cube_lattice(N, K)
    
    pt_new = pt
    for point in pt:
        d = Distance(point) 
        if K/2 < d:
            pt_new = delete_point(pt_new, point)[0]
    
    global filling_factor
    filling_factor = len(pt_new) / N
    print('filling factor = %s' % filling_factor)
    
    return pt_new



#Dumbbell cloud (two equivalent spheres touching each other at one point)
def dumbbell(N=30000, radius_sphere=2, qr=True, kwhalton=False):
    
    K = 2 * radius_sphere
    
    pt_cube1 = cube(N, K, qr, kwhalton, dim=3)
    pt_cube1[:,2] += radius_sphere
    pt_cube2 = cube(N, K, qr, kwhalton, dim=3)
    pt_cube2[:,2] -= radius_sphere
    
    pt1 = pt_cube1
    for point in pt_cube1:
        #Distance from the origin
        d = np.sqrt(point[0]**2 + point[1]**2 + (point[2]-radius_sphere)**2)
        if radius_sphere < d:
            pt1 = delete_point(pt1, point)[0]
    
    pt2 = pt_cube2
    for point in pt_cube2:
        #Distance from the origin
        d = np.sqrt(point[0]**2 + point[1]**2 + (point[2]+radius_sphere)**2)
        if radius_sphere < d:
            pt2 = delete_point(pt2, point)[0]
                
    filling_factor1 = len(pt1) / N
    filling_factor2 = len(pt2) / N
    global filling_factor
    filling_factor = (filling_factor1 + filling_factor1) / 2
    print('filling factor = %s' % filling_factor)
    
    return np.concatenate((pt1, pt2), axis=0)



#Vertical cuboid
#aspect ratio = 1 -> cube, aspect ratio > 1 -> cuboid
#N - number of scatterers
#K - edge length
def vertical_cuboid(N, K, aspect_ratio, qr=True, kwhalton=False):
    
    global filling_factor
    filling_factor = 1
    
    #quasi-random, Sobol
    if qr==True and kwhalton==False:
        M = K*sobol_seq.i4_sobol_generate(3, N)-(K/2)
        Mx = M[:, 0]
        My = M[:, 1]
        Mz = aspect_ratio * M[:, 2]
        return np.vstack((Mx, My, Mz)).T
    
    #quasi-random, Halton
    elif qr==True and kwhalton==True:
        M = K*debyer.kwhalton(N,3)-(K/2)
        Mx = M[:, 0]
        My = M[:, 1]
        Mz = aspect_ratio * M[:, 2]
        return np.vstack((Mx, My, Mz)).T
    
    #true-random
    else:
        x = K*np.random.rand(N,1)-(K/2)
        x = x[:,0]
        y = K*np.random.rand(N,1)-(K/2)
        y = y[:,0]
        z = aspect_ratio*(K*np.random.rand(N,1)-(K/2))
        z = z[:,0]
        return np.vstack((x,y,z)).T


    
#Horizontal cuboid
#aspect ratio = 1 -> cube, aspect ratio > 1 -> cuboid
#N - number of scatterers
#K - edge length
def cuboid_lateral(N, K, aspect_ratio, qr=True, kwhalton=False):
    
    global filling_factor
    filling_factor = 1
    
    #quasi-random, Sobol
    if qr==True and kwhalton==False:
        M = K*sobol_seq.i4_sobol_generate(3, N)-(K/2)
        Mx = M[:, 0]
        My = aspect_ratio * M[:, 1]
        Mz = M[:, 2]
        return np.vstack((Mx, My, Mz)).T
    
    elif qr==True and kwhalton==True:
        M = K*debyer.kwhalton(N,3)-(K/2)
        Mx = M[:, 0]
        My = aspect_ratio * M[:, 1]
        Mz = M[:, 2]
        return np.vstack((Mx, My, Mz)).T
    
    #true-random
    else:
        x = K*np.random.rand(N,1)-(K/2)
        x = x[:,0]
        y = aspect_ratio*(K*np.random.rand(N,1)-(K/2))
        y = y[:,0]
        z = K*np.random.rand(N,1)-(K/2)
        z = z[:,0]
        return np.vstack((x,y,z)).T

    

#Cylinder cloud
#aspect ratio = length / radius
def cylinder(N, aspect_ratio, K=2, qr=True, kwhalton=False):
    
    #aspect ratio = length / radius -> length / diameter
    aspect_ratio = aspect_ratio / 2 
    
    pt = vertical_cuboid(N, K, aspect_ratio, qr, kwhalton)
    
    pt_new = pt
    for point in pt:
        d = np.sqrt(point[0]**2 + point[1]**2)
        if K/2 < d:
            pt_new = delete_point(pt_new, point)[0]
    
    global filling_factor
    filling_factor = len(pt_new) / N
    print('filling factor = %s' % filling_factor)

    return pt_new



#R - cylinder radius
#L - cylinder length
def cylinder_lateral(N, R, L, qr=True, kwhalton=False):
    
    K = 2*R 
    aspect_ratio = L / K
    pt = cuboid_lateral(N, K, aspect_ratio, qr, kwhalton)

    pt_new = pt
    for point in pt:
        d = np.sqrt(point[0]**2 + point[2]**2)
        if R < d:
            pt_new = delete_point(pt_new, point)[0]
    
    global filling_factor
    filling_factor = len(pt_new) / N
    print('filling factor = %s' % filling_factor)
   
    return pt_new


#computes pair distance histogram of cloud
def histo(cloud, Density=True):
    distances = distance.cdist(cloud, cloud, 'euclidean')
    distances = distances[0 <= distances]
    hist, bins = np.histogram(distances, bins='auto', density=Density)
    bin_center = (bins[1:] + bins[:-1]) / 2
    return bin_center, hist

