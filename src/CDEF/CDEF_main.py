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

#debyer module
from . import debyer
from .debyer import read_stl

#module including different point clouds
from . import cloud
from . import sobol_seq


#Building cloud out arbitrary stl-files
#mesh as 3D tensor with rows as edge point coordinates
def bounding_box_mesh(mesh):
    xcoord = mesh[:,:,0].flatten()
    ycoord = mesh[:,:,1].flatten()
    zcoord = mesh[:,:,2].flatten()
    x_size=np.amax(xcoord)-np.amin(xcoord)
    y_size=np.amax(ycoord)-np.amin(ycoord)
    z_size=np.amax(zcoord)-np.amin(zcoord)
    return abs(np.array([x_size, y_size, z_size])) 


#point cloud and corresponding filling fraction of arbitrary particle mesh
#N - number of scatterers
#returns cloud + filling factor
def stl_cloud(mesh, N, sequence='halton'):
    
    generators = {
            'halton': lambda N : debyer.kwhalton(N, 3),
            'sobol':  lambda N : sobol_seq.i4_sobol_generate(3, N),
            'random': lambda N : np.random.rand(N, 3)}

    generator = generators[sequence]

    cube = generator(N)
    
    pointcloud = debyer.makepoints(mesh, cube) 
    
    global filling_factor # uargs
    filling_factor = len(pointcloud) / N
    
    return pointcloud


#Computation of single-particle scattering profile using numerical Debye integration
#q-interval goes from q_ini to q_end in q_step steps
#nob - number of bins of the pair distance histogram
#bin_range (init -> end) of empty bins
def scattering_mono(pt, q_ini = 0.001, q_end = 100, q_step = 0.01 , selfcorrelation=True, rbins=1000, cutoff = 0, sinc_damp = 0, zerobinstart = 0, zerobinend = -1):
    
    #box needs to be global since it will be used by "scattering_poly"
    global box 
    box = cloud.bounding_box(pt)
    
    #using Debyer function to speed up calculation
    #data = debyer.debyer_ff(pt, nob, bin_init, bin_end, q_ini, q_end, q_step)
    data = debyer.debyer_ff(pt, q_ini, q_end, q_step, 
            selfcorrelation = selfcorrelation, rbins=rbins, 
            cutoff = cutoff, sinc_damp = sinc_damp,
            zerobinstart = zerobinstart, zerobinend = zerobinend)   
 
    return data

#Poly-disperse scattering profiles according to specific size distribution with mean R0 and std sigma
#unitscattering - single-particle scattering profile
#q - q-vector on which values scattering_poly will be evaluated
#Nsamples - number of single-particle profiles that shall be sumed up
def scattering_poly(unitscattering, q, R0, sigma, Nsamples, Gaussian=True):
    
    volume_bounding_box = box[0]*box[1]*box[2]

    volume_of_cloud = volume_bounding_box * filling_factor
    
    #particle dimension(s) that shall be rescaled/fitted
    selected_dimension_bounding_box = np.amax(box) #maximal edge length for instance
    
    #Random number generator
    if Gaussian==True:
        radii = np.random.normal(R0, abs(sigma), Nsamples) #number-weighted
    #lognormal distribution
    else:
        #parameters of lognormal-distributed particle size
        E = R0 #expectation value
        VAR = sigma**2 #variance
        #parameters of normal-distributed log(particle size)
        sigma2 = np.sqrt(np.log(VAR/E**2 + 1))
        mu = np.log(E) - (sigma2**2)/2
        radii = np.random.lognormal(mu, sigma2, Nsamples) #number-weighted
    
    
    qknown = unitscattering[:, 0] 
    Ilog   = np.log(unitscattering[:,1])
    
    result = np.zeros_like(q) 
    
    #Summing up single-particle profiles
    for radius in radii:
        rscaled = radius / (selected_dimension_bounding_box / 2) 
        qscaled = qknown / rscaled
        Iscaled = np.exp(np.interp(q, qscaled, Ilog)) * (volume_of_cloud * rscaled**3)**2
        result += Iscaled 
    
    result = result / Nsamples
    
    return np.column_stack((q, result))



#Model function with parameters N_C, R0, sigma, c0
def scattering_model(unitscattering, q, N_C, R0, sigma, c0, Gaussian=True):

    result = scattering_poly(unitscattering, q, R0, sigma, 3000, Gaussian) #by default, we add 3000 single-particle profiles
    
    result[:,1] *= N_C   #Constant which containes information about number concentration and electron contrast
    
    result[:,1] += c0    #constant scattering background
    
    return result



#Chi_squared
#params - fit parameters
#data - experimental data which we intend to fit
def chi_squared(params, data, unitscattering, Gaussian=True):
    
    N_C, R0, sigma, c0 = params
    
    q = data[:,0]
    I = data[:,1]
    Ierr = data[:,2]
    
    I_theo = scattering_model(unitscattering, q, N_C, R0, sigma, c0, Gaussian)[:,1]
    
    Chi = (1/(len(I_theo)-len(params))) * np.sum(((I - I_theo) / Ierr)**2)


    return Chi

