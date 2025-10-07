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
from scipy.stats import norm, lognorm

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
    # print(box)
    # print(f"np.amax(box)/2 = {np.amax(box)/2}")


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
def scattering_poly(unitscattering, q, R0, sigma, Nsamples, distribution='gaussian'):
    
    volume_bounding_box = box[0]*box[1]*box[2]

    volume_of_cloud = volume_bounding_box * filling_factor
    
    #particle dimension(s) that shall be rescaled/fitted
    selected_dimension_bounding_box = np.amax(box) #maximal edge length for instance
    
    #Random number generator
    if distribution=='gaussian':
        radii = np.random.normal(R0, abs(sigma), Nsamples) #number-weighted
    #lognormal distribution
    elif distribution=='lognormal':
        #parameters of lognormal-distributed particle size
        E = R0 #expectation value
        VAR = sigma**2 #variance
        #parameters of normal-distributed log(particle size)
        sigma2 = np.sqrt(np.log(VAR/E**2 + 1))
        mu = np.log(E) - (sigma2**2)/2
        radii = np.random.lognormal(mu, sigma2, Nsamples) #number-weighted
    else:
        raise ValueError(f'distribution can be either gaussian or lognormal (got >{distribution})<')
    
    
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


def lognormal_pdf(mean, std, N=1000, k=10):
    """
    Generate a lognormal PDF on a logarithmically spaced grid.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution (of ln(x)).
    std : float
        Standard deviation of the underlying normal distribution (of ln(x)).
    N : int
        Number of points in the grid.
    k : float
        Range multiplier for how many sigma to include.

    Returns
    -------
    x : ndarray
        Log-spaced grid.
    pdf : ndarray
        Corresponding lognormal PDF values.
    """

    sigma = np.sqrt(np.log(std**2/mean**2 + 1))
    mu = np.log(mean) - (sigma**2)/2
    
    mode = np.exp(mu - sigma**2) # sample around mode for symmetry

    xmin = np.exp(np.log(mode) - k*sigma)
    xmax = np.exp(np.log(mode) + k*sigma)

    # logarithmically spaced grid
    x = np.logspace(np.log10(xmin), np.log10(xmax), N)

    # Scipy uses shape parameter = sigma, scale = exp(mu)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    # normalize numerically to ensure ∫pdf dx = 1
    # r and pdf may contain NaNs
    mask = np.isfinite(pdf) * np.isfinite(x)  # True for finite values
    pdf /= np.trapz(pdf[mask], x[mask])

    return x, pdf


def normal_pdf(mean, std, N=1000, k=10):
    """
    Generate a normal PDF on a linearly spaced grid.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std : float
        Standard deviation of the normal distribution.
    N : int
        Number of points in the grid.
    k : float
        Range multiplier for how many sigma to include.

    Returns
    -------
    x : ndarray
        Linearly spaced grid.
    pdf : ndarray
        Corresponding normal PDF values.
    """

    xmin = np.max([0, mean - k*std])  # avoid negative radii
    xmax = mean + k*std

    # linearly spaced grid
    x = np.linspace(xmin, xmax, N)

    pdf = norm.pdf(x, loc=mean, scale=std)

    # normalize numerically to ensure ∫pdf dx = 1
    # r and pdf may contain NaNs
    mask = np.isfinite(pdf) * np.isfinite(x)  # True for finite values
    pdf /= np.trapz(pdf[mask], x[mask])

    return x, pdf


def scattering_poly_pdf(unitscattering, q, R0, sigma, Nsamples=1000, distribution='gaussian'):
    
    volume_bounding_box = box[0]*box[1]*box[2]

    volume_of_cloud = volume_bounding_box * filling_factor
    
    #particle dimension(s) that shall be rescaled/fitted
    selected_dimension_bounding_box = np.amax(box) #maximal edge length for instance
    
    #Random number generator
    if distribution=='gaussian':
        radii, pdf = normal_pdf(R0, sigma, N=Nsamples)
    #lognormal distribution
    elif distribution=='lognormal':
        radii, pdf = lognormal_pdf(R0, sigma, N=Nsamples)
    else:
        raise ValueError(f'distribution can be either gaussian or lognormal (got >{distribution})<')
    
    # normalize pdf to account for numerical inaccuracy
    # r and pdf may contain NaNs
    mask = np.isfinite(pdf) * np.isfinite(radii)  # True for finite values
    # Normalize
    pdf /= np.trapz(pdf[mask], radii[mask])
    
    qknown = unitscattering[:, 0] 
    Ilog   = np.log(unitscattering[:,1])
    
    result = np.zeros_like(q)
    
    #Summing up single-particle profiles
    for i, radius in enumerate(radii):
        rscaled = radius / (selected_dimension_bounding_box / 2) 
        qscaled = qknown / rscaled
        Iscaled = np.exp(np.interp(q, qscaled, Ilog)) * (volume_of_cloud * rscaled**3)**2
        result += Iscaled * pdf[i]
    
    return np.column_stack((q, result))


def scattering_poly(unitscattering, q, R0, sigma, Nsamples, distribution='gaussian'):
    
    volume_bounding_box = box[0]*box[1]*box[2]

    volume_of_cloud = volume_bounding_box * filling_factor
    
    #particle dimension(s) that shall be rescaled/fitted
    selected_dimension_bounding_box = np.amax(box) #maximal edge length for instance
    
    #Random number generator
    if distribution=='gaussian':
        radii = np.random.normal(R0, abs(sigma), Nsamples) #number-weighted
    #lognormal distribution
    elif distribution=='lognormal':
        #parameters of lognormal-distributed particle size
        E = R0 #expectation value
        VAR = sigma**2 #variance
        #parameters of normal-distributed log(particle size)
        sigma2 = np.sqrt(np.log(VAR/E**2 + 1))
        mu = np.log(E) - (sigma2**2)/2
        radii = np.random.lognormal(mu, sigma2, Nsamples) #number-weighted
    else:
        raise ValueError(f'distribution can be either gaussian or lognormal (got >{distribution})<')
    
    
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
def scattering_model(unitscattering, q, N_C, R0, sigma, c0, distribution='gaussian'):

    # result = scattering_poly(unitscattering, q, R0, sigma, 3000, distribution) #by default, we add 3000 single-particle profiles
    result = scattering_poly_pdf(unitscattering, q, R0, sigma, 1000, distribution) #by default, we add 3000 single-particle profiles
    
    result[:,1] *= N_C   #Constant which containes information about number concentration and electron contrast
    
    result[:,1] += c0    #constant scattering background
    
    return result



#Chi_squared
#params - fit parameters
#data - experimental data which we intend to fit
def chi_squared(params, data, unitscattering, distribution):
    
    N_C, R0, sigma, c0 = params
    
    q = data[:,0]
    I = data[:,1]
    Ierr = data[:,2]
    
    I_theo = scattering_model(unitscattering, q, N_C, R0, sigma, c0, distribution)[:,1]
    
    Chi = (1/(len(I_theo)-len(params))) * np.sum(((I - I_theo) / Ierr)**2)


    return Chi


#Chi_squared
#params - fit parameters
#data - experimental data which we intend to fit
def chi_squared_model(params, data, model, model_args, distribution):
    
    N_C, R0, sigma, c0, *model_params = params # variable number of model parameters
    
    q = data[:,0]
    I = data[:,1]
    Ierr = data[:,2]
    
    if isinstance(model_args, dict):
        unitscattering = model(*model_params, **model_args)  # Call the user-defined model function
    elif isinstance(model_args, tuple):
        unitscattering = model(*model_params, *model_args)  # Call the user-defined model function
    else:
        raise ValueError("model_args must be a dictionary or a tuple")
    

    I_theo = scattering_model(unitscattering, q, N_C, R0, sigma, c0, distribution)[:,1]
    
    Chi = (1/(len(I_theo)-len(params))) * np.sum(((I - I_theo) / Ierr)**2)


    return Chi


def neg_log_likelihood(theta, data, model, model_args, distribution):
    """
    theta: array of parameters [N_C, R0, sigma, c0, ...model_params, log_f]
    data: (n,3) array with columns [q, I, Ierr]
    model: function for unit scattering
    distribution: type of distribution (e.g. Gaussian)
    """

    # Unpack parameters
    N_C, R0, sigma, c0, log_f, *model_params  = theta

    if isinstance(model_args, dict):
        if len(model_params) > 0:
            unitscattering = model(*model_params, **model_args)  # Call the user-defined model function
        else:
            unitscattering = model(**model_args)  # Call the user-defined model function
    elif isinstance(model_args, tuple):
        if len(model_params) > 0:
            unitscattering = model(*model_params, *model_args)  # Call the user-defined model function
        else:
            unitscattering = model(*model_args)  # Call the user-defined model function
    else:
        raise ValueError("model_args must be a dictionary or a tuple")

    q = data[:, 0]
    I = data[:, 1]
    Ierr = data[:, 3]

    # Compute theoretical intensity
    I_Mod = scattering_model(unitscattering, q, N_C, R0, sigma, c0, distribution)[:, 1]
    # I_Mod = c0 + N_C * scattering_poly(unitscattering, q, R0, sigma, 3000, distribution)[:, 1]

    # Variance model (same form as linear example, adapt if needed)
    sigma2 = Ierr**2 + I_Mod**2 * np.exp(2 * log_f)

    # negative Gaussian log-likelihood
    return - (-0.5 * np.sum((I - I_Mod) ** 2 / sigma2 + np.log(sigma2)))