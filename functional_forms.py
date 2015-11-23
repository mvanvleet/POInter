#!/usr/local/bin/python
"""Contains functional forms used in the main fit_ff_parameters module.
"""

import numpy as np
from sympy import sqrt, exp, sin, cos, pi
import sympy as sym
from sympy.utilities import lambdify
from scipy.misc import factorial
import sys

####################################################################################################    
def get_eij(component,aij,rij,bij,functional_form='stone',slater_correction=True):
    '''Calls the relevant calc_energy routine that will compute the
    interaction energy between atoms with parameters Ai, Aj, bij, and pairwise
    distance rij. Component is an indexing number that maps as follows:

    Index       Energy Component
    0           Exchange
    1           Electrostatics
    2           Induction
    3           Delta-HF (dhf)
    4           Disperison
    5           Residual Errors
    6           Total Energy

    The slater_correction option only matters for Dispersion.
    '''

    if component == 0:
        return get_exchange_energy(aij,rij,bij,functional_form)
    elif component == 1:
        # Technically the electrostatic component also contains a multipolar
        # component to the energy, but this has already been subtracted off as
        # a hard constraint.
        return get_charge_penetration_energy(aij,rij,bij,functional_form)
    elif component == 2:
        return get_charge_penetration_energy(aij,rij,bij,functional_form)
    elif component == 3:
        return get_charge_penetration_energy(aij,rij,bij,functional_form)
    elif component == 4:
        # For dispersion, what we really should have entered is the cij
        # coefficients as a list, rather than a single aij parameter. 
        cij = aij
        return get_dispersion_energy(cij,rij,bij,slater_correction)
    elif component == 5:
        return get_charge_penetration_energy(aij,rij,bij,functional_form)
    else:
        print 'Unknown energy component!'
        sys.exit()

####################################################################################################    


####################################################################################################    
def get_exchange_energy(aij,rij,bij,functional_form='stone',k=0.001):
    '''For a given pair of atoms i and j, with associated distance rij and
    exponent bij, computes the exchange energy of the pair according to an
    exponential functional form.
    '''

    if functional_form == 'stone':
        return k*exp(-bij*(rij-aij))
    elif functional_form == 'born-mayer':
        return aij*exp(-bij*rij)
    else:
        raise NotImplementedError('Unknown functional form')
####################################################################################################    


####################################################################################################    
def get_charge_penetration_energy(aij,rij,bij,functional_form='stone',k=0.001):
    '''For a given pair of atoms i and j, with associated distance rij and
    exponent bij, computes the charge penetration portion of the
    electrostatic/induction energy of the pair according to an exponential
    functional form. The charge penetration energy is assumed to be strictly
    attractive.
    '''

    if functional_form == 'stone':
        return -k*exp(-bij*(rij-aij))
    elif functional_form == 'born-mayer':
        return -aij*exp(-bij*rij)
    else:
        raise NotImplementedError('Unknown functional form')
####################################################################################################    


####################################################################################################    
def get_multipole_energy(qi,qj,rij,bij,slater_correction=False):
    '''For a given pair of atoms i and j, with associated distance rij and
    exponent bij, computes the multipole portion of the electrostatic
    energy of the pair according to a Coulombic functional form.

    Worth mentioning is our use of the standard Tang-Toennies damping factor,
    see 
    (1) McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. A 2013, 117, 2053-066.
    (2) Tang, K. T.; Toennies, J. P. J. Chem. Phys. 1984, 80, 3726-3741.
    (3) Tang, K. T.; Peter Toennies, J. Surf. Sci. 1992, 279, L203-206.

    This damping factor depends (see ref. 3) on the form of the repulsive part
    of the potential, with y = -d/dr(ln V_repulsive).

    '''
    damping_factor = get_damping_factor(rij,bij,1,slater_correction)
    return damping_factor*(qi*qj/rij)
####################################################################################################    


####################################################################################################    
def get_dispersion_energy(Cij,rij,bij,slater_correction):
    '''Ci is array containing C6 through C12 coefficients.
    '''

    dispersion_energy = 0.0
    for i in range(6,14,2): # Compute C6, C8, C10, C12 dispersion energies
        dispersion_energy -= get_damping_factor(rij,bij,i,slater_correction)*Cij[i/2-3]/(rij**i)
        #dispersion_energy -= get_damping_factor(y*rij,i)*Cij[i/2-3]/(rij**i)

    return dispersion_energy
####################################################################################################    


####################################################################################################    
def get_damping_factor(rij,bij,n,slater_correction):
    '''Computes the standard Tang-Toennies damping factor, see
    (1) McDaniel, J. G.; Schmidt, J. R. J. Phys. Chem. A 2013, 117, 2053-066.
    (2) Tang, K. T.; Toennies, J. P. J. Chem. Phys. 1984, 80, 3726-3741.
    (3) Tang, K. T.; Peter Toennies, J. Surf. Sci. 1992, 279, L203-206.
    This damping factor depends (see ref. 3) on the form of the repulsive part
    of the potential, with x=y*r and y = -d/dr(ln V_repulsive).

    Input: x, a parameter dependent on the repulsive part of the potential
           n, the order of the damping correction
    '''

    if slater_correction:
        y = bij - (2*bij**2*rij + 3*bij)/(bij**2*rij**2 + 3*bij*rij + 3)
    else:
        y = bij
    x = y*rij

    sum = 1.0 # Account for n = 0 term
    for i in range(1,n+1):
        sum += (x**i)/factorial(i)

    return 1.0 - np.exp(-x)*sum
####################################################################################################    


####################################################################################################    
def get_exact_slater_overlap(bi,bj,rij):
    '''Exact form of the slater overlap. Only applies if bi != bj;
    otherwise a DivideByZero error will occur. 

    For the exact radial correction, formula obtained from 
    (1) Rosen, N.  Phys. Rev. Lett. 1931, 38, 255-276.  
    and a communication with Alston Misquitta. 
    '''

    u = (bi + bj)/2
    t = (bi - bj)/2

    # Note that prefactor differs slightly from AJM derivation in order to
    # keep proportional to the approximate slater overlap
    #prefactor = pi*0.25*rij**3
    #prefactor = 0.25*rij**3
    prefactor = 0.25*sqrt(bi*bj)**3
    #prefactor = 1

    term1 = (exp(t*rij) - exp(-t*rij))*(rij**2 + 2*rij/u + 2/u**2)
    term2 = -exp(t*rij)*(rij**2 - 2*rij/t + 2/t**2)
    term3 = exp(-t*rij)*(rij**2 + 2*rij/t + 2/t**2)
    polynomial = 1/(t*u*rij)*(term1 + term2 + term3)

    # Sqrt to account for the fact that this will be computed twice.
    return prefactor*polynomial
####################################################################################################    


####################################################################################################    
def get_approximate_slater_overlap(bij,rij,normalized=True):
    '''Computes the approximate form of the slater overlap, which is only
    formally exact for cases where bi=bj.
    '''

    #normalized = True

    if normalized:
        return ((bij*rij)**2)/3 + bij*rij + 1
    else:
        return rij**2/bij + 3*rij/bij**2 + 3/bij**3
    
####################################################################################################    


####################################################################################################    
## LIST OF SPHERICAL HARMONIC FUNCTIONS ##
# http://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

# L=0 Spherical Harmonics: S orbital
def y00(theta, phi):  return 1  #s

# L=1 Spherical Harmonics: Pz, Py, and Px orbitals, respectively
def y10(theta, phi):  return cos(phi)             #p_z
def y11s(theta, phi): return sin(theta)*sin(phi)  #p_y
def y11c(theta, phi): return cos(theta)*sin(phi)  #p_x  

# L=2 Spherical Harmonics: Dz2, Dyz, Dxz, Dxy, Dx2-y2 orbitals, respectively
def y20(theta, phi):  return 0.5*(3*cos(phi)**2 - 1)                #d_z2
def y21s(theta, phi): return np.sqrt(3)*sin(2*pi)*sin(theta)        #d_yz
def y21c(theta, phi): return np.sqrt(3)*sin(2*phi)*cos(theta)       #d_xz
def y22s(theta, phi): return np.sqrt(0.75)*sin(phi)**2*sin(2*theta) #d_xy
def y22c(theta, phi): return np.sqrt(0.75)*sin(phi)**2*cos(2*theta) #d_x2-y2

    ## y30 = cos(phi)*(2*cos(phi)**2 -3*sin(phi)**2)
    ## y32 = sin(phi)*sin(2*phi)*cos(2*theta)

sym_real_sph_harm = { 'y00' : y00 ,\
                  'y10' : y10 ,\
                  'y11s': y11s,\
                  'y11c': y11c,\
                  'y20' : y20 ,\
                  'y21s': y21s,\
                  'y21c': y21c,\
                  'y22s': y22s,\
                  'y22c': y22c, }

# Numpy routines of spherical harmonics
theta, phi = sym.symbols('theta phi')
np_real_sph_harm = {
        k : lambdify((theta,phi),v(theta,phi),modules='numpy') 
            for (k,v) in sym_real_sph_harm.items() }

####################################################################################################    



####################################################################################################    
def get_anisotropic_ai(sph_harm,a,Aangular,rij,theta,phi):
    '''For a given pair of atoms i and j, with associated distance (rad),
    azimuthal angle (theta), and polar angle (phi), computes the anisotropic
    exchange energy of the pair with respect to the functional form given
    below.

    Input:
        a, a float; scaling factor for anisotropic exchange coefficient
        #Arad, a list; radial parameters
        Aangular, a list; angular parameters
        rij, a float; pairwise distance between atoms (units of bohr)
        theta, a float; angle theta, user defined, range 0 - 2*pi
        phi, a float; angle phi, user defined, range 0 - pi

    Output:
        A, a float; A(r,theta,phi) coefficient
    '''

    A = 1

    # Determine if theta, phi values are numerical arrays or symbols, as we
    # need to use different function dictionaries to handle the two data types
    if type(theta).__module__ == np.__name__:
        real_sph_harm = np_real_sph_harm
    else:
        real_sph_harm = sym_real_sph_harm


    for (aang,yn) in zip(Aangular,sph_harm):
        A += aang*real_sph_harm[yn](theta,phi)

    A *= a

    return A
####################################################################################################    


####################################################################################################    
def weight(energy,Eff_mu=0.005,Eff_kt=0.001):
    '''Uses the Fermi-Dirac distribution to yield a relative weighting
    function given an energy and effective mu and KT parameters.

    energy should be a 1d array; Eff_mu and Eff_kt are floats.
    Output is a 1d array of the same size as energy.
    '''
    # Ensure energy input is a numpy array
    energy = np.array(energy)
    return 1./(np.exp((energy-Eff_mu)/Eff_kt)+1.)
####################################################################################################    


